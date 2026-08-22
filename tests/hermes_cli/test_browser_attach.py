"""Tests for hermes_cli.browser_attach — Electron discovery + session registry."""

from __future__ import annotations

import json
import os

import pytest

from hermes_cli import browser_attach as ba


# ── Electron detection ─────────────────────────────────────────────


class TestIsElectronExecutable:
    def test_app_asar_next_to_binary(self, tmp_path):
        exe = tmp_path / "MyApp" / "myapp"
        resources = tmp_path / "MyApp" / "resources"
        resources.mkdir(parents=True)
        (resources / "app.asar").write_bytes(b"")
        exe.parent.mkdir(exist_ok=True)
        exe.write_bytes(b"")
        assert ba.is_electron_executable(str(exe)) is True

    def test_unpacked_app_package_json(self, tmp_path):
        exe = tmp_path / "MyApp" / "myapp"
        app_dir = tmp_path / "MyApp" / "resources" / "app"
        app_dir.mkdir(parents=True)
        (app_dir / "package.json").write_text("{}")
        exe.write_bytes(b"")
        assert ba.is_electron_executable(str(exe)) is True

    def test_macos_bundle_layout(self, tmp_path):
        macos_dir = tmp_path / "Obsidian.app" / "Contents" / "MacOS"
        resources = tmp_path / "Obsidian.app" / "Contents" / "Resources"
        macos_dir.mkdir(parents=True)
        resources.mkdir(parents=True)
        (resources / "app.asar").write_bytes(b"")
        exe = macos_dir / "Obsidian"
        exe.write_bytes(b"")
        assert ba.is_electron_executable(str(exe)) is True

    def test_plain_binary_is_not_electron(self, tmp_path):
        exe = tmp_path / "bin" / "vim"
        exe.parent.mkdir()
        exe.write_bytes(b"")
        assert ba.is_electron_executable(str(exe)) is False

    def test_empty_path(self):
        assert ba.is_electron_executable("") is False


class TestCmdlineParsing:
    def test_debug_port_extracted(self):
        assert ba.debug_port_from_cmdline(["/x/app", "--remote-debugging-port=9333"]) == 9333

    def test_port_zero_rejected(self):
        assert ba.debug_port_from_cmdline(["/x/app", "--remote-debugging-port=0"]) is None

    def test_no_port(self):
        assert ba.debug_port_from_cmdline(["/x/app", "--flag"]) is None

    def test_child_detected_by_type_flag(self):
        assert ba.is_electron_child(["/x/app", "--type=renderer"]) is True
        assert ba.is_electron_child(["/x/app", "--type=gpu-process"]) is True

    def test_main_process_is_not_child(self):
        assert ba.is_electron_child(["/x/app", "--remote-debugging-port=1"]) is False


class TestNames:
    def test_display_name_from_bundle(self):
        assert (
            ba.app_display_name("/Applications/Obsidian.app/Contents/MacOS/Obsidian", "x")
            == "Obsidian"
        )

    def test_display_name_fallback(self):
        assert ba.app_display_name("/usr/lib/slack/slack", "slack") == "slack"

    def test_session_slug_sanitizes(self):
        assert ba.session_slug("Visual Studio Code") == "visual-studio-code"

    def test_session_slug_empty_fallback(self):
        assert ba.session_slug("///") == "app"

    def test_session_slug_always_matches_browser_exec_grammar(self):
        # browser_exec rejects sessions not matching _SESSION_RE (alnum
        # first char) — a slug that fails it would be registered but
        # unreachable, silent dead state.
        from tools.browser_use_cli import _SESSION_RE

        for raw in ("_private app", "-dash", "My Obsidian", "app!", "日本語アプリ"):
            assert _SESSION_RE.match(ba.session_slug(raw)), raw


# ── Registry round-trip ────────────────────────────────────────────


@pytest.fixture
def registry_home(tmp_path, monkeypatch):
    monkeypatch.setattr(ba, "registry_path", lambda: str(tmp_path / "browser-sessions.json"))
    return tmp_path


class TestRegistry:
    def test_round_trip(self, registry_home):
        ba.save_session_endpoint("obsidian", "http://127.0.0.1:9333", "Obsidian")
        assert ba.resolve_session_endpoint("obsidian") == "http://127.0.0.1:9333"
        sessions = ba.load_registry()
        assert sessions["obsidian"]["app"] == "Obsidian"

    def test_missing_file_is_empty(self, registry_home):
        assert ba.load_registry() == {}
        assert ba.resolve_session_endpoint("nope") is None

    def test_corrupt_file_is_empty(self, registry_home):
        with open(ba.registry_path(), "w") as fh:
            fh.write("{not json")
        assert ba.load_registry() == {}

    def test_entries_without_url_dropped(self, registry_home):
        with open(ba.registry_path(), "w") as fh:
            json.dump({"sessions": {"bad": {"app": "X"}, "good": {"cdp_url": "http://h:1"}}}, fh)
        assert set(ba.load_registry()) == {"good"}

    def test_remove(self, registry_home):
        ba.save_session_endpoint("s1", "http://127.0.0.1:1", "A")
        assert ba.remove_session_endpoint("s1") is True
        assert ba.remove_session_endpoint("s1") is False
        assert ba.resolve_session_endpoint("s1") is None

    def test_atomic_write_leaves_no_tmp(self, registry_home):
        ba.save_session_endpoint("s1", "http://127.0.0.1:1", "A")
        assert not os.path.exists(ba.registry_path() + ".tmp")


# ── browser_exec backend resolution honors the registry ───────────


class TestResolveBackendRegistry:
    def _patch_browser_tool(self, monkeypatch):
        import tools.browser_use_cli as bu_cli

        monkeypatch.setattr(
            "tools.browser_tool._get_cdp_override", lambda: "", raising=False
        )
        monkeypatch.setattr(
            "tools.browser_tool._get_cloud_provider", lambda: None, raising=False
        )
        return bu_cli

    def test_named_session_uses_registry(self, registry_home, monkeypatch):
        bu_cli = self._patch_browser_tool(monkeypatch)
        ba.save_session_endpoint("obsidian", "http://127.0.0.1:9333", "Obsidian")
        env: dict = {"BU_NAME": "obsidian"}
        assert bu_cli._resolve_backend_cdp(env, "t1", session_name="obsidian") is None
        assert env["BU_CDP_URL"] == "http://127.0.0.1:9333"
        # App session is private: no own-tab preamble tab leaked into the app.
        assert env.get(bu_cli._PRIVATE_BROWSER_SENTINEL) == "1"
        # Electron rejects Target.createTarget, which the harness's named-
        # daemon path calls — app sessions must run the default-name daemon
        # isolated via a per-session harness home instead.
        assert "BU_NAME" not in env
        assert env["BH_HOME"].endswith(os.path.join("app-sessions", "obsidian"))

    def test_ws_endpoint_uses_ws_env(self, registry_home, monkeypatch):
        bu_cli = self._patch_browser_tool(monkeypatch)
        ba.save_session_endpoint("app", "ws://127.0.0.1:9333/devtools/browser/x", "App")
        env: dict = {}
        assert bu_cli._resolve_backend_cdp(env, "t1", session_name="app") is None
        assert env["BU_CDP_WS"] == "ws://127.0.0.1:9333/devtools/browser/x"

    def test_unregistered_session_falls_through(self, registry_home, monkeypatch):
        bu_cli = self._patch_browser_tool(monkeypatch)
        env: dict = {}
        assert bu_cli._resolve_backend_cdp(env, "t1", session_name="other") is None
        assert "BU_CDP_URL" not in env and "BU_CDP_WS" not in env

    def test_unnamed_call_ignores_registry(self, registry_home, monkeypatch):
        bu_cli = self._patch_browser_tool(monkeypatch)
        ba.save_session_endpoint("obsidian", "http://127.0.0.1:9333", "Obsidian")
        env: dict = {}
        assert bu_cli._resolve_backend_cdp(env, "t1") is None
        assert "BU_CDP_URL" not in env

    def test_explicit_env_wins_over_registry(self, registry_home, monkeypatch):
        bu_cli = self._patch_browser_tool(monkeypatch)
        ba.save_session_endpoint("obsidian", "http://127.0.0.1:9333", "Obsidian")
        env = {"BU_CDP_URL": "http://127.0.0.1:1111"}
        assert bu_cli._resolve_backend_cdp(env, "t1", session_name="obsidian") is None
        assert env["BU_CDP_URL"] == "http://127.0.0.1:1111"

    def test_registry_outranks_global_override(self, registry_home, monkeypatch):
        import tools.browser_use_cli as bu_cli

        monkeypatch.setattr(
            "tools.browser_tool._get_cdp_override",
            lambda: "http://127.0.0.1:9222",
            raising=False,
        )
        monkeypatch.setattr(
            "tools.browser_tool._get_cloud_provider", lambda: None, raising=False
        )
        ba.save_session_endpoint("obsidian", "http://127.0.0.1:9333", "Obsidian")
        env: dict = {}
        assert bu_cli._resolve_backend_cdp(env, "t1", session_name="obsidian") is None
        assert env["BU_CDP_URL"] == "http://127.0.0.1:9333"
