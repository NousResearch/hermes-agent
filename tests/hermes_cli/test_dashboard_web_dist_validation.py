"""Regression tests: `hermes dashboard` validates HERMES_WEB_DIST before serving.

A custom HERMES_WEB_DIST without --skip-build previously skipped BOTH the
build and any validation, so the server started and served 404s with no
obvious cause (same failure mode as issue #23817, reached via the env-var
path instead of --skip-build). The env-var branch must now fail fast when
the dist has no index.html, and proceed when it does.

Design credit: PR #17845 (@Caelier).
"""

import os
import sys
import types

import pytest


@pytest.fixture()
def main_mod():
    import hermes_cli.main as main
    return main


def _args(**over):
    base = {
        "host": "127.0.0.1",
        "port": 0,
        "no_open": True,
        "open_profile": None,
        "skip_build": False,
        "headless_backend": False,
        "tui": False,
    }
    base.update(over)
    return types.SimpleNamespace(**base)


def _wire_common(main_mod, monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: "default"
    )
    monkeypatch.setattr(main_mod, "_sync_bundled_skills_quietly", lambda: None)
    monkeypatch.setitem(sys.modules, "fastapi", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "hermes_logging",
        types.SimpleNamespace(setup_logging=lambda **_k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.start_background_mcp_discovery",
        lambda **_k: None,
    )


def test_env_dist_without_index_exits(main_mod, monkeypatch, tmp_path, capsys):
    """HERMES_WEB_DIST pointing at a dist with no index.html must exit 1,
    not start a server that 404s."""
    _wire_common(main_mod, monkeypatch)
    empty_dist = tmp_path / "empty_dist"
    empty_dist.mkdir()
    monkeypatch.setenv("HERMES_WEB_DIST", str(empty_dist))

    started = []
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.web_server",
        types.SimpleNamespace(start_server=lambda **k: started.append(k)),
    )
    builds = []
    monkeypatch.setattr(
        main_mod, "_build_web_ui", lambda *a, **k: builds.append(a) or True
    )

    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_dashboard(_args())

    assert exc.value.code == 1
    assert started == []
    assert builds == []  # env var set -> build skipped, validation is the gate
    out = capsys.readouterr().out
    assert "HERMES_WEB_DIST" in out and str(empty_dist) in out




# ---------------------------------------------------------------------------
# --skip-build recovery (issue #59288): a missing dist under --skip-build
# should warn and attempt ONE recovery build via _build_web_ui before the
# fatal exit, instead of hard-failing immediately.
# ---------------------------------------------------------------------------


def test_skip_build_missing_dist_attempts_one_recovery_build(
    main_mod, monkeypatch, tmp_path, capsys
):
    """--skip-build + missing index.html triggers exactly one recovery build;
    when the build produces a dist, the server starts."""
    _wire_common(main_mod, monkeypatch)
    monkeypatch.delenv("HERMES_WEB_DIST", raising=False)
    project_root = tmp_path / "proj"
    dist = project_root / "hermes_cli" / "web_dist"
    dist.mkdir(parents=True)
    monkeypatch.setattr(main_mod, "PROJECT_ROOT", project_root)

    started = []
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.web_server",
        types.SimpleNamespace(start_server=lambda **k: started.append(k)),
    )

    builds = []

    def fake_build(web_dir, *, fatal=False):
        builds.append((web_dir, fatal))
        (dist / "index.html").write_text("<html></html>", encoding="utf-8")
        return True

    monkeypatch.setattr(main_mod, "_build_web_ui", fake_build)

    main_mod.cmd_dashboard(_args(skip_build=True))

    assert len(builds) == 1  # exactly ONE recovery build
    assert builds[0][0] == project_root / "web"
    assert len(started) == 1
    out = capsys.readouterr().out
    assert "recovery build" in out.lower()




# ---------------------------------------------------------------------------
# Desktop-inherited env isolation (issue #52945 / supersedes #52948, #67402)
# ---------------------------------------------------------------------------


class TestDesktopInheritedEnvIsolation:
    """Browser `hermes dashboard` must not serve the Desktop renderer."""

    def test_browser_dashboard_strips_inherited_desktop_flag_and_asar_dist(
        self, main_mod, monkeypatch, tmp_path
    ):
        asar_dist = tmp_path / "app.asar" / "dist"
        asar_dist.mkdir(parents=True)
        (asar_dist / "index.html").write_text("<html></html>", encoding="utf-8")
        monkeypatch.setenv("HERMES_DESKTOP", "1")
        monkeypatch.setenv("HERMES_WEB_DIST", str(asar_dist))
        monkeypatch.setenv("HERMES_SERVE_HEADLESS", "1")

        captured = {}

        def fake_start_server(**kwargs):
            captured["env_desktop"] = os.environ.get("HERMES_DESKTOP")
            captured["env_web_dist"] = os.environ.get("HERMES_WEB_DIST")
            captured["env_headless"] = os.environ.get("HERMES_SERVE_HEADLESS")
            raise SystemExit(0)

        _wire_common(main_mod, monkeypatch)
        monkeypatch.setattr(main_mod, "_build_web_ui", lambda *a, **k: True)
        monkeypatch.setitem(
            sys.modules,
            "hermes_cli.web_server",
            types.SimpleNamespace(start_server=fake_start_server),
        )

        with pytest.raises(SystemExit) as exc:
            main_mod.cmd_dashboard(_args(no_open=True, headless_backend=False, skip_build=True))
        assert exc.value.code == 0
        assert captured.get("env_desktop") is None
        assert captured.get("env_web_dist") is None
        assert captured.get("env_headless") is None

    def test_headless_desktop_serve_keeps_desktop_flag(self, main_mod, monkeypatch, tmp_path):
        asar_dist = tmp_path / "app.asar" / "dist"
        asar_dist.mkdir(parents=True)
        (asar_dist / "index.html").write_text("<html></html>", encoding="utf-8")
        monkeypatch.setenv("HERMES_DESKTOP", "1")
        monkeypatch.setenv("HERMES_WEB_DIST", str(asar_dist))

        captured = {}

        def fake_start_server(**kwargs):
            captured["env_desktop"] = os.environ.get("HERMES_DESKTOP")
            captured["env_web_dist"] = os.environ.get("HERMES_WEB_DIST")
            raise SystemExit(0)

        _wire_common(main_mod, monkeypatch)
        monkeypatch.setattr(main_mod, "_build_web_ui", lambda *a, **k: True)
        monkeypatch.setitem(
            sys.modules,
            "hermes_cli.web_server",
            types.SimpleNamespace(start_server=fake_start_server),
        )

        with pytest.raises(SystemExit):
            main_mod.cmd_dashboard(_args(no_open=True, headless_backend=True, skip_build=True))
        assert captured.get("env_desktop") == "1"
        assert captured.get("env_web_dist") == str(asar_dist)










