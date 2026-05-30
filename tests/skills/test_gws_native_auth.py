"""Unit tests for the shared gws-native credential helpers (_gws_auth.py)."""

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/_gws_auth.py"
)


@pytest.fixture
def gws_auth(monkeypatch):
    # The scripts dir must be importable (the module has no heavy deps).
    monkeypatch.syspath_prepend(str(MODULE_PATH.parent))
    spec = importlib.util.spec_from_file_location("gws_auth_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestGwsBinary:
    def test_honors_env_override(self, gws_auth, monkeypatch):
        monkeypatch.setenv("HERMES_GWS_BIN", "/custom/gws")
        assert gws_auth.gws_binary() == "/custom/gws"

    def test_falls_back_to_path_lookup(self, gws_auth, monkeypatch):
        monkeypatch.delenv("HERMES_GWS_BIN", raising=False)
        monkeypatch.setattr(gws_auth.shutil, "which", lambda _name: "/usr/local/bin/gws")
        assert gws_auth.gws_binary() == "/usr/local/bin/gws"

    def test_none_when_not_installed(self, gws_auth, monkeypatch):
        monkeypatch.delenv("HERMES_GWS_BIN", raising=False)
        monkeypatch.setattr(gws_auth.shutil, "which", lambda _name: None)
        assert gws_auth.gws_binary() is None


class TestGwsNativeAuthed:
    def test_false_when_binary_missing(self, gws_auth, monkeypatch):
        monkeypatch.setattr(gws_auth, "gws_binary", lambda: None)
        assert gws_auth.gws_native_authed() is False

    def test_false_on_nonzero_exit(self, gws_auth, monkeypatch):
        monkeypatch.setattr(gws_auth, "gws_binary", lambda: "/usr/local/bin/gws")
        monkeypatch.setattr(gws_auth.subprocess, "run", lambda *a, **k: FakeResult(returncode=1))
        assert gws_auth.gws_native_authed() is False

    def test_true_on_valid_output(self, gws_auth, monkeypatch):
        monkeypatch.setattr(gws_auth, "gws_binary", lambda: "/usr/local/bin/gws")
        monkeypatch.setattr(
            gws_auth.subprocess,
            "run",
            lambda *a, **k: FakeResult(stdout=json.dumps({"token_valid": True, "has_refresh_token": True})),
        )
        assert gws_auth.gws_native_authed() is True

    def test_false_on_expired_token(self, gws_auth, monkeypatch):
        monkeypatch.setattr(gws_auth, "gws_binary", lambda: "/usr/local/bin/gws")
        monkeypatch.setattr(
            gws_auth.subprocess,
            "run",
            lambda *a, **k: FakeResult(stdout=json.dumps({"token_valid": False, "has_refresh_token": True})),
        )
        assert gws_auth.gws_native_authed() is False

    def test_false_without_refresh_token(self, gws_auth, monkeypatch):
        monkeypatch.setattr(gws_auth, "gws_binary", lambda: "/usr/local/bin/gws")
        monkeypatch.setattr(
            gws_auth.subprocess,
            "run",
            lambda *a, **k: FakeResult(stdout=json.dumps({"token_valid": True, "has_refresh_token": False})),
        )
        assert gws_auth.gws_native_authed() is False

    def test_false_on_non_json(self, gws_auth, monkeypatch):
        monkeypatch.setattr(gws_auth, "gws_binary", lambda: "/usr/local/bin/gws")
        monkeypatch.setattr(gws_auth.subprocess, "run", lambda *a, **k: FakeResult(stdout="not json"))
        assert gws_auth.gws_native_authed() is False

    def test_false_when_subprocess_raises(self, gws_auth, monkeypatch):
        monkeypatch.setattr(gws_auth, "gws_binary", lambda: "/usr/local/bin/gws")

        def boom(*a, **k):
            raise OSError("spawn failed")

        monkeypatch.setattr(gws_auth.subprocess, "run", boom)
        assert gws_auth.gws_native_authed() is False


class TestGwsLiveCheck:
    def test_false_when_binary_missing(self, gws_auth, monkeypatch):
        monkeypatch.setattr(gws_auth, "gws_binary", lambda: None)
        ok, detail = gws_auth.gws_live_check()
        assert ok is False
        assert "not installed" in detail

    def test_ok_on_success(self, gws_auth, monkeypatch):
        monkeypatch.setattr(gws_auth, "gws_binary", lambda: "/usr/local/bin/gws")
        monkeypatch.setattr(gws_auth.subprocess, "run", lambda *a, **k: FakeResult(stdout="{}"))
        ok, _ = gws_auth.gws_live_check()
        assert ok is True

    def test_returns_error_on_failure(self, gws_auth, monkeypatch):
        monkeypatch.setattr(gws_auth, "gws_binary", lambda: "/usr/local/bin/gws")
        monkeypatch.setattr(
            gws_auth.subprocess,
            "run",
            lambda *a, **k: FakeResult(returncode=1, stderr="disabled_client"),
        )
        ok, detail = gws_auth.gws_live_check()
        assert ok is False
        assert "disabled_client" in detail

    def test_does_not_pin_hermes_credentials_file(self, gws_auth, monkeypatch):
        """gws-native checks must NOT force GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE."""
        captured = {}
        monkeypatch.setattr(gws_auth, "gws_binary", lambda: "/usr/local/bin/gws")

        def fake_run(cmd, *a, **k):
            captured["env"] = k.get("env")
            return FakeResult(stdout="{}")

        monkeypatch.setattr(gws_auth.subprocess, "run", fake_run)
        gws_auth.gws_live_check()
        # No explicit env passed → inherits the process env, never pins the Hermes token path.
        assert captured["env"] is None
