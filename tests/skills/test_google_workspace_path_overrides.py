"""Tests for Google Workspace credential path environment overrides."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts"
)


def _load_script_module(script_name: str, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / script_name)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_google_api_uses_profile_defaults_when_overrides_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("GOOGLE_TOKEN_PATH", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET_PATH", raising=False)

    module = _load_script_module("google_api.py", "google_api_default_paths_test")

    assert module.TOKEN_PATH == tmp_path / "google_token.json"
    assert module.CLIENT_SECRET_PATH == tmp_path / "google_client_secret.json"


def test_google_api_uses_nonblank_path_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("GOOGLE_TOKEN_PATH", str(tmp_path / "tokens" / "token.json"))
    monkeypatch.setenv(
        "GOOGLE_CLIENT_SECRET_PATH",
        str(tmp_path / "secrets" / "client_secret.json"),
    )

    module = _load_script_module("google_api.py", "google_api_override_paths_test")

    assert module.TOKEN_PATH == tmp_path / "tokens" / "token.json"
    assert module.CLIENT_SECRET_PATH == tmp_path / "secrets" / "client_secret.json"


def test_google_api_treats_blank_overrides_as_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("GOOGLE_TOKEN_PATH", "")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET_PATH", "   ")

    module = _load_script_module("google_api.py", "google_api_blank_paths_test")

    assert module.TOKEN_PATH == tmp_path / "google_token.json"
    assert module.CLIENT_SECRET_PATH == tmp_path / "google_client_secret.json"


def test_setup_uses_nonblank_path_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("GOOGLE_TOKEN_PATH", str(tmp_path / "token.json"))
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET_PATH", str(tmp_path / "secret.json"))
    monkeypatch.setenv("GOOGLE_PENDING_PATH", str(tmp_path / "pending.json"))

    module = _load_script_module("setup.py", "google_setup_override_paths_test")

    assert module.TOKEN_PATH == tmp_path / "token.json"
    assert module.CLIENT_SECRET_PATH == tmp_path / "secret.json"
    assert module.PENDING_AUTH_PATH == tmp_path / "pending.json"


def test_setup_treats_blank_overrides_as_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("GOOGLE_TOKEN_PATH", "")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET_PATH", "\t")
    monkeypatch.setenv("GOOGLE_PENDING_PATH", "   ")

    module = _load_script_module("setup.py", "google_setup_blank_paths_test")

    assert module.TOKEN_PATH == tmp_path / "google_token.json"
    assert module.CLIENT_SECRET_PATH == tmp_path / "google_client_secret.json"
    assert module.PENDING_AUTH_PATH == tmp_path / "google_oauth_pending.json"
