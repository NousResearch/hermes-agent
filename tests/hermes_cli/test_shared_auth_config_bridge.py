"""config.yaml → internal env bridge for shared xAI OAuth activation.

User-facing switch is shared_auth.providers (AGENTS.md env-var-for-config).
The shared-store engine still reads HERMES_XAI_SHARED_AUTH /
HERMES_SHARED_AUTH_PROVIDERS; this module only tests the bridge + config
writers. Engine gate body stays env-driven.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from hermes_cli import auth
from hermes_cli.config import (
    apply_shared_auth_config_to_env,
    disable_shared_auth_provider,
    enable_shared_auth_provider,
)


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so config writes never touch the live install.

    Also hard-resets the internal bridge env vars on teardown. The bridge
    writes ``os.environ`` directly (like TERMINAL_CWD), so monkeypatch alone
    does not always undo force-exports and later suite tests would see a
    leaked gate-on.
    """
    home = tmp_path / "home"
    home.mkdir()
    hermes = tmp_path / ".hermes"
    hermes.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    monkeypatch.setenv("HOME", str(home))
    # Start gate-off for every test unless the test sets env itself.
    for key in (
        "HERMES_XAI_SHARED_AUTH",
        "HERMES_SHARED_AUTH_PROVIDERS",
        "HERMES_SHARED_AUTH_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
        os.environ.pop(key, None)

    yield hermes

    # Teardown: bridge force-exports bypass monkeypatch tracking.
    for key in (
        "HERMES_XAI_SHARED_AUTH",
        "HERMES_SHARED_AUTH_PROVIDERS",
        "HERMES_SHARED_AUTH_DIR",
    ):
        os.environ.pop(key, None)


def _write_config(hermes_home: Path, payload: dict) -> Path:
    path = hermes_home / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_bridge_from_config_enables_gate(isolated_home, monkeypatch):
    """config.yaml shared_auth.providers:[xai-oauth] → bridge → gate True."""
    _write_config(
        isolated_home,
        {"shared_auth": {"providers": ["xai-oauth"]}},
    )
    # Prove gate is off before bridge.
    assert auth._xai_shared_auth_enabled() is False
    assert "HERMES_XAI_SHARED_AUTH" not in os.environ

    apply_shared_auth_config_to_env()

    assert os.environ.get("HERMES_XAI_SHARED_AUTH") == "1"
    assert "xai-oauth" in os.environ.get("HERMES_SHARED_AUTH_PROVIDERS", "")
    assert auth._xai_shared_auth_enabled() is True


def test_bridge_gate_off_when_shared_auth_absent(isolated_home):
    """Absent shared_auth + no env → bridge sets nothing → gate False.

    Byte-identical legacy: do not invent HERMES_* vars when config is silent.
    """
    _write_config(isolated_home, {"model": "some-model"})
    before = {
        "HERMES_XAI_SHARED_AUTH": os.environ.get("HERMES_XAI_SHARED_AUTH"),
        "HERMES_SHARED_AUTH_PROVIDERS": os.environ.get("HERMES_SHARED_AUTH_PROVIDERS"),
        "HERMES_SHARED_AUTH_DIR": os.environ.get("HERMES_SHARED_AUTH_DIR"),
    }

    apply_shared_auth_config_to_env()

    assert os.environ.get("HERMES_XAI_SHARED_AUTH") == before["HERMES_XAI_SHARED_AUTH"]
    assert (
        os.environ.get("HERMES_SHARED_AUTH_PROVIDERS")
        == before["HERMES_SHARED_AUTH_PROVIDERS"]
    )
    assert os.environ.get("HERMES_SHARED_AUTH_DIR") == before["HERMES_SHARED_AUTH_DIR"]
    assert auth._xai_shared_auth_enabled() is False


def test_bridge_gate_off_when_providers_empty(isolated_home):
    """Empty providers list must not set or clear env (gate-off invariant)."""
    _write_config(isolated_home, {"shared_auth": {"providers": []}})
    apply_shared_auth_config_to_env()
    assert "HERMES_XAI_SHARED_AUTH" not in os.environ
    assert "HERMES_SHARED_AUTH_PROVIDERS" not in os.environ
    assert auth._xai_shared_auth_enabled() is False


def test_bridge_does_not_clobber_power_user_env_when_config_absent(
    isolated_home, monkeypatch
):
    """Power-user / tests that set HERMES_XAI_SHARED_AUTH directly still work."""
    _write_config(isolated_home, {"model": "x"})
    monkeypatch.setenv("HERMES_XAI_SHARED_AUTH", "1")
    assert auth._xai_shared_auth_enabled() is True

    apply_shared_auth_config_to_env()

    # Bridge must not pop a deliberate env override when config is silent.
    assert os.environ.get("HERMES_XAI_SHARED_AUTH") == "1"
    assert auth._xai_shared_auth_enabled() is True


def test_bridge_optional_dir(isolated_home, tmp_path):
    shared_dir = tmp_path / "custom-shared"
    _write_config(
        isolated_home,
        {
            "shared_auth": {
                "providers": ["xai-oauth"],
                "dir": str(shared_dir),
            }
        },
    )
    apply_shared_auth_config_to_env()
    assert os.environ.get("HERMES_SHARED_AUTH_DIR") == str(shared_dir)


def test_enable_shared_writes_config_not_env_file(isolated_home):
    """enable-shared adds xai-oauth to config.yaml providers (not .env)."""
    _write_config(isolated_home, {"model": "m"})
    providers = enable_shared_auth_provider("xai-oauth")
    assert "xai-oauth" in providers

    raw = yaml.safe_load((isolated_home / "config.yaml").read_text(encoding="utf-8"))
    assert "xai-oauth" in raw["shared_auth"]["providers"]
    # Must not write a .env for this non-secret flag.
    env_path = isolated_home / ".env"
    if env_path.exists():
        assert "HERMES_XAI_SHARED_AUTH" not in env_path.read_text(encoding="utf-8")
    # Current process bridge must flip the engine gate on.
    assert auth._xai_shared_auth_enabled() is True
    assert os.environ.get("HERMES_XAI_SHARED_AUTH") == "1"


def test_disable_shared_removes_config_key(isolated_home):
    """disable-shared removes xai-oauth from config.yaml providers."""
    _write_config(
        isolated_home,
        {"shared_auth": {"providers": ["xai-oauth", "other"]}},
    )
    enable_shared_auth_provider("xai-oauth")  # ensure env bridged
    remaining = disable_shared_auth_provider("xai-oauth")
    assert remaining == ["other"]

    raw = yaml.safe_load((isolated_home / "config.yaml").read_text(encoding="utf-8"))
    assert raw["shared_auth"]["providers"] == ["other"]


def test_disable_shared_clears_empty_section(isolated_home):
    _write_config(
        isolated_home,
        {"shared_auth": {"providers": ["xai-oauth"]}},
    )
    enable_shared_auth_provider("xai-oauth")
    remaining = disable_shared_auth_provider("xai-oauth")
    assert remaining == []

    raw = yaml.safe_load((isolated_home / "config.yaml").read_text(encoding="utf-8")) or {}
    shared = raw.get("shared_auth")
    if shared is not None:
        assert not shared.get("providers")
    # Process gate off after disable.
    assert auth._xai_shared_auth_enabled() is False
    assert "HERMES_XAI_SHARED_AUTH" not in os.environ


def test_engine_gate_body_still_env_driven(monkeypatch, isolated_home):
    """Regression: _xai_shared_auth_enabled must keep reading env vars only."""
    # Config on disk is irrelevant if env is not bridged and not set.
    _write_config(
        isolated_home,
        {"shared_auth": {"providers": ["xai-oauth"]}},
    )
    monkeypatch.delenv("HERMES_XAI_SHARED_AUTH", raising=False)
    monkeypatch.delenv("HERMES_SHARED_AUTH_PROVIDERS", raising=False)
    # Without calling the bridge, engine gate stays off even if config exists.
    assert auth._xai_shared_auth_enabled() is False
    monkeypatch.setenv("HERMES_SHARED_AUTH_PROVIDERS", "xai-oauth")
    assert auth._xai_shared_auth_enabled() is True
