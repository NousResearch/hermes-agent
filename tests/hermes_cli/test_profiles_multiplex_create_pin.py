"""Machine-level Bot create must pin secondary API-server disable under multiplex.

Desktop New Agent / ``profiles.create`` / POST /api/profiles can clone or
mirror the launch profile's API-server credential. Without an explicit
``platforms.api_server.enabled: false`` pin, the multiplexer treats the new
profile as a second listener and skips it.

CLI ``create_profile()`` must stay untouched so standalone named gateways
keep working.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hermes_cli.profiles import (
    create_profile,
    launch_gateway_is_multiplexed,
    normalize_created_profile_for_launch_multiplex,
    pin_secondary_profile_api_server_disabled,
)


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    monkeypatch.delenv("API_SERVER_KEY", raising=False)
    return default_home


def _write_launch_multiplex(home: Path, *, key: str = "repro-api-server-key-16+") -> None:
    (home / "config.yaml").write_text(
        "gateway:\n  multiplex_profiles: true\n"
        "platforms:\n  api_server:\n    enabled: true\n",
        encoding="utf-8",
    )
    (home / ".env").write_text(f"API_SERVER_KEY={key}\n", encoding="utf-8")


def _api_server_enabled(cfg_path: Path):
    if not cfg_path.is_file():
        return None
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    api = ((data.get("platforms") or {}).get("api_server") or {})
    return api.get("enabled")


def _port_binding_platforms(profile_home: Path) -> list[str]:
    from agent.secret_scope import (
        build_profile_secret_scope,
        reset_secret_scope,
        set_secret_scope,
    )
    from gateway.config import load_gateway_config, platform_binds_port
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(profile_home))
    secret = set_secret_scope(build_profile_secret_scope(profile_home))
    try:
        cfg = load_gateway_config()
    finally:
        reset_secret_scope(secret)
        reset_hermes_home_override(token)
    return sorted(
        platform.value
        for platform, pcfg in cfg.platforms.items()
        if pcfg.enabled and platform_binds_port(platform.value, pcfg.extra)
    )


def test_launch_gateway_is_multiplexed_reads_default_root(profile_env):
    assert launch_gateway_is_multiplexed() is False
    _write_launch_multiplex(profile_env)
    assert launch_gateway_is_multiplexed() is True


def test_launch_gateway_env_override_wins(profile_env, monkeypatch):
    _write_launch_multiplex(profile_env)
    monkeypatch.setenv("GATEWAY_MULTIPLEX_PROFILES", "0")
    assert launch_gateway_is_multiplexed() is False
    monkeypatch.setenv("GATEWAY_MULTIPLEX_PROFILES", "1")
    assert launch_gateway_is_multiplexed() is True


def test_pin_writes_explicit_false_and_stops_force_enable(profile_env):
    _write_launch_multiplex(profile_env)
    created = create_profile(
        "clonebot", clone_from="default", clone_config=True, no_alias=True
    )
    assert _api_server_enabled(created / "config.yaml") is True
    assert "api_server" in _port_binding_platforms(created)

    wrote = pin_secondary_profile_api_server_disabled(created)
    assert wrote is True
    assert _api_server_enabled(created / "config.yaml") is False
    assert _port_binding_platforms(created) == []
    env_text = (created / ".env").read_text(encoding="utf-8")
    assert "API_SERVER_KEY=" in env_text


def test_normalize_pins_only_when_launch_is_multiplexed(profile_env):
    (profile_env / "config.yaml").write_text(
        "platforms:\n  api_server:\n    enabled: true\n",
        encoding="utf-8",
    )
    created = create_profile(
        "standalone", clone_from="default", clone_config=True, no_alias=True
    )
    result = normalize_created_profile_for_launch_multiplex(created, name="standalone")
    assert result["multiplex"] is False
    assert result["applied"] is False
    assert _api_server_enabled(created / "config.yaml") is True


def test_cli_create_profile_is_not_silently_rewritten(profile_env):
    _write_launch_multiplex(profile_env)
    created = create_profile("clibot", no_alias=True)
    assert _api_server_enabled(created / "config.yaml") is None
    env_text = (created / ".env").read_text(encoding="utf-8")
    assert "API_SERVER_KEY=" not in env_text
    assert _port_binding_platforms(created) == []
