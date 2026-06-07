"""Load + fail-closed validation of gateway.profile_routing."""

import pytest

from gateway.config import load_gateway_config


def _write(tmp_path, yaml_text):
    (tmp_path / "config.yaml").write_text(yaml_text)


def _mk_profile(tmp_path, name):
    (tmp_path / "profiles" / name).mkdir(parents=True, exist_ok=True)


def test_no_profile_routing_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write(tmp_path, "telegram:\n  channel_models: {}\n")
    assert load_gateway_config().profile_routing is None


def test_valid_route_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _mk_profile(tmp_path, "research")
    _write(tmp_path, """
gateway:
  profile_routing:
    routes:
      - { platform: telegram, chat_id: "-100", thread_id: 42, profile: research }
""")
    cfg = load_gateway_config()
    assert cfg.profile_routing["routes"][0]["profile"] == "research"


def test_unknown_profile_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write(tmp_path, """
gateway:
  profile_routing:
    routes:
      - { platform: telegram, profile: ghost }
""")
    with pytest.raises(ValueError, match="ghost"):
        load_gateway_config()


def test_duplicate_exact_match_routes_raise(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _mk_profile(tmp_path, "a")
    _mk_profile(tmp_path, "b")
    _write(tmp_path, """
gateway:
  profile_routing:
    routes:
      - { platform: discord, channel_id: "c1", profile: a }
      - { platform: discord, channel_id: "c1", profile: b }
""")
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        load_gateway_config()


def test_invalid_profile_name_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write(tmp_path, """
gateway:
  profile_routing:
    routes:
      - { platform: telegram, profile: "Bad Name!" }
""")
    with pytest.raises(ValueError):
        load_gateway_config()
