"""WhatsApp's config.yaml keys are policy, not a setup: only the WHATSAPP_ENABLED opt-in or an explicit
``enabled: true`` turns WhatsApp on. Real ``load_gateway_config`` against a temp HERMES_HOME and managed scope."""

import json

import pytest

from gateway.config import Platform, load_gateway_config
from hermes_cli import managed_scope

_POLICY = {"allow_admin_from": ["nobody"], "user_allowed_commands": ["new", "stop"]}
_OWN_POLICY = {"free_response_chats": ["123@g.us"]}


def _load(monkeypatch, tmp_path, *, managed=None, profile=None, env=None):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(json.dumps(profile or {}), encoding="utf-8")
    managed_dir = tmp_path / "managed"
    managed_dir.mkdir()
    if managed is not None:
        (managed_dir / "config.yaml").write_text(json.dumps(managed), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_dir))
    monkeypatch.delenv("WHATSAPP_ENABLED", raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    # Without Node.js the WhatsApp dependency probe falls back to the lazy-install policy, which is not under test.
    monkeypatch.setattr("pm.lazy_installs_allowed", lambda: True)
    managed_scope.invalidate_managed_cache()
    return load_gateway_config()


@pytest.mark.parametrize("source", [
    pytest.param({"managed": {"platforms": {"whatsapp": {"extra": _POLICY}}}}, id="managed-scope"),
    pytest.param({"profile": {"whatsapp": _POLICY}}, id="profile-config"),
    pytest.param({"profile": {"whatsapp": _OWN_POLICY}}, id="whatsapp-only-key"),
    pytest.param({"profile": {"platforms": {"whatsapp": {"extra": _OWN_POLICY}}}}, id="whatsapp-only-key-in-extra"),
])
def test_a_policy_alone_does_not_enable_whatsapp(monkeypatch, tmp_path, source):
    config = _load(monkeypatch, tmp_path, **source)

    assert Platform.WHATSAPP not in config.get_connected_platforms()


@pytest.mark.parametrize(("profile", "env"), [
    pytest.param({"whatsapp": _POLICY}, {"WHATSAPP_ENABLED": "true"}, id="env-opt-in"),
    pytest.param({"platforms": {"whatsapp": {"enabled": True, "extra": _POLICY}}}, {}, id="enabled-true"),
])
def test_an_opt_in_enables_whatsapp_and_keeps_its_policy(monkeypatch, tmp_path, profile, env):
    config = _load(monkeypatch, tmp_path, profile=profile, env=env)

    assert Platform.WHATSAPP in config.get_connected_platforms()
    assert config.platforms[Platform.WHATSAPP].extra["user_allowed_commands"] == ["new", "stop"]
