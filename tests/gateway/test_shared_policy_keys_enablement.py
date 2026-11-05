"""A slash or allowlist policy is not a platform setup: the plugin enable sweep must not count the
shared policy keys the loader bridges into every platform's ``extra``. Real ``load_gateway_config``
against a temp HERMES_HOME and managed scope."""

import json

import pytest

from gateway.config import Platform, load_gateway_config
from hermes_cli import managed_scope

_POLICY = {"allow_admin_from": ["nobody"], "user_allowed_commands": ["new", "stop"]}


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
])
def test_a_policy_alone_does_not_enable_whatsapp(monkeypatch, tmp_path, source):
    config = _load(monkeypatch, tmp_path, **source)

    assert Platform.WHATSAPP not in config.get_connected_platforms()


def test_the_opt_in_still_enables_whatsapp_with_its_policy(monkeypatch, tmp_path):
    config = _load(monkeypatch, tmp_path, profile={"whatsapp": _POLICY}, env={"WHATSAPP_ENABLED": "true"})

    assert Platform.WHATSAPP in config.get_connected_platforms()
    assert config.platforms[Platform.WHATSAPP].extra["user_allowed_commands"] == ["new", "stop"]
