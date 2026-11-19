"""The primary home's YAML must expand under the PRIMARY's secret scope, not the caller's (#148606).

``_primary_profile_routes_for_current_home`` runs inside a satellite's profile scope (the gateway
drain walks satellites under ``_profile_runtime_scope``) but reads the PRIMARY home's config.yaml.
``${env:...}`` refs resolve against the bound scope, fail-closed, so the primary's refs used to
resolve against the satellite's .env: a missing key re-logged "keeping the literal placeholder" every
tick, and a satellite key of the same name was substituted into the primary's config.
"""

import logging

import hermes_yaml as yaml
import pytest

from agent.secret_scope import reset_secret_scope, set_secret_scope
from cron.scheduler_preflight import _primary_profile_routes_for_current_home
from hermes_cli import managed_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def satellite_home(tmp_path, monkeypatch):
    root = tmp_path / "root"
    sat_home = root / "profiles" / "sat"
    sat_home.mkdir(parents=True)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "no-managed"))
    monkeypatch.delenv("ROUTE_CHAT_ID", raising=False)
    managed_scope.invalidate_managed_cache()
    (root / "config.yaml").write_text(yaml.safe_dump({"gateway": {
        "multiplex_profiles": True,
        "profile_routes": [{"name": "sat-tg", "platform": "telegram",
                            "chat_id": "${env:ROUTE_CHAT_ID}", "profile": "sat"}],
    }}), encoding="utf-8")
    (root / ".env").write_text("ROUTE_CHAT_ID=-100primary\n", encoding="utf-8")
    token = set_hermes_home_override(str(sat_home))
    try:
        yield root, sat_home
    finally:
        reset_hermes_home_override(token)
        managed_scope.invalidate_managed_cache()


def _under_satellite_scope(sat_home, secrets):
    token = set_secret_scope(secrets, profile_home=str(sat_home))
    try:
        return _primary_profile_routes_for_current_home()
    finally:
        reset_secret_scope(token)


def test_primary_refs_resolve_from_primary_env_not_satellite(satellite_home):
    _, sat_home = satellite_home
    routes = _under_satellite_scope(sat_home, {"ROUTE_CHAT_ID": "-100satellite"})
    assert [r.chat_id for r in routes] == ["-100primary"]


def test_satellite_without_key_does_not_log_placeholder(satellite_home, caplog):
    _, sat_home = satellite_home
    with caplog.at_level(logging.WARNING):
        routes = _under_satellite_scope(sat_home, {})
    assert [r.chat_id for r in routes] == ["-100primary"]
    assert "literal placeholder" not in caplog.text


def test_caller_scope_restored_after_lookup(satellite_home):
    from agent.secret_scope import get_secret
    _, sat_home = satellite_home
    token = set_secret_scope({"ROUTE_CHAT_ID": "-100satellite"}, profile_home=str(sat_home))
    try:
        _primary_profile_routes_for_current_home()
        assert get_secret("ROUTE_CHAT_ID") == "-100satellite"
    finally:
        reset_secret_scope(token)


def test_real_profile_bind_of_satellite_does_not_leak_into_primary_refs(satellite_home):
    """The gateway drain binds the satellite's WHOLE scope (``_profile_runtime_scope``), not a bare mapping."""
    from agent.runtime_scope import bind_profile_scope
    from hermes_constants import get_hermes_home
    _, sat_home = satellite_home
    (sat_home / ".env").write_text("ROUTE_CHAT_ID=-100satellite\n", encoding="utf-8")
    with bind_profile_scope(sat_home, hydrate_secrets=False):
        routes = _primary_profile_routes_for_current_home()
        assert str(get_hermes_home()) == str(sat_home)
    assert [r.chat_id for r in routes] == ["-100primary"]


@pytest.mark.usefixtures("satellite_home")
def test_unscoped_satellite_process_env_does_not_fill_primary_refs(monkeypatch):
    """A satellite CLI process (no scope bound) carries its own .env in os.environ; the primary's
    refs must still come from the primary's .env."""
    monkeypatch.setenv("ROUTE_CHAT_ID", "-100satellite")
    assert [r.chat_id for r in _primary_profile_routes_for_current_home()] == ["-100primary"]
