"""Managed-scope ``gateway.profile_routes`` must reach the satellite cron path (#121212).

The primary gateway reads routes through the layered loader (user file + managed overlay), so
on a centrally-managed install the routes live only in ``/etc/hermes/config.yaml``. The
satellite-side helper shared by preflight rescue and delivery-time ``SharedRouteAdapters`` read
the raw user file alone, false-blocking every routed job and failing delivery closed.
"""

import hermes_yaml as yaml
import pytest

from cron.scheduler_preflight import (
    SharedRouteAdapters,
    _delivery_platform_routed_from_primary_gateway,
    _primary_profile_routes_for_current_home,
)
from hermes_cli import managed_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _routes(*entries):
    return {"gateway": {"multiplex_profiles": True, "profile_routes": list(entries)}}


WA_ROUTE = {"name": "sat-wa", "platform": "whatsapp", "chat_id": "123@g.us", "profile": "sat"}
TG_ROUTE = {"name": "sat-tg", "platform": "telegram", "chat_id": "-100", "profile": "sat"}


@pytest.fixture
def satellite_home(tmp_path, monkeypatch):
    """Serve profile ``sat`` under a primary root whose managed scope pins the routes."""
    root = tmp_path / "root"
    sat_home = root / "profiles" / "sat"
    sat_home.mkdir(parents=True)
    managed_dir = tmp_path / "managed"
    managed_dir.mkdir()
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_dir))
    managed_scope.invalidate_managed_cache()
    token = set_hermes_home_override(str(sat_home))
    try:
        yield root, managed_dir
    finally:
        reset_hermes_home_override(token)
        managed_scope.invalidate_managed_cache()


def test_managed_scope_routes_reach_satellite_preflight_and_delivery(satellite_home):
    root, managed_dir = satellite_home
    (root / "config.yaml").write_text(
        yaml.safe_dump({"platforms": {"whatsapp": {"enabled": True}}}), encoding="utf-8")
    (managed_dir / "config.yaml").write_text(yaml.safe_dump(_routes(WA_ROUTE)), encoding="utf-8")

    assert _delivery_platform_routed_from_primary_gateway("whatsapp")
    shared = SharedRouteAdapters({"whatsapp": object()}, _primary_profile_routes_for_current_home())
    assert shared  # delivery-time fallback no longer fails closed
    assert shared.get("whatsapp", {"chat_id": "123@g.us"}) is not None


def test_managed_routes_replace_user_file_routes(satellite_home):
    root, managed_dir = satellite_home
    (root / "config.yaml").write_text(yaml.safe_dump(_routes(TG_ROUTE)), encoding="utf-8")
    (managed_dir / "config.yaml").write_text(yaml.safe_dump(_routes(WA_ROUTE)), encoding="utf-8")

    assert [r.platform for r in _primary_profile_routes_for_current_home()] == ["whatsapp"]
