"""Regression tests for #135667 — a cron origin captured in a thread keeps its parent
channel, so a parent-channel ``gateway.profile_routes`` entry authorizes shared-bot
origin delivery into that thread under the same AND-semantics as inbound routing.

Before the fix, ``_origin_from_env`` dropped ``HERMES_SESSION_PARENT_CHAT_ID`` and
``SharedRouteAdapters.get`` matched routes without a parent, so the thread's own id was
the only anchor and a parent-channel route missed — the satellite's own disabled
platform block then vetoed the delivery with "not configured/enabled".
"""
from unittest.mock import MagicMock

import hermes_yaml as yaml

from cron.scheduler_delivery import _resolve_single_delivery_target, _resolve_target_transport
from cron.scheduler_preflight import SharedRouteAdapters, _primary_profile_routes_for_current_home
from gateway.config import Platform, PlatformConfig
from gateway.session_context import clear_session_vars, set_session_vars
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.cronjob_job_args import _origin_from_env

PARENT = "111111111111111111"
THREAD = "222222222222222222"


def _thread_session_vars():
    return set_session_vars(
        platform="discord", chat_id=THREAD, thread_id=THREAD, parent_chat_id=PARENT,
        chat_type="thread", async_delivery=True)


def test_origin_capture_keeps_parent_channel():
    tokens = _thread_session_vars()
    try:
        origin = _origin_from_env()
    finally:
        clear_session_vars(tokens)
    assert origin["chat_id"] == THREAD and origin["thread_id"] == THREAD
    assert origin["parent_chat_id"] == PARENT
    # the parent survives into the delivery target, where route matching reads it
    target = _resolve_single_delivery_target(
        {"id": "repro", "deliver": "origin", "origin": origin}, "origin")
    assert target["chat_id"] == THREAD and target["thread_id"] == THREAD
    assert target["parent_chat_id"] == PARENT


def test_origin_capture_without_parent_stays_absent():
    tokens = set_session_vars(platform="discord", chat_id=PARENT, async_delivery=True)
    try:
        origin = _origin_from_env()
    finally:
        clear_session_vars(tokens)
    assert origin["parent_chat_id"] is None
    target = _resolve_single_delivery_target(
        {"id": "repro", "deliver": "origin", "origin": origin}, "origin")
    assert "parent_chat_id" not in target


def _satellite_shared_routes(tmp_path, monkeypatch, route: dict):
    root = tmp_path / "root"
    sat_home = root / "profiles" / "fitness"
    sat_home.mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump({
        "gateway": {"multiplex_profiles": True, "profile_routes": [route]},
    }), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    primary = object()
    token = set_hermes_home_override(str(sat_home))
    try:
        shared = SharedRouteAdapters(
            {Platform.DISCORD: primary}, _primary_profile_routes_for_current_home())
    finally:
        reset_hermes_home_override(token)
    return shared, primary


def _satellite_config():
    config = MagicMock()
    config.platforms = {Platform.DISCORD: PlatformConfig(enabled=False)}
    return config


def test_parent_channel_route_authorizes_thread_origin(tmp_path, monkeypatch):
    """A route keyed on the parent channel serves the thread's origin delivery through the
    primary adapter — the same target an explicit ``discord:PARENT:THREAD`` address is
    authorized for (#135667)."""
    shared, primary = _satellite_shared_routes(
        tmp_path, monkeypatch, {"name": "fit", "platform": "discord", "chat_id": PARENT, "profile": "fitness"})
    target = {"platform": "discord", "chat_id": THREAD, "thread_id": THREAD, "parent_chat_id": PARENT}
    resolved, error = _resolve_target_transport(
        {"id": "repro"}, Platform.DISCORD, "discord", target, shared, _satellite_config())
    assert error is None
    assert resolved[2] is primary


def test_thread_origin_without_parent_fails_closed_on_parent_route(tmp_path, monkeypatch):
    """Older origins stamped without parent metadata must not authorize against a
    parent-channel route — the parent is never guessed from the thread id (#135667)."""
    shared, primary = _satellite_shared_routes(
        tmp_path, monkeypatch, {"name": "fit", "platform": "discord", "chat_id": PARENT, "profile": "fitness"})
    target = {"platform": "discord", "chat_id": THREAD, "thread_id": THREAD}
    resolved, error = _resolve_target_transport(
        {"id": "repro"}, Platform.DISCORD, "discord", target, shared, _satellite_config())
    assert resolved is None and "not configured/enabled" in error


def test_unrelated_parent_fails_closed(tmp_path, monkeypatch):
    """A thread under a different parent channel is not authorized by this route."""
    shared, primary = _satellite_shared_routes(
        tmp_path, monkeypatch, {"name": "fit", "platform": "discord", "chat_id": PARENT, "profile": "fitness"})
    target = {"platform": "discord", "chat_id": "333333333333333333",
              "thread_id": "333333333333333333", "parent_chat_id": "444444444444444444"}
    resolved, error = _resolve_target_transport(
        {"id": "repro"}, Platform.DISCORD, "discord", target, shared, _satellite_config())
    assert resolved is None and "not configured/enabled" in error
