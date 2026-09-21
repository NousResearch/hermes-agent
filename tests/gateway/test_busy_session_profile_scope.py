"""Regression for #103717: a secondary multiplex profile's busy-session
follow-ups must authorize against THAT profile's allowlist, not whatever
``os.environ`` happens to hold (the primary profile's first-writer-bridged
value under multiplex).

``_make_profile_message_handler`` (the cold path) wraps ``_handle_message``
in ``_async_profile_runtime_scope`` before authorization runs. Until this
fix, ``_make_profile_busy_session_handler`` (the busy path) stamped
``source.profile`` but never entered that scope, so ``_is_user_authorized``
fell through to ``os.environ`` and read the wrong profile's
``FEISHU_ALLOWED_USERS``.
"""
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.pairing import PairingStore
from gateway.session import SessionSource


@pytest.fixture
def mux_home(tmp_path, monkeypatch):
    from agent import secret_scope

    home = tmp_path / "hh"
    (home / "profiles" / "secondary").mkdir(parents=True)
    (home / ".env").write_text("FEISHU_ALLOWED_USERS=primary-user\n")
    (home / "profiles" / "secondary" / ".env").write_text(
        "FEISHU_ALLOWED_USERS=secondary-user\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in ("FEISHU_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS"):
        monkeypatch.delenv(key, raising=False)
    prev = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    yield home
    secret_scope.set_multiplex_active(prev)


def _runner(home):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.config.platforms = {Platform.FEISHU: PlatformConfig(enabled=True, extra={})}
    runner.pairing_store = PairingStore(profile="default")
    runner.pairing_stores = {"default": runner.pairing_store}
    runner._profile_adapters = {"secondary": {}}
    runner.adapters = {}
    return runner


def _feishu_event(user_id):
    from gateway.platforms.base import MessageEvent, MessageType

    source = SessionSource(
        platform=Platform.FEISHU,
        user_id=user_id,
        user_name=user_id,
        chat_id="oc_dm",
        chat_type="dm",
        profile=None,
    )
    return MessageEvent(text="follow-up", message_type=MessageType.TEXT, source=source, message_id="m1")


@pytest.mark.asyncio
async def test_busy_handler_authorizes_against_secondary_profile_allowlist(mux_home):
    """Secondary profile owner's follow-up while their session is busy must be
    authorized against the SECONDARY profile's own allowlist."""
    runner = _runner(mux_home)

    # Isolate exactly the authorization decision the busy path gates on
    # (see gateway/run_busy.py::_handle_active_session_busy_message's first
    # check) without pulling in queueing/steer machinery unrelated to the bug.
    async def _stub_busy_message(event, session_key):
        return runner._is_user_authorized(event.source)

    runner._handle_active_session_busy_message = _stub_busy_message

    handler = runner._make_profile_busy_session_handler("secondary")

    authorized_event = _feishu_event("secondary-user")
    assert await handler(authorized_event, "sk") is True

    intruder_event = _feishu_event("primary-user")
    assert await handler(intruder_event, "sk") is False
