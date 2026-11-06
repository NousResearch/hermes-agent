"""Flat Matrix cron seeds must not create participant-less orphan sessions."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cron.scheduler_delivery import (
    _prepare_target_delivery,
    _seed_cron_channel_session,
    _seed_live_delivery_sessions,
)
from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource, build_session_key


@pytest.mark.parametrize("origin_kind", ["cli", "api", "partial", "chat", "dm"])
def test_matrix_inchannel_missing_origin_user_fails_closed(origin_kind, caplog):
    is_dm = origin_kind == "dm"
    user_id = "@alice:example.org" if origin_kind == "chat" else None
    origin = {
        "platform": "matrix", "chat_id": "!room:example.org",
        "chat_type": "dm" if is_dm else "group", "user_id": user_id,
    }
    job = {"id": "matrix-cron", "origin": origin, "attach_to_session": True}
    if origin_kind == "cli":
        job.pop("origin")
    elif origin_kind == "api":
        job["origin"] = None
    adapter = SimpleNamespace(
        supports_inchannel_continuable=True, _session_store=MagicMock())
    pconfig = PlatformConfig(enabled=True, extra={"cron_continuable_surface": "in_channel"})
    target = {
        "platform": "matrix", "chat_id": "!room:example.org",
        "_resolved_from": "origin_fallback" if origin_kind in {"cli", "api"} else "origin",
    }
    with patch("cron.scheduler_delivery._resolve_target_transport", return_value=(
        (None, pconfig, adapter, {}), None
    )), patch("gateway.mirror.mirror_to_session", return_value=True):
        delivery = _prepare_target_delivery(
            job, target, adapters={}, loop=SimpleNamespace(is_running=lambda: True),
            config=None, notify_delivery=False, mirror_enabled=True,
            mirror_text="Brief", delivery_errors=[])
        assert delivery is not None
        assert delivery.in_channel_surface
        expected = bool(is_dm or user_id)
        assert delivery.inchannel_continuable is expected
        _seed_live_delivery_sessions(delivery, delivered_message_id=None)
    if expected:
        source = adapter._session_store.get_or_create_session.call_args[0][0]
        reply = SessionSource(
            platform=Platform.MATRIX, chat_id="!room:example.org",
            chat_type="dm" if is_dm else "group", user_id="@alice:example.org")
        assert build_session_key(source, group_sessions_per_user=True) == build_session_key(
            reply, group_sessions_per_user=True)
    else:
        adapter._session_store.get_or_create_session.assert_not_called()
        assert "seed skipped" in caplog.text


def test_flat_seed_helper_refuses_missing_group_participant(caplog):
    adapter = SimpleNamespace(_session_store=MagicMock())
    with patch("gateway.mirror.mirror_to_session", return_value=True) as mirror:
        assert not _seed_cron_channel_session(
            {"id": "matrix-cron"}, adapter, "matrix", "!room:example.org", "Brief",
            is_dm=False, user_id=None)
    adapter._session_store.get_or_create_session.assert_not_called()
    mirror.assert_not_called()
    assert "user_id" in caplog.text
