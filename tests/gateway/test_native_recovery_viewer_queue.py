"""Native restart recovery accepts already-authorized local viewer work in the same FIFO."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.session_authority import SessionAuthority


@pytest.mark.asyncio
async def test_recovery_validates_queued_viewer_payload_instead_of_rejecting_it(monkeypatch):
    from gateway.config import Platform
    from gateway.platforms.event import SessionSource

    sid = "native-session"
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="chat", user_id="native-user", profile="default")
    route = "telegram:default:chat"
    native = {
        "status": "terminal",
        "payload": {"native_text_v1": {"source": {}, "route": route}},
        "principal_id": "messaging:native",
        "request_id": "native",
    }
    viewer = {
        "status": "queued",
        "payload": {
            "text": "viewer follow-up",
            "local_operator_v1": {
                "profile_id": "default",
                "session_id": sid,
                "principal_id": "authenticated-viewer",
            },
        },
        "principal_id": "authenticated-viewer",
        "request_id": "viewer",
    }

    monkeypatch.setattr(
        "gateway.session_authority.list_session_admissions",
        lambda db, session_id, pending_only=False: [native, viewer],
    )

    async def check_native_route(runner, payload, target, available_source, adapter):
        return source, route

    monkeypatch.setattr("gateway.session_envelope.check_native_route", check_native_route)

    scheduled = []
    fake = SimpleNamespace(
        db=object(),
        profile_id="default",
        runner=SimpleNamespace(session_store=SimpleNamespace()),
        sessions={},
        physical_target=lambda ref: "physical",
        _require_admission_open=lambda: None,
        _schedule=lambda ref: scheduled.append(ref.session_id),
    )

    result = await SessionAuthority.recover_native_sessions(
        fake, [(sid, source, object())])

    assert result == {sid: "ready"}
    assert fake.sessions[sid].source is source
    assert fake.sessions[sid].route == route
    assert scheduled == [sid]
