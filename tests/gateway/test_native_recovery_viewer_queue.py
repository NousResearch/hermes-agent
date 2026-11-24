"""Native restart recovery accepts viewer work actually admitted by the authority."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.session import SessionSource
from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_contract import SessionActor, SessionRef, Submission
from gateway.config import Platform
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def _runner():
    return SimpleNamespace(_draining=False, session_store=SimpleNamespace())


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat",
        user_id="native-user",
        profile="default",
    )


def _native_row(db, epoch, sid, route):
    admitted, _created = db.admissions.admit(
        epoch=epoch,
        principal_id="messaging:native",
        session_id=sid,
        request_id="native",
        payload={"native_text_v1": {"source": {}, "route": route}},
    )
    started = db.admissions.claim(epoch=epoch, session_id=sid)
    db.admissions.settle(
        epoch=epoch,
        admission_id=admitted.admission_id,
        generation=started.generation,
        outcome="completed",
    )


@pytest.mark.asyncio
async def test_submit_persists_viewer_provenance_and_restart_accepts_it(monkeypatch, tmp_path):
    sid = "native-session"
    route = "telegram:default:chat"
    source = _source()
    db = storage_handle(tmp_path)
    try:
        seed_session(db, sid, "telegram")
        epoch = db.runtime.begin_epoch(instance_id="first")
        _native_row(db, epoch, sid, route)

        authority = SessionAuthority(
            _runner(), profile_id="default", instance_id="first", epoch=epoch,
            persistence=storage_handle(tmp_path))
        authority.sessions[sid] = LiveSession(source, route)
        authority._publish_pending = lambda ref: None
        authority._schedule = lambda ref: None

        actor = SessionActor(
            "authenticated-viewer", "default", frozenset({"session:submit"}), "viewer")
        await authority.submit(
            actor,
            Submission(
                request_id="viewer",
                ref=SessionRef("default", sid),
                payload={"text": "viewer follow-up"},
                intent="queue",
            ),
        )

        rows = db.admissions.list(session_id=sid, pending_only=False)
        viewer = next(row for row in rows if row.request_id == "viewer")
        assert viewer.payload["local_operator_v1"] == {
            "profile_id": "default",
            "session_id": sid,
            "principal_id": "authenticated-viewer",
        }

        async def check_native_route(runner, payload, target, available_source, adapter):
            return source, route

        monkeypatch.setattr(
            "gateway.session_envelope.check_native_route", check_native_route)

        recovered = SessionAuthority(
            _runner(), profile_id="default", instance_id="restart", epoch=epoch,
            persistence=storage_handle(tmp_path))
        scheduled = []
        recovered._schedule = lambda ref: scheduled.append(ref.session_id)

        result = await recovered.recover_native_sessions([(sid, source, object())])

        assert result == {sid: "ready"}
        assert recovered.sessions[sid].source is source
        assert recovered.sessions[sid].route == route
        assert scheduled == [sid]
    finally:
        db.close()
