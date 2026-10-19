"""Native restart recovery accepts viewer work actually admitted by the authority."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.session import SessionSource
from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_contract import Principal, SessionRef, Submission
from gateway.config import Platform
from hermes_state import SessionDB
from hermes_state_runtime import (
    admit_session_input,
    begin_runtime_epoch,
    claim_session_input,
    list_session_admissions,
    settle_session_input,
)


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
    admitted = admit_session_input(
        db,
        epoch=epoch,
        principal_id="messaging:native",
        session_id=sid,
        request_id="native",
        payload={"native_text_v1": {"source": {}, "route": route}},
    )
    started = claim_session_input(db, epoch=epoch, session_id=sid)
    settle_session_input(
        db,
        epoch=epoch,
        admission_id=admitted["admission_id"],
        generation=started["generation"],
        outcome="completed",
    )


@pytest.mark.asyncio
async def test_submit_persists_viewer_provenance_and_restart_accepts_it(monkeypatch, tmp_path):
    sid = "native-session"
    route = "telegram:default:chat"
    source = _source()
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session(sid, source="telegram")
        epoch = begin_runtime_epoch(db, instance_id="first")
        _native_row(db, epoch, sid, route)

        authority = SessionAuthority(
            _runner(), profile_id="default", instance_id="first", db=db, epoch=epoch)
        authority.sessions[sid] = LiveSession(source, route)
        authority._publish_pending = lambda ref: None
        authority._schedule = lambda ref: None

        actor = Principal(
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

        rows = list_session_admissions(db, session_id=sid, pending_only=False)
        viewer = next(row for row in rows if row["request_id"] == "viewer")
        assert viewer["payload"]["local_operator_v1"] == {
            "profile_id": "default",
            "session_id": sid,
            "principal_id": "authenticated-viewer",
        }

        async def check_native_route(runner, payload, target, available_source, adapter):
            return source, route

        monkeypatch.setattr(
            "gateway.session_envelope.check_native_route", check_native_route)

        recovered = SessionAuthority(
            _runner(), profile_id="default", instance_id="restart", db=db, epoch=epoch)
        scheduled = []
        recovered._schedule = lambda ref: scheduled.append(ref.session_id)

        result = await recovered.recover_native_sessions([(sid, source, object())])

        assert result == {sid: "ready"}
        assert recovered.sessions[sid].source is source
        assert recovered.sessions[sid].route == route
        assert scheduled == [sid]
    finally:
        db.close()
