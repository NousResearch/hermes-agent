"""Security fences for durable operator provenance on native-session recovery."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle

from gateway.session import SessionSource
from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_contract import SessionActor, SessionRef, Submission
from gateway.config import Platform
from persistence.core.errors import AuthorityRefusal


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
        epoch=epoch, principal_id="messaging:native", session_id=sid,
        request_id="native",
        payload={"native_text_v1": {"source": {}, "route": route}},
    )
    started = db.admissions.claim(epoch=epoch, session_id=sid)
    db.admissions.settle(
        epoch=epoch, admission_id=admitted.admission_id,
        generation=started.generation, outcome="completed")


@pytest.mark.asyncio
async def test_client_cannot_submit_private_operator_provenance(tmp_path):
    sid = "native-session"
    source = _source()
    db = storage_handle(tmp_path)
    try:
        seed_session(db, sid, "telegram")
        epoch = db.runtime.begin_epoch(instance_id="first")
        authority = SessionAuthority(
            _runner(), profile_id="default", instance_id="first", epoch=epoch,
            persistence=storage_handle(tmp_path))
        authority.sessions[sid] = LiveSession(source, "telegram:default:chat")
        authority._publish_pending = lambda ref: None
        authority._schedule = lambda ref: None

        actor = SessionActor(
            "authenticated-viewer", "default", frozenset({"session:submit"}), "viewer")
        with pytest.raises(AuthorityRefusal) as caught:
            await authority.submit(
                actor,
                Submission(
                    request_id="forged",
                    ref=SessionRef("default", sid),
                    payload={
                        "text": "viewer follow-up",
                        "local_operator_v1": {
                            "profile_id": "default",
                            "session_id": sid,
                            "principal_id": "authenticated-viewer",
                        },
                    },
                    intent="queue",
                ),
            )
        assert caught.value.reason == "invalid_params"
        assert not db.admissions.list(session_id=sid, pending_only=False)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_recovery_rejects_mismatched_operator_provenance(monkeypatch, tmp_path):
    sid = "native-session"
    route = "telegram:default:chat"
    source = _source()
    db = storage_handle(tmp_path)
    try:
        seed_session(db, sid, "telegram")
        epoch = db.runtime.begin_epoch(instance_id="first")
        _native_row(db, epoch, sid, route)
        db.admissions.admit(
            epoch=epoch, principal_id="authenticated-viewer", session_id=sid,
            request_id="viewer",
            payload={
                "text": "viewer follow-up",
                "local_operator_v1": {
                    "profile_id": "default",
                    "session_id": sid,
                    "principal_id": "someone-else",
                },
            },
        )

        async def check_native_route(runner, payload, target, available_source, adapter):
            return source, route

        monkeypatch.setattr(
            "gateway.session_envelope.check_native_route", check_native_route)

        recovered = SessionAuthority(
            _runner(), profile_id="default", instance_id="restart", epoch=epoch,
            persistence=storage_handle(tmp_path))
        assert await recovered.recover_native_sessions(
            [(sid, source, object())]
        ) == {sid: "permission_denied"}
    finally:
        db.close()


@pytest.mark.asyncio
async def test_native_route_failure_still_blocks_recovery(monkeypatch, tmp_path):
    sid = "native-session"
    source = _source()
    db = storage_handle(tmp_path)
    try:
        seed_session(db, sid, "telegram")
        epoch = db.runtime.begin_epoch(instance_id="first")
        _native_row(db, epoch, sid, "telegram:default:chat")

        async def reject_route(runner, payload, target, available_source, adapter):
            raise AuthorityRefusal("admission_conflict")

        monkeypatch.setattr(
            "gateway.session_envelope.check_native_route", reject_route)

        recovered = SessionAuthority(
            _runner(), profile_id="default", instance_id="restart", epoch=epoch,
            persistence=storage_handle(tmp_path))
        assert await recovered.recover_native_sessions(
            [(sid, source, object())]
        ) == {sid: "admission_conflict"}
    finally:
        db.close()
