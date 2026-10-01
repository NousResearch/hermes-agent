"""Bulk Bot receipt recovery isolates bad files without weakening exact-id failures."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from gateway.session_bot import _migrate, recover_bot_deliveries
from gateway.session_results import retain_result
from hermes_state import SessionDB
from hermes_state_runtime import (
    RuntimeStoreError,
    admit_session_input,
    begin_runtime_epoch,
    claim_session_input,
)
from tools.bot_live_delivery import _locked, _read, _write


def _settled_admission(db, epoch, request_id, text, reply):
    admission = admit_session_input(
        db,
        epoch=epoch,
        principal_id="owner",
        session_id="bot",
        request_id=request_id,
        payload={"text": text},
    )
    row = claim_session_input(db, epoch=epoch, session_id="bot")
    retain_result(
        db,
        epoch=epoch,
        row=row,
        result={"result": {"final_response": reply}, "usage": {}},
    )
    return admission


@pytest.mark.asyncio
async def test_recovery_skips_structurally_corrupt_canonical_neighbor(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("bot", source="gui")
        epoch = begin_runtime_epoch(db, instance_id="owner")
        healthy = _settled_admission(db, epoch, "healthy", "hello", "recovered")
        damaged = _settled_admission(db, epoch, "damaged", "bad", "should-not-be-read")

        healthy_key = "a" * 32
        damaged_key = "b" * 32
        with _locked(tmp_path) as root:
            _write(
                root / f"{healthy_key}.json",
                {
                    "status": "canonical",
                    "admission_id": healthy["admission_id"],
                    "delivery_id": healthy_key,
                    "profile_home": str(tmp_path),
                    "session_id": "bot",
                    "principal_id": "owner",
                    "message": "hello",
                },
            )
            _write(
                root / f"{damaged_key}.json",
                {
                    "status": "canonical",
                    "admission_id": damaged["admission_id"],
                    "delivery_id": damaged_key,
                    "profile_home": str(tmp_path),
                    "session_id": "bot",
                    "principal_id": "owner",
                    # Valid JSON and a real admission, but structurally incomplete.
                },
            )

        await recover_bot_deliveries(SimpleNamespace(db=db, waiters={}))

        with _locked(tmp_path) as root:
            recovered = _read(root / f"{healthy_key}.json")
            skipped = _read(root / f"{damaged_key}.json")
        assert recovered["status"] == "settled"
        assert recovered["reply"] == "recovered"
        assert skipped["status"] == "canonical"
        assert "message" not in skipped
    finally:
        db.close()


@pytest.mark.asyncio
async def test_recovery_does_not_hide_real_admission_lookup_failure(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("bot", source="gui")
        begin_runtime_epoch(db, instance_id="owner")
        key = "c" * 32
        with _locked(tmp_path) as root:
            _write(
                root / f"{key}.json",
                {
                    "status": "canonical",
                    "admission_id": "missing-admission",
                    "delivery_id": key,
                    "profile_home": str(tmp_path),
                    "session_id": "bot",
                    "principal_id": "owner",
                    "message": "valid receipt, missing durable admission",
                },
            )

        with pytest.raises(RuntimeStoreError) as caught:
            await recover_bot_deliveries(SimpleNamespace(db=db, waiters={}))
        assert caught.value.reason == "storage_unavailable"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_legacy_migration_skips_structurally_corrupt_neighbor(monkeypatch, tmp_path):
    healthy_key = "d" * 32
    damaged_key = "e" * 32
    owner = {
        "profile_home": str(tmp_path),
        "session_id": "legacy-session",
        "lease_id": "departed-owner",
        "live_session_id": "old-native",
    }
    with _locked(tmp_path) as root:
        _write(
            root / f"{healthy_key}.json",
            {
                "id": healthy_key,
                "delivery_id": healthy_key,
                "status": "queued",
                "created_at": 1,
                "sequence": 1,
                "owner": owner,
                "message": "healthy legacy",
            },
        )
        _write(root / f"{damaged_key}.json", {"owner": {}})

        authority = SimpleNamespace(
            db=SimpleNamespace(get_compression_tip=lambda session_id: "tip"),
        )
        ref = SimpleNamespace(session_id="tip")
        live = object()
        entry = SimpleNamespace(session_id="tip")
        monkeypatch.setattr(
            "gateway.session_bot._target",
            lambda authority, actor: (ref, live, entry),
        )

        @contextmanager
        def inactive_session(*args, **kwargs):
            yield False

        monkeypatch.setattr(
            "hermes_cli.active_sessions.active_session_liveness_guard",
            inactive_session,
        )

        migrated = []

        async def fake_admit(
            authority,
            actor,
            home,
            root,
            key,
            message,
            ref,
            live,
            entry,
            author=None,
            legacy=None,
        ):
            migrated.append((key, message, legacy["status"]))

        monkeypatch.setattr("gateway.session_bot._admit", fake_admit)

        await _migrate(authority, object(), tmp_path, root)

    assert migrated == [(healthy_key, "healthy legacy", "queued")]
