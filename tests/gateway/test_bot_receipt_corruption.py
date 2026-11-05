"""Bulk Bot receipt recovery isolates bad files without weakening exact-id failures."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from gateway.session_authority import admission_row
from gateway.session_bot import _migrate, recover_bot_deliveries
from gateway.session_results import retain_result
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_session
from tools.bot_live_delivery import _locked, _read, _write


def _settled_admission(db, epoch, request_id, text, reply):
    admission, _ = db.admissions.admit(
        epoch=epoch,
        principal_id="owner",
        session_id="bot",
        request_id=request_id,
        payload={"text": text},
    )
    row = admission_row(db.admissions.claim(epoch=epoch, session_id="bot"))
    retain_result(
        db,
        epoch=epoch,
        row=row,
        result={"result": {"final_response": reply}, "usage": {}},
    )
    return admission


@pytest.mark.asyncio
async def test_recovery_skips_structurally_corrupt_canonical_neighbor(tmp_path):
    db = open_persistence(profile_scope(home=tmp_path))
    try:
        seed_session(db, "bot", "gui")
        epoch = db.runtime.begin_epoch(instance_id="owner")
        healthy = _settled_admission(db, epoch, "healthy", "hello", "recovered")
        damaged = _settled_admission(db, epoch, "damaged", "bad", "should-not-be-read")

        healthy_key = "a" * 32
        damaged_key = "b" * 32
        with _locked(tmp_path) as root:
            _write(
                root / f"{healthy_key}.json",
                {
                    "status": "canonical",
                    "admission_id": healthy.admission_id,
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
                    "admission_id": damaged.admission_id,
                    "delivery_id": damaged_key,
                    "profile_home": str(tmp_path),
                    "session_id": "bot",
                    "principal_id": "owner",
                    # Valid JSON and a real admission, but structurally incomplete.
                },
            )

        await recover_bot_deliveries(SimpleNamespace(db=db, persistence=db, waiters={}))

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
async def test_recovery_does_not_hide_real_admission_lookup_failure(tmp_path, caplog):
    """A live session missing a receipt's admission is a storage fault: recovery logs it by
    delivery and reason and leaves the receipt as it was, but never stops the owner's start."""
    db = open_persistence(profile_scope(home=tmp_path))
    try:
        seed_session(db, "bot", "gui")
        db.runtime.begin_epoch(instance_id="owner")
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

        await recover_bot_deliveries(SimpleNamespace(db=db, persistence=db, waiters={}))
        assert f"Bot delivery {key} not recovered: storage_unavailable" in caplog.text
        with _locked(tmp_path) as root:
            assert _read(root / f"{key}.json")["status"] == "canonical"
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
            persistence=SimpleNamespace(lineage=SimpleNamespace(tip=lambda session_id: "tip")),
        )
        ref = SimpleNamespace(session_id="tip")
        live = object()
        entry = SimpleNamespace(session_id="tip")
        async def target(authority, actor):
            return ref, live, entry
        monkeypatch.setattr("gateway.session_bot._target", target)

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
            notification_category="result",
        ):
            migrated.append((key, message, legacy["status"]))

        monkeypatch.setattr("gateway.session_bot._admit", fake_admit)

    await _migrate(authority, object(), tmp_path, root)

    assert migrated == [(healthy_key, "healthy legacy", "queued")]
