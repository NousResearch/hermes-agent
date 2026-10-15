"""A pool loaded before ``hermes auth remove`` drops tombstoned rows from memory (#138756).

The store-side tombstone keeps the removed row off disk, but the process that loaded
the pool earlier still holds it in ``self._entries``; without syncing on flush it keeps
 selecting — and reflushing — the removed credential until it restarts."""

from pathlib import Path

import pytest

from agent.credential_pool import load_pool
from hermes_cli.auth import read_credential_pool, write_credential_pool


@pytest.fixture(autouse=True)
def isolated_external_auth_stores(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared"))


def _row(row_id: str, priority: int = 0) -> dict:
    return {
        "id": row_id,
        "label": row_id,
        "auth_type": "api_key",
        "source": "manual",
        "priority": priority,
        "access_token": f"fixture-access-{row_id}",
    }


def test_flush_drops_tombstoned_row_from_memory_and_selection():
    write_credential_pool(
        "openrouter", [_row("keep", priority=1), _row("revoked", priority=0)]
    )
    pool = load_pool("openrouter")
    assert pool.select().id == "revoked"  # priority 0 wins pre-removal

    # `hermes auth remove` in another process tombstones the row on disk.
    write_credential_pool(
        "openrouter", [_row("keep", priority=1)], removed_ids=["revoked"]
    )
    assert [r["id"] for r in read_credential_pool("openrouter")] == ["keep"]

    pool._persist()  # this process's next flush
    assert sorted(e.id for e in pool.entries()) == ["keep"]
    selected = pool.select()
    assert selected is not None and selected.id == "keep"


def test_flush_clears_current_id_pointing_at_removed_row():
    write_credential_pool(
        "openrouter", [_row("keep", priority=1), _row("revoked", priority=0)]
    )
    pool = load_pool("openrouter")
    assert pool.select().id == "revoked"
    assert pool._current_id == "revoked"

    write_credential_pool(
        "openrouter", [_row("keep", priority=1)], removed_ids=["revoked"]
    )
    pool._persist()
    assert "revoked" not in {e.id for e in pool.entries()}
    assert pool._current_id is None


def test_flush_keeps_rows_missing_for_other_reasons():
    # A reference-only row (no token pair on disk) is absent from the written set's
    # token pairs but must not be dropped: only tombstoned ids sync out of memory.
    write_credential_pool(
        "openrouter", [_row("keep", priority=1), _row("revoked", priority=0)]
    )
    pool = load_pool("openrouter")
    # No tombstone anywhere: a flush keeps every entry, removed-id or not.
    pool._persist()
    assert sorted(e.id for e in pool.entries()) == ["keep", "revoked"]
