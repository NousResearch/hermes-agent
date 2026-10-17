"""A pool loaded before ``hermes auth remove`` drops tombstoned rows from memory (#138756).

The store-side tombstone keeps the removed row off disk, but the process that loaded
the pool earlier still holds it in ``self._entries``; without syncing on flush it keeps
 selecting — and reflushing — the removed credential until it restarts."""

from pathlib import Path

import pytest

import hermes_constants
from agent.credential_pool import PooledCredential, load_pool
from hermes_cli import auth
from hermes_cli.auth import read_credential_pool, write_credential_pool


@pytest.fixture(autouse=True)
def isolated_external_auth_stores(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared"))


@pytest.fixture
def borrowed_root_pool(tmp_path, monkeypatch):
    """A named profile (no rows of its own) borrowing the root's single-use rows.

    Drives the REAL path resolution (HERMES_HOME switch + default-root memo reset,
    the way ``tests/agent/test_credential_pool_profile_oauth_fork.py`` does) so the
    borrowed-root write-through in ``persist_pool_entries`` engages for real. Returns
    a writer that persists through the ROOT store, the way ``hermes auth remove``
    in the root does.
    """
    root = tmp_path / "hermes-root"
    profile = root / "profiles" / "work"
    profile.mkdir(parents=True)
    # Keep the host's Claude Code login and env credentials out of the pool seeds,
    # and point HOME away from the root so the pytest seat belt in
    # ``_guarded_global_root`` allows the root write-through.
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "not-the-root"))
    for var in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)

    def _scope(home: Path) -> None:
        monkeypatch.setenv("HERMES_HOME", str(home))
        hermes_constants._default_hermes_root_memo = None
        auth._global_auth_store_cache = None

    _scope(root)

    def _write_root(rows, **kwargs):
        _scope(root)
        try:
            write_credential_pool("anthropic", rows, **kwargs)
        finally:
            _scope(profile)

    return _write_root


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


def test_borrowed_root_flush_drops_tombstoned_row_from_profile_memory(borrowed_root_pool):
    # Profile mode: the borrowed single-use pool persists through the root store, and
    # the root's ``hermes auth remove`` tombstones THERE; reading the profile's own
    # store finds nothing and would keep the removed row selectable until restart.
    borrowed_root_pool([_row("keep", priority=1), _row("revoked", priority=0)])
    pool = load_pool("anthropic")  # borrows the root's rows via the global fallback
    assert pool.select().id == "revoked"  # priority 0 wins pre-removal

    # `hermes auth remove` in the root tombstones the row in the ROOT's auth.json.
    borrowed_root_pool([_row("keep", priority=1)], removed_ids=["revoked"])
    assert [r["id"] for r in read_credential_pool("anthropic")] == ["keep"]

    pool._persist()  # the profile process's next flush
    assert sorted(e.id for e in pool.entries()) == ["keep"]
    selected = pool.select()
    assert selected is not None and selected.id == "keep"


def test_borrowed_root_flush_keeps_rows_without_a_root_tombstone(borrowed_root_pool):
    # A row the root never had (e.g. one seeded locally in the profile process) is
    # missing from the written set on every flush; without a root tombstone it must
    # not be dropped — only tombstoned ids sync out of memory.
    borrowed_root_pool([_row("keep", priority=1)])
    pool = load_pool("anthropic")
    pool._entries.append(PooledCredential.from_dict("anthropic", _row("seeded", priority=2)))

    pool._persist()
    assert sorted(e.id for e in pool.entries()) == ["keep", "seeded"]
