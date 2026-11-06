"""Removal tombstones keep ``hermes auth remove`` stable across stale writers (#138756).

A long-running process that loaded the pool before the removal still holds the row in
memory; without the tombstone its next flush resurrects the removed credential."""
import json
import time
from pathlib import Path

import pytest

from hermes_cli.auth import read_credential_pool, write_credential_pool


@pytest.fixture(autouse=True)
def isolated_external_auth_stores(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared"))


def _row(row_id: str) -> dict:
    return {"id": row_id, "label": row_id, "auth_type": "api_key", "source": "manual",
            "priority": 0, "access_token": f"fixture-access-{row_id}"}


def _auth_path() -> Path:
    path = Path.home() / ".hermes" / "auth.json"
    if not path.exists():
        from hermes_cli.auth import _auth_file_path
        path = _auth_file_path()
    return path


def _auth_json() -> dict:
    path = _auth_path()
    return json.loads(path.read_text()) if path.exists() else {}


def _ids(rows) -> list:
    return [r["id"] for r in rows if isinstance(r, dict) and r.get("id")]


def test_stale_process_cannot_resurrect_removed_row():
    write_credential_pool("openrouter", [_row("keep"), _row("revoked")])
    stale_in_memory = [_row("keep"), _row("revoked")]  # process B loaded both rows

    write_credential_pool("openrouter", [_row("keep")], removed_ids=["revoked"])
    assert _ids(read_credential_pool("openrouter")) == ["keep"]

    written = write_credential_pool("openrouter", stale_in_memory)  # B's next flush
    assert _ids(written) == ["keep"]
    assert _ids(read_credential_pool("openrouter")) == ["keep"]


def test_removal_keeps_concurrently_added_row():
    write_credential_pool("openrouter", [_row("keep"), _row("revoked")])
    write_credential_pool("openrouter", [_row("keep")], removed_ids=["revoked"])

    # A concurrent `auth add` lands on disk, then the stale writer flushes a
    # snapshot that predates it: the merge-back must keep the added row while
    # the tombstone still suppresses the removed one.
    write_credential_pool("openrouter", [_row("keep"), _row("fresh-add")])
    written = write_credential_pool("openrouter", [_row("keep"), _row("revoked")])
    assert sorted(_ids(written)) == ["fresh-add", "keep"]
    assert sorted(_ids(read_credential_pool("openrouter"))) == ["fresh-add", "keep"]


def test_readded_credential_with_fresh_id_survives_tombstone():
    write_credential_pool("openrouter", [_row("old-id")])
    write_credential_pool("openrouter", [], removed_ids=["old-id"])

    # `auth add` mints a new uuid id for the same label; the tombstone must not eat it.
    written = write_credential_pool("openrouter", [_row("new-id")])
    assert _ids(written) == ["new-id"]
    assert _ids(read_credential_pool("openrouter")) == ["new-id"]


def test_expired_tombstone_stops_filtering_and_is_dropped():
    write_credential_pool("openrouter", [_row("revoked")])
    write_credential_pool("openrouter", [], removed_ids=["revoked"])

    store = _auth_json()
    store["credential_pool_removals"]["openrouter"]["revoked"] = time.time() - 8 * 24 * 3600
    _auth_path().write_text(json.dumps(store))

    written = write_credential_pool("openrouter", [_row("revoked")])
    assert _ids(written) == ["revoked"]
    assert "openrouter" not in _auth_json().get("credential_pool_removals", {})


def test_plain_write_creates_no_tombstone_section():
    write_credential_pool("openrouter", [_row("keep")])
    assert "credential_pool_removals" not in _auth_json()


def test_removal_ids_read_back_unexpired_and_filtered():
    from hermes_cli.auth import credential_pool_removal_ids

    write_credential_pool("openrouter", [_row("keep")], removed_ids=["revoked", "other"])
    store = _auth_json()
    store["credential_pool_removals"]["openrouter"]["expired"] = time.time() - 8 * 24 * 3600
    _auth_path().write_text(json.dumps(store))

    assert credential_pool_removal_ids("openrouter") == {"revoked", "other"}
    assert credential_pool_removal_ids("openrouter", ["revoked", "unrelated"]) == {"revoked"}
    assert credential_pool_removal_ids("openrouter", []) == set()
    assert credential_pool_removal_ids("absent-provider") == set()
