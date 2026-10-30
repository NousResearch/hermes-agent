"""End-to-end tests for shared Hermes PKCE grants across real profile homes.

Unlike ``test_credential_pool_shared_pkce.py`` (unit tests around the pool
object), these tests drive the real persistence path: two ``HERMES_HOME``
layouts (``<root>`` and ``<root>/profiles/<name>``), real ``auth.json`` files,
real ``load_pool()`` / ``select()`` / ``_refresh_entry()``.  Only the network
refresh call (``refresh_anthropic_oauth_pure``) is mocked.
"""

import json
import os
from pathlib import Path

import pytest

ROOT_NAME = "hermes-root"
WORKER_NAME = "worker"
LABEL = "cg.chae-claude"
PAST_MS = 1_000
FUTURE_MS = 4_102_444_800_000  # 2100-01-01


def _copy_entry(label=LABEL, access="copied-access", refresh="copied-refresh", expires=FUTURE_MS):
    return {
        "id": f"id-{label}",
        "label": label,
        "auth_type": "oauth",
        "priority": 0,
        "source": "manual:hermes_pkce",
        "access_token": access,
        "refresh_token": refresh,
        "expires_at_ms": expires,
    }


def _write_store(home: Path, entries: list) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": {"anthropic": entries}}, indent=2)
    )


def _read_store(home: Path) -> list:
    return json.loads((home / "auth.json").read_text())["credential_pool"]["anthropic"]


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """Real root + worker profile layout. Returns (root, worker)."""
    root = tmp_path / ROOT_NAME
    worker = root / "profiles" / WORKER_NAME
    root.mkdir(parents=True)
    worker.mkdir(parents=True)
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("hermes_cli.auth.is_provider_explicitly_configured", lambda pid: True)
    # Keep the real ~/.claude and legacy singleton readers out of the picture.
    monkeypatch.setattr("agent.anthropic_adapter.read_claude_code_credentials", lambda: None)
    monkeypatch.setattr("agent.anthropic_adapter.read_hermes_oauth_credentials", lambda: None)
    return root, worker


def _use_home(monkeypatch, home: Path):
    monkeypatch.setenv("HERMES_HOME", str(home))


def _load(monkeypatch, home: Path):
    _use_home(monkeypatch, home)
    from agent.credential_pool import load_pool

    return load_pool("anthropic")


def _shared_file(root: Path, label: str = LABEL) -> Path:
    safe = label.replace(" ", "_")
    return root / ".anthropic_oauth" / f"{safe}.json"


def _mock_refresh(monkeypatch, *, valid: set, counter: list):
    """Refresh succeeds only for tokens in ``valid``; each success rotates."""

    def _refresh(refresh_token, *, use_json=False):
        counter.append(refresh_token)
        if refresh_token not in valid:
            raise RuntimeError("HTTP Error 404: Not Found")
        new_refresh = f"{refresh_token}-rot"
        valid.discard(refresh_token)
        valid.add(new_refresh)
        return {
            "access_token": f"access-from-{refresh_token}",
            "refresh_token": new_refresh,
            "expires_at_ms": FUTURE_MS,
        }

    monkeypatch.setattr("agent.anthropic_adapter.refresh_anthropic_oauth_pure", _refresh)


# ---------------------------------------------------------------------------


def test_root_seeds_shared_grant_and_worker_copy_becomes_reference(homes, monkeypatch):
    root, worker = homes
    _write_store(root, [_copy_entry(access="root-access", refresh="root-refresh")])
    _write_store(worker, [_copy_entry(access="stale-access", refresh="stale-refresh")])

    root_pool = _load(monkeypatch, root)
    assert [e.source for e in root_pool.entries()] == ["hermes_pkce_shared"]
    shared = json.loads(_shared_file(root).read_text())
    assert shared["refreshToken"] == "root-refresh"
    assert oct(_shared_file(root).stat().st_mode & 0o777) == oct(0o600)

    worker_pool = _load(monkeypatch, worker)
    rows = _read_store(worker)
    assert [r["source"] for r in rows] == ["hermes_pkce_shared"]
    assert rows[0]["label"] == LABEL
    # The worker's stale copy must never leak into the shared source.
    assert json.loads(_shared_file(root).read_text())["refreshToken"] == "root-refresh"

    selected = worker_pool.select()
    assert selected is not None
    assert selected.label == LABEL
    assert selected.access_token == "root-access"
    assert selected.refresh_token == "root-refresh"


def test_worker_loaded_before_root_keeps_its_copy_usable(homes, monkeypatch):
    """Order independence: no shared grant yet -> copy stays a copy, not a dangling ref."""
    root, worker = homes
    _write_store(worker, [_copy_entry(access="copy-access", refresh="copy-refresh")])

    pool = _load(monkeypatch, worker)
    assert not _shared_file(root).exists()
    rows = _read_store(worker)
    assert rows[0]["source"] == "manual:hermes_pkce"
    assert rows[0]["access_token"] == "copy-access"
    selected = pool.select()
    assert selected is not None and selected.access_token == "copy-access"

    # Root comes online later and publishes; the worker converts on its next load.
    _write_store(root, [_copy_entry(access="root-access", refresh="root-refresh")])
    _load(monkeypatch, root)
    pool2 = _load(monkeypatch, worker)
    assert _read_store(worker)[0]["source"] == "hermes_pkce_shared"
    assert pool2.select().access_token == "root-access"


def test_refresh_in_one_profile_is_visible_to_the_other_without_refreshing(homes, monkeypatch):
    root, worker = homes
    _write_store(root, [_copy_entry(access="root-access", refresh="root-refresh", expires=PAST_MS)])
    _write_store(worker, [_copy_entry(access="stale", refresh="stale", expires=PAST_MS)])
    calls: list = []
    _mock_refresh(monkeypatch, valid={"root-refresh"}, counter=calls)

    root_pool = _load(monkeypatch, root)
    _load(monkeypatch, worker)  # migrate the worker row to a shared reference

    # Root refreshes (expired) -> publishes rotated pair to the shared file.
    chosen = root_pool.select()
    assert chosen is not None and chosen.access_token == "access-from-root-refresh"
    assert calls == ["root-refresh"]
    shared = json.loads(_shared_file(root).read_text())
    assert shared["refreshToken"] == "root-refresh-rot"

    # Worker's next select() picks up the rotated pair from the file: no refresh call.
    worker_pool = _load(monkeypatch, worker)
    picked = worker_pool.select()
    assert picked is not None
    assert picked.access_token == "access-from-root-refresh"
    assert picked.refresh_token == "root-refresh-rot"
    assert calls == ["root-refresh"]
    # Reference rows never persist token material into the profile auth.json.
    row = _read_store(worker)[0]
    assert row["source"] == "hermes_pkce_shared"
    assert not row.get("access_token") and not row.get("refresh_token")


def test_rotated_shared_grant_is_adopted_under_lock_without_posting(homes, monkeypatch):
    """Concurrent-refresh loser: the shared file already moved on -> adopt, no POST."""
    root, worker = homes
    _write_store(root, [_copy_entry(access="root-access", refresh="root-refresh")])
    _write_store(worker, [_copy_entry(access="stale", refresh="stale")])
    _load(monkeypatch, root)
    worker_pool = _load(monkeypatch, worker)
    hydrated = worker_pool.select()  # reference rows hydrate from the shared file on selection
    assert hydrated is not None and hydrated.source == "hermes_pkce_shared"
    assert hydrated.refresh_token == "root-refresh"

    # Another profile rotates the grant after this pool hydrated.
    from agent.anthropic_adapter import write_shared_hermes_oauth_credentials

    _use_home(monkeypatch, root)
    assert write_shared_hermes_oauth_credentials(
        {"accessToken": "newer-access", "refreshToken": "newer-refresh", "expiresAt": FUTURE_MS},
        label=LABEL,
    )
    _use_home(monkeypatch, worker)
    calls: list = []
    _mock_refresh(monkeypatch, valid={"newer-refresh"}, counter=calls)

    refreshed = worker_pool._refresh_entry(hydrated, force=True)
    assert refreshed is not None
    assert calls == []  # in-lock re-sync picked up the rotated pair; no network call
    assert refreshed.access_token == "newer-access"
    assert refreshed.refresh_token == "newer-refresh"


def test_revoked_refresh_token_syncs_from_shared_and_retries_once(homes, monkeypatch):
    """Server-side revocation with an out-of-band rotation -> 404, re-sync, single retry."""
    root, worker = homes
    _write_store(root, [_copy_entry(access="root-access", refresh="root-refresh")])
    _write_store(worker, [_copy_entry(access="stale", refresh="stale")])
    _load(monkeypatch, root)
    worker_pool = _load(monkeypatch, worker)
    hydrated = worker_pool.select()
    assert hydrated is not None and hydrated.refresh_token == "root-refresh"

    from agent.anthropic_adapter import write_shared_hermes_oauth_credentials

    calls: list = []

    def _refresh(refresh_token, *, use_json=False):
        calls.append(refresh_token)
        if refresh_token == "root-refresh":
            # An external writer (not holding our lock) rotated the grant and
            # published it between our sync and our POST.
            _use_home(monkeypatch, root)
            assert write_shared_hermes_oauth_credentials(
                {"accessToken": "ext-access", "refreshToken": "ext-refresh", "expiresAt": FUTURE_MS},
                label=LABEL,
            )
            _use_home(monkeypatch, worker)
            raise RuntimeError("HTTP Error 404: Not Found")
        assert refresh_token == "ext-refresh"
        return {"access_token": "access-from-ext", "refresh_token": "ext-refresh-rot", "expires_at_ms": FUTURE_MS}

    monkeypatch.setattr("agent.anthropic_adapter.refresh_anthropic_oauth_pure", _refresh)

    refreshed = worker_pool._refresh_entry(hydrated, force=True)
    assert refreshed is not None
    assert calls == ["root-refresh", "ext-refresh"]  # 404 -> sync -> exactly one retry
    assert refreshed.access_token == "access-from-ext"
    assert json.loads(_shared_file(root).read_text())["refreshToken"] == "ext-refresh-rot"
    row = _read_store(worker)[0]
    assert row["source"] == "hermes_pkce_shared"
    assert not row.get("access_token") and not row.get("refresh_token")


def test_two_labels_migrate_independently_without_cross_hydration(homes, monkeypatch):
    root, worker = homes
    other = "second-account"
    _write_store(
        root,
        [
            _copy_entry(access="a-access", refresh="a-refresh"),
            {**_copy_entry(other, access="b-access", refresh="b-refresh"), "id": "id-b", "priority": 1},
        ],
    )
    _write_store(
        worker,
        [
            _copy_entry(access="old-a", refresh="old-a"),
            {**_copy_entry(other, access="old-b", refresh="old-b"), "id": "id-b", "priority": 1},
        ],
    )
    _load(monkeypatch, root)
    assert json.loads(_shared_file(root).read_text())["refreshToken"] == "a-refresh"
    assert json.loads(_shared_file(root, other).read_text())["refreshToken"] == "b-refresh"

    worker_pool = _load(monkeypatch, worker)
    available = worker_pool._available_entries()  # hydrates every reference row
    by_label = {e.label: e for e in available}
    assert by_label[LABEL].access_token == "a-access"
    assert by_label[other].access_token == "b-access"
    assert {e.source for e in available} == {"hermes_pkce_shared"}


def test_malformed_shared_file_skips_reference_with_warning_and_keeps_row(homes, monkeypatch, caplog):
    root, worker = homes
    _write_store(root, [_copy_entry(access="root-access", refresh="root-refresh")])
    _write_store(worker, [_copy_entry(access="stale", refresh="stale")])
    _load(monkeypatch, root)
    _load(monkeypatch, worker)

    _shared_file(root).write_text("{not json")
    with caplog.at_level("WARNING"):
        pool = _load(monkeypatch, worker)
        assert pool.select() is None
    assert any("shared OAuth source is missing" in r.getMessage() for r in caplog.records)
    # Row is preserved (not pruned) so it recovers once the source is restored.
    assert _read_store(worker)[0]["source"] == "hermes_pkce_shared"

    _use_home(monkeypatch, root)
    from agent.anthropic_adapter import write_shared_hermes_oauth_credentials

    assert write_shared_hermes_oauth_credentials(
        {"accessToken": "restored", "refreshToken": "restored-r", "expiresAt": FUTURE_MS}, label=LABEL
    )
    assert _load(monkeypatch, worker).select().access_token == "restored"


def test_empty_profile_discovers_every_shared_label_without_copies(homes, monkeypatch):
    """A profile with no pool rows (e.g. a freshly created worker) must see all
    published grants by label -- no `auth add`, no copied tokens."""
    root, worker = homes
    other = "second account"  # label with a character the filename sanitizer rewrites
    from agent.anthropic_adapter import write_shared_hermes_oauth_credentials

    _use_home(monkeypatch, root)
    assert write_shared_hermes_oauth_credentials(
        {"accessToken": "a-access", "refreshToken": "a-refresh", "expiresAt": FUTURE_MS}, label=LABEL
    )
    assert write_shared_hermes_oauth_credentials(
        {"accessToken": "b-access", "refreshToken": "b-refresh", "expiresAt": FUTURE_MS}, label=other
    )
    _write_store(worker, [])

    pool = _load(monkeypatch, worker)
    available = pool._available_entries()
    by_label = {e.label: e for e in available}
    assert set(by_label) == {LABEL, other}
    assert {e.source for e in available} == {"hermes_pkce_shared"}
    assert by_label[LABEL].access_token == "a-access"
    assert by_label[other].access_token == "b-access"
    assert pool.select() is not None

    rows = _read_store(worker)
    assert sorted(r["label"] for r in rows) == sorted([LABEL, other])
    assert all(r["source"] == "hermes_pkce_shared" for r in rows)
    assert all(not r.get("access_token") and not r.get("refresh_token") for r in rows)

    # Reloading must not duplicate rows or collapse the two labels into one.
    pool2 = _load(monkeypatch, worker)
    assert len(_read_store(worker)) == 2
    assert {e.label for e in pool2._available_entries()} == {LABEL, other}


def test_empty_profile_discovers_legacy_grants_map_file(homes, monkeypatch):
    """Pre-directory layout: `<root>/.anthropic_oauth.json` holding a grants map."""
    root, worker = homes
    (root / ".anthropic_oauth.json").write_text(
        json.dumps({"grants": {LABEL: {"accessToken": "legacy-access", "refreshToken": "legacy-refresh", "expiresAt": FUTURE_MS}}})
    )
    _write_store(worker, [])
    pool = _load(monkeypatch, worker)
    picked = pool.select()
    assert picked is not None
    assert picked.label == LABEL and picked.access_token == "legacy-access"
    assert _read_store(worker)[0]["source"] == "hermes_pkce_shared"
