"""The borrowed ``claude_code`` row is a reference, never a token authority.

``claude_code`` is absent from ``_PERSISTABLE_PROVIDER_SOURCES``, so
``sanitize_borrowed_credential_payload`` strips ``access_token`` and
``refresh_token`` before the pool row reaches ``auth.json``: what survives on
disk is provenance, status and a ``secret_fingerprint``.  ``load_pool()``
re-hydrates the live pair from ``~/.claude/.credentials.json`` on every load,
which is what makes the singleton -- not the pool store -- authoritative for
this source.

Two failure modes follow from forgetting that, and both are covered here:

1. ``_sync_anthropic_entry_from_pool_store()`` re-reads the persisted row
   during refresh.  For a borrowed source that row has *no* tokens, so it
   "differs" from the live entry and was adopted as though another process had
   rotated the pair -- blanking a usable credential and returning before
   ``_claude_code_credentials_lock()`` and the authoritative re-read were ever
   entered.
2. ``_available_entries()`` only refused to lease empty *API-key* rows, so the
   blanked OAuth entry stayed selectable and would have been sent as an empty
   bearer.

The existing race/write-through suites build ``CredentialPool`` objects
directly or back the store with unsanitized in-memory rows, so neither crosses
the real persistence boundary.  Every test below starts from ``load_pool()``
reading an actually persisted, actually sanitized row.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace as dc_replace

import pytest

from agent import anthropic_credentials as AA
from agent.credential_persistence import sanitize_borrowed_credential_payload
from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    CredentialPool,
    PooledCredential,
    load_pool,
)

_EXPIRED_MS = 1_000

_STALE_ACCESS = "sk-ant-oat01-borrowed-stale"
_STALE_REFRESH = "sk-ant-ort01-borrowed-stale"
_ROTATED_ACCESS = "sk-ant-oat01-borrowed-rotated"
_ROTATED_REFRESH = "sk-ant-ort01-borrowed-rotated"


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Real on-disk HERMES_HOME so ``load_pool()`` re-reads what it persisted."""
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "hermes_cli.auth.is_provider_explicitly_configured", lambda pid: True
    )
    return home


@pytest.fixture
def claude_credentials(tmp_path, monkeypatch):
    """Point the ``claude_code`` singleton at a tmp file holding a stale pair."""
    cred_path = tmp_path / "claude" / ".credentials.json"
    cred_path.parent.mkdir(parents=True, exist_ok=True)
    cred_path.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": _STALE_ACCESS,
                    "refreshToken": _STALE_REFRESH,
                    "expiresAt": _EXPIRED_MS,
                    "scopes": ["user:inference", "user:profile"],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(AA, "claude_code_credentials_path", lambda: cred_path)
    monkeypatch.setattr(AA, "_read_claude_code_credentials_from_keychain", lambda: None)
    return cred_path


def _persisted_rows(home):
    store = json.loads((home / "auth.json").read_text(encoding="utf-8"))
    return store.get("credential_pool", {}).get("anthropic", [])


def _claude_pair(cred_path):
    data = json.loads(cred_path.read_text(encoding="utf-8"))["claudeAiOauth"]
    return data["accessToken"], data["refreshToken"]


def _rotating_refresh(refresh_token, **_kw):
    return {
        "access_token": _ROTATED_ACCESS,
        "refresh_token": _ROTATED_REFRESH,
        "expires_at_ms": int(time.time() * 1000) + 3_600_000,
    }


def test_persisted_claude_code_row_carries_no_token_material(
    hermes_home, claude_credentials
):
    """Baseline: the row the refresh path re-reads really is sanitized.

    Every other test in this file only means something if the disk row is
    token-less, so assert the boundary rather than assuming it.
    """
    pool = load_pool("anthropic")

    live = [e for e in pool._entries if e.source == "claude_code"]
    assert len(live) == 1
    assert live[0].access_token == _STALE_ACCESS, (
        "load_pool must hydrate the live pair from the singleton"
    )

    rows = [r for r in _persisted_rows(hermes_home) if r.get("source") == "claude_code"]
    assert len(rows) == 1
    assert not rows[0].get("access_token")
    assert not rows[0].get("refresh_token")
    assert str(rows[0].get("secret_fingerprint", "")).startswith("sha256:")
    assert sanitize_borrowed_credential_payload(rows[0], "anthropic") == rows[0]


def test_pool_store_sync_never_adopts_a_borrowed_row(hermes_home, claude_credentials):
    """The sanitized row must not be mistaken for a rotation by another process."""
    pool = load_pool("anthropic")
    entry = next(e for e in pool._entries if e.source == "claude_code")

    synced = pool._sync_anthropic_entry_from_pool_store(entry)

    assert synced is entry, "a borrowed row is a reference, not token authority"
    assert synced.access_token == _STALE_ACCESS
    assert synced.refresh_token == _STALE_REFRESH


def test_refresh_from_persisted_sanitized_row_keeps_the_full_pair(
    hermes_home, claude_credentials, monkeypatch
):
    """The production ``load -> sanitize -> refresh`` path refreshes, not blanks.

    Exactly one POST and one authoritative write, the returned entry carries
    the complete rotated pair, and the shared credentials file is the copy that
    was updated.
    """
    posts = []
    writes = []

    def _counting_refresh(refresh_token, **kwargs):
        posts.append(refresh_token)
        return _rotating_refresh(refresh_token, **kwargs)

    real_write = AA._write_claude_code_credentials

    def _counting_write(access_token, refresh_token, expires_at_ms):
        writes.append(refresh_token)
        return real_write(access_token, refresh_token, expires_at_ms)

    monkeypatch.setattr(AA, "refresh_anthropic_oauth_pure", _counting_refresh)
    monkeypatch.setattr(AA, "_write_claude_code_credentials", _counting_write)

    pool = load_pool("anthropic")
    entry = next(e for e in pool._entries if e.source == "claude_code")

    refreshed = pool._refresh_entry(entry, force=True)

    assert refreshed is not None, "the refresh must not be abandoned"
    assert refreshed.access_token == _ROTATED_ACCESS
    assert refreshed.refresh_token == _ROTATED_REFRESH
    assert posts == [_STALE_REFRESH], f"expected exactly one POST, got {posts}"
    assert writes == [_ROTATED_REFRESH], f"expected exactly one commit, got {writes}"
    assert _claude_pair(claude_credentials) == (_ROTATED_ACCESS, _ROTATED_REFRESH)


def test_refresh_reaches_the_shared_credentials_lock(
    hermes_home, claude_credentials, monkeypatch
):
    """``claude_code`` must always take the path-keyed lock before deciding.

    That lock is what serializes profiles sharing one
    ``~/.claude/.credentials.json``; an adopt-and-return shortcut firing first
    would leave the cross-profile race exactly where it was.
    """
    taken = []
    real_lock = CredentialPool._claude_code_credentials_lock

    def _tracking_lock(self):
        taken.append(True)
        return real_lock(self)

    monkeypatch.setattr(CredentialPool, "_claude_code_credentials_lock", _tracking_lock)
    monkeypatch.setattr(AA, "refresh_anthropic_oauth_pure", _rotating_refresh)

    pool = load_pool("anthropic")
    entry = next(e for e in pool._entries if e.source == "claude_code")
    pool._refresh_entry(entry, force=True)

    assert taken, "the authoritative re-read must happen under the shared-file lock"


def test_empty_oauth_entry_is_never_leased(hermes_home, claude_credentials):
    """A token-less OAuth row must not be selectable as an empty bearer.

    The pre-existing guard covered ``AUTH_TYPE_API_KEY`` only, so an OAuth row
    that failed to hydrate went straight into the available list.
    """
    pool = load_pool("anthropic")
    entry = next(e for e in pool._entries if e.source == "claude_code")
    blanked = dc_replace(entry, access_token="", refresh_token="")
    pool._replace_entry(entry, blanked)

    available, _pending = pool._available_entries(clear_expired=False, refresh=False)

    assert all(e.access_token for e in available), (
        "an OAuth entry with no access token must never be leased"
    )
    assert blanked.id not in {e.id for e in available}


def test_selection_after_refresh_leases_only_hydrated_entries(
    hermes_home, claude_credentials, monkeypatch
):
    """End-to-end: refresh through selection leaves a usable, non-empty lease."""
    monkeypatch.setattr(AA, "refresh_anthropic_oauth_pure", _rotating_refresh)

    pool = load_pool("anthropic")
    available, _pending = pool._available_entries(clear_expired=True, refresh=True)

    assert available, "the credential must survive the refresh, not be dropped"
    assert all(e.access_token for e in available)
    assert any(e.access_token == _ROTATED_ACCESS for e in available)


def test_hermes_pkce_row_still_syncs_from_the_pool_store(monkeypatch):
    """The borrowed-source refusal must not disable pool-owned adoption.

    ``hermes_pkce`` *is* pool-owned, so its persisted row keeps its tokens and
    stays a legitimate rotation witness for another pool instance.
    """
    rotated = {
        "id": "anthropic-pkce",
        "label": "anthropic oauth",
        "auth_type": AUTH_TYPE_OAUTH,
        "priority": 0,
        "source": "hermes_pkce",
        "access_token": _ROTATED_ACCESS,
        "refresh_token": _ROTATED_REFRESH,
        "expires_at_ms": int(time.time() * 1000) + 3_600_000,
    }
    monkeypatch.setattr(
        "agent.credential_pool.read_credential_pool", lambda provider=None: [rotated]
    )

    entry = PooledCredential(
        provider="anthropic",
        id="anthropic-pkce",
        label="anthropic oauth",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="hermes_pkce",
        access_token=_STALE_ACCESS,
        refresh_token=_STALE_REFRESH,
        expires_at_ms=_EXPIRED_MS,
    )
    pool = CredentialPool("anthropic", [entry])

    synced = pool._sync_anthropic_entry_from_pool_store(entry)

    assert synced.access_token == _ROTATED_ACCESS
    assert synced.refresh_token == _ROTATED_REFRESH
