"""Regression coverage for #148164: sticky DEAD must apply only to the 401
that follows a dead refresh grant, not to any later status.

``_mark_exhausted`` used to compute ``terminal = entry.last_status ==
STATUS_DEAD or ...``, so once a row went DEAD (e.g. the persist-failure
quarantine) and was later revived by a re-auth sync, the next ordinary 429
rate limit re-marked it DEAD with no failure reason — a spurious re-auth
loop instead of a cooldown. The comment above the branch always scoped the
intent to "the 401 that follows"; the code now matches it.
"""

from __future__ import annotations

import json
import time

import pytest

from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    STATUS_DEAD,
    STATUS_EXHAUSTED,
    CredentialPool,
    PooledCredential,
)

_EXPIRED_MS = 1_000


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so nothing touches the real auth store."""
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "hermes_cli.auth.is_provider_explicitly_configured", lambda pid: True
    )
    return home


def _dead_entry() -> PooledCredential:
    """Synthetic OAuth row that is already STATUS_DEAD, as after a quarantine."""
    return PooledCredential(
        provider="anthropic",
        id="anthropic-1",
        label="anthropic oauth",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="claude_code",
        access_token="«redacted:sk-…»",
        refresh_token="«redacted:sk-…»",
        expires_at_ms=_EXPIRED_MS,
        last_status=STATUS_DEAD,
        last_status_at=time.time() - 60,
    )


def test_429_on_previously_dead_entry_cools_down_instead_of_remarking_dead(hermes_home):
    """A revived-then-throttled credential gets a cooldown, not a re-auth loop."""
    pool = CredentialPool("anthropic", [_dead_entry()])
    entry = pool.entries()[0]

    after = pool._mark_exhausted(entry, 429, persist=False)

    assert after.last_status == STATUS_EXHAUSTED, (
        "a transient 429 must bench the credential, never re-mark it DEAD — "
        "DEAD on 429 is the spurious re-auth loop from #148164"
    )


def test_401_following_dead_grant_stays_terminal(hermes_home):
    """The documented intent is preserved: the 401 after a dead grant stays DEAD."""
    pool = CredentialPool("anthropic", [_dead_entry()])
    entry = pool.entries()[0]

    after = pool._mark_exhausted(entry, 401, {"reason": "invalid_grant"}, persist=False)

    assert after.last_status == STATUS_DEAD, (
        "the 401 that confirms a dead refresh grant must remain terminal"
    )


def test_reasonless_401_on_dead_entry_also_stays_terminal(hermes_home):
    """Sticky applies to the 401 status itself, independent of the error reason."""
    pool = CredentialPool("anthropic", [_dead_entry()])
    entry = pool.entries()[0]

    after = pool._mark_exhausted(entry, 401, None, persist=False)

    assert after.last_status == STATUS_DEAD
