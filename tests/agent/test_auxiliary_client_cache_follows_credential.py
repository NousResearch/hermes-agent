"""Regression coverage: the auxiliary client cache key follows the pooled credential (#113022).

Points 1/2/5 of the issue: a token rotated or refreshed by another process must yield a new
client instead of a 401 round trip on the cached one, including while a per-model cooldown
hides the entry from ``peek()``.
"""

import time
from pathlib import Path

import pytest

import agent.anthropic_credentials as anth_cred
import agent.auxiliary_client as aux
from hermes_cli.auth import write_credential_pool

MODEL = "claude-sonnet-4-5"


def _seed(provider: str, token: str, *, model_cooldown: str | None = None) -> None:
    row = {
        "id": "entry-1", "label": "pooled", "auth_type": "oauth", "priority": 1, "source": "manual",
        "access_token": token, "refresh_token": f"rt-{token}", "expires_at": time.time() + 3600,
    }
    if model_cooldown:
        row["model_cooldowns"] = {model_cooldown: time.time() + 600}
    write_credential_pool(provider, [row])


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    # A manual Anthropic row enables external singleton discovery. HERMES_HOME
    # alone does not isolate Claude Code's HOME/CLAUDE_CONFIG_DIR credential file.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claude"))
    monkeypatch.setattr(aux, "_client_cache", {})
    # HERMES_HOME only redirects Hermes-owned state; the borrowed Claude Code
    # reader still consults ~/.claude/.credentials.json and the macOS Keychain,
    # so an ambient login on the host would seed a second un-cooled-down pool
    # entry and break the cooldown assertions below (#114424).
    monkeypatch.setattr(anth_cred, "read_claude_code_credentials", lambda: None)
    return tmp_path / "hermes"




@pytest.mark.parametrize("provider", ["anthropic", "openai-codex"])
def test_pool_token_rotation_changes_cache_key_and_hides_the_secret(isolated_home, provider):
    """Same entry id, new token -> different key; the token itself never enters the key."""
    _seed(provider, "tok-old")
    before = aux._client_cache_key(provider, async_mode=False, model=MODEL)
    _seed(provider, "tok-new")
    after = aux._client_cache_key(provider, async_mode=False, model=MODEL)

    assert before != after
    flat = repr(before) + repr(after)
    assert "tok-old" not in flat and "tok-new" not in flat
    # unchanged credentials still hit the same entry (control)
    assert aux._client_cache_key(provider, async_mode=False, model=MODEL) == after


def test_pool_cache_hint_follows_rotation_during_per_model_cooldown(isolated_home):
    """A per-model cooldown hides the entry from peek(); the hint must still track the token."""
    _seed("anthropic", "tok-old", model_cooldown=MODEL)
    pool = aux.load_pool("anthropic")
    assert [(entry.id, entry.source) for entry in pool.entries()] == [("entry-1", "manual")]
    assert pool.token_is_blocked("tok-old", model=MODEL)
    assert pool.peek() is None
    before = aux._pool_cache_hint("anthropic")
    _seed("anthropic", "tok-new", model_cooldown=MODEL)
    rotated_pool = aux.load_pool("anthropic")
    assert [(entry.id, entry.source) for entry in rotated_pool.entries()] == [("entry-1", "manual")]
    assert rotated_pool.token_is_blocked("tok-new", model=MODEL)
    assert rotated_pool.peek() is None
    after = aux._pool_cache_hint("anthropic")

    assert before.startswith("anthropic::") and after.startswith("anthropic::")
    assert before != after
    assert "tok-" not in before + after
    assert aux._pool_cache_hint("anthropic") == after


def _seed_two(provider: str, first_token: str, second_token: str) -> None:
    rows = []
    for idx, token in ((1, first_token), (2, second_token)):
        rows.append({
            "id": f"entry-{idx}", "label": f"pooled-{idx}", "auth_type": "oauth", "priority": idx,
            "source": "manual", "access_token": token, "refresh_token": f"rt-{token}",
            "expires_at": time.time() + 3600,
        })
    write_credential_pool(provider, rows)


def test_sibling_entry_rotation_does_not_churn_selected_entry_key(isolated_home):
    """Only the peeked entry's key is digested: rotating entry-2 keeps entry-1's client cached."""
    _seed_two("anthropic", "tok-a1", "tok-b1")
    before = aux._pool_cache_hint("anthropic")
    assert before.startswith("anthropic:entry-1:")
    _seed_two("anthropic", "tok-a1", "tok-b2")  # sibling rotated, selection unchanged
    assert aux._pool_cache_hint("anthropic") == before
    _seed_two("anthropic", "tok-a2", "tok-b2")  # the selected entry itself rotated
    after = aux._pool_cache_hint("anthropic")
    assert after.startswith("anthropic:entry-1:") and after != before
    assert "tok-" not in before + after
