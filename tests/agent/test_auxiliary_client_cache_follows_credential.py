"""Regression coverage: the auxiliary client cache key follows the pooled credential (#113022).

Points 1/2/5 of the issue: a token rotated or refreshed by another process must yield a new
client instead of a 401 round trip on the cached one, including while a per-model cooldown
hides the entry from ``peek()``.
"""

import time
from dataclasses import replace
from unittest.mock import patch

import httpx
import pytest

import agent.anthropic_credentials as anth_cred
import agent.auxiliary_client as aux
from agent.credential_pool import CredentialPool, PooledCredential
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
    before = aux._pool_cache_hint("anthropic")
    _seed("anthropic", "tok-new", model_cooldown=MODEL)
    after = aux._pool_cache_hint("anthropic")

    assert before.startswith("anthropic::") and after.startswith("anthropic::")
    assert before != after
    assert "tok-" not in before + after


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


OR_MODEL = "openai/gpt-target"


def _completion(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, request=request, json={
        "id": "x", "object": "chat.completion", "created": 0, "model": OR_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    })


def _two_key_pool_calls(change_k0):
    """Two ``_get_cached_client`` calls on a real two-key pool; returns (same client?, tokens sent).

    k0 is the pool's current entry and carries an unrelated model's cooldown; ``change_k0`` edits it
    between the calls the way another process's refresh or a 429 on the requested model would.
    """
    keys = [
        PooledCredential("openrouter", "k0", "k0", "api_key", 0, "manual", "tok-k0",
                         model_cooldowns={"openai/other": time.time() + 600}),
        PooledCredential("openrouter", "k1", "k1", "api_key", 1, "manual", "tok-k1"),
    ]
    sent: list[str] = []

    def handler(request):
        sent.append(request.headers["authorization"].removeprefix("Bearer "))
        return _completion(request)

    # load_pool re-reads the store on every call: a fresh pool over the current rows.
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client, \
         patch.object(aux, "load_pool", side_effect=lambda _p: CredentialPool("openrouter", [replace(k) for k in keys])), \
         patch.object(aux, "_openai_http_client_kwargs", return_value={"http_client": http_client}):
        first, _ = aux._get_cached_client("openrouter", OR_MODEL)
        assert first is not None
        first.chat.completions.create(model=OR_MODEL, messages=[{"role": "user", "content": "hi"}])
        keys[0] = change_k0(keys[0])
        second, _ = aux._get_cached_client("openrouter", OR_MODEL)
        assert second is not None
        second.chat.completions.create(model=OR_MODEL, messages=[{"role": "user", "content": "hi"}])
    return first is second, sent


def test_cached_pool_client_rebuilds_when_its_model_aware_key_is_refreshed(isolated_home):
    reused, sent = _two_key_pool_calls(lambda k0: replace(k0, access_token="tok-k0-new"))
    assert not reused
    assert sent == ["tok-k0", "tok-k0-new"]


def test_cached_pool_client_rebuilds_when_its_key_cools_down_for_the_model(isolated_home):
    reused, sent = _two_key_pool_calls(lambda k0: replace(k0, model_cooldowns={OR_MODEL: time.time() + 600}))
    assert not reused
    assert sent == ["tok-k0", "tok-k1"]


def test_cached_pool_client_is_reused_while_its_key_is_unchanged(isolated_home):
    reused, sent = _two_key_pool_calls(lambda k0: k0)
    assert reused
    assert sent == ["tok-k0", "tok-k0"]
