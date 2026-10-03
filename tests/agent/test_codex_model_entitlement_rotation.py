"""A Codex ChatGPT-account model entitlement 400 rotates to the next pool credential (#71970).

The exact normalized rejection benches only (credential, model) and hands the next eligible entry
back; every other 400 stays a plain request failure. Once every entry rejects the model, the
single-credential handling from #106475 takes over.
"""
import json
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.error_classifier import FailoverReason, classify_api_error

MODEL = "gpt-5.3-codex"
OTHER_MODEL = "gpt-5.3-codex-mini"
TOKENS = ("tok-account-a", "tok-account-b")
ZEN_MODEL = "claude-opus-5"


class _Err(Exception):
    def __init__(self, status, body):
        self.status_code = status
        self.body = body
        self.response = types.SimpleNamespace(status_code=status, headers={}, text=json.dumps(body), json=lambda: body)
        self.message = f"Error code: {status} - {json.dumps(body)}"
        super().__init__(self.message)


def _entitlement_400():
    return _Err(400, {"detail": f"The '{MODEL}' model is not supported when using Codex with a ChatGPT account."})


@pytest.fixture
def pool(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    root.mkdir()
    (tmp_path / "fakehome").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"openai-codex": [
        {"id": f"cred-{i}", "label": f"acct-{i}", "auth_type": "oauth", "priority": i, "source": "manual",
         "access_token": tok, "refresh_token": f"rt-{i}", "expires_at_ms": 4_000_000_000_000}
        for i, tok in enumerate(TOKENS)
    ]}}))
    from agent.credential_pool import load_pool
    return load_pool("openai-codex")


def test_entitlement_400_benches_only_that_model_and_rotates(pool):
    verdict = classify_api_error(_entitlement_400(), provider="openai-codex", model=MODEL)
    assert verdict.reason == FailoverReason.model_entitlement
    assert verdict.should_rotate_credential and verdict.should_fallback and not verdict.retryable

    generic = classify_api_error(_Err(400, {"detail": "Invalid request: bad field"}), provider="openai-codex", model=MODEL)
    assert generic.reason == FailoverReason.format_error and not generic.should_rotate_credential

    # Drive the production recovery entry point (turn recovery -> recover_with_credential_pool),
    # not the pool directly: the classifier verdict must reach the model-scoped bench.
    assert pool.select(model=MODEL).id == "cred-0"
    agent = types.SimpleNamespace(
        provider="openai-codex", model=MODEL, base_url="https://chatgpt.com/backend-api/codex",
        api_key=TOKENS[0], _credential_pool=pool, _credential_pool_entry_id="cred-0",
        _swap_credential=MagicMock(return_value=True),
    )
    assert recover_with_credential_pool(
        agent, status_code=400, has_retried_429=False, classified_reason=verdict.reason,
    ) == (True, False)
    agent._swap_credential.assert_called_once()
    assert agent._swap_credential.call_args.args[0].id == "cred-1"
    first = pool.entries()[0]
    assert first.last_status is None  # credential-wide state untouched: other models stay usable
    assert set(first.model_cooldowns) == {MODEL}
    # An entitlement is a plan property, not a window: no hourly re-probe, only reset clears it.
    assert first.model_cooldowns[MODEL] > time.time() + 24 * 3600
    assert pool.select(model=OTHER_MODEL).id == "cred-0"
    assert pool.reset_statuses() >= 1 and not pool.entries()[0].model_cooldowns


def test_all_entries_rejecting_falls_back_to_session_marker(pool):
    from agent.fallback_cooldown import _is_entitlement_rejected, _mark_entitlement_rejected_model

    agent = types.SimpleNamespace(
        provider="openai-codex", model=MODEL, _credential_pool=pool,
        _buffer_diagnostic_status=lambda *_a, **_k: None,
    )
    pool.mark_exhausted_and_rotate(
        status_code=400, api_key_hint=TOKENS[0], credential_id="cred-0", failure_reason="model_entitlement", model=MODEL,
    )
    # cred-1 is still eligible for the model: rotation owns the recovery, no session-wide marker.
    assert _mark_entitlement_rejected_model(agent, _entitlement_400()) is False
    assert not _is_entitlement_rejected(agent, "openai-codex", MODEL)

    assert pool.mark_exhausted_and_rotate(
        status_code=400, api_key_hint=TOKENS[1], credential_id="cred-1", failure_reason="model_entitlement", model=MODEL,
    ) is None
    assert _mark_entitlement_rejected_model(agent, _entitlement_400()) is True
    assert _is_entitlement_rejected(agent, "openai-codex", MODEL)


def _zen_pool(tmp_path, monkeypatch):
    root = tmp_path / "zen-root"
    root.mkdir()
    (tmp_path / "zenfakehome").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "zenfakehome"))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("OPENCODE_ZEN_API_KEY", raising=False)
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"opencode-zen": [
        {"id": f"cred-zen-{i}", "label": f"zen-{i}", "auth_type": "oauth", "priority": i, "source": "manual",
         "access_token": tok, "refresh_token": f"rt-{i}", "expires_at_ms": 4_000_000_000_000}
        for i, tok in enumerate(("zen-key-a", "zen-key-b"))
    ]}}))
    from agent.credential_pool import load_pool
    return load_pool("opencode-zen")


def test_zen_model_access_403_benches_only_that_model(tmp_path, monkeypatch):
    """Regression for #124021: Zen relays an upstream model gate as 403 'Upstream request
    failed: Model access is disabled' (error type ``api_error``) — the MODEL was rejected,
    not the credential. A working key must never be marked auth-failed."""
    zen_pool = _zen_pool(tmp_path, monkeypatch)

    err = _Err(403, {"type": "error", "error": {
        "type": "api_error", "message": "Upstream request failed: Model access is disabled"}})
    verdict = classify_api_error(err, provider="opencode-zen", model=ZEN_MODEL)
    assert verdict.reason == FailoverReason.model_entitlement

    # Drive the production recovery entry point: the verdict must reach the model-scoped
    # bench, not the credential-wide auth-failed marking.
    assert zen_pool.select(model=ZEN_MODEL).id == "cred-zen-0"
    agent = types.SimpleNamespace(
        provider="opencode-zen", model=ZEN_MODEL, base_url="https://opencode.ai/zen",
        api_key="zen-key-a", _credential_pool=zen_pool, _credential_pool_entry_id="cred-zen-0",
        _swap_credential=MagicMock(return_value=True),
    )
    assert recover_with_credential_pool(
        agent, status_code=403, has_retried_429=False, classified_reason=verdict.reason,
    ) == (True, False)
    agent._swap_credential.assert_called_once()
    first = zen_pool.entries()[0]
    assert first.last_status is None  # the key stays working: no auth-failed marking
    assert set(first.model_cooldowns) == {ZEN_MODEL}


def test_zen_credential_403_without_model_gate_stays_auth(tmp_path, monkeypatch):
    """A real credential refusal keeps the auth verdict (and its re-auth guidance)."""
    zen_pool = _zen_pool(tmp_path, monkeypatch)

    err = _Err(403, {"type": "error", "error": {
        "type": "authentication_error", "message": "invalid api key"}})
    verdict = classify_api_error(err, provider="opencode-zen", model=ZEN_MODEL)
    assert verdict.reason != FailoverReason.model_entitlement
    assert zen_pool.entries()[0].last_status is None
