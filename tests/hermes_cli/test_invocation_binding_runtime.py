"""Strict invocation binding (#60): runtime resolution picks the bound pool entry, never the selected one,
and the single-query stream protocol never reports a verified account."""

import json
from types import SimpleNamespace

import pytest

from agent.credential_pool import STATUS_DEAD, PooledCredential
from agent.invocation_binding import InvocationBinding, InvocationBindingError
from hermes_cli import runtime_provider as rp
from hermes_cli.stream_json import StreamJsonEmitter

BASE = "https://api.anthropic.com"


def _binding(**overrides):
    fields = dict(
        invocation_id="inv-1", credential_id="cred-b", provider="anthropic", model="claude-sonnet-4-6",
        reasoning_effort="high",
    )
    return InvocationBinding(**{**fields, **overrides})


def _entry(entry_id, token, **extra):
    return PooledCredential(
        provider="anthropic", id=entry_id, label=entry_id, auth_type="api_key", priority=extra.pop("priority", 0),
        source="manual", access_token=token, base_url=BASE, **extra,
    )


class _Pool:
    """Pool double whose ``select()`` would hand out the highest-priority entry (cred-a)."""

    provider = "anthropic"

    def __init__(self, *entries):
        self._entries = list(entries)
        self.select_calls = 0

    def entries(self):
        return list(self._entries)

    def has_credentials(self):
        return True

    def select(self, **kwargs):
        self.select_calls += 1
        return self._entries[0]


@pytest.fixture
def pool(monkeypatch):
    fake = _Pool(_entry("cred-a", "sk-ant-api03-a", priority=0), _entry("cred-b", "sk-ant-api03-b", priority=1))
    monkeypatch.setattr(rp, "load_pool", lambda provider: fake)
    return fake


def test_bound_runtime_is_the_named_entry_not_the_selected_one(pool):
    runtime = rp.resolve_runtime_provider(requested="anthropic", invocation_binding=_binding())

    assert runtime["api_key"] == "sk-ant-api03-b"
    assert runtime["credential_pool"] is pool
    assert pool.select_calls == 0


def test_bound_runtime_refuses_instead_of_substituting(pool):
    with pytest.raises(InvocationBindingError, match="^credential_not_found$"):
        rp.resolve_runtime_provider(requested="anthropic", invocation_binding=_binding(credential_id="cred-zzz"))
    with pytest.raises(InvocationBindingError, match="^provider_mismatch$"):
        rp.resolve_runtime_provider(requested="openrouter", invocation_binding=_binding())
    pool.entries()[1].last_status = STATUS_DEAD
    with pytest.raises(InvocationBindingError, match="^credential_missing$"):
        rp.resolve_runtime_provider(requested="anthropic", invocation_binding=_binding())
    assert pool.select_calls == 0


def test_stream_json_reports_requested_then_rejected_and_never_verified(capsys):
    emitter = StreamJsonEmitter(model="claude-sonnet-4-6", session_id="sess-1", binding=_binding())
    emitter.emit_result({"failed": True, "error": "credentials or agent init failed"}, session_id="sess-1", exit_code=1)

    init, result = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert init["invocation_binding"] == {
        "invocation_id": "inv-1", "digest": _binding().digest, "status": "requested"}
    reference = result["invocation_binding"]
    assert reference["status"] == "rejected" and reference["account_verified"] is False
    assert reference["request_count"] == 0 and result["exit_code"] == 1


def test_stream_json_without_a_binding_is_unchanged(capsys):
    emitter = StreamJsonEmitter(model="m", session_id="s")
    emitter.emit_result({"final_response": "hi"}, session_id="s")
    init, result = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert "invocation_binding" not in init and "invocation_binding" not in result
