"""Strict invocation binding (#60): the request boundary fails closed on credential/provider/model/effort drift.

Everything runs against the real request seams (``_interruptible_api_call``, ``_create_request_*_client``,
``_swap_credential``, the pool lookup and a real temporary ``SessionDB``) with fake SDK clients; nothing here
touches a network, a credential store or the controller. A bound request is only ever allowed to reach a
client built from the bound pool entry's key, and even then the record says the account is unattested.
"""

import json
import threading
from types import SimpleNamespace

import pytest

import run_agent
from agent.credential_pool import CredentialPool, PooledCredential
from agent.invocation_binding import (
    InvocationBinding, InvocationBindingError, bind_agent, binding_from_mapping, load_binding_file,
    result_reference, run_bound_request, wire_reasoning_effort,
)
from hermes_state import SessionDB

OPENAI_BASE = "https://api.openai.com/v1"
ANTHROPIC_BASE = "https://api.anthropic.com"
WIRE = {"model": "gpt-test", "messages": [], "reasoning_effort": "high"}
ANTHROPIC_WIRE = {
    "model": "claude-sonnet-4-6", "messages": [],
    "thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "high"},
}


def _binding(**overrides):
    fields = dict(
        invocation_id="inv-1", credential_id="cred-a", provider="openai", model="gpt-test", reasoning_effort="high",
    )
    return InvocationBinding(**{**fields, **overrides})


def _pool(provider="openai", base=OPENAI_BASE, **tokens):
    """Pool shell without disk/keyring I/O; ``tokens`` maps entry id -> runtime key, in priority order."""
    pool = CredentialPool.__new__(CredentialPool)
    pool._lock = threading.RLock()
    pool.provider = provider
    pool._current_id = None
    pool._active_leases = {}
    pool._entries = [
        PooledCredential(
            provider=provider, id=entry_id, label=entry_id, auth_type="api_key", priority=rank, source="manual",
            access_token=token, base_url=base,
        )
        for rank, (entry_id, token) in enumerate(tokens.items())
    ]
    return pool


class _Client:
    """Fake OpenAI-style SDK client recording the key it was built with and every request it sends."""

    def __init__(self, api_key, sent):
        self.api_key = api_key
        self._client = SimpleNamespace(is_closed=False)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.responses = SimpleNamespace()
        self._sent = sent

    def _create(self, **kwargs):
        self._sent.append({"api_key": self.api_key, "kwargs": kwargs})
        return {"ok": True}

    def close(self):
        self._client.is_closed = True


class _Factory:
    def __init__(self):
        self.built, self.sent = [], []

    def __call__(self, **kwargs):
        self.built.append(kwargs.get("api_key"))
        return _Client(kwargs.get("api_key"), self.sent)


@pytest.fixture
def db(tmp_path):
    database = SessionDB(db_path=tmp_path / "state.db")
    database.create_session("sess-1", "cli", model="gpt-test")
    yield database
    database.close()


@pytest.fixture
def factory(monkeypatch):
    fake = _Factory()
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", fake)
    return fake


def _agent(db, pool, binding, *, key="sk-a", session_id="sess-1", factory=None, bind=True):
    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent.api_mode, agent.provider, agent.base_url, agent.model = "chat_completions", "openai", OPENAI_BASE, "gpt-test"
    agent.log_prefix, agent.quiet_mode = "", True
    agent._interrupt_requested, agent._interrupt_message = False, None
    agent._client_lock = threading.RLock()
    agent._client_kwargs = {"api_key": key, "base_url": OPENAI_BASE}
    agent.api_key = key
    agent.client = _Client("shared", [] if factory is None else factory.sent)
    agent.stream_delta_callback = agent._stream_callback = agent.reasoning_callback = agent.status_callback = None
    agent._credential_pool = pool
    agent._session_db, agent._session_db_created, agent.session_id = db, True, session_id
    agent._ensure_db_session = lambda: None  # the fixture created the row; strict mode must still see it
    if bind:
        bind_agent(agent, binding)
    return agent


def _anthropic_agent(db, pool, binding, *, key="sk-ant-a", monkeypatch=None):
    agent = _agent(db, pool, binding, bind=False)
    agent.api_mode, agent.provider, agent.base_url, agent.model = (
        "anthropic_messages", "anthropic", ANTHROPIC_BASE, "claude-sonnet-4-6")
    agent._anthropic_api_key, agent._anthropic_base_url, agent.api_key = key, ANTHROPIC_BASE, key
    agent._oauth_1m_beta_disabled = False
    agent.built_keys = []

    def _build(client_key):
        agent.built_keys.append(client_key[1])
        return SimpleNamespace(is_closed=False, token=client_key[1])
    agent._build_anthropic_client_for_key = _build
    bind_agent(agent, binding)
    return agent


def _refused(code):
    return pytest.raises(InvocationBindingError, match=f"^{code}$")


def _record(db, session_id="sess-1"):
    return db.get_invocation_binding(session_id)


# ── credential id contract ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", [None, "", "   ", 7])
def test_missing_credential_id_is_refused(value):
    with _refused("credential_id_missing"):
        _binding(credential_id=value)
    with _refused("credential_id_missing"):
        binding_from_mapping({"version": 1, "invocation_id": "inv-1", "provider": "openai", "model": "m",
                              "reasoning_effort": "high"})


@pytest.mark.parametrize("value", ["a b", "../etc", "x" * 129, "-lead", "id\n", "ünï"])
def test_invalid_credential_id_is_refused(value):
    with _refused("credential_id_invalid"):
        _binding(credential_id=value)


def test_binding_context_file_is_strict(tmp_path):
    good = tmp_path / "ctx.json"
    good.write_text(json.dumps({"version": 1, "invocation_id": "inv-1", "credential_id": "cred-a",
                                "provider": "OpenAI", "model": "gpt-test", "reasoning_effort": "HIGH"}))
    assert load_binding_file(str(good)) == _binding()

    for name, body in {
        "extra": json.dumps({"version": 1, "invocation_id": "i", "credential_id": "c", "provider": "p", "model": "m",
                             "reasoning_effort": "high", "api_key": "sk-leak"}),
        "version": json.dumps({"version": 2, "invocation_id": "i", "credential_id": "c", "provider": "p",
                               "model": "m", "reasoning_effort": "high"}),
        "junk": "{not json", "list": "[]",
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(body)
        with _refused("binding_context_invalid"):
            load_binding_file(str(path))
    with _refused("binding_context_invalid"):
        load_binding_file(str(tmp_path / "missing.json"))


def test_requesting_account_attestation_is_refused_not_faked():
    with _refused("account_attestation_unsupported"):
        _binding(require_account_attestation=True)


# ── request boundary: GREEN and the evidence it leaves ───────────────────────────────────────────


def test_matching_request_reaches_bound_key_and_is_recorded_unattested(db, factory):
    agent = _agent(db, _pool(**{"cred-a": "sk-a", "cred-b": "sk-b"}), _binding(), factory=factory)

    assert agent._interruptible_api_call(dict(WIRE)) == {"ok": True}

    assert [call["api_key"] for call in factory.sent] == ["sk-a"]
    record = _record(db)
    assert (record["request_count"], record["responded_count"], record["failed_count"]) == (1, 1, 0)
    assert record["last_request"]["credential_id"] == "cred-a"
    assert record["last_request"]["wire_reasoning_effort"] == "high"
    assert record["last_request"]["state"] == "responded"
    assert "sk-a" not in json.dumps(record)
    assert record["account_attestation"] == "unsupported" and record["account_verified"] is False
    reference = result_reference(agent, _binding(), "sess-1")
    assert reference["status"] == "bound_unattested" and reference["account_verified"] is False


@pytest.mark.parametrize(
    ("extra_body", "code"),
    [
        ({"model": "gpt-other"}, "model_mismatch"),
        ({"reasoning_effort": "low"}, "reasoning_mismatch"),
        ({"model": "gpt-other", "reasoning_effort": "low"}, "model_mismatch"),
        ({"reasoning": {"effort": "low"}}, "reasoning_mismatch"),
        ([{"model": "gpt-other"}], "request_overrides_invalid"),
    ],
)
def test_openai_extra_body_cannot_override_bound_model_or_effort(db, factory, extra_body, code):
    agent = _agent(db, _pool(**{"cred-a": "sk-a"}), _binding(), factory=factory)

    with _refused(code):
        agent._interruptible_api_call({**WIRE, "extra_body": extra_body})

    _assert_nothing_built_or_sent(factory)
    assert _record(db) is None


def test_wire_effort_is_read_from_the_payload_only():
    assert wire_reasoning_effort("anthropic_messages", ANTHROPIC_WIRE) == "high"
    # budget-token thinking and omitted thinking (haiku) prove no effort
    assert wire_reasoning_effort("anthropic_messages", {"thinking": {"type": "enabled", "budget_tokens": 16000}}) is None
    assert wire_reasoning_effort("anthropic_messages", {"model": "claude-haiku-4-5"}) is None
    assert wire_reasoning_effort("codex_responses", {"reasoning": {"effort": "High"}}) == "high"
    assert wire_reasoning_effort("chat_completions", {"extra_body": {"reasoning": {"effort": "high"}}}) == "high"
    assert wire_reasoning_effort("chat_completions", {"messages": []}) is None
    assert wire_reasoning_effort("bedrock_converse", {"reasoning_effort": "high"}) is None


# ── RED cases: every refusal is before any client exists, and nothing is sent ─────────────────────


def _assert_nothing_built_or_sent(factory):
    assert factory.built == [] and factory.sent == []


def test_wrong_account_key_is_refused_before_a_client_is_built(db, factory):
    agent = _agent(db, _pool(**{"cred-a": "sk-a", "cred-b": "sk-b"}), _binding(), factory=factory)
    agent._client_kwargs["api_key"] = agent.api_key = "sk-b"  # the other pool account

    with _refused("credential_mismatch"):
        agent._interruptible_api_call(dict(WIRE))

    _assert_nothing_built_or_sent(factory)
    assert _record(db) is None  # refused before the row was ever bound: result reads "rejected"
    assert result_reference(agent, _binding(), "sess-1")["status"] == "rejected"


def test_key_outside_the_pool_and_shared_keys_are_refused(db, factory):
    agent = _agent(db, _pool(**{"cred-a": "sk-a"}), _binding(), factory=factory)
    agent._client_kwargs["api_key"] = agent.api_key = "sk-from-env"
    with _refused("credential_mismatch"):
        agent._interruptible_api_call(dict(WIRE))

    # two entries holding one key cannot prove which account answers
    agent = _agent(db, _pool(**{"cred-a": "sk-a", "cred-dup": "sk-a"}), _binding(), factory=factory, bind=False)
    with _refused("credential_mismatch"):
        bind_agent(agent, _binding())
    _assert_nothing_built_or_sent(factory)


def test_stale_agent_key_that_differs_from_client_material_is_refused(db, factory):
    agent = _agent(db, _pool(**{"cred-a": "sk-a", "cred-b": "sk-b"}), _binding(), factory=factory)
    agent.api_key = "sk-b"  # a lagging field must not be able to launder a request
    with _refused("credential_mismatch"):
        agent._interruptible_api_call(dict(WIRE))
    _assert_nothing_built_or_sent(factory)


def test_missing_pool_or_unknown_credential_is_refused(db, factory):
    with _refused("credential_pool_missing"):
        _agent(db, None, _binding(), factory=factory)
    with _refused("credential_not_found"):
        _agent(db, _pool(**{"cred-b": "sk-b"}), _binding(), factory=factory)
    _assert_nothing_built_or_sent(factory)


def test_wrong_provider_and_wrong_model_are_refused(db, factory):
    pool = _pool(**{"cred-a": "sk-a"})
    with _refused("provider_mismatch"):
        _agent(db, pool, _binding(provider="openrouter"), factory=factory)
    with _refused("model_mismatch"):
        _agent(db, pool, _binding(model="gpt-other"), factory=factory)

    agent = _agent(db, pool, _binding(), factory=factory)
    with _refused("model_mismatch"):
        agent._interruptible_api_call({**WIRE, "model": "gpt-cheaper"})  # e.g. changed after build
    agent.provider = "openrouter"  # fallback/failover rewrote the route
    with _refused("provider_mismatch"):
        agent._interruptible_api_call(dict(WIRE))
    _assert_nothing_built_or_sent(factory)


def test_fallback_activation_is_refused(db, factory):
    agent = _agent(db, _pool(**{"cred-a": "sk-a"}), _binding(), factory=factory)
    agent._fallback_activated = True
    with _refused("fallback_active"):
        agent._interruptible_api_call(dict(WIRE))
    _assert_nothing_built_or_sent(factory)


def test_wrong_or_unproven_effective_reasoning_is_refused(db, factory):
    agent = _agent(db, _pool(**{"cred-a": "sk-a"}), _binding(), factory=factory)
    agent.reasoning_config = {"enabled": True, "effort": "high"}  # configured intent is not wire proof

    with _refused("reasoning_mismatch"):
        agent._interruptible_api_call({**WIRE, "reasoning_effort": "medium"})
    with _refused("reasoning_unproven"):
        agent._interruptible_api_call({"model": "gpt-test", "messages": []})
    _assert_nothing_built_or_sent(factory)


def test_violations_are_visible_in_the_native_record(db, factory):
    agent = _agent(db, _pool(**{"cred-a": "sk-a"}), _binding(), factory=factory)
    assert agent._interruptible_api_call(dict(WIRE)) == {"ok": True}
    with _refused("reasoning_mismatch"):
        agent._interruptible_api_call({**WIRE, "reasoning_effort": "low"})

    record = _record(db)
    assert record["violations"] == ["reasoning_mismatch"]
    assert len(factory.sent) == 1
    assert result_reference(agent, _binding(), "sess-1")["status"] == "violation"


# ── retry / rotation / refresh ───────────────────────────────────────────────────────────────────


def test_retry_after_rotation_to_another_credential_is_refused_and_unsent(db, factory):
    pool = _pool(**{"cred-a": "sk-a", "cred-b": "sk-b"})
    agent = _agent(db, pool, _binding(), factory=factory)
    assert agent._interruptible_api_call(dict(WIRE)) == {"ok": True}

    other = pool.entries()[1]
    assert agent._swap_credential(other) is False  # pool rotation handed us cred-b: refused, nothing changed
    assert agent._client_kwargs["api_key"] == "sk-a" and agent.api_key == "sk-a"

    # even a path that bypasses the swap guard cannot reach the wire with the other key
    agent._client_kwargs["api_key"] = agent.api_key = "sk-b"
    with _refused("credential_mismatch"):
        agent._interruptible_api_call(dict(WIRE))
    assert [call["api_key"] for call in factory.sent] == ["sk-a"]


def test_refresh_of_the_same_credential_is_allowed_and_the_new_key_is_what_is_sent(db, factory):
    pool = _pool(**{"cred-a": "sk-a", "cred-b": "sk-b"})
    agent = _agent(db, pool, _binding(), factory=factory)
    assert agent._interruptible_api_call(dict(WIRE)) == {"ok": True}

    refreshed = pool.entries()[0]
    refreshed.access_token = "sk-a-refreshed"  # OAuth refresh rewrote the SAME pool entry
    assert agent._swap_credential(refreshed) is True
    assert agent._interruptible_api_call(dict(WIRE)) == {"ok": True}

    assert [call["api_key"] for call in factory.sent] == ["sk-a", "sk-a-refreshed"]
    assert _record(db)["request_count"] == 2


def test_refresh_that_substitutes_an_account_is_refused(db, factory):
    pool = _pool(**{"cred-a": "sk-a", "cred-b": "sk-b"})
    agent = _agent(db, pool, _binding(), factory=factory)

    # singleton / env / vertex style adoption of a foreign token
    assert agent._adopt_openai_credentials("sk-b", OPENAI_BASE, reason="env_credential_refresh") is False
    assert agent._apply_copilot_token("sk-b", None, reason="copilot_credential_refresh") is False
    assert agent._client_kwargs["api_key"] == "sk-a" and agent.api_key == "sk-a"

    # a "refreshed" entry that is actually the other account's id
    imposter = pool.entries()[1]
    assert agent._swap_credential(imposter) is False
    _assert_nothing_built_or_sent(factory)


def test_anthropic_global_token_refresh_cannot_swap_the_account(db, monkeypatch):
    pool = _pool(provider="anthropic", base=ANTHROPIC_BASE, **{"cred-a": "sk-ant-a", "cred-b": "sk-ant-b"})
    binding = _binding(provider="anthropic", model="claude-sonnet-4-6")
    calls = []
    monkeypatch.setattr(
        "agent.anthropic_credentials.resolve_anthropic_token", lambda **kw: calls.append(kw) or "sk-ant-b")
    agent = _anthropic_agent(db, pool, binding)

    assert agent._try_refresh_anthropic_client_credentials() is False
    assert calls == [] and agent._anthropic_api_key == "sk-ant-a"

    # the same refresher still swaps accounts for an unbound agent: strict mode is the only thing stopping it
    legacy = _anthropic_agent(db, pool, None)
    legacy._anthropic_client = SimpleNamespace(close=lambda: None)
    legacy._build_direct_anthropic_client = lambda token, base: SimpleNamespace(token=token)
    assert legacy._try_refresh_anthropic_client_credentials() is True
    assert legacy._anthropic_api_key == "sk-ant-b" and len(calls) == 1


def test_anthropic_request_client_is_built_only_from_the_bound_key(db, monkeypatch):
    pool = _pool(provider="anthropic", base=ANTHROPIC_BASE, **{"cred-a": "sk-ant-a", "cred-b": "sk-ant-b"})
    binding = _binding(provider="anthropic", model="claude-sonnet-4-6")
    monkeypatch.setattr("agent.anthropic_credentials.resolve_anthropic_token", lambda **kw: "sk-ant-b")
    agent = _anthropic_agent(db, pool, binding)

    run_bound_request(agent, dict(ANTHROPIC_WIRE), lambda: agent._create_request_anthropic_client(reason="t"))
    assert agent.built_keys == ["sk-ant-a"]

    agent._anthropic_api_key = agent.api_key = "sk-ant-b"  # substitution before the next attempt
    with _refused("credential_mismatch"):
        run_bound_request(agent, dict(ANTHROPIC_WIRE), lambda: agent._create_request_anthropic_client(reason="t"))
    assert agent.built_keys == ["sk-ant-a"]

    # a request that reaches client creation by another route is still checked at that exact point
    with _refused("credential_mismatch"):
        agent._create_request_anthropic_client(reason="direct")
    assert agent.built_keys == ["sk-ant-a"]


def test_anthropic_effort_must_be_adaptive_wire_effort(db):
    pool = _pool(provider="anthropic", base=ANTHROPIC_BASE, **{"cred-a": "sk-ant-a"})
    agent = _anthropic_agent(db, pool, _binding(provider="anthropic", model="claude-sonnet-4-6"))
    send = lambda: "sent"
    assert run_bound_request(agent, dict(ANTHROPIC_WIRE), send) == "sent"
    haiku_like = {k: v for k, v in ANTHROPIC_WIRE.items() if k not in ("thinking", "output_config")}
    with _refused("reasoning_unproven"):
        run_bound_request(agent, haiku_like, send)
    with _refused("reasoning_mismatch"):
        run_bound_request(agent, {**ANTHROPIC_WIRE, "output_config": {"effort": "max"}}, send)


# ── streaming entry shares the same gate ─────────────────────────────────────────────────────────


def test_streaming_entry_is_gated_too(db, factory):
    agent = _agent(db, _pool(**{"cred-a": "sk-a", "cred-b": "sk-b"}), _binding(), factory=factory)
    agent._client_kwargs["api_key"] = agent.api_key = "sk-b"
    with _refused("credential_mismatch"):
        agent._interruptible_streaming_api_call(dict(WIRE))
    _assert_nothing_built_or_sent(factory)


# ── native session / result: persistence failure, restart, duplicates, idempotency ─────────────────


def test_unavailable_native_session_blocks_the_request(db, factory):
    agent = _agent(db, _pool(**{"cred-a": "sk-a"}), _binding(), factory=factory)
    agent._session_db_created = False
    with _refused("native_session_unavailable"):
        agent._interruptible_api_call(dict(WIRE))

    agent._session_db_created, agent._persist_disabled = True, True
    with _refused("native_session_unavailable"):
        agent._interruptible_api_call(dict(WIRE))

    agent._persist_disabled, agent.session_id = False, "sess-missing-row"
    with _refused("native_session_missing"):
        agent._interruptible_api_call(dict(WIRE))
    _assert_nothing_built_or_sent(factory)


def test_failed_terminal_write_discards_the_response(db, factory, monkeypatch):
    agent = _agent(db, _pool(**{"cred-a": "sk-a"}), _binding(), factory=factory)

    def _boom(*args, **kwargs):
        raise RuntimeError("disk full sk-leak")

    monkeypatch.setattr(db, "settle_invocation_request", _boom)
    with _refused("native_record_failed"):
        agent._interruptible_api_call(dict(WIRE))
    record = _record(db)
    assert record["request_count"] == 1 and record["responded_count"] == 0
    assert record["last_request"]["state"] == "prepared"  # in flight forever: never admissible
    assert result_reference(agent, _binding(), "sess-1")["status"] == "incomplete"


def test_failed_request_is_recorded_and_not_admissible(db, factory):
    pool = _pool(**{"cred-a": "sk-a"})
    agent = _agent(db, pool, _binding(), factory=factory)

    def _fail():
        raise TimeoutError("provider timeout with sk-a inside")

    with pytest.raises(TimeoutError):
        run_bound_request(agent, dict(WIRE), _fail)
    record = _record(db)
    assert (record["request_count"], record["responded_count"], record["failed_count"]) == (1, 0, 1)
    assert record["last_request"]["state"] == "failed" and record["last_request"]["error_type"] == "TimeoutError"
    assert "sk-a" not in json.dumps(record)
    assert result_reference(agent, _binding(), "sess-1")["status"] == "incomplete"


def test_restart_on_the_same_session_is_idempotent_and_continues_the_ordinal(db, factory):
    pool = _pool(**{"cred-a": "sk-a"})
    first = _agent(db, pool, _binding(), factory=factory)
    first._interruptible_api_call(dict(WIRE))
    bound_at = _record(db)["bound_at"]

    restarted = _agent(db, pool, _binding(), factory=factory)  # new process, same invocation and session
    restarted._interruptible_api_call(dict(WIRE))

    record = _record(db)
    assert record["bound_at"] == bound_at  # write-once: not re-stamped
    assert (record["request_count"], record["responded_count"]) == (2, 2)


def test_stale_session_with_a_different_binding_is_refused(db, factory):
    pool = _pool(**{"cred-a": "sk-a", "cred-b": "sk-b"})
    _agent(db, pool, _binding(), factory=factory)._interruptible_api_call(dict(WIRE))

    for changed in (_binding(model="gpt-test", invocation_id="inv-other"), _binding(reasoning_effort="low")):
        stale = _agent(db, pool, changed, factory=factory)
        with _refused("binding_conflict"):
            stale._interruptible_api_call({**WIRE, "reasoning_effort": changed.reasoning_effort})
    assert len(factory.sent) == 1


def test_dispatch_reuse_of_an_invocation_in_another_session_is_refused(db, factory):
    pool = _pool(**{"cred-a": "sk-a"})
    _agent(db, pool, _binding(), factory=factory)._interruptible_api_call(dict(WIRE))

    db.create_session("sess-2", "cli", model="gpt-test")
    duplicate = _agent(db, pool, _binding(), session_id="sess-2", factory=factory)
    with _refused("invocation_reused"):
        duplicate._interruptible_api_call(dict(WIRE))
    assert len(factory.sent) == 1
    assert _record(db, "sess-2") is None


def test_unbound_agent_behaves_as_before_and_writes_no_evidence(db, factory):
    pool = _pool(**{"cred-a": "sk-a", "cred-b": "sk-b"})
    agent = _agent(db, pool, None, key="sk-b", factory=factory)
    assert agent._interruptible_api_call({"model": "gpt-test", "messages": []}) == {"ok": True}
    assert [call["api_key"] for call in factory.sent] == ["sk-b"]
    assert _record(db) is None


# ── a binding refusal is final for the turn loop ─────────────────────────────────────────────────


def test_binding_refusal_is_terminal_in_the_api_error_handler():
    from agent.turn_api_error import handle_api_error

    agent = SimpleNamespace(thinking_callback=None)
    verdict = handle_api_error(
        agent, api_error=InvocationBindingError("credential_mismatch"), _retry=None, thinking_spinner=None,
        messages=[], api_messages=[], api_kwargs={}, system_message=None, active_system_prompt=None,
        conversation_history=[], approx_tokens=0, retry_count=0, max_retries=3, compression_attempts=0,
        max_compression_attempts=3, api_call_count=1, api_request_id="r", api_start_time=0.0,
        effective_task_id="t", turn_id="u",
    )
    assert verdict.action == "return"
    assert verdict.result["failed"] is True and verdict.result["failure_retryable"] is False
    assert verdict.result["error"] == "credential_mismatch"
    assert verdict.result["failure_reason"] == "invocation_binding_violation"
