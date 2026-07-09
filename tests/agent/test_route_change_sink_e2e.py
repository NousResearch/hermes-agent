"""E2E-ish integration: the failover + recovery CALL SITES write the durable
route-change sink and honor the two-flag chat gate.

Unlike test_route_change_announce.py (which unit-tests the leaf helpers), this
drives ``try_activate_fallback`` and ``restore_primary_runtime`` end-to-end
against a temp HERMES_HOME and asserts the on-disk sink lines + the CB-2/CB-3
recovery invariants:

- CB-2: the recovery sink line's OLD side is the FALLBACK route being left, not
  the primary (a primary→primary line is garbage).
- CB-3: a recovery to a primary whose _primary_runtime snapshot LACKS
  reasoning_config must not KeyError and must still write its line.
"""

import os
import types

import pytest

import agent.auxiliary_client as ac
from agent.chat_completion_helpers import try_activate_fallback
from agent.agent_runtime_helpers import restore_primary_runtime
from agent.error_classifier import FailoverReason


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ac.clear_runtime_main()
    try:
        yield
    finally:
        ac.clear_runtime_main()


def _sink_lines(home):
    p = os.path.join(home, "state", "model-route-changes.log")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as fh:
        return fh.read().splitlines()


class _Compressor:
    def __init__(self, context_length):
        self.context_length = context_length
        self.threshold_percent = 0.5

    def update_model(self, *, model, context_length, **kw):
        self.context_length = context_length


def _fake_agent(*, model, provider, base_url, api_mode, context_length=1_000_000):
    a = types.SimpleNamespace()
    a.model = model
    a.provider = provider
    a.base_url = base_url
    a.api_mode = api_mode
    a.api_key = "primary-key"
    a.reasoning_config = {"enabled": True, "effort": "high"}
    a.reasoning_effort = "high"
    a._config_context_length = context_length
    a._fallback_activated = False
    a._transport_cache = {}
    a._credential_pool = None
    a.context_compressor = _Compressor(context_length)
    a._primary_runtime = None
    a._fallback_index = 0
    a._rate_limited_until = 0.0
    a._fallback_chain = [{"provider": "openai-codex", "model": "gpt-5.5",
                          "reasoning_effort": "xhigh"}]
    a.fallback_model = list(a._fallback_chain)
    a._snapshot_primary_runtime = lambda: None
    a._restore_primary_runtime = lambda: None
    a._try_activate_fallback = lambda *x, **k: False
    a._anthropic_prompt_cache_policy = lambda **k: (False, False)
    a._ensure_lmstudio_runtime_loaded = lambda: None
    a._is_azure_openai_url = lambda u: False
    a._is_direct_openai_url = lambda u: False
    a._provider_model_requires_responses_api = lambda *x, **k: True
    a._buffer_status = lambda *x, **k: None
    a._replace_primary_openai_client = lambda **k: None
    a._announced = []
    a.status_callback = lambda kind, msg: a._announced.append((kind, msg))
    a._emit_status = lambda message: a._announced.append(("lifecycle", message))
    a._vprint = lambda *x, **k: None
    a.log_prefix = ""
    a._last_fallback_announced = None
    return a


def _patch_resolver(monkeypatch):
    fb_client = types.SimpleNamespace(
        api_key="codex-token",
        base_url="https://chatgpt.com/backend-api/codex/",
        _custom_headers=None,
        default_headers=None,
    )
    monkeypatch.setattr(
        ac, "resolve_provider_client",
        lambda provider, model, **kw: (fb_client, model),
        raising=False,
    )
    import agent.chat_completion_helpers as cch
    monkeypatch.setattr(
        cch, "get_model_context_length",
        lambda *a, **k: 272_000, raising=False,
    )


def test_failover_writes_sink_line_with_effort(tmp_path, monkeypatch):
    _patch_resolver(monkeypatch)
    ac.set_runtime_main("claude-pool", "claude-opus-4-8",
                        base_url="http://127.0.0.1:18810/anthropic",
                        api_key="primary-key", api_mode="anthropic_messages")
    agent = _fake_agent(model="claude-opus-4-8", provider="claude-pool",
                        base_url="http://127.0.0.1:18810/anthropic",
                        api_mode="anthropic_messages")
    assert try_activate_fallback(agent, reason=FailoverReason.overloaded) is True
    lines = _sink_lines(str(tmp_path))
    assert len(lines) == 1, lines
    # primary high → fallback entry xhigh → effort suffix on both sides.
    assert "failover claude-pool/claude-opus-4-8@high -> openai-codex/gpt-5.5@xhigh" in lines[0]


def test_failover_gate_off_suppresses_chat_but_writes_sink(tmp_path, monkeypatch):
    _patch_resolver(monkeypatch)
    # config.yaml under HERMES_HOME with the gate OFF.
    (tmp_path / "config.yaml").write_text("model:\n  announce_route_change: false\n")
    ac.set_runtime_main("claude-pool", "claude-opus-4-8",
                        base_url="http://127.0.0.1:18810/anthropic",
                        api_key="primary-key", api_mode="anthropic_messages")
    agent = _fake_agent(model="claude-opus-4-8", provider="claude-pool",
                        base_url="http://127.0.0.1:18810/anthropic",
                        api_mode="anthropic_messages")
    try_activate_fallback(agent, reason=FailoverReason.overloaded)
    # Chat announce suppressed…
    announces = [m for (k, m) in agent._announced if "fallback" in str(m).lower()]
    assert announces == [], announces
    # …but the durable sink line is still written.
    assert len(_sink_lines(str(tmp_path))) == 1


# ── recovery (restore_primary_runtime) ──────────────────────────────────────

def _recovery_agent(*, primary_reasoning=("dict", "high"), on_fallback=True):
    """Agent sitting on a fallback route with a _primary_runtime snapshot to
    restore. primary_reasoning controls whether the snapshot HAS reasoning_config
    (CB-3): ('dict','high') → present; ('missing',None) → key absent (legacy)."""
    a = types.SimpleNamespace()
    # Currently on the FALLBACK route (gpt-5.5 @ openai-codex, xhigh).
    a.model = "gpt-5.5"
    a.provider = "openai-codex"
    a.base_url = "https://chatgpt.com/backend-api/codex/"
    a.api_mode = "codex_responses"
    a.api_key = "codex-token"
    a.reasoning_config = {"enabled": True, "effort": "xhigh"}
    a._fallback_activated = on_fallback
    a._fallback_index = 1
    a._rate_limited_until = 0.0
    a._transport_cache = {}
    a._credential_pool = None
    a._use_native_cache_layout = False
    a._last_fallback_announced = (("claude-app", "claude-opus-4-8"),
                                  ("openai-codex", "gpt-5.5"))
    a._announced = []
    a._emit_status = lambda m: a._announced.append(m)

    class _CC:
        def update_model(self, **kw):
            pass
    a.context_compressor = _CC()
    a._create_openai_client = lambda *x, **k: object()

    rt = {
        "model": "claude-opus-4-8",
        "provider": "claude-app",
        "base_url": "http://127.0.0.1:18810/anthropic",
        "api_mode": "anthropic_messages",
        "api_key": "primary-key",
        "client_kwargs": {},
        "use_prompt_caching": True,
        "anthropic_api_key": "ak",
        "anthropic_base_url": "http://127.0.0.1:18810/anthropic",
        "is_anthropic_oauth": False,
        "compressor_model": "claude-opus-4-8",
        "compressor_context_length": 200_000,
        "compressor_base_url": "http://127.0.0.1:18810/anthropic",
        "compressor_api_key": "primary-key",
        "compressor_provider": "claude-app",
        "use_native_cache_layout": True,
    }
    if primary_reasoning[0] == "dict":
        rt["reasoning_config"] = {"enabled": True, "effort": primary_reasoning[1]}
    # else: intentionally OMIT reasoning_config (legacy keyless snapshot → CB-3).
    a._primary_runtime = rt

    # Anthropic client build path.
    import agent.agent_runtime_helpers as arh
    a._is_anthropic = True
    return a, arh


def test_recovery_restores_primary_route(tmp_path, monkeypatch):
    """CB-2 (repointed twice): restore_primary_runtime returns the agent to its
    primary route AND (SPEC 2026-07-08 prologue-recovery, supersedes #238's
    INV-7) emits the restore-leg recovery INLINE — at the only moment the
    restored route exists when the turn later re-fails-over (the refusing-pin
    case). The durable sink line is written gate-independently; chat emit is
    gated on announce_recovery (off here → sink only). The end-of-turn gateway
    site is persist-only now. Announce coverage:
    tests/gateway/test_reinit_recovery_announce.py.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent, arh = _recovery_agent(primary_reasoning=("dict", "high"))
    monkeypatch.setattr(
        arh, "build_anthropic_client", lambda *a, **k: object(), raising=False)
    assert restore_primary_runtime(agent) is True
    # State restored to the primary.
    assert agent.model == "claude-opus-4-8"
    assert agent.provider == "claude-app"
    assert agent._fallback_activated is False
    # Inline restore emit: sink line always (gate-independent); chat emit is
    # gated on announce_recovery which is OFF in this env → no chat line.
    assert agent._announced == [], agent._announced
    lines = _sink_lines(str(tmp_path))
    assert len(lines) == 1 and " recovery " in lines[0], lines


def test_recovery_keyless_primary_no_crash(tmp_path, monkeypatch):
    """CB-3 (repointed): a legacy _primary_runtime WITHOUT reasoning_config
    recovers and does not KeyError."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent, arh = _recovery_agent(primary_reasoning=("missing", None))
    monkeypatch.setattr(
        arh, "build_anthropic_client", lambda *a, **k: object(), raising=False)
    # Must not raise.
    assert restore_primary_runtime(agent) is True
    assert agent.model == "claude-opus-4-8"
    assert agent.provider == "claude-app"


def test_recovery_restore_announces_when_gate_on(tmp_path, monkeypatch):
    """SPEC 2026-07-08 (supersedes #238 INV-7): with announce_recovery=true the
    restore leg announces INLINE with the (restore) rider — chronologically
    correct (before any later failover this turn), visible under a refusing
    /model pin. Sink line written as well.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model:\n  announce_recovery: true\n")
    agent, arh = _recovery_agent(primary_reasoning=("dict", "high"))
    monkeypatch.setattr(
        arh, "build_anthropic_client", lambda *a, **k: object(), raising=False)
    restore_primary_runtime(agent)
    assert len(agent._announced) == 1, agent._announced
    assert agent._announced[0].startswith("🔄 Model recovery (restore): ")
    assert len(_sink_lines(str(tmp_path))) == 1
