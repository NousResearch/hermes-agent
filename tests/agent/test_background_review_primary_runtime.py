"""Background-review fork pins to the primary runtime during fallback (#105825).

After a provider failover, the post-turn background review/curator fork would inherit
the fallback provider/model via ``agent.provider``/``agent.model`` because
``restore_primary_runtime()`` only fires at the NEXT user turn's start
(``turn_context.py``). ``_resolve_review_runtime`` now re-derives the fork's
identity from ``agent._primary_runtime`` when ``_fallback_activated`` is set so
the review/curator pass runs on the primary model.
"""

from __future__ import annotations

from unittest.mock import patch

from agent.background_review import (
    _BackgroundReviewRun,
    _ReviewForkState,
    _resolve_review_runtime,
    _run_review_fork,
    _select_review_pool_credential,
)


class _FakeAgent:
    """Minimal stand-in exposing the attributes _resolve_review_runtime reads."""

    def __init__(self, **attrs):
        self.provider = attrs.get("provider", "openai")
        self.model = attrs.get("model", "gpt-primary")
        self._credential_pool = attrs.get("_credential_pool", None)
        self.request_overrides = attrs.get("request_overrides", {})
        self.max_tokens = attrs.get("max_tokens", None)
        self.acp_command = attrs.get("acp_command", None)
        self.acp_args = attrs.get("acp_args", [])
        self._fallback_activated = attrs.get("_fallback_activated", False)
        self._provider_fallback_active = attrs.get("_provider_fallback_active", self._fallback_activated)
        self._rate_limited_until = attrs.get("_rate_limited_until", 0)
        self._primary_runtime = attrs.get("_primary_runtime", None)
        self._fallback_runtime = attrs.get("_fallback_runtime", None)

    def _current_main_runtime(self):
        # When fallback is active, the "current main runtime" is the fallback's.
        return self._fallback_runtime or {
            "provider": self.provider, "model": self.model,
            "api_key": "sk-fallback", "base_url": "https://fallback.invalid/v1",
            "api_mode": "openai",
        }


PRIMARY_SNAPSHOT = {
    "provider": "openai",
    "model": "gpt-primary",
    "api_key": "sk-primary",
    "base_url": "https://primary.invalid/v1",
    "api_mode": "openai",
    "request_overrides": {"temperature": 0.3},
}

FALLBACK_RUNTIME = {
    "provider": "anthropic",
    "model": "claude-fallback",
    "api_key": "sk-fallback",
    "base_url": "https://fallback.invalid/v1",
    "api_mode": "openai",
}


def test_fallback_active_returns_primary_identity():
    """When _fallback_activated is True, the fork must use the primary snapshot, not the
    inherited fallback provider/model."""
    agent = _FakeAgent(
        provider="anthropic", model="claude-fallback",  # inherited fallback
        _fallback_activated=True,
        _primary_runtime=PRIMARY_SNAPSHOT,
        _fallback_runtime=FALLBACK_RUNTIME,
    )
    with patch("agent.credential_pool.load_pool", side_effect=Exception("no auth.json")):
        rt = _resolve_review_runtime(agent, task_cfg={})
    assert rt["provider"] == "openai", f"expected primary provider, got {rt['provider']!r}"
    assert rt["model"] == "gpt-primary", f"expected primary model, got {rt['model']!r}"
    assert rt["api_key"] == "sk-primary", f"expected primary api_key, got {rt['api_key']!r}"
    assert rt["base_url"] == "https://primary.invalid/v1"
    # No primary pool loadable in the test env -> None (fork falls back to direct api_key).
    assert rt["credential_pool"] is None
    assert rt["request_overrides"] == {"temperature": 0.3}
    # Different provider/model means a cold routed fork; never copy the fallback prompt/parity knobs.
    assert rt["routed"] is True


def test_no_fallback_returns_current_runtime():
    """Without an active fallback, behavior is unchanged: the fork inherits the parent's live
    runtime (the pre-fix path)."""
    agent = _FakeAgent(
        provider="openai", model="gpt-primary",  # primary == current (no fallback)
        _fallback_activated=False,
        _primary_runtime=None,
        _fallback_runtime=FALLBACK_RUNTIME,
    )
    rt = _resolve_review_runtime(agent, task_cfg={})
    # No fallback -> identity = parent_runtime (the fallback dict returned by _current_main_runtime
    # in this fake; in real life it is the live runtime).
    assert rt["provider"] == "anthropic"
    assert rt["model"] == "claude-fallback"
    assert rt["api_key"] == "sk-fallback"
    assert rt["routed"] is False


class _FakePool:
    def __init__(self, provider, selected=None, next_available_at=None):
        self.provider = provider
        self.selected = selected
        self._next_available_at = next_available_at
        self.peek_calls = 0
        self.select_calls = 0

    def next_available_at(self):
        return self._next_available_at

    def has_credentials(self):
        return self.selected is not None or self._next_available_at is not None

    def has_available(self):
        return self.selected is not None

    def peek(self):
        self.peek_calls += 1
        return self.selected

    def select(self):
        self.select_calls += 1
        return self.selected


class _FakeEntry:
    provider = "openai"
    runtime_api_key = "sk-healthy-primary"
    access_token = "sk-healthy-primary"
    runtime_base_url = "https://healthy-primary.invalid/v1"
    base_url = runtime_base_url


def test_fallback_active_loads_primary_pool():
    """When the primary credential pool is loadable, the fork gets it (not the fallback's pool)
    so a cross-provider fallback pool can't trip the provider-mismatch guard."""
    agent = _FakeAgent(
        provider="anthropic", model="claude-fallback",
        _fallback_activated=True,
        _primary_runtime=PRIMARY_SNAPSHOT,
        _fallback_runtime=FALLBACK_RUNTIME,
    )
    fake_pool = _FakePool("openai", _FakeEntry())
    with patch("agent.credential_pool.load_pool", return_value=fake_pool), \
         patch("agent.credential_pool.credential_pool_matches_provider", return_value=True):
        rt = _resolve_review_runtime(agent, task_cfg={})
    assert rt["provider"] == "openai"
    assert rt["model"] == "gpt-primary"
    assert rt["credential_pool"] is fake_pool
    assert rt["select_pool_on_admission"] is True
    assert fake_pool.peek_calls == 0
    assert fake_pool.select_calls == 0
    # Construction is side-effect free; the admitted request swaps to the selected entry below.
    assert rt["api_key"] == "sk-primary"
    assert rt["base_url"] == "https://primary.invalid/v1"


def test_primary_pool_reset_keeps_live_fallback():
    """A persisted pool reset is the authoritative restore gate after local cooldown elapses."""
    import time

    agent = _FakeAgent(
        provider="anthropic", model="claude-fallback",
        _fallback_activated=True, _provider_fallback_active=True,
        _primary_runtime=PRIMARY_SNAPSHOT,
        _fallback_runtime=FALLBACK_RUNTIME,
    )
    exhausted_pool = _FakePool("openai", next_available_at=time.time() + 60)
    with patch("agent.credential_pool.load_pool", return_value=exhausted_pool), \
         patch("agent.credential_pool.credential_pool_matches_provider", return_value=True):
        rt = _resolve_review_runtime(agent, task_cfg={})
    assert rt["provider"] == "anthropic"
    assert rt["model"] == "claude-fallback"
    assert rt["credential_pool"] is agent._credential_pool
    assert rt["routed"] is False


def test_empty_primary_pool_keeps_direct_snapshot_credential():
    """Production load_pool returns an empty pool object, not None, for direct-key providers."""
    agent = _FakeAgent(
        provider="anthropic", model="claude-fallback",
        _fallback_activated=True, _provider_fallback_active=True,
        _primary_runtime=PRIMARY_SNAPSHOT,
        _fallback_runtime=FALLBACK_RUNTIME,
    )
    empty_pool = _FakePool("openai")
    with patch("agent.credential_pool.load_pool", return_value=empty_pool), \
         patch("agent.credential_pool.credential_pool_matches_provider", return_value=True):
        rt = _resolve_review_runtime(agent, task_cfg={})
    assert rt["provider"] == "openai"
    assert rt["api_key"] == "sk-primary"
    assert rt["credential_pool"] is None
    assert rt["select_pool_on_admission"] is False
    assert rt["routed"] is True


def test_primary_pool_without_available_entry_keeps_live_fallback():
    """A populated but unusable pool must not fall back to the stale snapshot key."""
    agent = _FakeAgent(
        provider="anthropic", model="claude-fallback",
        _fallback_activated=True, _provider_fallback_active=True,
        _primary_runtime=PRIMARY_SNAPSHOT,
        _fallback_runtime=FALLBACK_RUNTIME,
    )
    unavailable_pool = _FakePool("openai")
    unavailable_pool.has_credentials = lambda: True
    with patch("agent.credential_pool.load_pool", return_value=unavailable_pool), \
         patch("agent.credential_pool.credential_pool_matches_provider", return_value=True):
        rt = _resolve_review_runtime(agent, task_cfg={})
    assert rt["provider"] == "anthropic"
    assert rt["api_key"] == "sk-fallback"
    assert rt["routed"] is False


def test_same_model_different_endpoint_is_routed():
    """Endpoint/API-mode changes are cold identities even when provider/model labels match."""
    primary = {
        **PRIMARY_SNAPSHOT,
        "provider": "openai",
        "model": "shared-model",
        "base_url": "https://primary.invalid/v1",
        "api_mode": "codex_responses",
    }
    fallback = {
        **FALLBACK_RUNTIME,
        "provider": "openai",
        "model": "shared-model",
        "base_url": "https://fallback.invalid/v1",
        "api_mode": "chat_completions",
    }
    agent = _FakeAgent(
        provider="openai", model="shared-model",
        _fallback_activated=True, _provider_fallback_active=True,
        _primary_runtime=primary, _fallback_runtime=fallback,
    )
    with patch("agent.credential_pool.load_pool", return_value=None):
        rt = _resolve_review_runtime(agent, task_cfg={})
    assert rt["base_url"] == "https://primary.invalid/v1"
    assert rt["api_mode"] == "codex_responses"
    assert rt["routed"] is True


def test_active_primary_cooldown_keeps_live_fallback():
    """A review must not bypass the same primary cooldown respected by turn restoration."""
    import time

    agent = _FakeAgent(
        provider="anthropic", model="claude-fallback",
        _fallback_activated=True, _provider_fallback_active=True,
        _rate_limited_until=time.monotonic() + 60,
        _primary_runtime=PRIMARY_SNAPSHOT,
        _fallback_runtime=FALLBACK_RUNTIME,
    )
    rt = _resolve_review_runtime(agent, task_cfg={})
    assert rt["provider"] == "anthropic"
    assert rt["model"] == "claude-fallback"
    assert rt["api_key"] == "sk-fallback"
    assert rt["routed"] is False


def test_model_once_restore_flag_does_not_masquerade_as_provider_fallback():
    """_fallback_activated is shared by /model --once; provenance must use its dedicated flag."""
    agent = _FakeAgent(
        provider="anthropic", model="claude-once",
        _fallback_activated=True, _provider_fallback_active=False,
        _primary_runtime=PRIMARY_SNAPSHOT,
        _fallback_runtime=FALLBACK_RUNTIME,
    )
    rt = _resolve_review_runtime(agent, task_cfg={})
    assert rt["provider"] == "anthropic"
    assert rt["model"] == "claude-fallback"
    assert rt["routed"] is False


def test_pool_selection_is_deferred_until_request_admission():
    """Resolution peeks only; admission leases and applies exactly one credential."""
    pool = _FakePool("openai", _FakeEntry())

    class _ReviewAgent:
        _credential_pool = pool
        swapped = None

        def _swap_credential(self, entry):
            self.swapped = entry

    review_agent = _ReviewAgent()
    _select_review_pool_credential(review_agent)
    assert pool.peek_calls == 0
    assert pool.select_calls == 1
    assert review_agent.swapped is pool.selected


def test_cancelled_after_fork_build_does_not_select_pool_or_call_provider():
    """Cancellation between construction and admission has no pool or provider side effects."""
    calls = []

    class _ReviewAgent:
        _session_messages = []

        def run_conversation(self, **kwargs):
            calls.append("provider")

        def release_clients(self):
            pass

    review_agent = _ReviewAgent()
    runtime = {"select_pool_on_admission": True}
    run = _BackgroundReviewRun()
    run.cancel_requested.set()
    state = _ReviewForkState()
    parent = type("Parent", (), {})()

    with patch(
        "agent.background_review.build_cache_parity_fork",
        return_value=(review_agent, runtime, True),
    ), patch(
        "agent.background_review._select_review_pool_credential",
        side_effect=lambda fork: calls.append("select"),
    ), patch("agent.background_review._track_review_fork"), patch(
        "agent.background_review.finish_background_review_run"
    ), patch("agent.background_review._review_tool_whitelist", return_value=(set(), set())), patch(
        "agent.background_review._snapshot_review_usage", return_value={}
    ), patch("agent.background_review._record_review_usage_to_parent"), patch(
        "hermes_cli.plugins.set_thread_tool_whitelist"
    ), patch("hermes_cli.plugins.clear_thread_tool_whitelist"):
        _run_review_fork(parent, [], "review", {}, run, state)

    assert calls == []


def test_parent_agent_untouched_on_fork_resolve():
    """Resolving the fork runtime must NOT mutate the parent agent's fallback state: the parent
    stays on the fallback until the next user turn restores the primary, so on_session_end /
    usage attribution for the current turn still records the fallback it actually used."""
    agent = _FakeAgent(
        provider="anthropic", model="claude-fallback",
        _fallback_activated=True,
        _primary_runtime=PRIMARY_SNAPSHOT,
        _fallback_runtime=FALLBACK_RUNTIME,
    )
    with patch("agent.credential_pool.load_pool", side_effect=Exception("no auth.json")):
        _resolve_review_runtime(agent, task_cfg={})
    # Parent must still report the fallback it ran the turn on.
    assert agent._fallback_activated is True
    assert agent.provider == "anthropic"
    assert agent.model == "claude-fallback"
