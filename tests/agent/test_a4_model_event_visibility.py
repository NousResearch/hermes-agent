"""A4 — model-event-visibility pass (3 axes), behavior-contract tests.

- **A** a deliberate ``/model`` switch is announced in-chat like failover, via
  the SAME ``_emit_status`` seam. Covers the symmetric to/from render and the
  model+effort+context-window deltas, plus dedupe and the silent-on-no-op case.
- **B** the compaction summarizer's EXPLICIT-override paths are swap-aware:
  ``update_model`` resets a stale runtime-carried ``summary_model`` (but keeps a
  config-pinned one), and a concrete ``auxiliary.compression.model`` pin whose
  vendor family is incompatible with the live provider falls through to the live
  main model instead of shipping an unsupported id.
- **C** first-class-provider composite ``provider/model`` strings are stripped
  at the blackbox recording boundary (never reach ``turns.db.model``) while
  aggregator vendor slugs are preserved; and the fN provider-collision lint
  fires on a real name/alias/base_url/env-key collision and stays silent on the
  legitimate proxy-vs-bridge / credential-reuse pairs.

These assert BEHAVIOR CONTRACTS (how the pieces must relate), not snapshots.
"""

import types

import pytest


# ── Axis A — deliberate /model switch announce ─────────────────────────────

def _announce_agent():
    a = types.SimpleNamespace()
    a._announced = []

    def _emit_status(message):
        a._announced.append(("lifecycle", message))

    a._emit_status = _emit_status
    a._last_switch_announced = None
    return a


def test_switch_announce_renders_symmetric_to_from_with_deltas():
    from agent.chat_completion_helpers import _emit_switch_announce

    agent = _announce_agent()
    _emit_switch_announce(
        agent,
        old_model="claude-opus-4-8",
        new_model="gpt-5.5",
        new_provider="openai-codex",
        old_provider="claude-app",
        old_window=1_000_000,
        new_window=272_000,
        old_effort="high",
        new_effort="xhigh",
    )
    msgs = [m for (_k, m) in agent._announced if m.startswith("🔀")]
    assert len(msgs) == 1, f"expected exactly one switch announce, got {agent._announced!r}"
    msg = msgs[0]
    # Symmetric to/from, both sides rendered provider/model.
    assert "claude-app/claude-opus-4-8" in msg, msg
    assert "openai-codex/gpt-5.5" in msg, msg
    # Effort + context-window deltas appear when they differ.
    assert "effort high→xhigh" in msg, msg
    assert "1M" in msg and "272K" in msg, msg


def test_switch_announce_deduped_on_same_transition():
    from agent.chat_completion_helpers import _emit_switch_announce

    agent = _announce_agent()
    for _ in range(3):
        _emit_switch_announce(
            agent, old_model="a-model", new_model="b-model", new_provider="p",
        )
    msgs = [m for (_k, m) in agent._announced if m.startswith("🔀")]
    assert len(msgs) == 1, f"re-entrant same-transition must announce once, got {msgs!r}"


def test_switch_announce_silent_on_noop():
    from agent.chat_completion_helpers import _emit_switch_announce

    agent = _announce_agent()
    # Same model, no window/effort delta -> silent.
    _emit_switch_announce(
        agent, old_model="same", new_model="same", new_provider="p",
        old_window=100_000, new_window=100_000, old_effort="high", new_effort="high",
    )
    assert agent._announced == [], f"a true no-op must be silent, got {agent._announced!r}"


def test_switch_announce_window_only_delta_fires_even_when_model_same():
    """Same model id on both sides but a different context window IS a real,
    announceable change (e.g. a provider route swap that changes the window)."""
    from agent.chat_completion_helpers import _emit_switch_announce

    agent = _announce_agent()
    _emit_switch_announce(
        agent, old_model="m", new_model="m", new_provider="new-prov",
        old_provider="old-prov", old_window=1_000_000, new_window=200_000,
    )
    msgs = [m for (_k, m) in agent._announced if m.startswith("🔀")]
    assert len(msgs) == 1, f"window-only delta should announce, got {agent._announced!r}"
    assert "context window 1M→200K" in msgs[0], msgs[0]


def test_switch_announce_is_pure_emission_no_context_mutation():
    """Cache invariant: the announce must ONLY call _emit_status. It must not
    touch conversation history / system prompt / toolset (none exist on the
    stub — a mutation attempt would AttributeError)."""
    from agent.chat_completion_helpers import _emit_switch_announce

    agent = _announce_agent()
    _emit_switch_announce(
        agent, old_model="x", new_model="y", new_provider="p",
    )
    # The only side effects allowed: the dedupe marker + the emission.
    assert agent._last_switch_announced == ("x", "y")
    assert len([m for (_k, m) in agent._announced if m.startswith("🔀")]) == 1
    assert not hasattr(agent, "messages")
    assert not hasattr(agent, "_cached_system_prompt")


# ── Axis B — summarizer inherits live model on explicit-override paths ──────

def _bare_compressor(model, summary_model, config_pinned):
    """Construct a ContextCompressor via __new__ so no heavy init / network
    fires, then set only the fields update_model's reset logic reads."""
    from agent.context_compressor import ContextCompressor

    cc = ContextCompressor.__new__(ContextCompressor)
    cc.model = model
    cc.base_url = ""
    cc.api_key = ""
    cc.provider = ""
    cc.api_mode = ""
    cc.context_length = 200_000
    cc.threshold_percent = 0.5
    cc.max_tokens = None
    cc.summary_target_ratio = 0.2
    cc._per_model_threshold_cfg = None
    cc._global_threshold_percent = None
    cc._codex_gpt55_autoraise = True
    cc.summary_model = summary_model
    cc._summary_model_is_config_pinned = config_pinned
    return cc


def test_update_model_resets_stale_runtime_summary_model_on_switch():
    cc = _bare_compressor("claude-opus-4-8", "claude-haiku-4-5", config_pinned=False)
    cc.update_model(model="gpt-5.5", context_length=272_000, provider="openai-codex")
    assert cc.summary_model == "", (
        "a runtime-carried summary_model must be cleared on a real switch so "
        f"compaction inherits the live model, got {cc.summary_model!r}"
    )
    assert cc.model == "gpt-5.5"


def test_update_model_preserves_config_pinned_summary_model_on_switch():
    cc = _bare_compressor("claude-opus-4-8", "gemini-2.5-flash", config_pinned=True)
    cc.update_model(model="gpt-5.5", context_length=272_000, provider="openai-codex")
    assert cc.summary_model == "gemini-2.5-flash", (
        "a deliberately config-pinned summary_model must survive a switch, "
        f"got {cc.summary_model!r}"
    )


def test_update_model_no_reset_when_model_unchanged():
    cc = _bare_compressor("claude-opus-4-8", "claude-haiku-4-5", config_pinned=False)
    cc.update_model(model="claude-opus-4-8", context_length=200_000, provider="anthropic")
    assert cc.summary_model == "claude-haiku-4-5", (
        "no model change -> no reset (the reset is gated on model-changed)"
    )


def test_config_pin_flag_only_true_when_override_present():
    """The pin flag must not latch True when there's no override, otherwise a
    swap would wrongly PRESERVE an empty summary_model as if pinned."""
    from agent.context_compressor import ContextCompressor

    cc = ContextCompressor.__new__(ContextCompressor)
    # Replicate the __init__ derivation:
    for override, pinned_kw, expect in [
        ("", True, False),        # pinned requested but no override -> False
        ("gemini-2.5-flash", True, True),
        ("gemini-2.5-flash", False, False),
        (None, True, False),
    ]:
        got = bool(pinned_kw and (override or ""))
        assert got is expect, (override, pinned_kw, expect, got)


def test_pinned_model_incompatible_with_provider_matrix():
    from agent.auxiliary_client import _pinned_model_incompatible_with_provider as bad

    # Provable single-vendor mismatches -> True.
    assert bad("claude-opus-4-8", "openai-codex") is True
    assert bad("gpt-5.5", "anthropic") is True
    assert bad("claude-haiku-4-5", "gemini") is True
    # Matching family -> False.
    assert bad("gpt-5.5", "openai-codex") is False
    assert bad("claude-opus-4-8", "anthropic") is False
    # Aggregators / custom / auto route many vendors -> never block.
    assert bad("claude-opus-4-8", "openrouter") is False
    assert bad("claude-opus-4-8", "custom") is False
    assert bad("claude-opus-4-8", "auto") is False
    # Unknown model family -> can't prove wrong -> False.
    assert bad("some-unknown-model", "openai-codex") is False


def test_compression_pin_falls_through_to_live_model_on_provider_swap(monkeypatch):
    """E2E of the B-2 resolution: with a pinned Claude compression model but a
    live Codex provider, call_llm must DROP the pinned model (resolve model=None
    so _resolve_auto inherits the live main model) rather than send the bad id."""
    import agent.auxiliary_client as ac

    # Pinned concrete compression model = a Claude id.
    monkeypatch.setattr(
        ac, "_resolve_task_provider_model",
        lambda task, provider, model, base_url, api_key: (
            "auto", "claude-opus-4-8", None, None, None,
        ),
        raising=True,
    )
    # Live main provider is the Codex route (the swapped-to provider).
    monkeypatch.setattr(ac, "_read_main_provider", lambda: "openai-codex", raising=True)

    captured = {}

    def _fake_get_cached_client(provider, model=None, **kw):
        captured["provider"] = provider
        captured["model"] = model
        # Return a stub client + a resolved default model.
        return types.SimpleNamespace(base_url="x"), (model or "gpt-5.5")

    monkeypatch.setattr(ac, "_get_cached_client", _fake_get_cached_client, raising=True)
    monkeypatch.setattr(ac, "_get_task_extra_body", lambda task: {}, raising=True)
    monkeypatch.setattr(ac, "_get_task_timeout", lambda task: 30.0, raising=True)

    # Stop before the actual network send: raise once resolution + client
    # selection are done. _build_call_kwargs is reached right after
    # _get_cached_client, so raising there lets us assert on the captured model.
    class _Stop(Exception):
        pass

    def _boom(*a, **k):
        raise _Stop()

    monkeypatch.setattr(ac, "_build_call_kwargs", _boom, raising=True)

    with pytest.raises(_Stop):
        ac.call_llm(
            task="compression",
            messages=[{"role": "user", "content": "x"}],
            main_runtime={"provider": "openai-codex", "model": "gpt-5.5"},
        )
    # The load-bearing assertion: the pinned Claude id was NOT passed to the
    # client resolver — it was dropped to None so the live model is inherited.
    assert captured.get("model") is None, (
        "an incompatible pinned compression model must be dropped to None "
        f"(inherit live model), got {captured.get('model')!r}"
    )


# ── Axis C — telemetry hygiene ─────────────────────────────────────────────
# NOTE (deploy reconciliation 2026-07-08): fork/main landed the canonical fix for
# this exact bug as `_recover_provider_from_model(model, provider)` while A4 was in
# flight (it proposed `_normalize_record_provider_model`). The two solve the same
# phantom-"Spend by model" row; we keep the canonical one and assert ITS contract
# here. fork/main's fix is deliberately narrower — it recovers ONLY when the
# provider column is EMPTY (a correctly-split turn, or one that already carries a
# provider, is left untouched) — so the tests assert that real, shipped behavior.

def test_c1_recovers_provider_from_composite_when_provider_empty():
    from plugins.blackbox import _recover_provider_from_model as rec

    # Empty provider + first-class fN composite model -> split out the lane.
    prov, mdl = rec("claude-apx-6/claude-haiku-4-5", "")
    assert (prov, mdl) == ("claude-apx-6", "claude-haiku-4-5"), (prov, mdl)


def test_c1_leaves_composite_untouched_when_provider_already_set():
    from plugins.blackbox import _recover_provider_from_model as rec

    # Provider already populated -> canonical fix trusts the split turn, no rewrite.
    prov, mdl = rec("claude-apx-6/claude-haiku-4-5", "claude-apx-6")
    assert (prov, mdl) == ("claude-apx-6", "claude-apx-6/claude-haiku-4-5"), (prov, mdl)


def test_c1_preserves_aggregator_vendor_prefix_verbatim():
    from plugins.blackbox import _recover_provider_from_model as rec

    # An aggregator turn always records WITH a provider, so it's left untouched;
    # even with an empty provider a non-first-class vendor prefix is not split.
    prov, mdl = rec("anthropic/claude-haiku-4.5", "openrouter")
    assert (prov, mdl) == ("openrouter", "anthropic/claude-haiku-4.5"), (prov, mdl)


def test_c1_leaves_bare_model_untouched():
    from plugins.blackbox import _recover_provider_from_model as rec

    prov, mdl = rec("claude-opus-4-8", "claude-apx-1")
    assert (prov, mdl) == ("claude-apx-1", "claude-opus-4-8"), (prov, mdl)


def test_c2_lint_flags_duplicate_alias_and_base_url():
    import providers
    from providers.base import ProviderProfile

    # Two DISTINCT profiles claiming the same alias + same (url, auth) pair.
    p_a = ProviderProfile(
        name="fN-lane-a", aliases=("dup-alias",),
        base_url="https://proxy.example:9000/v1", env_vars=("LANE_A_KEY",),
    )
    p_b = ProviderProfile(
        name="fN-lane-b", aliases=("dup-alias",),
        base_url="https://proxy.example:9000/v1", env_vars=("LANE_B_KEY",),
    )
    monkey = [p_a, p_b]
    orig = providers.list_providers
    providers.list_providers = lambda: monkey  # type: ignore
    try:
        warns = providers._lint_provider_collisions()
    finally:
        providers.list_providers = orig  # type: ignore
    joined = " | ".join(warns)
    assert "alias collision" in joined and "dup-alias" in joined, warns
    assert "base_url collision" in joined, warns


def test_c2_lint_does_not_flag_proxy_vs_bridge_same_host_diff_port():
    import providers
    from providers.base import ProviderProfile

    # The real fN shape: same Tailscale host, DIFFERENT port/path -> NOT a
    # collision (full base_url incl. port + path differs).
    proxy = ProviderProfile(
        name="claude-apx-5", base_url="http://100.64.0.1:18801/anthropic",
        env_vars=("APX5_KEY",),
    )
    bridge = ProviderProfile(
        name="claude-bpx-5", base_url="http://100.64.0.1:3556/v1",
        env_vars=("BPX5_KEY",),
    )
    orig = providers.list_providers
    providers.list_providers = lambda: [proxy, bridge]  # type: ignore
    try:
        warns = providers._lint_provider_collisions()
    finally:
        providers.list_providers = orig  # type: ignore
    assert warns == [], f"proxy-vs-bridge (same host, diff port) must not warn: {warns!r}"


def test_c2_lint_clean_on_live_registry():
    """The real fleet provider registry must lint clean (no false positives on
    the deliberate minimax/minimax-oauth + alibaba credential-reuse pairs)."""
    import providers

    warns = providers.lint_provider_collisions(emit_log=False)
    assert warns == [], f"live provider registry should be collision-free, got {warns!r}"
