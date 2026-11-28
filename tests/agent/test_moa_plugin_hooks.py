from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import auxiliary_client, moa_loop
from agent.error_classifier import classify_api_error
from agent.shell_hooks import _parse_hooks
from hermes_cli import plugins as plugins_mod


def _register_temp_hook(monkeypatch, hook_name: str, callback):
    mgr = plugins_mod.get_plugin_manager()
    monkeypatch.setattr(mgr, "_discovered", True)
    existing = list(mgr._hooks.get(hook_name, []))
    monkeypatch.setitem(mgr._hooks, hook_name, [*existing, callback])


def test_moa_fanout_verdict_accept_cache_false_skips_caching(monkeypatch):
    seen_payloads = []

    def _verdict(**kwargs):
        seen_payloads.append(kwargs)
        return {"action": "accept", "cache": False}

    _register_temp_hook(monkeypatch, "moa_fanout_verdict", _verdict)
    monkeypatch.setattr(
        moa_loop,
        "_run_references_parallel",
        lambda slots, *a, **kw: [(moa_loop._slot_label(s), "[failed: timeout]", None) for s in slots],
    )
    monkeypatch.setattr(
        moa_loop,
        "_resolve_preset_cached",
        lambda _name: (
            {
                "enabled": True,
                "reference_models": [{"provider": "openrouter", "model": "ref-a", "enabled": True}],
                "aggregator": {"provider": "openrouter", "model": "agg-main"},
            },
            {},
        ),
    )

    facade = moa_loop.MoAChatCompletions("review", reference_callback=None, agent=None)
    prepared = facade.prepare([{"role": "user", "content": "hello"}])

    assert len(seen_payloads) == 1
    assert seen_payloads[0]["attempt"] == 1
    assert seen_payloads[0]["preset_name"] == "review"
    assert seen_payloads[0]["reference_outputs"][0]["failed"] is True
    assert facade._ref_cache_key is None
    assert facade._ref_cache_outputs == []
    assert isinstance(prepared, dict)


def test_moa_fanout_verdict_retry_merges_healthy_and_substitutes_secondary(monkeypatch):
    calls = []

    def _fake_parallel(slots, *a, **kw):
        labels = [moa_loop._slot_label(s) for s in slots]
        calls.append(labels)
        out = []
        for s in slots:
            lbl = moa_loop._slot_label(s)
            if "ref-good" in lbl:
                out.append((lbl, "healthy first advice", None))
            elif "ref-backup" in lbl:
                out.append((lbl, "secondary backup advice", None))
            else:
                out.append((lbl, "[failed: 429 rate limited]", None))
        return out

    def _verdict(attempt, reference_outputs, **kwargs):
        failed = [r for r in reference_outputs if r["failed"]]
        if failed and attempt == 1:
            return {
                "action": "retry",
                "slots": [{"provider": "openrouter", "model": "ref-backup", "enabled": True}],
                "delay_s": 0.01,
                "merge_healthy": True,
            }
        return {"action": "accept", "cache": True}

    _register_temp_hook(monkeypatch, "moa_fanout_verdict", _verdict)
    monkeypatch.setattr(moa_loop, "_run_references_parallel", _fake_parallel)
    monkeypatch.setattr(
        moa_loop,
        "_resolve_preset_cached",
        lambda _name: (
            {
                "enabled": True,
                "reference_models": [
                    {"provider": "openrouter", "model": "ref-good", "enabled": True},
                    {"provider": "openrouter", "model": "ref-flaky", "enabled": True},
                ],
                "aggregator": {"provider": "openrouter", "model": "agg-main"},
            },
            {},
        ),
    )

    facade = moa_loop.MoAChatCompletions("review", reference_callback=None, agent=None)
    prepared = facade.prepare([{"role": "user", "content": "hello"}])

    assert calls == [
        ["openrouter:ref-good", "openrouter:ref-flaky"],
        ["openrouter:ref-backup"],
    ]
    guidance = prepared["guidance"] or ""
    assert "healthy first advice" in guidance
    assert "secondary backup advice" in guidance
    assert len(facade._ref_cache_outputs) == 2


def test_moa_fanout_verdict_abort_halts_before_aggregator_and_classifies_non_retryable(monkeypatch):
    agg_called = {"n": 0}

    def _forbid_agg(**kwargs):
        if kwargs.get("task") == "moa_aggregator":
            agg_called["n"] += 1
        raise AssertionError("aggregator must not run after abort verdict")

    _register_temp_hook(
        monkeypatch,
        "moa_fanout_verdict",
        lambda **kw: {"action": "abort", "message": "MoA reference outage: halted fail-closed"},
    )
    monkeypatch.setattr(
        moa_loop,
        "_run_references_parallel",
        lambda slots, *a, **kw: [(moa_loop._slot_label(s), "[failed: 503]", None) for s in slots],
    )
    monkeypatch.setattr(moa_loop, "call_llm", _forbid_agg)
    monkeypatch.setattr(
        moa_loop,
        "_resolve_preset_cached",
        lambda _name: (
            {
                "enabled": True,
                "reference_models": [{"provider": "openrouter", "model": "ref-a", "enabled": True}],
                "aggregator": {"provider": "openrouter", "model": "agg-main"},
            },
            {},
        ),
    )

    facade = moa_loop.MoAChatCompletions("review", reference_callback=None, agent=None)
    prepared = facade.prepare([{"role": "user", "content": "hello"}])
    with pytest.raises(Exception) as exc_info:
        facade.create(_moa_prepared_request=prepared, messages=[{"role": "user", "content": "hello"}])

    assert "MoA reference outage: halted fail-closed" in str(exc_info.value)
    assert agg_called["n"] == 0
    classified = classify_api_error(exc_info.value, provider="moa", model="review")
    assert classified.retryable is False
    assert classified.should_fallback is False

    # Also verify one-shot aggregate_moa_context aborts before aggregator synthesis
    with pytest.raises(Exception) as oneshot_exc:
        moa_loop.aggregate_moa_context(
            user_prompt="hello",
            api_messages=[{"role": "user", "content": "hello"}],
            reference_models=[{"provider": "openrouter", "model": "ref-a", "enabled": True}],
            aggregator={"provider": "openrouter", "model": "agg-main"},
        )
    assert "MoA reference outage: halted fail-closed" in str(oneshot_exc.value)
    assert agg_called["n"] == 0


def test_auxiliary_fallback_policy_blocks_moa_reference_impersonation(monkeypatch):
    monkeypatch.setattr(auxiliary_client, "_read_main_provider", lambda: "moa")
    monkeypatch.setattr(auxiliary_client, "_read_main_model", lambda: "review")
    monkeypatch.setattr(auxiliary_client, "_resolve_moa_aggregator", lambda _preset: ("openrouter", "agg-main"))
    monkeypatch.setattr(auxiliary_client, "_custom_health_base_url", lambda _p: "")
    monkeypatch.setattr(auxiliary_client, "_is_provider_unhealthy", lambda *_a, **_kw: False)
    monkeypatch.setattr(
        auxiliary_client,
        "resolve_provider_client",
        lambda **_kw: (SimpleNamespace(name="agg-client"), "agg-main"),
    )

    _register_temp_hook(
        monkeypatch,
        "auxiliary_fallback_policy",
        lambda task=None, **_kw: {"allow": False, "reason": "no aggregator impersonation"}
        if task == "moa_reference"
        else None,
    )

    blocked = auxiliary_client._try_main_agent_model_fallback("custom-ref", task="moa_reference", reason="rate_limit")
    assert blocked == (None, None, "")

    allowed = auxiliary_client._try_main_agent_model_fallback("custom-ref", task="titling", reason="rate_limit")
    assert allowed[1] == "agg-main"


def test_new_hooks_refused_for_shell_hooks():
    specs, complete = _parse_hooks(
        {
            "moa_fanout_verdict": [{"command": "echo hi"}],
            "auxiliary_fallback_policy": [{"command": "echo hi"}],
        }
    )
    assert specs == []
    assert complete is True


def test_moa_fanout_verdict_raising_callback_fails_open_and_warns_on_competing_directives(monkeypatch, caplog):
    def _boom(**_kw):
        raise RuntimeError("plugin bug")

    _register_temp_hook(monkeypatch, "moa_fanout_verdict", _boom)
    _register_temp_hook(monkeypatch, "moa_fanout_verdict", lambda **_kw: {"action": "accept", "cache": False})
    _register_temp_hook(monkeypatch, "moa_fanout_verdict", lambda **_kw: {"action": "abort", "message": "shadowed"})

    verdict = plugins_mod.get_moa_fanout_verdict(
        preset_name="review",
        preset_config={},
        aggregator={"provider": "openrouter", "model": "agg"},
        active_slots=[{"provider": "openrouter", "model": "ref-a"}],
        reference_outputs=[{"label": "openrouter:ref-a", "text": "ok", "failed": False}],
        attempt=1,
    )
    assert verdict == {"action": "accept", "cache": False}
    assert "skipped 1 valid directive(s)" in caplog.text
