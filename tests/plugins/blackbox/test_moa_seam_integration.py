"""MoA execution → turn finalizer → Blackbox SQLite seam coverage."""

from __future__ import annotations

import importlib
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.conversation_loop import _build_moa_pricing_calls
from agent.turn_finalizer import finalize_turn
from agent.usage_pricing import CanonicalUsage, CostResult
import plugins.blackbox.cost as cost_mod
from plugins.blackbox.cost import compute_turn_cost


class _Agent:
    def __init__(self):
        self.max_iterations = 90
        self.iteration_budget = SimpleNamespace(remaining=10, used=1, max_total=90)
        self.quiet_mode = True
        self.model = "default"
        self.provider = "moa"
        self.base_url = ""
        self.session_id = "moa-seam"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0, context_length=1_000_000)
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []

    def _handle_max_iterations(self, *_args):
        raise AssertionError("not expected")

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, _messages):
        pass

    def _persist_session(self, *_args, **_kwargs):
        pass

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kwargs):
        pass


def _response(content: str, prompt_tokens: int, completion_tokens: int):
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage, model="fake")


def test_moa_execution_persists_priced_explicit_model_through_finalizer(
    tmp_path, monkeypatch
):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: default
  presets:
    default:
      enabled: true
      reference_models:
        - provider: openrouter
          model: anthropic/claude-opus-4.8
        - provider: openrouter
          model: openai/gpt-5.5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_constants
    import plugins.blackbox as bb
    import plugins.blackbox.store as store

    repo_root = Path(__file__).resolve().parents[3]
    assert Path(bb.__file__).resolve().is_relative_to(repo_root)

    importlib.reload(hermes_constants)
    importlib.reload(store)
    importlib.reload(bb)
    bb._sessions.clear()
    monkeypatch.setattr(
        bb,
        "_config",
        lambda: {
            "enabled": True,
            "cost_alert_threshold_usd": 999.0,
            "store_text": False,
            "record_subagents": True,
        },
    )
    monkeypatch.setattr(bb, "store", store, raising=False)
    monkeypatch.setitem(sys.modules, "plugins.blackbox.store", store)
    monkeypatch.setattr(bb, "_turn_id", lambda: "turn_moa_seam")

    def fake_call_llm(**kwargs):
        if kwargs.get("task") == "moa_reference":
            return _response("advice", 1_000, 100)
        return _response("acted", 400, 40)

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    from agent.moa_loop import MoAClient

    client = MoAClient("default")
    response = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "clean the db"}],
    )
    reference_usage, _ = client.consume_reference_usage()
    reference_calls = client.consume_reference_pricing_calls()
    aggregator_usage = CanonicalUsage(
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
    )
    combined_usage = aggregator_usage + reference_usage
    slot = client.last_aggregator_slot
    assert slot is not None
    pricing_calls = _build_moa_pricing_calls(
        reference_calls,
        aggregator_usage,
        aggregator_model=slot["model"],
        aggregator_provider=slot["provider"],
        aggregator_base_url=slot.get("base_url"),
    )
    turn_call = {
        "input_tokens": combined_usage.input_tokens,
        "output_tokens": combined_usage.output_tokens,
        "cache_read_tokens": combined_usage.cache_read_tokens,
        "cache_write_tokens": combined_usage.cache_write_tokens,
        "reasoning_tokens": combined_usage.reasoning_tokens,
        "prompt_tokens": combined_usage.prompt_tokens,
        "completion_tokens": combined_usage.output_tokens,
        "total_tokens": combined_usage.total_tokens,
        "latency_s": 0.1,
        "composition": None,
        "pricing_calls": pricing_calls,
    }
    rates = {
        "anthropic/claude-opus-4.8": Decimal("0.01"),
        "openai/gpt-5.5": Decimal("0.02"),
    }
    seen_routes = []

    def fake_estimate(model, usage, *, provider=None, base_url=None):
        seen_routes.append((model, provider, base_url))
        amount = rates[model]
        return CostResult(
            amount_usd=amount,
            status="estimated",
            source="official_docs_snapshot",
            label=f"~${amount}",
            cost_input_usd=amount,
            cost_output_usd=Decimal("0"),
            cost_cache_read_usd=Decimal("0"),
            cost_cache_write_usd=Decimal("0"),
        )

    monkeypatch.setattr(cost_mod, "estimate_usage_cost", fake_estimate)
    expected_routes = [
        (call["model"], call["provider"], call.get("base_url"))
        for call in pricing_calls
    ]
    expected_cost, expected_status, _ = compute_turn_cost(
        "moa/default", "moa", None, [turn_call]
    )
    assert expected_cost == pytest.approx(0.04)

    def invoke_hook(name, **kwargs):
        if name == "on_session_end":
            bb._on_session_end(**kwargs)
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    bb._on_session_start(session_id="moa-seam")
    finalize_turn(
        _Agent(),
        final_response=response.choices[0].message.content,
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "clean the db"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="clean the db",
        original_user_message="clean the db",
        _should_review_memory=False,
        _turn_exit_reason="text_response(stop)",
        _turn_calls=[turn_call],
    )

    row = store.get_turn("turn_moa_seam")
    assert row is not None
    assert row["provider"] == "moa"
    assert row["model"] == "moa/default"
    assert row["api_calls"] == 1
    assert row["input_tokens"] == combined_usage.input_tokens
    assert row["output_tokens"] == combined_usage.output_tokens
    assert row["cost_status"] == expected_status == "estimated"
    assert row["cost_usd"] == pytest.approx(expected_cost)
    assert seen_routes == expected_routes * 2
