"""Regression coverage for host-owned context-engine compression budgets."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.conversation_compression import (
    apply_context_engine_compression_budget,
    check_compression_model_feasibility,
)


class BudgetAwareEngine:
    """Minimal external engine opting into Hermes' optional budget contract."""

    def __init__(self, context_length: int, threshold_tokens: int) -> None:
        self.context_length = context_length
        self.threshold_tokens = threshold_tokens
        self.threshold_percent = 0.50
        self.budgets: list[tuple[int, int, str]] = []

    def set_compression_budget(
        self, context_capacity: int, trigger_tokens: int, *, reason: str
    ) -> None:
        self.budgets.append((context_capacity, trigger_tokens, reason))
        self.threshold_tokens = trigger_tokens


def _agent(engine: BudgetAwareEngine, *, max_tokens: int | None = None):
    return SimpleNamespace(
        compression_enabled=True,
        context_compressor=engine,
        max_tokens=max_tokens,
        model="test-main-model",
        provider="test-provider",
        _compression_threshold_percent=0.50,
        _auxiliary_compression_context_cap=None,
        _aux_compression_context_length_config=None,
        _custom_providers=[],
        _compression_warning=None,
        _emit_status=MagicMock(),
        _current_main_runtime=lambda: {},
    )


def _aux_client() -> MagicMock:
    client = MagicMock()
    client.base_url = "https://aux.example/v1"
    client.api_key = "test-key"
    return client


def test_budget_aware_engine_receives_builtin_capacity_and_trigger():
    """Host handoff must preserve the built-in output reservation semantics."""
    engine = BudgetAwareEngine(context_length=1_000_000, threshold_tokens=0)
    agent = _agent(engine, max_tokens=200_000)

    assert apply_context_engine_compression_budget(
        agent, 1_000_000, reason="model_init"
    )

    assert engine.budgets == [(800_000, 400_000, "model_init")]


def test_auxiliary_cap_survives_transition_then_releases_after_revalidation():
    """Do not expose a new runtime to an unproven auxiliary context window."""
    nominal_context = 1_050_000
    auxiliary_context = 272_000
    engine = BudgetAwareEngine(
        context_length=nominal_context, threshold_tokens=525_000
    )
    agent = _agent(engine)

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("test-provider", None, None, None, None),
        ),
        patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(_aux_client(), "small-aux"),
        ),
        patch(
            "agent.model_metadata.get_model_context_length",
            return_value=auxiliary_context,
        ),
    ):
        check_compression_model_feasibility(agent)

    assert agent._auxiliary_compression_context_cap == 270_000
    assert engine.budgets[-1] == (
        270_000,
        270_000,
        "auxiliary_context",
    )

    assert apply_context_engine_compression_budget(
        agent, nominal_context, reason="fallback_model"
    )
    assert engine.budgets[-1] == (
        270_000,
        270_000,
        "fallback_model",
    )

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("test-provider", None, None, None, None),
        ),
        patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(_aux_client(), "large-aux"),
        ),
        patch(
            "agent.model_metadata.get_model_context_length",
            return_value=nominal_context,
        ),
    ):
        check_compression_model_feasibility(agent)

    assert agent._auxiliary_compression_context_cap is None
    assert engine.budgets[-1] == (
        nominal_context,
        525_000,
        "auxiliary_context_restored",
    )
