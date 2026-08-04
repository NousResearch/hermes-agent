"""Regression coverage for host-owned context-engine compression budgets."""

from types import SimpleNamespace

from agent.conversation_compression import apply_context_engine_compression_budget


class BudgetAwareEngine:
    """Minimal external engine explicitly accepting the host budget contract."""

    def __init__(self, *, threshold_percent: float = 0.50) -> None:
        self.threshold_percent = threshold_percent
        self.threshold_tokens = 321_000
        self.budgets: list[tuple[int, int, str]] = []

    def set_compression_budget(
        self, context_capacity: int, trigger_tokens: int, *, reason: str
    ) -> bool:
        self.budgets.append((context_capacity, trigger_tokens, reason))
        self.threshold_tokens = trigger_tokens
        return True


class LegacyEngine:
    """An existing engine with no host-budget hook keeps its policy."""

    def __init__(self) -> None:
        self.threshold_percent = 0.75
        self.threshold_tokens = 321_000


def _agent(
    engine,
    *,
    max_tokens: int | None = None,
    model: str = "test-model",
    model_thresholds: dict[str, float] | None = None,
    threshold_tokens_cap: int | None = None,
):
    return SimpleNamespace(
        context_compressor=engine,
        max_tokens=max_tokens,
        model=model,
        _compression_threshold_percent=0.50,
        _compression_model_thresholds=model_thresholds or {},
        _compression_threshold_tokens_cap=threshold_tokens_cap,
    )


def test_budget_aware_engine_receives_builtin_capacity_and_trigger():
    """Budget handoff preserves ContextCompressor output-reservation semantics."""
    engine = BudgetAwareEngine()

    assert apply_context_engine_compression_budget(
        _agent(engine, max_tokens=200_000), 1_000_000, reason="model_init"
    )

    assert engine.budgets == [(800_000, 400_000, "model_init")]


def test_budget_aware_engine_matches_model_override_and_absolute_cap():
    """The handoff applies the same override, reservation, and cap ordering."""
    engine = BudgetAwareEngine()

    assert apply_context_engine_compression_budget(
        _agent(
            engine,
            max_tokens=200_000,
            model="vendor/special-model",
            model_thresholds={"model": 0.90},
            threshold_tokens_cap=300_000,
        ),
        1_000_000,
        reason="model_switch",
    )

    # 90% of the reserved 800K input budget is 720K; the host cap wins.
    assert engine.budgets == [(800_000, 300_000, "model_switch")]


def test_host_uses_no_plugin_cap_when_its_cap_is_unset():
    """An opted-in engine cannot replace an unset host cap with its own."""
    engine = BudgetAwareEngine()
    engine.threshold_tokens_cap = 250_000

    assert apply_context_engine_compression_budget(
        _agent(engine, max_tokens=200_000), 1_000_000, reason="model_init"
    )

    assert engine.budgets == [(800_000, 400_000, "model_init")]


def test_legacy_engine_without_hook_keeps_its_policy():
    """Pre-contract engines remain unaffected by the optional extension point."""
    engine = LegacyEngine()

    assert not apply_context_engine_compression_budget(
        _agent(engine, max_tokens=200_000), 1_000_000, reason="model_init"
    )
    assert engine.threshold_tokens == 321_000
    assert engine.threshold_percent == 0.75


def test_budget_hook_must_explicitly_accept_the_handoff():
    """An engine that returns False is not treated as budget-aware."""

    class RejectingEngine(BudgetAwareEngine):
        def set_compression_budget(self, *args, **kwargs) -> bool:
            return False

    engine = RejectingEngine()

    assert not apply_context_engine_compression_budget(
        _agent(engine), 1_000_000, reason="model_init"
    )
    assert engine.threshold_tokens == 321_000
