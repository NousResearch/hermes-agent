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
    compression_max_tokens: int | None = None,
):
    return SimpleNamespace(
        context_compressor=engine,
        max_tokens=max_tokens,
        model=model,
        _compression_threshold_percent=0.50,
        _compression_model_thresholds=model_thresholds or {},
        _compression_threshold_tokens_cap=threshold_tokens_cap,
        _compression_max_tokens=compression_max_tokens,
    )


def test_budget_aware_engine_receives_builtin_capacity_and_trigger():
    """Budget handoff preserves ContextCompressor output-reservation semantics."""
    engine = BudgetAwareEngine()

    assert apply_context_engine_compression_budget(
        _agent(engine, max_tokens=200_000), 1_000_000, reason="model_init"
    )

    assert engine.budgets == [(800_000, 400_000, "model_init")]


def test_provider_transitions_recompute_the_unset_native_gemini_reserve():
    """Budget handoffs follow the active provider instead of an init-time cache."""
    engine = BudgetAwareEngine()
    agent = _agent(engine)
    agent.provider = "openai"
    agent.base_url = "https://api.openai.com/v1"

    assert apply_context_engine_compression_budget(agent, 1_000_000, reason="model_switch")
    agent.provider = "gemini"
    agent.base_url = "https://generativelanguage.googleapis.com/v1beta"
    assert apply_context_engine_compression_budget(agent, 1_000_000, reason="model_switch")
    agent.provider = "openai"
    agent.base_url = "https://api.openai.com/v1"
    assert apply_context_engine_compression_budget(agent, 1_000_000, reason="model_switch")

    assert engine.budgets == [
        (1_000_000, 500_000, "model_switch"),
        (934_465, 467_232, "model_switch"),
        (1_000_000, 500_000, "model_switch"),
    ]

    from agent.agent_init import _compressor_max_tokens

    reverse_engine = BudgetAwareEngine()
    reverse_agent = _agent(reverse_engine)
    reverse_agent.provider = "gemini"
    reverse_agent.base_url = "https://generativelanguage.googleapis.com/v1beta"
    reverse_agent._compression_max_tokens = _compressor_max_tokens(reverse_agent)
    reverse_agent.provider = "openai"
    reverse_agent.base_url = "https://api.openai.com/v1"

    assert apply_context_engine_compression_budget(reverse_agent, 1_000_000, reason="model_switch")
    assert reverse_engine.budgets == [(1_000_000, 500_000, "model_switch")]


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


def test_ollama_post_init_clamp_resyncs_opted_in_engine():
    """Ollama's served num_ctx replaces the initial model-window budget."""
    from agent.agent_init import _configure_ollama_num_ctx

    engine = BudgetAwareEngine()
    engine.context_length = 131_072
    engine.update_model = lambda model, context_length, **kwargs: setattr(engine, "context_length", context_length)
    agent = _agent(engine)
    agent.base_url = "http://127.0.0.1:11434/v1"
    agent.api_key = ""
    agent.provider = "ollama"
    agent.api_mode = "chat_completions"
    agent.quiet_mode = True

    _configure_ollama_num_ctx(agent, {"ollama_num_ctx": 65_536}, None)

    assert engine.budgets == [(65_536, 55_705, "ollama_num_ctx")]


def test_local_window_growth_resyncs_opted_in_engine():
    """A managed local expansion immediately gives the engine the larger budget."""
    from agent.turn_context_compaction import _apply_grown_window

    engine = BudgetAwareEngine()
    engine.update_model = lambda model, context_length, **kwargs: setattr(engine, "context_length", context_length)
    agent = _agent(engine)
    agent.base_url = "http://127.0.0.1:8080/v1"
    agent.api_key = ""
    agent.provider = "llamacpp"
    agent.api_mode = "chat_completions"
    agent._buffer_status = lambda message: None

    _apply_grown_window(agent, engine, 131_072)

    assert engine.budgets == [(131_072, 98_304, "local_window_growth")]


def test_long_context_recovery_advertises_only_the_effective_capacity():
    """Tier recovery hands off the clamped compressor window, never a presumed one."""
    from agent.turn_recovery import _cap_long_context_tier

    engine = BudgetAwareEngine()
    engine.context_length = 300_000
    engine.update_model = lambda model, context_length, **kwargs: setattr(engine, "context_length", context_length)
    agent = _agent(engine)
    agent.base_url = ""
    agent.api_key = ""
    agent.provider = "anthropic"
    agent.api_mode = "anthropic_messages"
    agent._buffer_vprint = lambda message: None

    _cap_long_context_tier(agent)

    assert engine.budgets == [(200_000, 150_000, "long_context_tier")]
