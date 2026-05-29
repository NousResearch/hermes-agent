(hermes-agent) void@Pongo:~/lab/git/hermes_agent$ uv run --extra dev python -m pytest tests/provider_gateway -q
      Built hermes-agent @ file:///home/void/lab/git/hermes_agent
Uninstalled 1 package in 0.88ms
Installed 1 package in 1ms
...................FFF....FFF...............................F......... [ 90%]
.......                                                                [100%]
================================== FAILURES ==================================
____________ test_fallback_standard_linear_when_gateway_disabled _____________

mock_resolve = <MagicMock name='resolve_provider_client' id='135758701416128'>

    @patch("agent.auxiliary_client.resolve_provider_client")
    def test_fallback_standard_linear_when_gateway_disabled(mock_resolve) -> None:
        """Test that try_activate_fallback remains linear when gateway is disabled."""
        agent = MockAgent()
        agent._provider_gateway_config = GatewayConfig(enabled=False)
    
        fake_client = MagicMock()
        fake_client.api_key = "key-or"
        fake_client.base_url = "https://openrouter.ai/api/v1"
        mock_resolve.return_value = (fake_client, "gpt-4o")
    
        # Call fallback
>       success = try_activate_fallback(agent, reason=FailoverReason.rate_limit)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/provider_gateway/test_integration.py:66: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

agent = <test_integration.MockAgent object at 0x7b78c92c4440>
reason = <FailoverReason.rate_limit: 'rate_limit'>

    def try_activate_fallback(agent, reason: "FailoverReason | None" = None) -> bool:
        """Switch to the next fallback model/provider in the chain.
    
        Called when the current model is failing after retries.  Swaps the
        OpenAI client, model slug, and provider in-place so the retry loop
        can continue with the new backend.  Advances through the chain on
        each call; returns False when exhausted.
    
        Uses the centralized provider router (resolve_provider_client) for
        auth resolution and client construction — no duplicated provider→key
        mappings.
        """
        if reason in {FailoverReason.rate_limit, FailoverReason.billing}:
            # Only start cooldown when leaving the primary provider.  If we're
            # already on a fallback and chain-switching, the primary wasn't the
            # source of the 429 so the cooldown should not be reset/extended.
            fallback_already_active = bool(getattr(agent, "_fallback_activated", False))
            current_provider = (getattr(agent, "provider", "") or "").strip().lower()
>           primary_provider = ((agent._primary_runtime or {}).get("provider") or "").strip().lower()
                                 ^^^^^^^^^^^^^^^^^^^^^^
E           AttributeError: 'MockAgent' object has no attribute '_primary_runtime'

agent/chat_completion_helpers.py:1066: AttributeError
_____________ test_fallback_dynamic_routing_when_gateway_enabled _____________

mock_resolve = <MagicMock name='resolve_provider_client' id='135758701425872'>

    @patch("agent.auxiliary_client.resolve_provider_client")
    def test_fallback_dynamic_routing_when_gateway_enabled(mock_resolve) -> None:
        """Test that try_activate_fallback routes intelligently when gateway is enabled."""
        agent = MockAgent()
        agent._provider_gateway_config = GatewayConfig(
            enabled=True,
            routing_strategy="lowest-cost",
        )
    
        # Let's seed Circuit Breaker with latencies
        breaker = get_circuit_breaker(agent)
        breaker.record_success("openrouter", latency_ms=400.0)
        breaker.record_success("ollama", latency_ms=50.0)
    
        fake_client = MagicMock()
        fake_client.api_key = "key-ollama"
        fake_client.base_url = "http://localhost:11434"
        mock_resolve.return_value = (fake_client, "llama3.2")
    
        # With lowest-cost (or round-robin skipping to free Ollama), dynamic routing should prefer Ollama!
>       success = try_activate_fallback(agent, reason=FailoverReason.rate_limit)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/provider_gateway/test_integration.py:96: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

agent = <test_integration.MockAgent object at 0x7b78c90a4410>
reason = <FailoverReason.rate_limit: 'rate_limit'>

    def try_activate_fallback(agent, reason: "FailoverReason | None" = None) -> bool:
        """Switch to the next fallback model/provider in the chain.
    
        Called when the current model is failing after retries.  Swaps the
        OpenAI client, model slug, and provider in-place so the retry loop
        can continue with the new backend.  Advances through the chain on
        each call; returns False when exhausted.
    
        Uses the centralized provider router (resolve_provider_client) for
        auth resolution and client construction — no duplicated provider→key
        mappings.
        """
        if reason in {FailoverReason.rate_limit, FailoverReason.billing}:
            # Only start cooldown when leaving the primary provider.  If we're
            # already on a fallback and chain-switching, the primary wasn't the
            # source of the 429 so the cooldown should not be reset/extended.
            fallback_already_active = bool(getattr(agent, "_fallback_activated", False))
            current_provider = (getattr(agent, "provider", "") or "").strip().lower()
>           primary_provider = ((agent._primary_runtime or {}).get("provider") or "").strip().lower()
                                 ^^^^^^^^^^^^^^^^^^^^^^
E           AttributeError: 'MockAgent' object has no attribute '_primary_runtime'

agent/chat_completion_helpers.py:1066: AttributeError
_________________ test_fallback_skips_circuit_open_provider __________________

mock_resolve = <MagicMock name='resolve_provider_client' id='135758699481136'>

    @patch("agent.auxiliary_client.resolve_provider_client")
    def test_fallback_skips_circuit_open_provider(mock_resolve) -> None:
        """Test that fallback skips a provider if its circuit breaker is OPEN."""
        agent = MockAgent()
        agent._provider_gateway_config = GatewayConfig(
            enabled=True,
            routing_strategy="round-robin",
        )
    
        # Trip circuit for openrouter (the first fallback candidate)
        breaker = get_circuit_breaker(agent)
        for _ in range(5):
            breaker.record_failure("openrouter")
        assert breaker.is_available("openrouter") is False
    
        fake_client = MagicMock()
        fake_client.api_key = "key-ollama"
        fake_client.base_url = "http://localhost:11434"
        mock_resolve.return_value = (fake_client, "llama3.2")
    
        # Try fallback. Since openrouter is circuit-open, it should skip it and select ollama!
>       success = try_activate_fallback(agent, reason=FailoverReason.rate_limit)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/provider_gateway/test_integration.py:127: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

agent = <test_integration.MockAgent object at 0x7b78c90a4f50>
reason = <FailoverReason.rate_limit: 'rate_limit'>

    def try_activate_fallback(agent, reason: "FailoverReason | None" = None) -> bool:
        """Switch to the next fallback model/provider in the chain.
    
        Called when the current model is failing after retries.  Swaps the
        OpenAI client, model slug, and provider in-place so the retry loop
        can continue with the new backend.  Advances through the chain on
        each call; returns False when exhausted.
    
        Uses the centralized provider router (resolve_provider_client) for
        auth resolution and client construction — no duplicated provider→key
        mappings.
        """
        if reason in {FailoverReason.rate_limit, FailoverReason.billing}:
            # Only start cooldown when leaving the primary provider.  If we're
            # already on a fallback and chain-switching, the primary wasn't the
            # source of the 429 so the cooldown should not be reset/extended.
            fallback_already_active = bool(getattr(agent, "_fallback_activated", False))
            current_provider = (getattr(agent, "provider", "") or "").strip().lower()
>           primary_provider = ((agent._primary_runtime or {}).get("provider") or "").strip().lower()
                                 ^^^^^^^^^^^^^^^^^^^^^^
E           AttributeError: 'MockAgent' object has no attribute '_primary_runtime'

agent/chat_completion_helpers.py:1066: AttributeError
----------------------------- Captured log call ------------------------------
WARNING  provider_gateway.circuit_breaker:circuit_breaker.py:140 Circuit breaker for provider openrouter TRIPPED to OPEN after 5 consecutive failures
___________________ test_complete_calls_litellm_completion ___________________

    @patch("provider_gateway.litellm_backend._LITELLM_AVAILABLE", True)
    @patch("sys.modules", {"litellm": MagicMock()})
    def test_complete_calls_litellm_completion() -> None:
        """Test that complete correctly delegates parameters to litellm.completion."""
        import litellm
    
        mock_resp = MagicMock()
        litellm.completion = MagicMock(return_value=mock_resp)
    
        messages = [{"role": "user", "content": "hi"}]
>       res = backend.complete(
            "openai/gpt-4o",
            messages,
            api_key="my-key",
            api_base="https://my-base.com",
            temperature=0.5,
        )

tests/provider_gateway/test_litellm.py:47: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

model = 'openai/gpt-4o', messages = [{'content': 'hi', 'role': 'user'}]
api_key = 'my-key', api_base = 'https://my-base.com', stream = False
kwargs = {'temperature': 0.5}

    def complete(
        model: str,
        messages: list[dict[str, Any]],
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        stream: bool = False,
        **kwargs,
    ) -> Any:
        """Call a model using LiteLLM completion API.
    
        Requires 'litellm' to be installed.
        """
        if not _LITELLM_AVAILABLE:
>           raise ImportError(
                "The 'litellm' package is not installed. "
                "Please install it using: pip install hermes-agent[gateway]"
            )
E           ImportError: The 'litellm' package is not installed. Please install it using: pip install hermes-agent[gateway]

provider_gateway/litellm_backend.py:45: ImportError
__________________ test_estimate_cost_calculates_correctly ___________________

    @patch("provider_gateway.litellm_backend._LITELLM_AVAILABLE", True)
    @patch("sys.modules", {"litellm": MagicMock()})
    def test_estimate_cost_calculates_correctly() -> None:
        """Test that estimate_cost returns float cost when litellm is available."""
        import litellm
    
        litellm.completion_cost = MagicMock(return_value=0.015)
    
        res = backend.estimate_cost("openai/gpt-4o", 1000, 500)
>       assert res == 0.015
E       assert 0.0 == 0.015

tests/provider_gateway/test_litellm.py:75: AssertionError
___________________ test_list_models_returns_correct_list ____________________

    @patch("provider_gateway.litellm_backend._LITELLM_AVAILABLE", True)
    @patch("sys.modules", {"litellm": MagicMock()})
    def test_list_models_returns_correct_list() -> None:
        """Test that list_models returns mock models when litellm is available."""
        import litellm
    
        litellm.model_list = ["openai/gpt-4o", "anthropic/claude-3"]
    
        res = backend.list_models()
>       assert res == ["openai/gpt-4o", "anthropic/claude-3"]
E       AssertionError: assert [] == ['openai/gpt-...pic/claude-3']
E         
E         Right contains 2 more items, first extra item: 'openai/gpt-4o'
E         Use -v to get more diff

tests/provider_gateway/test_litellm.py:92: AssertionError
____________________ test_runtime_updates_circuit_breaker ____________________

    def test_runtime_updates_circuit_breaker() -> None:
        """Test that runtime success/error calls update the circuit breaker health."""
        from provider_gateway.runtime import get_circuit_breaker
        tracker = CapturingTracker()
        agent = SimpleNamespace(
            _provider_gateway_config=GatewayConfig(
                enabled=True,
                track_usage=True,
                track_cost=False,
            ),
            _provider_usage_tracker=tracker,
            provider="test-circuit-provider",
            model="claude-sonnet",
            api_mode="chat_completions",
            session_id="session-1",
        )
    
        # Initially not tracked or CLOSED
        breaker = get_circuit_breaker(agent)
        assert breaker.is_available("test-circuit-provider") is True
    
        # Record success
        record_provider_response_usage(
            agent,
            SimpleNamespace(usage=_response_usage()),
            latency_seconds=0.15,
        )
        health = breaker.get_health("test-circuit-provider")
        assert health is not None
>       assert health.state == CircuitState.CLOSED
                               ^^^^^^^^^^^^
E       NameError: name 'CircuitState' is not defined

tests/provider_gateway/test_runtime.py:471: NameError
========================== short test summary info ===========================
FAILED tests/provider_gateway/test_integration.py::test_fallback_standard_linear_when_gateway_disabled - AttributeError: 'MockAgent' object has no attribute '_primary_runtime'
FAILED tests/provider_gateway/test_integration.py::test_fallback_dynamic_routing_when_gateway_enabled - AttributeError: 'MockAgent' object has no attribute '_primary_runtime'
FAILED tests/provider_gateway/test_integration.py::test_fallback_skips_circuit_open_provider - AttributeError: 'MockAgent' object has no attribute '_primary_runtime'
FAILED tests/provider_gateway/test_litellm.py::test_complete_calls_litellm_completion - ImportError: The 'litellm' package is not installed. Please install it us...
FAILED tests/provider_gateway/test_litellm.py::test_estimate_cost_calculates_correctly - assert 0.0 == 0.015
FAILED tests/provider_gateway/test_litellm.py::test_list_models_returns_correct_list - AssertionError: assert [] == ['openai/gpt-...pic/claude-3']
FAILED tests/provider_gateway/test_runtime.py::test_runtime_updates_circuit_breaker - NameError: name 'CircuitState' is not defined
7 failed, 70 passed in 0.84s