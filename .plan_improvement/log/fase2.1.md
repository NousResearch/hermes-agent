(hermes-agent) void@Pongo:~/lab/git/hermes_agent$ uv run --extra dev python -m pytest tests/provider_gateway -q
...................F.................................................. [ 95%]
...                                                                    [100%]
================================== FAILURES ==================================
____________ test_fallback_standard_linear_when_gateway_disabled _____________

mock_resolve = <MagicMock name='resolve_provider_client' id='134712995727072'>

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
        success = try_activate_fallback(agent, reason=FailoverReason.rate_limit)
    
        assert success is True
        assert agent._fallback_index == 1
        # Standard linear selection selected the first item (openrouter/gpt-4o)
        assert agent.provider == "openrouter"
>       assert agent.model == "gpt-4o"
E       AssertionError: assert 'openai/gpt-4o' == 'gpt-4o'
E         
E         - gpt-4o
E         + openai/gpt-4o

tests/provider_gateway/test_integration.py:73: AssertionError
========================== short test summary info ===========================
FAILED tests/provider_gateway/test_integration.py::test_fallback_standard_linear_when_gateway_disabled - AssertionError: assert 'openai/gpt-4o' == 'gpt-4o'
1 failed, 72 passed in 0.91s