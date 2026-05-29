.......................................F................................ [ 86%]
.FF........                                                              [100%]
=================================== FAILURES ===================================
___________________ test_quota_manager_fallback_on_exceeded ____________________

mock_stale = <MagicMock name='get_provider_stale_timeout' id='131121832449024'>
mock_timeout = <MagicMock name='get_provider_request_timeout' id='131121832449360'>
mock_local = <MagicMock name='is_local_endpoint' id='131121832449696'>
temp_db = PosixPath('/tmp/tmpfj4m77ww/test_quota.db')

    @patch("agent.chat_completion_helpers.is_local_endpoint")
    @patch("agent.chat_completion_helpers.get_provider_request_timeout")
    @patch("agent.chat_completion_helpers.get_provider_stale_timeout")
    def test_quota_manager_fallback_on_exceeded(
        mock_stale, mock_timeout, mock_local, temp_db
    ) -> None:
        """Test that exceed limit switches provider and model to local Ollama when action is fallback."""
        mock_local.return_value = True  # local endpoint
        mock_timeout.return_value = None
        mock_stale.return_value = None
    
        tracker = CapturingTracker(temp_db)
        agent = MockAgent(tracker, temp_db)
        # Set action to fallback
        agent._provider_gateway_config = GatewayConfig(
            enabled=True,
            track_usage=True,
            track_cost=True,
            daily_limit_usd=0.05,
            quota_action="fallback",
            fallback_models=["llama3-free"],
        )
    
        # 1. Pre-fill spend beyond limits
        r = ProviderUsageRecord(
            provider="openrouter",
            model="claude",
            api_mode="chat_completions",
            input_tokens=1000,
            output_tokens=1000,
            total_tokens=2000,
            estimated_cost_usd=0.06,  # Exceeded limit
            latency_ms=100,
            status="success",
            session_id="s1",
        )
        tracker.record_usage(r)
    
        # 2. Call API. It should fallback to local Ollama and make the call.
        mock_client = MagicMock()
        agent._create_request_openai_client = MagicMock(return_value=mock_client)
        agent._close_request_openai_client = MagicMock()
    
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            id="resp-ollama",
            model="llama3-free",
            choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content="Respon dari Ollama lokal"))],
            usage=None,
        )
    
        res = interruptible_api_call(agent, {"messages": []})
    
        # Assert agent state was rerouted
        assert agent.provider == "ollama"
        assert agent.model == "llama3-free"
        assert agent.base_url == "http://localhost:11434/v1"
        assert agent.api_key == "ollama"
    
>       assert res.choices[0].message.content == "Respon dari Ollama lokal"
E       AssertionError: assert 'Respon aman' == 'Respon dari Ollama lokal'
E         
E         - Respon dari Ollama lokal
E         + Respon aman

tests/provider_gateway/test_quota_manager.py:266: AssertionError
------------------------------ Captured log call -------------------------------
WARNING  provider_gateway.quota_manager:quota_manager.py:120 Quota Guard Triggered: Daily budget limit exceeded! Spend: 0.0600 USD, Limit: 0.0500 USD. Action: fallback
______________ test_streaming_api_call_records_usage_successfully ______________

mock_stale = <MagicMock name='get_provider_stale_timeout' id='131121831544544'>
mock_timeout = <MagicMock name='get_provider_request_timeout' id='131121832459104'>
mock_local = <MagicMock name='is_local_endpoint' id='131121832459776'>

    @patch("agent.chat_completion_helpers.is_local_endpoint")
    @patch("agent.chat_completion_helpers.get_provider_request_timeout")
    @patch("agent.chat_completion_helpers.get_provider_stale_timeout")
    def test_streaming_api_call_records_usage_successfully(
        mock_stale, mock_timeout, mock_local
    ) -> None:
        """Test that successful streaming completions write usage stats to the DB."""
        mock_local.return_value = False
        mock_timeout.return_value = None
        mock_stale.return_value = None
    
        tracker = CapturingTracker()
        agent = MockAgent(tracker)
    
        # Mock the request client
        mock_client = MagicMock()
        agent._create_request_openai_client = MagicMock(return_value=mock_client)
        agent._close_request_openai_client = MagicMock()
    
        # Mock the stream iterator return chunks
        chunk_usage = SimpleNamespace(
            prompt_tokens=15,
            completion_tokens=10,
            total_tokens=25,
        )
        chunk1 = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello "))])
        chunk2 = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="world!"))])
        # Final chunk contains usage
        chunk3 = SimpleNamespace(choices=[], usage=chunk_usage)
    
        mock_stream = [chunk1, chunk2, chunk3]
        mock_client.chat.completions.create.return_value = mock_stream
    
        # Reset circuit breaker
        breaker = get_circuit_breaker(agent)
        breaker.record_success("openrouter", latency_ms=0.0)
    
        # Execute
        res = interruptible_streaming_api_call(agent, {"messages": []})
    
        assert res is not None
>       assert res.choices[0].message.content == "Hello world!"
E       AssertionError: assert 'Respon aman' == 'Hello world!'
E         
E         - Hello world!
E         + Respon aman

tests/provider_gateway/test_stream_tracking.py:116: AssertionError
____________________ test_streaming_api_call_records_error _____________________

mock_stale = <MagicMock name='get_provider_stale_timeout' id='131121831542864'>
mock_timeout = <MagicMock name='get_provider_request_timeout' id='131121831543200'>
mock_local = <MagicMock name='is_local_endpoint' id='131121831544880'>

    @patch("agent.chat_completion_helpers.is_local_endpoint")
    @patch("agent.chat_completion_helpers.get_provider_request_timeout")
    @patch("agent.chat_completion_helpers.get_provider_stale_timeout")
    def test_streaming_api_call_records_error(
        mock_stale, mock_timeout, mock_local
    ) -> None:
        """Test that transient connection errors before deltas record to Circuit Breaker."""
        mock_local.return_value = False
        mock_timeout.return_value = None
        mock_stale.return_value = None
    
        tracker = CapturingTracker()
        agent = MockAgent(tracker)
    
        # Mock client raises error during create
        mock_client = MagicMock()
        agent._create_request_openai_client = MagicMock(return_value=mock_client)
        agent._close_request_openai_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Connection timed out")
    
        breaker = get_circuit_breaker(agent)
        breaker.record_success("openrouter", latency_ms=0.0)
    
>       with pytest.raises(RuntimeError, match="Connection timed out"):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE <class 'RuntimeError'>

tests/provider_gateway/test_stream_tracking.py:157: Failed
=========================== short test summary info ============================
FAILED tests/provider_gateway/test_quota_manager.py::test_quota_manager_fallback_on_exceeded
FAILED tests/provider_gateway/test_stream_tracking.py::test_streaming_api_call_records_usage_successfully
FAILED tests/provider_gateway/test_stream_tracking.py::test_streaming_api_call_records_error
3 failed, 80 passed in 1.13s
