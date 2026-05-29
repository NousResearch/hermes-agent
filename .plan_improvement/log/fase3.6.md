........................................................................ [ 86%]
.FF........                                                              [100%]
=================================== FAILURES ===================================
______________ test_streaming_api_call_records_usage_successfully ______________

mock_stale = <MagicMock name='get_provider_stale_timeout' id='128287265497712'>
mock_timeout = <MagicMock name='get_provider_request_timeout' id='128287265498720'>
mock_local = <MagicMock name='is_local_endpoint' id='128287265498384'>

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
        chunk1 = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello ", tool_calls=None))])
        chunk2 = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="world!", tool_calls=None))])
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
E       AssertionError: assert None == 'Hello world!'
E        +  where None = namespace(role='assistant', content=None, tool_calls=None, reasoning_content=None).content
E        +    where namespace(role='assistant', content=None, tool_calls=None, reasoning_content=None) = namespace(index=0, message=namespace(role='assistant', content=None, tool_calls=None, reasoning_content=None), finish_reason='length').message

tests/provider_gateway/test_stream_tracking.py:128: AssertionError
------------------------------ Captured log call -------------------------------
WARNING  agent.chat_completion_helpers:chat_completion_helpers.py:2285 Streaming failed after partial delivery, not retrying: 'types.SimpleNamespace' object has no attribute 'finish_reason'
WARNING  agent.chat_completion_helpers:chat_completion_helpers.py:2576 Partial stream delivered before error; returning length-truncated stub with 0 chars of recovered content so the loop can continue from where the stream died: 'types.SimpleNamespace' object has no attribute 'finish_reason'
____________________ test_streaming_api_call_records_error _____________________

mock_stale = <MagicMock name='get_provider_stale_timeout' id='128287181201360'>
mock_timeout = <MagicMock name='get_provider_request_timeout' id='128287181203376'>
mock_local = <MagicMock name='is_local_endpoint' id='128287181203712'>

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
    
        with pytest.raises(RuntimeError, match="Connection timed out"):
            interruptible_streaming_api_call(agent, {"messages": []})
    
        # Verify error was logged
        assert len(tracker.records) == 1
        record = tracker.records[0]
        assert record.provider == "openrouter"
        assert record.status == "error"
        assert record.error_type == "RuntimeError"
    
        # Verify Circuit Breaker was updated
        health = breaker.get_health("openrouter")
>       assert health.total_requests == 1
E       assert 2 == 1
E        +  where 2 = <provider_gateway.circuit_breaker.ProviderHealth object at 0x74ad2fca22d0>.total_requests

tests/provider_gateway/test_stream_tracking.py:181: AssertionError
=========================== short test summary info ============================
FAILED tests/provider_gateway/test_stream_tracking.py::test_streaming_api_call_records_usage_successfully
FAILED tests/provider_gateway/test_stream_tracking.py::test_streaming_api_call_records_error
2 failed, 81 passed in 1.69s
