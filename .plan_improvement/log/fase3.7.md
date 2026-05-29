........................................................................ [ 86%]
.F.........                                                              [100%]
=================================== FAILURES ===================================
______________ test_streaming_api_call_records_usage_successfully ______________

mock_stale = <MagicMock name='get_provider_stale_timeout' id='139496649532016'>
mock_timeout = <MagicMock name='get_provider_request_timeout' id='139496649533024'>
mock_local = <MagicMock name='is_local_endpoint' id='139496649532688'>

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
        chunk1 = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello ", tool_calls=None), finish_reason=None)])
        chunk2 = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="world!", tool_calls=None), finish_reason=None)])
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
        assert res.choices[0].message.content == "Hello world!"
        assert res.usage == chunk_usage
    
        # Verify usage was captured by the gateway
        assert len(tracker.records) == 1
        record = tracker.records[0]
        assert record.provider == "openrouter"
        assert record.model == "anthropic/claude-sonnet-4.6"
        assert record.status == "success"
        assert record.total_tokens == 25
        assert record.latency_ms > 0.0
    
        # Verify health updated
        health = breaker.get_health("openrouter")
>       assert health.total_requests == 1
E       assert 2 == 1
E        +  where 2 = <provider_gateway.circuit_breaker.ProviderHealth object at 0x7edf182afee0>.total_requests

tests/provider_gateway/test_stream_tracking.py:142: AssertionError
=========================== short test summary info ============================
FAILED tests/provider_gateway/test_stream_tracking.py::test_streaming_api_call_records_usage_successfully
1 failed, 82 passed in 1.56s
