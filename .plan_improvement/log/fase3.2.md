......................................FF........FF.............FF....... [ 86%]
.FF........                                                              [100%]
=================================== FAILURES ===================================
____________________ test_quota_manager_blocks_on_exceeded _____________________

mock_stale = <MagicMock name='get_provider_stale_timeout' id='136461282643648'>
mock_timeout = <MagicMock name='get_provider_request_timeout' id='136461282643312'>
mock_local = <MagicMock name='is_local_endpoint' id='136461282640288'>
temp_db = PosixPath('/tmp/tmpau68y5_4/test_quota.db')

    @patch("agent.chat_completion_helpers.is_local_endpoint")
    @patch("agent.chat_completion_helpers.get_provider_request_timeout")
    @patch("agent.chat_completion_helpers.get_provider_stale_timeout")
    def test_quota_manager_blocks_on_exceeded(
        mock_stale, mock_timeout, mock_local, temp_db
    ) -> None:
        """Test that a request is blocked and raises QuotaExceededError when limits are exceeded."""
        mock_local.return_value = False
        mock_timeout.return_value = None
        mock_stale.return_value = None
    
        tracker = CapturingTracker(temp_db)
        agent = MockAgent(tracker, temp_db)
    
        # 1. Pre-fill spend database beyond daily_limit (daily_limit is 0.05)
        r = ProviderUsageRecord(
            provider="openrouter",
            model="claude",
            api_mode="chat_completions",
            input_tokens=1000,
            output_tokens=1000,
            total_tokens=2000,
            estimated_cost_usd=0.06,  # Exceeded daily limit!
            latency_ms=100,
            status="success",
            session_id="s1",
        )
        tracker.record_usage(r)
    
        # 2. Try calling API. It should be blocked.
        mock_client = MagicMock()
        agent._create_request_openai_client = MagicMock(return_value=mock_client)
    
>       with pytest.raises(QuotaExceededError, match="Daily budget limit exceeded"):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE <class 'provider_gateway.quota_manager.QuotaExceededError'>

tests/provider_gateway/test_quota_manager.py:201: Failed
___________________ test_quota_manager_fallback_on_exceeded ____________________

mock_stale = <MagicMock name='get_provider_stale_timeout' id='136461282644656'>
mock_timeout = <MagicMock name='get_provider_request_timeout' id='136461282646672'>
mock_local = <MagicMock name='is_local_endpoint' id='136461282647008'>
temp_db = PosixPath('/tmp/tmp2dsnej9o/test_quota.db')

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
>       assert agent.provider == "ollama"
E       AssertionError: assert 'openrouter' == 'ollama'
E         
E         - ollama
E         + openrouter

tests/provider_gateway/test_quota_manager.py:261: AssertionError
_______ test_interruptible_api_call_records_success_when_gateway_enabled _______

    def test_interruptible_api_call_records_success_when_gateway_enabled() -> None:
        tracker = CapturingTracker()
        response = SimpleNamespace(usage=_response_usage(), choices=[SimpleNamespace()])
        agent = FakeAgent(response, tracker)
    
        result = interruptible_api_call(agent, {"model": agent.model, "messages": []})
    
>       assert result is response
E       AssertionError: assert namespace(id='cache-5e7e007f-9396-40c9-93e3-56322593800c', model='anthropic/claude-sonnet-4.6', choices=[namespace(ind...(role='assistant', content='Respon aman', tool_calls=None, reasoning_content=None), finish_reason='stop')], usage=None) is namespace(usage=namespace(prompt_tokens=13, completion_tokens=7, total_tokens=20, prompt_tokens_details=namespace(cached_tokens=3)), choices=[namespace()])

tests/provider_gateway/test_runtime.py:191: AssertionError
________ test_interruptible_api_call_records_error_when_gateway_enabled ________

    def test_interruptible_api_call_records_error_when_gateway_enabled() -> None:
        tracker = CapturingTracker()
        agent = FakeAgent(ValueError("provider exploded"), tracker)
    
>       with pytest.raises(ValueError, match="provider exploded"):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE <class 'ValueError'>

tests/provider_gateway/test_runtime.py:201: Failed
____________________ test_semantic_cache_disabled_by_config ____________________

mock_stale = <MagicMock name='get_provider_stale_timeout' id='136461282654064'>
mock_timeout = <MagicMock name='get_provider_request_timeout' id='136461282654400'>
mock_local = <MagicMock name='is_local_endpoint' id='136461281870912'>
temp_db = PosixPath('/tmp/tmp9y_9idcr/test_cache.db')

    @patch("agent.chat_completion_helpers.is_local_endpoint")
    @patch("agent.chat_completion_helpers.get_provider_request_timeout")
    @patch("agent.chat_completion_helpers.get_provider_stale_timeout")
    def test_semantic_cache_disabled_by_config(
        mock_stale, mock_timeout, mock_local, temp_db
    ) -> None:
        """Test that cache is bypassed entirely when the gateway config is disabled."""
        mock_local.return_value = False
        mock_timeout.return_value = None
        mock_stale.return_value = None
    
        tracker = CapturingTracker()
        agent = MockAgent(tracker, temp_db)
>       agent._provider_gateway_config.enabled = False  # Disable gateway!
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/provider_gateway/test_semantic_cache.py:172: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = GatewayConfig(enabled=True, backend='native', track_usage=True, track_cost=False, routing_strategy='round-robin', fallback_models=[], daily_limit_usd=None, monthly_limit_usd=None, quota_action='block')
name = 'enabled', value = False

>   ???
E   dataclasses.FrozenInstanceError: cannot assign to field 'enabled'

<string>:23: FrozenInstanceError
______________________ test_semantic_cache_streaming_hit _______________________

mock_stale = <MagicMock name='get_provider_stale_timeout' id='136461282655408'>
mock_timeout = <MagicMock name='get_provider_request_timeout' id='136461282655744'>
mock_local = <MagicMock name='is_local_endpoint' id='136461282654736'>
temp_db = PosixPath('/tmp/tmpe880ltvr/test_cache.db')

    @patch("agent.chat_completion_helpers.is_local_endpoint")
    @patch("agent.chat_completion_helpers.get_provider_request_timeout")
    @patch("agent.chat_completion_helpers.get_provider_stale_timeout")
    def test_semantic_cache_streaming_hit(
        mock_stale, mock_timeout, mock_local, temp_db
    ) -> None:
        """Test that a cache hit on streaming triggers on_first_delta, fires stream deltas, and returns mock response."""
        mock_local.return_value = False
        mock_timeout.return_value = None
        mock_stale.return_value = None
    
        tracker = CapturingTracker()
        agent = MockAgent(tracker, temp_db)
    
        # Directly store a response in cache first
        messages = [{"role": "user", "content": "Bagaimana cuaca?"}]
        cache = agent._provider_semantic_cache
        cache.set_cached_response(agent, messages, "Cuaca sangat cerah.")
    
        # Mock first delta callback
        first_delta_called = {"yes": False}
        def on_first_delta():
            first_delta_called["yes"] = True
    
        # Call streaming API call
>       res = interruptible_streaming_api_call(agent, {"messages": messages}, on_first_delta=on_first_delta)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/provider_gateway/test_semantic_cache.py:221: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
agent/chat_completion_helpers.py:2612: in interruptible_streaming_api_call
    raise result["error"]
agent/chat_completion_helpers.py:2219: in _call
    result["response"] = _call_chat_completions()
                         ^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    def _call_chat_completions():
        """Stream a chat completions response."""
        import httpx as _httpx
        # Per-provider / per-model request_timeout_seconds (from config.yaml)
        # wins over the HERMES_API_TIMEOUT env default if the user set it.
        _provider_timeout_cfg = get_provider_request_timeout(agent.provider, agent.model)
        _base_timeout = (
            _provider_timeout_cfg
            if _provider_timeout_cfg is not None
            else float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
        )
        # Read timeout: config wins here too.  Otherwise use
        # HERMES_STREAM_READ_TIMEOUT (default 120s) for cloud providers.
        if _provider_timeout_cfg is not None:
            _stream_read_timeout = _provider_timeout_cfg
        else:
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
            # Local providers (Ollama, llama.cpp, vLLM) can take minutes for
            # prefill on large contexts before producing the first token.
            # Auto-increase the httpx read timeout unless the user explicitly
            # overrode HERMES_STREAM_READ_TIMEOUT.
            if _stream_read_timeout == 120.0 and agent.base_url and is_local_endpoint(agent.base_url):
                _stream_read_timeout = _base_timeout
                logger.debug(
                    "Local provider detected (%s) — stream read timeout raised to %.0fs",
                    agent.base_url, _stream_read_timeout,
                )
        # Cap connect/pool at 60s even when provider timeout is higher.
        # connect/pool cover TCP handshake, not model inference.
        _conn_cap = min(_base_timeout, 60.0) if _provider_timeout_cfg is not None else 30.0
        stream_kwargs = {
            **api_kwargs,
            "stream": True,
            "stream_options": {"include_usage": True},
            "timeout": _httpx.Timeout(
                connect=_conn_cap,
                read=_stream_read_timeout,
                write=_base_timeout,
                pool=_conn_cap,
            ),
        }
        request_client = _set_request_client(
>           agent._create_request_openai_client(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                reason="chat_completion_stream_request",
                api_kwargs=stream_kwargs,
            )
        )
E       AttributeError: 'MockAgent' object has no attribute '_create_request_openai_client'

agent/chat_completion_helpers.py:1860: AttributeError
______________ test_streaming_api_call_records_usage_successfully ______________

mock_stale = <MagicMock name='get_provider_stale_timeout' id='136461262789936'>
mock_timeout = <MagicMock name='get_provider_request_timeout' id='136461262789264'>
mock_local = <MagicMock name='is_local_endpoint' id='136461262789600'>

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

mock_stale = <MagicMock name='get_provider_stale_timeout' id='136461262792960'>
mock_timeout = <MagicMock name='get_provider_request_timeout' id='136461262792624'>
mock_local = <MagicMock name='is_local_endpoint' id='136461262793296'>

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
FAILED tests/provider_gateway/test_quota_manager.py::test_quota_manager_blocks_on_exceeded
FAILED tests/provider_gateway/test_quota_manager.py::test_quota_manager_fallback_on_exceeded
FAILED tests/provider_gateway/test_runtime.py::test_interruptible_api_call_records_success_when_gateway_enabled
FAILED tests/provider_gateway/test_runtime.py::test_interruptible_api_call_records_error_when_gateway_enabled
FAILED tests/provider_gateway/test_semantic_cache.py::test_semantic_cache_disabled_by_config
FAILED tests/provider_gateway/test_semantic_cache.py::test_semantic_cache_streaming_hit
FAILED tests/provider_gateway/test_stream_tracking.py::test_streaming_api_call_records_usage_successfully
FAILED tests/provider_gateway/test_stream_tracking.py::test_streaming_api_call_records_error
8 failed, 75 passed in 1.44s
