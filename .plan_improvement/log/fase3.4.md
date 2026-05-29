........................................................................ [ 86%]
.FF........                                                              [100%]
=================================== FAILURES ===================================
______________ test_streaming_api_call_records_usage_successfully ______________

mock_stale = <MagicMock name='get_provider_stale_timeout' id='138388958487488'>
mock_timeout = <MagicMock name='get_provider_request_timeout' id='138388958488496'>
mock_local = <MagicMock name='is_local_endpoint' id='138388958488160'>

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
>       agent = MockAgent(tracker)
                ^^^^^^^^^^^^^^^^^^

tests/provider_gateway/test_stream_tracking.py:96: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <test_stream_tracking.MockAgent object at 0x7ddd30b62e40>
tracker = <test_stream_tracking.CapturingTracker object at 0x7ddd30b62cf0>
temp_db_path = None

    def __init__(self, tracker: CapturingTracker, temp_db_path: Path | None = None) -> None:
        self.provider = "openrouter"
        self.model = "anthropic/claude-sonnet-4.6"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_mode = "chat_completions"
        self.api_key = "key-123"
        self.session_id = "session-stream-test"
        self._provider_gateway_config = GatewayConfig(
            enabled=True,
            track_usage=True,
            track_cost=False,
        )
        self._provider_usage_tracker = tracker
        self._interrupt_requested = False
        self._disable_streaming = False
        self._ollama_num_ctx = None
        self._rate_limited_until = 0.0
    
        # Gunakan database cache sementara agar terisolasi dari database global
        from provider_gateway.semantic_cache import SemanticCache
        if temp_db_path is not None:
            self._provider_semantic_cache = SemanticCache(db_path=temp_db_path)
        else:
            import tempfile
            self._temp_dir = tempfile.TemporaryDirectory()
>           self._provider_semantic_cache = SemanticCache(db_path=Path(self._temp_dir.name) / "stream_mock_cache.db")
                                                                  ^^^^
E           NameError: name 'Path' is not defined

tests/provider_gateway/test_stream_tracking.py:48: NameError
____________________ test_streaming_api_call_records_error _____________________

mock_stale = <MagicMock name='get_provider_stale_timeout' id='138388958493536'>
mock_timeout = <MagicMock name='get_provider_request_timeout' id='138388957397744'>
mock_local = <MagicMock name='is_local_endpoint' id='138388957398080'>

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
>       agent = MockAgent(tracker)
                ^^^^^^^^^^^^^^^^^^

tests/provider_gateway/test_stream_tracking.py:155: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <test_stream_tracking.MockAgent object at 0x7ddd32cdae90>
tracker = <test_stream_tracking.CapturingTracker object at 0x7ddd32cdb110>
temp_db_path = None

    def __init__(self, tracker: CapturingTracker, temp_db_path: Path | None = None) -> None:
        self.provider = "openrouter"
        self.model = "anthropic/claude-sonnet-4.6"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_mode = "chat_completions"
        self.api_key = "key-123"
        self.session_id = "session-stream-test"
        self._provider_gateway_config = GatewayConfig(
            enabled=True,
            track_usage=True,
            track_cost=False,
        )
        self._provider_usage_tracker = tracker
        self._interrupt_requested = False
        self._disable_streaming = False
        self._ollama_num_ctx = None
        self._rate_limited_until = 0.0
    
        # Gunakan database cache sementara agar terisolasi dari database global
        from provider_gateway.semantic_cache import SemanticCache
        if temp_db_path is not None:
            self._provider_semantic_cache = SemanticCache(db_path=temp_db_path)
        else:
            import tempfile
            self._temp_dir = tempfile.TemporaryDirectory()
>           self._provider_semantic_cache = SemanticCache(db_path=Path(self._temp_dir.name) / "stream_mock_cache.db")
                                                                  ^^^^
E           NameError: name 'Path' is not defined

tests/provider_gateway/test_stream_tracking.py:48: NameError
=========================== short test summary info ============================
FAILED tests/provider_gateway/test_stream_tracking.py::test_streaming_api_call_records_usage_successfully
FAILED tests/provider_gateway/test_stream_tracking.py::test_streaming_api_call_records_error
2 failed, 81 passed in 1.30s
