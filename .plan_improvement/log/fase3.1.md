.....................................FFF......................FFF....... [ 86%]
.FF........                                                              [100%]
=================================== FAILURES ===================================
_______________________ test_quota_manager_within_limits _______________________

mock_stale = <MagicMock name='get_provider_stale_timeout' id='123230871490576'>
mock_timeout = <MagicMock name='get_provider_request_timeout' id='123230871493936'>
mock_local = <MagicMock name='is_local_endpoint' id='123230871495952'>
temp_db = PosixPath('/tmp/tmpxgygcldq/test_quota.db')

    @patch("agent.chat_completion_helpers.is_local_endpoint")
    @patch("agent.chat_completion_helpers.get_provider_request_timeout")
    @patch("agent.chat_completion_helpers.get_provider_stale_timeout")
    def test_quota_manager_within_limits(
        mock_stale, mock_timeout, mock_local, temp_db
    ) -> None:
        """Test that a request goes through fine when spend is within limits."""
        mock_local.return_value = False
        mock_timeout.return_value = None
        mock_stale.return_value = None
    
        tracker = CapturingTracker(temp_db)
        agent = MockAgent(tracker, temp_db)
    
        mock_client = MagicMock()
        agent._create_request_openai_client = MagicMock(return_value=mock_client)
        agent._close_request_openai_client = MagicMock()
    
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            id="resp-ok",
            model="anthropic/claude-sonnet-4.6",
            choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content="Respon aman"))],
            usage=None,
        )
    
>       res = interruptible_api_call(agent, {"messages": []})
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/provider_gateway/test_quota_manager.py:157: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

agent = <test_quota_manager.MockAgent object at 0x7013ea364d70>
api_kwargs = {'messages': []}

    def interruptible_api_call(agent, api_kwargs: dict):
        """
        Run the API call in a background thread so the main conversation loop
        can detect interrupts without waiting for the full HTTP round-trip.
    
        Each worker thread gets its own OpenAI client instance. Interrupts only
        close that worker-local client, so retries and other requests never
        inherit a closed transport.
    
        Includes a stale-call detector: if no response arrives within the
        configured timeout, the connection is killed and an error raised so
        the main retry loop can try again with backoff / credential rotation /
        provider fallback.
        """
        result = {"response": None, "error": None}
        request_client_holder = {"client": None, "owner_tid": None}
        request_client_lock = threading.Lock()
    
        def _set_request_client(client):
            with request_client_lock:
                request_client_holder["client"] = client
                # #29507: stamp the owning thread so a stranger-thread interrupt
                # only shuts the connection down rather than racing the worker
                # for FD ownership during ``client.close()``.
                request_client_holder["owner_tid"] = threading.get_ident()
            return client
    
        def _take_request_client():
            with request_client_lock:
                client = request_client_holder.get("client")
                request_client_holder["client"] = None
                request_client_holder["owner_tid"] = None
                return client
    
        def _close_request_client_once(reason: str) -> None:
            # #29507: dispatch on the calling thread.
            #
            # When ``_call`` (the worker) reaches its ``finally`` it owns the
            # close and we pop + fully close as before. When a *stranger* thread
            # (the interrupt-check loop, the stale-call detector) drives the
            # close, only shut the sockets down so the worker's blocked
            # ``recv``/``send`` unwinds with an ``EPIPE`` / EOF — and let the
            # worker close ``client`` from its own thread on its way out. That
            # avoids the FD-recycling race where the kernel reassigned a
            # just-closed TLS socket FD to ``kanban.db``, and the still-live SSL
            # BIO on the worker thread then wrote a 24-byte TLS application-data
            # record into the SQLite header (#29507).
            with request_client_lock:
                request_client = request_client_holder.get("client")
                owner_tid = request_client_holder.get("owner_tid")
                stranger_thread = (
                    request_client is not None
                    and owner_tid is not None
                    and owner_tid != threading.get_ident()
                )
                if not stranger_thread:
                    # Owning thread (or no recorded owner) → pop and fully close.
                    request_client_holder["client"] = None
                    request_client_holder["owner_tid"] = None
            if request_client is None:
                return
            if stranger_thread:
                agent._abort_request_openai_client(request_client, reason=reason)
            else:
                agent._close_request_openai_client(request_client, reason=reason)
    
        def _call():
            provider_gateway_started_at = time.monotonic()
            try:
                # ── [SUNTIKAN PROVIDER GATEWAY QUOTA CHECK] ──
                try:
                    from provider_gateway.runtime import get_quota_manager
                    quota = get_quota_manager(agent)
                    quota.check_quota(agent)
                except Exception as quota_exc:
                    from provider_gateway.quota_manager import QuotaExceededError
                    if isinstance(quota_exc, QuotaExceededError):
                        raise quota_exc
                    logger.debug("Provider gateway quota check failed: %s", quota_exc)
    
                if agent.api_mode == "codex_responses":
                    request_client = _set_request_client(
                        agent._create_request_openai_client(
                            reason="codex_stream_request",
                            api_kwargs=api_kwargs,
                        )
                    )
                    result["response"] = agent._run_codex_stream(
                        api_kwargs,
                        client=request_client,
                        on_first_delta=getattr(agent, "_codex_on_first_delta", None),
                    )
                elif agent.api_mode == "anthropic_messages":
                    result["response"] = agent._anthropic_messages_create(api_kwargs)
                elif agent.api_mode == "bedrock_converse":
                    # Bedrock uses boto3 directly — no OpenAI client needed.
                    # normalize_converse_response produces an OpenAI-compatible
                    # SimpleNamespace so the rest of the agent loop can treat
                    # bedrock responses like chat_completions responses.
                    from agent.bedrock_adapter import (
                        _get_bedrock_runtime_client,
                        invalidate_runtime_client,
                        is_stale_connection_error,
                        normalize_converse_response,
                    )
                    region = api_kwargs.pop("__bedrock_region__", "us-east-1")
                    api_kwargs.pop("__bedrock_converse__", None)
                    client = _get_bedrock_runtime_client(region)
                    try:
                        raw_response = client.converse(**api_kwargs)
                    except Exception as _bedrock_exc:
                        # Evict the cached client on stale-connection failures
                        # so the outer retry loop builds a fresh client/pool.
                        if is_stale_connection_error(_bedrock_exc):
                            invalidate_runtime_client(region)
                        raise
                    result["response"] = normalize_converse_response(raw_response)
                else:
                    # ── [SUNTIKAN PROVIDER GATEWAY SEMANTIC CACHE PREFLIGHT] ──
                    goto_api_call = True
                    try:
                        from provider_gateway.runtime import get_semantic_cache
                        cache = get_semantic_cache(agent)
                        cached_resp = cache.get_cached_response(agent, api_kwargs.get("messages", []))
                        if cached_resp is not None:
                            result["response"] = cached_resp
                            goto_api_call = False
                    except Exception as cache_exc:
                        logger.debug("Provider gateway cache lookup failed: %s", cache_exc)
    
                    if goto_api_call:
                        request_client = _set_request_client(
                            agent._create_request_openai_client(
                                reason="chat_completion_request",
                                api_kwargs=api_kwargs,
                            )
                        )
                        result["response"] = request_client.chat.completions.create(**api_kwargs)
                        try:
                            from provider_gateway.runtime import record_provider_response_usage
    
                            record_provider_response_usage(
                                agent,
                                result["response"],
                                latency_seconds=time.monotonic() - provider_gateway_started_at,
                            )
                        except Exception as gateway_exc:
                            logger.debug(
                                "Provider gateway response tracking failed: %s",
                                gateway_exc,
                            )
    
                        # ── [SUNTIKAN PROVIDER GATEWAY SEMANTIC CACHE STORE] ──
                        try:
                            from provider_gateway.runtime import get_semantic_cache
                            cache = get_semantic_cache(agent)
                            content = getattr(result["response"].choices[0].message, "content", None)
                            if content:
                                cache.set_cached_response(agent, api_kwargs.get("messages", []), content)
                        except Exception as cache_store_exc:
                            logger.debug("Provider gateway cache store failed: %s", cache_store_exc)
            except Exception as e:
                if agent.api_mode == "chat_completions":
                    try:
                        from provider_gateway.runtime import record_provider_error_usage
    
                        record_provider_error_usage(
                            agent,
                            e,
                            latency_seconds=time.monotonic() - provider_gateway_started_at,
                        )
                    except Exception as gateway_exc:
                        logger.debug(
                            "Provider gateway error tracking failed: %s",
                            gateway_exc,
                        )
                result["error"] = e
            finally:
                _close_request_client_once("request_complete")
    
        # ── Stale-call timeout (mirrors streaming stale detector) ────────
        # Non-streaming calls return nothing until the full response is
        # ready.  Without this, a hung provider can block for the full
        # httpx timeout (default 1800s) with zero feedback.  The stale
        # detector kills the connection early so the main retry loop can
        # apply richer recovery (credential rotation, provider fallback).
>       _stale_timeout = agent._compute_non_stream_stale_timeout(api_kwargs)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       AttributeError: 'MockAgent' object has no attribute '_compute_non_stream_stale_timeout'

agent/chat_completion_helpers.py:337: AttributeError
____________________ test_quota_manager_blocks_on_exceeded _____________________

mock_stale = <MagicMock name='get_provider_stale_timeout' id='123230836114496'>
mock_timeout = <MagicMock name='get_provider_request_timeout' id='123230836119200'>
mock_local = <MagicMock name='is_local_endpoint' id='123230836118528'>
temp_db = PosixPath('/tmp/tmp68fuad7x/test_quota.db')

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
    
        with pytest.raises(QuotaExceededError, match="Daily budget limit exceeded"):
>           interruptible_api_call(agent, {"messages": []})

tests/provider_gateway/test_quota_manager.py:196: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

agent = <test_quota_manager.MockAgent object at 0x7013ea2182d0>
api_kwargs = {'messages': []}

    def interruptible_api_call(agent, api_kwargs: dict):
        """
        Run the API call in a background thread so the main conversation loop
        can detect interrupts without waiting for the full HTTP round-trip.
    
        Each worker thread gets its own OpenAI client instance. Interrupts only
        close that worker-local client, so retries and other requests never
        inherit a closed transport.
    
        Includes a stale-call detector: if no response arrives within the
        configured timeout, the connection is killed and an error raised so
        the main retry loop can try again with backoff / credential rotation /
        provider fallback.
        """
        result = {"response": None, "error": None}
        request_client_holder = {"client": None, "owner_tid": None}
        request_client_lock = threading.Lock()
    
        def _set_request_client(client):
            with request_client_lock:
                request_client_holder["client"] = client
                # #29507: stamp the owning thread so a stranger-thread interrupt
                # only shuts the connection down rather than racing the worker
                # for FD ownership during ``client.close()``.
                request_client_holder["owner_tid"] = threading.get_ident()
            return client
    
        def _take_request_client():
            with request_client_lock:
                client = request_client_holder.get("client")
                request_client_holder["client"] = None
                request_client_holder["owner_tid"] = None
                return client
    
        def _close_request_client_once(reason: str) -> None:
            # #29507: dispatch on the calling thread.
            #
            # When ``_call`` (the worker) reaches its ``finally`` it owns the
            # close and we pop + fully close as before. When a *stranger* thread
            # (the interrupt-check loop, the stale-call detector) drives the
            # close, only shut the sockets down so the worker's blocked
            # ``recv``/``send`` unwinds with an ``EPIPE`` / EOF — and let the
            # worker close ``client`` from its own thread on its way out. That
            # avoids the FD-recycling race where the kernel reassigned a
            # just-closed TLS socket FD to ``kanban.db``, and the still-live SSL
            # BIO on the worker thread then wrote a 24-byte TLS application-data
            # record into the SQLite header (#29507).
            with request_client_lock:
                request_client = request_client_holder.get("client")
                owner_tid = request_client_holder.get("owner_tid")
                stranger_thread = (
                    request_client is not None
                    and owner_tid is not None
                    and owner_tid != threading.get_ident()
                )
                if not stranger_thread:
                    # Owning thread (or no recorded owner) → pop and fully close.
                    request_client_holder["client"] = None
                    request_client_holder["owner_tid"] = None
            if request_client is None:
                return
            if stranger_thread:
                agent._abort_request_openai_client(request_client, reason=reason)
            else:
                agent._close_request_openai_client(request_client, reason=reason)
    
        def _call():
            provider_gateway_started_at = time.monotonic()
            try:
                # ── [SUNTIKAN PROVIDER GATEWAY QUOTA CHECK] ──
                try:
                    from provider_gateway.runtime import get_quota_manager
                    quota = get_quota_manager(agent)
                    quota.check_quota(agent)
                except Exception as quota_exc:
                    from provider_gateway.quota_manager import QuotaExceededError
                    if isinstance(quota_exc, QuotaExceededError):
                        raise quota_exc
                    logger.debug("Provider gateway quota check failed: %s", quota_exc)
    
                if agent.api_mode == "codex_responses":
                    request_client = _set_request_client(
                        agent._create_request_openai_client(
                            reason="codex_stream_request",
                            api_kwargs=api_kwargs,
                        )
                    )
                    result["response"] = agent._run_codex_stream(
                        api_kwargs,
                        client=request_client,
                        on_first_delta=getattr(agent, "_codex_on_first_delta", None),
                    )
                elif agent.api_mode == "anthropic_messages":
                    result["response"] = agent._anthropic_messages_create(api_kwargs)
                elif agent.api_mode == "bedrock_converse":
                    # Bedrock uses boto3 directly — no OpenAI client needed.
                    # normalize_converse_response produces an OpenAI-compatible
                    # SimpleNamespace so the rest of the agent loop can treat
                    # bedrock responses like chat_completions responses.
                    from agent.bedrock_adapter import (
                        _get_bedrock_runtime_client,
                        invalidate_runtime_client,
                        is_stale_connection_error,
                        normalize_converse_response,
                    )
                    region = api_kwargs.pop("__bedrock_region__", "us-east-1")
                    api_kwargs.pop("__bedrock_converse__", None)
                    client = _get_bedrock_runtime_client(region)
                    try:
                        raw_response = client.converse(**api_kwargs)
                    except Exception as _bedrock_exc:
                        # Evict the cached client on stale-connection failures
                        # so the outer retry loop builds a fresh client/pool.
                        if is_stale_connection_error(_bedrock_exc):
                            invalidate_runtime_client(region)
                        raise
                    result["response"] = normalize_converse_response(raw_response)
                else:
                    # ── [SUNTIKAN PROVIDER GATEWAY SEMANTIC CACHE PREFLIGHT] ──
                    goto_api_call = True
                    try:
                        from provider_gateway.runtime import get_semantic_cache
                        cache = get_semantic_cache(agent)
                        cached_resp = cache.get_cached_response(agent, api_kwargs.get("messages", []))
                        if cached_resp is not None:
                            result["response"] = cached_resp
                            goto_api_call = False
                    except Exception as cache_exc:
                        logger.debug("Provider gateway cache lookup failed: %s", cache_exc)
    
                    if goto_api_call:
                        request_client = _set_request_client(
                            agent._create_request_openai_client(
                                reason="chat_completion_request",
                                api_kwargs=api_kwargs,
                            )
                        )
                        result["response"] = request_client.chat.completions.create(**api_kwargs)
                        try:
                            from provider_gateway.runtime import record_provider_response_usage
    
                            record_provider_response_usage(
                                agent,
                                result["response"],
                                latency_seconds=time.monotonic() - provider_gateway_started_at,
                            )
                        except Exception as gateway_exc:
                            logger.debug(
                                "Provider gateway response tracking failed: %s",
                                gateway_exc,
                            )
    
                        # ── [SUNTIKAN PROVIDER GATEWAY SEMANTIC CACHE STORE] ──
                        try:
                            from provider_gateway.runtime import get_semantic_cache
                            cache = get_semantic_cache(agent)
                            content = getattr(result["response"].choices[0].message, "content", None)
                            if content:
                                cache.set_cached_response(agent, api_kwargs.get("messages", []), content)
                        except Exception as cache_store_exc:
                            logger.debug("Provider gateway cache store failed: %s", cache_store_exc)
            except Exception as e:
                if agent.api_mode == "chat_completions":
                    try:
                        from provider_gateway.runtime import record_provider_error_usage
    
                        record_provider_error_usage(
                            agent,
                            e,
                            latency_seconds=time.monotonic() - provider_gateway_started_at,
                        )
                    except Exception as gateway_exc:
                        logger.debug(
                            "Provider gateway error tracking failed: %s",
                            gateway_exc,
                        )
                result["error"] = e
            finally:
                _close_request_client_once("request_complete")
    
        # ── Stale-call timeout (mirrors streaming stale detector) ────────
        # Non-streaming calls return nothing until the full response is
        # ready.  Without this, a hung provider can block for the full
        # httpx timeout (default 1800s) with zero feedback.  The stale
        # detector kills the connection early so the main retry loop can
        # apply richer recovery (credential rotation, provider fallback).
>       _stale_timeout = agent._compute_non_stream_stale_timeout(api_kwargs)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       AttributeError: 'MockAgent' object has no attribute '_compute_non_stream_stale_timeout'

agent/chat_completion_helpers.py:337: AttributeError
___________________ test_quota_manager_fallback_on_exceeded ____________________

mock_stale = <MagicMock name='get_provider_stale_timeout' id='123230836121216'>
mock_timeout = <MagicMock name='get_provider_request_timeout' id='123230836120544'>
mock_local = <MagicMock name='is_local_endpoint' id='123230836120880'>
temp_db = PosixPath('/tmp/tmpjub980xw/test_quota.db')

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
    
>       res = interruptible_api_call(agent, {"messages": []})
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/provider_gateway/test_quota_manager.py:252: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

agent = <test_quota_manager.MockAgent object at 0x7013ea2187d0>
api_kwargs = {'messages': []}

    def interruptible_api_call(agent, api_kwargs: dict):
        """
        Run the API call in a background thread so the main conversation loop
        can detect interrupts without waiting for the full HTTP round-trip.
    
        Each worker thread gets its own OpenAI client instance. Interrupts only
        close that worker-local client, so retries and other requests never
        inherit a closed transport.
    
        Includes a stale-call detector: if no response arrives within the
        configured timeout, the connection is killed and an error raised so
        the main retry loop can try again with backoff / credential rotation /
        provider fallback.
        """
        result = {"response": None, "error": None}
        request_client_holder = {"client": None, "owner_tid": None}
        request_client_lock = threading.Lock()
    
        def _set_request_client(client):
            with request_client_lock:
                request_client_holder["client"] = client
                # #29507: stamp the owning thread so a stranger-thread interrupt
                # only shuts the connection down rather than racing the worker
                # for FD ownership during ``client.close()``.
                request_client_holder["owner_tid"] = threading.get_ident()
            return client
    
        def _take_request_client():
            with request_client_lock:
                client = request_client_holder.get("client")
                request_client_holder["client"] = None
                request_client_holder["owner_tid"] = None
                return client
    
        def _close_request_client_once(reason: str) -> None:
            # #29507: dispatch on the calling thread.
            #
            # When ``_call`` (the worker) reaches its ``finally`` it owns the
            # close and we pop + fully close as before. When a *stranger* thread
            # (the interrupt-check loop, the stale-call detector) drives the
            # close, only shut the sockets down so the worker's blocked
            # ``recv``/``send`` unwinds with an ``EPIPE`` / EOF — and let the
            # worker close ``client`` from its own thread on its way out. That
            # avoids the FD-recycling race where the kernel reassigned a
            # just-closed TLS socket FD to ``kanban.db``, and the still-live SSL
            # BIO on the worker thread then wrote a 24-byte TLS application-data
            # record into the SQLite header (#29507).
            with request_client_lock:
                request_client = request_client_holder.get("client")
                owner_tid = request_client_holder.get("owner_tid")
                stranger_thread = (
                    request_client is not None
                    and owner_tid is not None
                    and owner_tid != threading.get_ident()
                )
                if not stranger_thread:
                    # Owning thread (or no recorded owner) → pop and fully close.
                    request_client_holder["client"] = None
                    request_client_holder["owner_tid"] = None
            if request_client is None:
                return
            if stranger_thread:
                agent._abort_request_openai_client(request_client, reason=reason)
            else:
                agent._close_request_openai_client(request_client, reason=reason)
    
        def _call():
            provider_gateway_started_at = time.monotonic()
            try:
                # ── [SUNTIKAN PROVIDER GATEWAY QUOTA CHECK] ──
                try:
                    from provider_gateway.runtime import get_quota_manager
                    quota = get_quota_manager(agent)
                    quota.check_quota(agent)
                except Exception as quota_exc:
                    from provider_gateway.quota_manager import QuotaExceededError
                    if isinstance(quota_exc, QuotaExceededError):
                        raise quota_exc
                    logger.debug("Provider gateway quota check failed: %s", quota_exc)
    
                if agent.api_mode == "codex_responses":
                    request_client = _set_request_client(
                        agent._create_request_openai_client(
                            reason="codex_stream_request",
                            api_kwargs=api_kwargs,
                        )
                    )
                    result["response"] = agent._run_codex_stream(
                        api_kwargs,
                        client=request_client,
                        on_first_delta=getattr(agent, "_codex_on_first_delta", None),
                    )
                elif agent.api_mode == "anthropic_messages":
                    result["response"] = agent._anthropic_messages_create(api_kwargs)
                elif agent.api_mode == "bedrock_converse":
                    # Bedrock uses boto3 directly — no OpenAI client needed.
                    # normalize_converse_response produces an OpenAI-compatible
                    # SimpleNamespace so the rest of the agent loop can treat
                    # bedrock responses like chat_completions responses.
                    from agent.bedrock_adapter import (
                        _get_bedrock_runtime_client,
                        invalidate_runtime_client,
                        is_stale_connection_error,
                        normalize_converse_response,
                    )
                    region = api_kwargs.pop("__bedrock_region__", "us-east-1")
                    api_kwargs.pop("__bedrock_converse__", None)
                    client = _get_bedrock_runtime_client(region)
                    try:
                        raw_response = client.converse(**api_kwargs)
                    except Exception as _bedrock_exc:
                        # Evict the cached client on stale-connection failures
                        # so the outer retry loop builds a fresh client/pool.
                        if is_stale_connection_error(_bedrock_exc):
                            invalidate_runtime_client(region)
                        raise
                    result["response"] = normalize_converse_response(raw_response)
                else:
                    # ── [SUNTIKAN PROVIDER GATEWAY SEMANTIC CACHE PREFLIGHT] ──
                    goto_api_call = True
                    try:
                        from provider_gateway.runtime import get_semantic_cache
                        cache = get_semantic_cache(agent)
                        cached_resp = cache.get_cached_response(agent, api_kwargs.get("messages", []))
                        if cached_resp is not None:
                            result["response"] = cached_resp
                            goto_api_call = False
                    except Exception as cache_exc:
                        logger.debug("Provider gateway cache lookup failed: %s", cache_exc)
    
                    if goto_api_call:
                        request_client = _set_request_client(
                            agent._create_request_openai_client(
                                reason="chat_completion_request",
                                api_kwargs=api_kwargs,
                            )
                        )
                        result["response"] = request_client.chat.completions.create(**api_kwargs)
                        try:
                            from provider_gateway.runtime import record_provider_response_usage
    
                            record_provider_response_usage(
                                agent,
                                result["response"],
                                latency_seconds=time.monotonic() - provider_gateway_started_at,
                            )
                        except Exception as gateway_exc:
                            logger.debug(
                                "Provider gateway response tracking failed: %s",
                                gateway_exc,
                            )
    
                        # ── [SUNTIKAN PROVIDER GATEWAY SEMANTIC CACHE STORE] ──
                        try:
                            from provider_gateway.runtime import get_semantic_cache
                            cache = get_semantic_cache(agent)
                            content = getattr(result["response"].choices[0].message, "content", None)
                            if content:
                                cache.set_cached_response(agent, api_kwargs.get("messages", []), content)
                        except Exception as cache_store_exc:
                            logger.debug("Provider gateway cache store failed: %s", cache_store_exc)
            except Exception as e:
                if agent.api_mode == "chat_completions":
                    try:
                        from provider_gateway.runtime import record_provider_error_usage
    
                        record_provider_error_usage(
                            agent,
                            e,
                            latency_seconds=time.monotonic() - provider_gateway_started_at,
                        )
                    except Exception as gateway_exc:
                        logger.debug(
                            "Provider gateway error tracking failed: %s",
                            gateway_exc,
                        )
                result["error"] = e
            finally:
                _close_request_client_once("request_complete")
    
        # ── Stale-call timeout (mirrors streaming stale detector) ────────
        # Non-streaming calls return nothing until the full response is
        # ready.  Without this, a hung provider can block for the full
        # httpx timeout (default 1800s) with zero feedback.  The stale
        # detector kills the connection early so the main retry loop can
        # apply richer recovery (credential rotation, provider fallback).
>       _stale_timeout = agent._compute_non_stream_stale_timeout(api_kwargs)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       AttributeError: 'MockAgent' object has no attribute '_compute_non_stream_stale_timeout'

agent/chat_completion_helpers.py:337: AttributeError
____________________ test_semantic_cache_basic_miss_and_hit ____________________

mock_stale = <MagicMock name='get_provider_stale_timeout' id='123230836127600'>
mock_timeout = <MagicMock name='get_provider_request_timeout' id='123230836127264'>
mock_local = <MagicMock name='is_local_endpoint' id='123230836127936'>
temp_db = PosixPath('/tmp/tmplbzju5jk/test_cache.db')

    @patch("agent.chat_completion_helpers.is_local_endpoint")
    @patch("agent.chat_completion_helpers.get_provider_request_timeout")
    @patch("agent.chat_completion_helpers.get_provider_stale_timeout")
    def test_semantic_cache_basic_miss_and_hit(
        mock_stale, mock_timeout, mock_local, temp_db
    ) -> None:
        """Test that a cache miss records to the cache, and a subsequent identical request hits."""
        mock_local.return_value = False
        mock_timeout.return_value = None
        mock_stale.return_value = None
    
        tracker = CapturingTracker()
        agent = MockAgent(tracker, temp_db)
    
        # 1. Mock API call behavior for cache miss
        mock_client = MagicMock()
        agent._create_request_openai_client = MagicMock(return_value=mock_client)
        agent._close_request_openai_client = MagicMock()
    
        mock_msg = SimpleNamespace(role="assistant", content="Jawaban pertama.")
        mock_choice = SimpleNamespace(message=mock_msg, finish_reason="stop")
        mock_response = SimpleNamespace(
            id="resp-123",
            model="anthropic/claude-sonnet-4.6",
            choices=[mock_choice],
            usage=None,
        )
        mock_client.chat.completions.create.return_value = mock_response
    
        # First call (Miss)
        messages = [{"role": "user", "content": "Pertanyaan unik"}]
>       res1 = interruptible_api_call(agent, {"messages": messages})
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/provider_gateway/test_semantic_cache.py:133: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

agent = <test_semantic_cache.MockAgent object at 0x7013ea367b60>
api_kwargs = {'messages': [{'content': 'Pertanyaan unik', 'role': 'user'}]}

    def interruptible_api_call(agent, api_kwargs: dict):
        """
        Run the API call in a background thread so the main conversation loop
        can detect interrupts without waiting for the full HTTP round-trip.
    
        Each worker thread gets its own OpenAI client instance. Interrupts only
        close that worker-local client, so retries and other requests never
        inherit a closed transport.
    
        Includes a stale-call detector: if no response arrives within the
        configured timeout, the connection is killed and an error raised so
        the main retry loop can try again with backoff / credential rotation /
        provider fallback.
        """
        result = {"response": None, "error": None}
        request_client_holder = {"client": None, "owner_tid": None}
        request_client_lock = threading.Lock()
    
        def _set_request_client(client):
            with request_client_lock:
                request_client_holder["client"] = client
                # #29507: stamp the owning thread so a stranger-thread interrupt
                # only shuts the connection down rather than racing the worker
                # for FD ownership during ``client.close()``.
                request_client_holder["owner_tid"] = threading.get_ident()
            return client
    
        def _take_request_client():
            with request_client_lock:
                client = request_client_holder.get("client")
                request_client_holder["client"] = None
                request_client_holder["owner_tid"] = None
                return client
    
        def _close_request_client_once(reason: str) -> None:
            # #29507: dispatch on the calling thread.
            #
            # When ``_call`` (the worker) reaches its ``finally`` it owns the
            # close and we pop + fully close as before. When a *stranger* thread
            # (the interrupt-check loop, the stale-call detector) drives the
            # close, only shut the sockets down so the worker's blocked
            # ``recv``/``send`` unwinds with an ``EPIPE`` / EOF — and let the
            # worker close ``client`` from its own thread on its way out. That
            # avoids the FD-recycling race where the kernel reassigned a
            # just-closed TLS socket FD to ``kanban.db``, and the still-live SSL
            # BIO on the worker thread then wrote a 24-byte TLS application-data
            # record into the SQLite header (#29507).
            with request_client_lock:
                request_client = request_client_holder.get("client")
                owner_tid = request_client_holder.get("owner_tid")
                stranger_thread = (
                    request_client is not None
                    and owner_tid is not None
                    and owner_tid != threading.get_ident()
                )
                if not stranger_thread:
                    # Owning thread (or no recorded owner) → pop and fully close.
                    request_client_holder["client"] = None
                    request_client_holder["owner_tid"] = None
            if request_client is None:
                return
            if stranger_thread:
                agent._abort_request_openai_client(request_client, reason=reason)
            else:
                agent._close_request_openai_client(request_client, reason=reason)
    
        def _call():
            provider_gateway_started_at = time.monotonic()
            try:
                # ── [SUNTIKAN PROVIDER GATEWAY QUOTA CHECK] ──
                try:
                    from provider_gateway.runtime import get_quota_manager
                    quota = get_quota_manager(agent)
                    quota.check_quota(agent)
                except Exception as quota_exc:
                    from provider_gateway.quota_manager import QuotaExceededError
                    if isinstance(quota_exc, QuotaExceededError):
                        raise quota_exc
                    logger.debug("Provider gateway quota check failed: %s", quota_exc)
    
                if agent.api_mode == "codex_responses":
                    request_client = _set_request_client(
                        agent._create_request_openai_client(
                            reason="codex_stream_request",
                            api_kwargs=api_kwargs,
                        )
                    )
                    result["response"] = agent._run_codex_stream(
                        api_kwargs,
                        client=request_client,
                        on_first_delta=getattr(agent, "_codex_on_first_delta", None),
                    )
                elif agent.api_mode == "anthropic_messages":
                    result["response"] = agent._anthropic_messages_create(api_kwargs)
                elif agent.api_mode == "bedrock_converse":
                    # Bedrock uses boto3 directly — no OpenAI client needed.
                    # normalize_converse_response produces an OpenAI-compatible
                    # SimpleNamespace so the rest of the agent loop can treat
                    # bedrock responses like chat_completions responses.
                    from agent.bedrock_adapter import (
                        _get_bedrock_runtime_client,
                        invalidate_runtime_client,
                        is_stale_connection_error,
                        normalize_converse_response,
                    )
                    region = api_kwargs.pop("__bedrock_region__", "us-east-1")
                    api_kwargs.pop("__bedrock_converse__", None)
                    client = _get_bedrock_runtime_client(region)
                    try:
                        raw_response = client.converse(**api_kwargs)
                    except Exception as _bedrock_exc:
                        # Evict the cached client on stale-connection failures
                        # so the outer retry loop builds a fresh client/pool.
                        if is_stale_connection_error(_bedrock_exc):
                            invalidate_runtime_client(region)
                        raise
                    result["response"] = normalize_converse_response(raw_response)
                else:
                    # ── [SUNTIKAN PROVIDER GATEWAY SEMANTIC CACHE PREFLIGHT] ──
                    goto_api_call = True
                    try:
                        from provider_gateway.runtime import get_semantic_cache
                        cache = get_semantic_cache(agent)
                        cached_resp = cache.get_cached_response(agent, api_kwargs.get("messages", []))
                        if cached_resp is not None:
                            result["response"] = cached_resp
                            goto_api_call = False
                    except Exception as cache_exc:
                        logger.debug("Provider gateway cache lookup failed: %s", cache_exc)
    
                    if goto_api_call:
                        request_client = _set_request_client(
                            agent._create_request_openai_client(
                                reason="chat_completion_request",
                                api_kwargs=api_kwargs,
                            )
                        )
                        result["response"] = request_client.chat.completions.create(**api_kwargs)
                        try:
                            from provider_gateway.runtime import record_provider_response_usage
    
                            record_provider_response_usage(
                                agent,
                                result["response"],
                                latency_seconds=time.monotonic() - provider_gateway_started_at,
                            )
                        except Exception as gateway_exc:
                            logger.debug(
                                "Provider gateway response tracking failed: %s",
                                gateway_exc,
                            )
    
                        # ── [SUNTIKAN PROVIDER GATEWAY SEMANTIC CACHE STORE] ──
                        try:
                            from provider_gateway.runtime import get_semantic_cache
                            cache = get_semantic_cache(agent)
                            content = getattr(result["response"].choices[0].message, "content", None)
                            if content:
                                cache.set_cached_response(agent, api_kwargs.get("messages", []), content)
                        except Exception as cache_store_exc:
                            logger.debug("Provider gateway cache store failed: %s", cache_store_exc)
            except Exception as e:
                if agent.api_mode == "chat_completions":
                    try:
                        from provider_gateway.runtime import record_provider_error_usage
    
                        record_provider_error_usage(
                            agent,
                            e,
                            latency_seconds=time.monotonic() - provider_gateway_started_at,
                        )
                    except Exception as gateway_exc:
                        logger.debug(
                            "Provider gateway error tracking failed: %s",
                            gateway_exc,
                        )
                result["error"] = e
            finally:
                _close_request_client_once("request_complete")
    
        # ── Stale-call timeout (mirrors streaming stale detector) ────────
        # Non-streaming calls return nothing until the full response is
        # ready.  Without this, a hung provider can block for the full
        # httpx timeout (default 1800s) with zero feedback.  The stale
        # detector kills the connection early so the main retry loop can
        # apply richer recovery (credential rotation, provider fallback).
>       _stale_timeout = agent._compute_non_stream_stale_timeout(api_kwargs)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       AttributeError: 'MockAgent' object has no attribute '_compute_non_stream_stale_timeout'

agent/chat_completion_helpers.py:337: AttributeError
____________________ test_semantic_cache_disabled_by_config ____________________

mock_stale = <MagicMock name='get_provider_stale_timeout' id='123230832035552'>
mock_timeout = <MagicMock name='get_provider_request_timeout' id='123230832034880'>
mock_local = <MagicMock name='is_local_endpoint' id='123230832035216'>
temp_db = PosixPath('/tmp/tmpj3wx2xi4/test_cache.db')

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

tests/provider_gateway/test_semantic_cache.py:166: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = GatewayConfig(enabled=True, backend='native', track_usage=True, track_cost=False, routing_strategy='round-robin', fallback_models=[], daily_limit_usd=None, monthly_limit_usd=None, quota_action='block')
name = 'enabled', value = False

>   ???
E   dataclasses.FrozenInstanceError: cannot assign to field 'enabled'

<string>:23: FrozenInstanceError
______________________ test_semantic_cache_streaming_hit _______________________

mock_stale = <MagicMock name='get_provider_stale_timeout' id='123230832034544'>
mock_timeout = <MagicMock name='get_provider_request_timeout' id='123230832035888'>
mock_local = <MagicMock name='is_local_endpoint' id='123230832036224'>
temp_db = PosixPath('/tmp/tmp14e8z1t1/test_cache.db')

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

tests/provider_gateway/test_semantic_cache.py:215: 
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

mock_stale = <MagicMock name='get_provider_stale_timeout' id='123230818872224'>
mock_timeout = <MagicMock name='get_provider_request_timeout' id='123230818871552'>
mock_local = <MagicMock name='is_local_endpoint' id='123230818871888'>

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
>       res = interruptible_streaming_api_call(agent, {"messages": []})
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/provider_gateway/test_stream_tracking.py:107: 
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
            agent._create_request_openai_client(
                reason="chat_completion_stream_request",
                api_kwargs=stream_kwargs,
            )
        )
        # Reset stale-stream timer so the detector measures from this
        # attempt's start, not a previous attempt's last chunk.
        last_chunk_time["t"] = time.time()
>       agent._touch_activity("waiting for provider response (streaming)")
        ^^^^^^^^^^^^^^^^^^^^^
E       AttributeError: 'MockAgent' object has no attribute '_touch_activity'

agent/chat_completion_helpers.py:1868: AttributeError
____________________ test_streaming_api_call_records_error _____________________

mock_stale = <MagicMock name='get_provider_stale_timeout' id='123230818876256'>
mock_timeout = <MagicMock name='get_provider_request_timeout' id='123230818875584'>
mock_local = <MagicMock name='is_local_endpoint' id='123230818875920'>

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
>           interruptible_streaming_api_call(agent, {"messages": []})

tests/provider_gateway/test_stream_tracking.py:152: 
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
            agent._create_request_openai_client(
                reason="chat_completion_stream_request",
                api_kwargs=stream_kwargs,
            )
        )
        # Reset stale-stream timer so the detector measures from this
        # attempt's start, not a previous attempt's last chunk.
        last_chunk_time["t"] = time.time()
>       agent._touch_activity("waiting for provider response (streaming)")
        ^^^^^^^^^^^^^^^^^^^^^
E       AttributeError: 'MockAgent' object has no attribute '_touch_activity'

agent/chat_completion_helpers.py:1868: AttributeError
=========================== short test summary info ============================
FAILED tests/provider_gateway/test_quota_manager.py::test_quota_manager_within_limits
FAILED tests/provider_gateway/test_quota_manager.py::test_quota_manager_blocks_on_exceeded
FAILED tests/provider_gateway/test_quota_manager.py::test_quota_manager_fallback_on_exceeded
FAILED tests/provider_gateway/test_semantic_cache.py::test_semantic_cache_basic_miss_and_hit
FAILED tests/provider_gateway/test_semantic_cache.py::test_semantic_cache_disabled_by_config
FAILED tests/provider_gateway/test_semantic_cache.py::test_semantic_cache_streaming_hit
FAILED tests/provider_gateway/test_stream_tracking.py::test_streaming_api_call_records_usage_successfully
FAILED tests/provider_gateway/test_stream_tracking.py::test_streaming_api_call_records_error
8 failed, 75 passed in 1.66s
