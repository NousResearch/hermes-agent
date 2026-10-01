"""Request preparation and recovery for auxiliary LLM calls."""

from __future__ import annotations

import functools
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

if TYPE_CHECKING:
    from agent.auxiliary_client import _ResolvedAuxRoute, _PreparedAuxRequest, _LadderStep


def _resolve_call_client(
    task: Optional[str], *, provider: Optional[str], model: Optional[str], base_url: Optional[str],
    api_key: Optional[str], resolved_provider: str, resolved_model: Optional[str],
    resolved_base_url: Optional[str], resolved_api_key: Optional[str],
    resolved_api_mode: Optional[str], main_runtime: Optional[dict[str, Any]], async_mode: bool,
) -> _ResolvedAuxRoute:
    """Resolve the client for one aux call: vision chain, or cached text client with the
    explicit-provider fallback_chain / auto-chain rescue; RuntimeError when nothing is configured."""
    from agent.auxiliary_client import (
        AuxiliaryClientUnavailable,
        _ResolvedAuxRoute,
        _effective_provider_for_client,
        _get_cached_client,
        _to_async_client,
        _try_configured_fallback_for_unavailable_client,
        logger,
        missing_provider_credentials_message,
        nous_credential_failure_detail,
        resolve_vision_provider_client,
    )

    effective_provider = resolved_provider
    if task == "vision":
        effective_provider, client, final_model = resolve_vision_provider_client(
            provider=resolved_provider if resolved_provider != "auto" else provider,
            model=resolved_model or model, base_url=resolved_base_url or base_url,
            api_key=resolved_api_key or api_key, async_mode=async_mode, main_runtime=main_runtime,
        )
        if client is None and resolved_provider != "auto" and not resolved_base_url:
            logger.warning("Vision provider %s unavailable, falling back to auto vision backends",
                           resolved_provider)
            effective_provider, client, final_model = resolve_vision_provider_client(
                provider="auto", model=resolved_model, async_mode=async_mode,
                main_runtime=main_runtime)
        if client is not None:
            resolved_provider = effective_provider or resolved_provider
    else:
        client, final_model = _get_cached_client(
            resolved_provider, resolved_model, async_mode=async_mode, base_url=resolved_base_url,
            api_key=resolved_api_key, api_mode=resolved_api_mode, main_runtime=main_runtime,
            task=task)
        effective_provider = _effective_provider_for_client(client, resolved_provider)
        if client is None:
            # Explicit provider with no credentials: honor the task fallback_chain before
            # raising (fallback entries may use OAuth / credential-pool auth).
            _explicit = (resolved_provider or "").strip().lower()
            if _explicit and _explicit not in {"auto", "openrouter", "custom"}:
                fb_client, fb_model, fb_label = _try_configured_fallback_for_unavailable_client(
                    task, _explicit)
                if fb_client is None:
                    nous_detail = nous_credential_failure_detail() if _explicit == "nous" else None
                    raise AuxiliaryClientUnavailable(
                        nous_detail or missing_provider_credentials_message(_explicit))
                client, final_model = fb_client, fb_model
                if async_mode:
                    client, final_model = _to_async_client(
                        fb_client, fb_model or "", is_vision=(task == "vision"))
                resolved_provider = fb_label or resolved_provider
                effective_provider = resolved_provider
            # Auto/custom with no credentials: walk the full auto chain (not just OpenRouter).
            # model=None so each provider uses its own default.
            if client is None and not resolved_base_url:
                logger.info("Auxiliary %s: provider %s unavailable, trying auto-detection chain",
                            task or "call", resolved_provider)
                client, final_model = _get_cached_client(
                    "auto", async_mode=async_mode, main_runtime=main_runtime, task=task)
                effective_provider = _effective_provider_for_client(client, "auto")
    if client is None:
        raise AuxiliaryClientUnavailable(f"No LLM provider configured for task={task} "
                                         f"provider={resolved_provider}. Run: hermes setup")
    return _ResolvedAuxRoute(client, final_model, resolved_provider, effective_provider)


def _prepare_aux_request(
    task: Optional[str], *, provider: Optional[str], model: Optional[str], base_url: Optional[str],
    api_key: Optional[str], main_runtime: dict[str, Any], messages: list,
    temperature: Optional[float], max_tokens: Optional[int], tools: Optional[list],
    timeout: Optional[float], extra_body: Optional[dict], reasoning_config: Optional[dict],
    extra_headers: Optional[dict[str, str]], api_mode: Optional[str],
    route_info: Optional[dict[str, str]], async_mode: bool,
) -> _PreparedAuxRequest:
    """Shared head of call_llm/async_call_llm: resolve route + client, publish it, build request kwargs.
    Sync-only: compression fast lane, per-request ``extra_headers``, and ``base_info`` falling
    back to the resolved base_url when the client exposes none."""
    from agent.auxiliary_client import (
        AsyncCodexAuxiliaryClient,
        CodexAuxiliaryClient,
        _PreparedAuxRequest,
        _build_call_kwargs,
        _compression_fast_lane_controls,
        _convert_openai_images_to_anthropic,
        _effective_aux_timeout,
        _fallback_provider_from_label,
        _get_auxiliary_task_config,
        _get_task_extra_body,
        _get_task_no_progress_timeout,
        _is_anthropic_compat_endpoint,
        _record_route_info,
        _resolve_task_provider_model,
        _set_relay_auxiliary_route,
        logger,
    )

    resolved_provider, resolved_model, resolved_base_url, resolved_api_key, resolved_api_mode = _resolve_task_provider_model(
        task, provider, model, base_url, api_key)
    if api_mode:
        resolved_api_mode = api_mode
    effective_extra_body = _get_task_extra_body(task)
    effective_extra_body.update(extra_body or {})
    client, final_model, resolved_provider, effective_provider = _resolve_call_client(
        task, provider=provider, model=model, base_url=base_url, api_key=api_key,
        resolved_provider=resolved_provider, resolved_model=resolved_model,
        resolved_base_url=resolved_base_url, resolved_api_key=resolved_api_key,
        resolved_api_mode=resolved_api_mode, main_runtime=main_runtime, async_mode=async_mode,
    )
    effective_timeout = _effective_aux_timeout(task, timeout)
    # Codex-Responses-only: real SDK clients reject an unrecognized ``no_progress_timeout``
    # kwarg, so only resolve/forward it when the route is actually a Codex stream (#108104).
    no_progress_timeout = (
        _get_task_no_progress_timeout(task)
        if isinstance(client, (CodexAuxiliaryClient, AsyncCodexAuxiliaryClient)) else None
    )
    request_provider = effective_provider or resolved_provider
    if not async_mode:
        compression_config = _get_auxiliary_task_config("compression") if task == "compression" else {}
        _, effective_extra_body = _compression_fast_lane_controls(
            task, actual_provider=request_provider, actual_model=final_model,
            requested_provider=provider, requested_model=model, route_config=compression_config,
            leak_guard_config=compression_config, max_tokens=max_tokens,
            extra_body=effective_extra_body,
        )
    _set_relay_auxiliary_route(request_provider, final_model, resolved_api_mode)
    _record_route_info(route_info, _fallback_provider_from_label(request_provider), final_model)
    if async_mode:
        base_info = str(getattr(client, "base_url", "") or "")
    else:
        base_info = str(getattr(client, "base_url", resolved_base_url) or "")
        if task:
            logger.info("Auxiliary %s: using %s (%s)%s",
                         task, request_provider or "auto", final_model or "default",
                         f" at {base_info}" if base_info and "openrouter" not in base_info else "")
    # Client's actual base_url so endpoint-specific temperature overrides work on
    # auto-detected routes (api.moonshot.ai vs api.kimi.com/coding).
    kwargs = _build_call_kwargs(
        request_provider, final_model, messages, temperature=temperature, max_tokens=max_tokens,
        tools=tools, timeout=effective_timeout, extra_body=effective_extra_body,
        reasoning_config=reasoning_config, base_url=base_info or resolved_base_url, task=task,
        no_progress_timeout=no_progress_timeout)
    if extra_headers:
        kwargs["extra_headers"] = dict(extra_headers)
    # Convert image blocks for Anthropic-compatible endpoints (e.g. MiniMax)
    client_base = str(getattr(client, "base_url", "") or "")
    if _is_anthropic_compat_endpoint(request_provider, client_base):
        kwargs["messages"] = _convert_openai_images_to_anthropic(kwargs["messages"])
    return _PreparedAuxRequest(
        client, final_model, kwargs, resolved_provider, request_provider, resolved_model,
        resolved_base_url, resolved_api_key, resolved_api_mode, effective_timeout,
        effective_extra_body, base_info)


def _aux_recovery_ladder(
    first_err: Exception, *, client: Any, kwargs: dict[str, Any], task: Optional[str],
    async_mode: bool, base_info: str, resolved_provider: str, resolved_model: Optional[str],
    resolved_base_url: Optional[str], resolved_api_key: Optional[str],
    resolved_api_mode: Optional[str], final_model: Optional[str], max_tokens: Optional[int],
    main_runtime: Optional[dict[str, Any]], route_info: Optional[dict[str, str]],
):
    """Ordered recovery rungs after the primary request failed (generator): parameter
    strips → Nous heal/refresh → credential refresh/pool rotation → provider fallback.
    Each rung returns a response, narrows ``first_err`` and falls through, or re-raises.
    Raises the narrowed ``first_err`` when exhausted (after evicting a connection-poisoned client)."""
    from agent.auxiliary_client import (
        _LadderRoute,
        _evict_cached_client_instance,
        _is_connection_error,
        _ladder_credential_rungs,
        _ladder_nous_rungs,
        _ladder_parameter_rungs,
        _ladder_provider_fallback,
        base_url_host_matches,
        logger,
    )

    tag = " (async)" if async_mode else ""
    route = _LadderRoute(
        client, task, tag, async_mode, base_info, resolved_provider, resolved_model,
        resolved_base_url, resolved_api_key, resolved_api_mode, final_model, main_runtime, route_info,
        kwargs.get("timeout"))
    resp, first_err, kwargs = yield from _ladder_parameter_rungs(first_err, route, kwargs, max_tokens)
    if first_err is None:
        return resp
    client_is_nous = (resolved_provider == "nous"
                      or base_url_host_matches(base_info, "inference-api.nousresearch.com"))
    resp, first_err = yield from _ladder_nous_rungs(first_err, route, kwargs, client_is_nous)
    if first_err is None:
        return resp
    resp, first_err = yield from _ladder_credential_rungs(first_err, route, kwargs, client_is_nous)
    if first_err is None:
        return resp
    resp = yield from _ladder_provider_fallback(first_err, route)
    if resp is not None:
        return resp
    # Connection/timeout errors poison the cached client (closed transport, half-read
    # stream); evict so the next aux call rebuilds a fresh one.
    # Reached only when no fallback answered, so the next auxiliary call rebuilds a fresh
    # client instead of reusing the dead one. ``first_err`` is the narrowed error from the
    # rungs above, not necessarily the original one. See issue #23432.
    # Mirror the sync path: drop poisoned clients on connection/timeout so the next aux call rebuilds. See
    # issue #23432.
    if _is_connection_error(first_err):
        try:
            _evict_cached_client_instance(client)
        except Exception:
            logger.debug("Auxiliary%s: cache eviction after connection error failed",
                         tag, exc_info=True)
    # The narrowed error is the actionable one (e.g. a 404 "requires credits" from the
    # retry after a healed 401), so surface it rather than the original.
    raise first_err


def _elapsed_ms(started_at: float, now: Optional[float] = None) -> int:
    """Whole milliseconds since ``started_at`` (clamped at 0)."""
    return max(0, int(((time.monotonic() if now is None else now) - started_at) * 1000))


def _stamp_latency_once(latency_info: Optional[dict[str, int]], key: str, started_at: float) -> None:
    """Record ``key`` in ``latency_info`` the first time it fires."""
    if latency_info is not None and key not in latency_info:
        latency_info[key] = _elapsed_ms(started_at)


def call_llm(
    task: str | None = None, *, provider: str | None = None, model: str | None = None, base_url: str | None = None,
    api_key: str | None = None, main_runtime: Optional[dict[str, Any]] = None, messages: list,
    temperature: Optional[float] = None, max_tokens: int | None = None, tools: list | None = None,
    timeout: float | None = None, extra_body: dict | None = None, reasoning_config: Optional[dict] = None,
    extra_headers: Optional[dict[str, str]] = None, api_mode: str | None = None, stream: bool = False,
    stream_options: dict | None = None, route_info: Optional[dict[str, str]] = None,
    latency_info: Optional[dict[str, int]] = None,
) -> Any:
    """Run an auxiliary LLM request, applying the configured task limit."""
    from agent.auxiliary_client import (
        _acquire_sync_aux_semaphore,
        _aux_dispatch,
        _aux_progress,
        _aux_provider_response,
        _aux_thread_local_hook,
        aux_progress_hook,
        scoped_runtime_main,
    )

    queue_started_at = time.monotonic()
    semaphore = _acquire_sync_aux_semaphore(task)
    if semaphore is not None:
        semaphore.acquire()
    request_started_at = time.monotonic()
    if latency_info is not None:
        latency_info["queue_wait_ms"] = _elapsed_ms(queue_started_at, request_started_at)
    prior_progress_hook = getattr(_aux_progress, "hook", None)
    try:
        with (
            scoped_runtime_main(main_runtime),
            aux_progress_hook(
                prior_progress_hook
                if callable(prior_progress_hook)
                else ((lambda: None) if latency_info is not None else None)
            ),
            _aux_thread_local_hook(_aux_dispatch, functools.partial(
                _stamp_latency_once, latency_info, "provider_dispatch_ms", request_started_at)),
            _aux_thread_local_hook(_aux_provider_response, functools.partial(
                _stamp_latency_once, latency_info, "time_to_first_progress_ms", request_started_at)),
        ):
            response = _call_llm_impl(
                task=task, provider=provider, model=model, base_url=base_url, api_key=api_key,
                main_runtime=main_runtime, messages=messages, temperature=temperature,
                max_tokens=max_tokens, tools=tools, timeout=timeout, extra_body=extra_body,
                reasoning_config=reasoning_config, extra_headers=extra_headers, api_mode=api_mode,
                stream=stream, stream_options=stream_options, route_info=route_info,
            )
        if stream and semaphore is not None:
            stream_semaphore = semaphore
            semaphore = None
            return _release_sync_semaphore_after_stream(response, stream_semaphore)
        return response
    finally:
        if latency_info is not None:
            latency_info["summary_generation_ms"] = _elapsed_ms(request_started_at)
        if semaphore is not None:
            semaphore.release()


def _release_sync_semaphore_after_stream(stream: Any, semaphore: threading.BoundedSemaphore):
    """Release a permit only after a streaming response is consumed or closed."""
    try:
        yield from stream
    finally:
        try:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        finally:
            semaphore.release()


def _plan_aux_call(
    task: Optional[str], *, async_mode: bool, provider: Optional[str], model: Optional[str],
    base_url: Optional[str], api_key: Optional[str], main_runtime: Optional[dict[str, Any]],
    messages: list, temperature: Optional[float], max_tokens: Optional[int], tools: Optional[list],
    timeout: Optional[float], extra_body: Optional[dict], reasoning_config: Optional[dict],
    extra_headers: Optional[dict[str, str]], api_mode: Optional[str],
    route_info: Optional[dict[str, str]],
) -> tuple[_PreparedAuxRequest, dict[str, Any], dict[str, Any]]:
    """Shared head of both call impls: prepare the request and bundle the kwargs the recovery
    drivers pass to ``_retry_same_provider_*`` / ``_call_fallback_candidate_*``. One immutable
    runtime snapshot for keying/resolution/retries/fallbacks, so a concurrent /model switch
    can't mix key and client from different runtimes."""
    from agent.auxiliary_client import (
        _normalize_main_runtime,
    )

    main_runtime = _normalize_main_runtime(main_runtime)
    req = _prepare_aux_request(
        task, provider=provider, model=model, base_url=base_url, api_key=api_key,
        main_runtime=main_runtime, messages=messages, temperature=temperature,
        max_tokens=max_tokens, tools=tools, timeout=timeout, extra_body=extra_body,
        reasoning_config=reasoning_config, extra_headers=extra_headers,
        api_mode=api_mode, route_info=route_info, async_mode=async_mode,
    )
    candidate_kwargs = dict(
        task=task, messages=messages, temperature=temperature, max_tokens=max_tokens,
        tools=tools, effective_timeout=req.effective_timeout,
        effective_extra_body=req.effective_extra_body, reasoning_config=reasoning_config,
    )
    retry_kwargs = dict(
        candidate_kwargs, resolved_base_url=req.resolved_base_url,
        resolved_api_key=req.resolved_api_key, resolved_api_mode=req.resolved_api_mode,
        main_runtime=main_runtime, final_model=req.final_model, extra_headers=extra_headers,
    )
    return req, retry_kwargs, candidate_kwargs


def _should_retry_same_provider(task: Optional[str], exc: Exception, tag: str) -> bool:
    """True when ``exc`` is a transient transport blip worth a same-provider retry; critical-path
    tasks skip it on a full-budget timeout (``_should_skip_same_provider_retry``) and go straight
    to fallback."""
    from agent.auxiliary_client import (
        _is_transient_transport_error,
        _should_skip_same_provider_retry,
        logger,
    )

    if not _is_transient_transport_error(exc):
        return False
    if _should_skip_same_provider_retry(task, exc):
        logger.info("Auxiliary %s%s: timeout on the critical path; "
                    "skipping same-provider retry and falling back: %s", task, tag, exc)
        return False
    return True


def _ladder_step_call(
    step: _LadderStep, req: _PreparedAuxRequest, retry_kwargs: dict[str, Any], candidate_kwargs: dict[str, Any],
) -> tuple[str, tuple, dict[str, Any]]:
    """Resolve a ladder step into ``(kind, args, kwargs)`` for the sync/async performer."""
    if step.kind == "call":
        return "call", step.args, dict(provider=req.resolved_provider, api_mode=req.resolved_api_mode)
    if step.kind == "retry_same_provider":
        retry_provider, retry_model = step.args
        return "retry", (), dict(retry_kwargs, resolved_provider=retry_provider, resolved_model=retry_model)
    return "fallback", step.args, candidate_kwargs


def _start_recovery_ladder(
    first_err: Exception, req: _PreparedAuxRequest, retry_kwargs: dict[str, Any], *,
    task: Optional[str], async_mode: bool, route_info: Optional[dict[str, str]],
):
    """Build the recovery-ladder generator for a failed primary request."""
    return _aux_recovery_ladder(
        first_err, client=req.client, kwargs=req.kwargs, task=task, async_mode=async_mode,
        base_info=req.base_info, resolved_provider=req.resolved_provider,
        resolved_model=req.resolved_model, resolved_base_url=req.resolved_base_url,
        resolved_api_key=req.resolved_api_key, resolved_api_mode=req.resolved_api_mode,
        final_model=req.final_model, max_tokens=retry_kwargs["max_tokens"],
        main_runtime=retry_kwargs["main_runtime"], route_info=route_info)


def _call_llm_impl(
    task: str | None = None, *, provider: str | None = None, model: str | None = None, base_url: str | None = None,
    api_key: str | None = None, main_runtime: Optional[dict[str, Any]] = None, messages: list,
    temperature: Optional[float] = None, max_tokens: int | None = None, tools: list | None = None,
    timeout: float | None = None, extra_body: dict | None = None, reasoning_config: Optional[dict] = None,
    extra_headers: Optional[dict[str, str]] = None, api_mode: str | None = None, stream: bool = False,
    stream_options: dict | None = None, route_info: Optional[dict[str, str]] = None,
) -> Any:
    """Centralized synchronous LLM call: resolve provider/model, auth, kwargs, fallbacks.
    task: aux task whose provider:model comes from config (ignored if provider set); api_mode
    overrides task config; timeout=None reads auxiliary.{task}.timeout; extra_headers override
    client defaults. stream=True returns the raw SDK stream (caller consumes/falls back)
    instead of a validated response. RuntimeError if no provider is configured."""
    from agent.auxiliary_client import (
        CodexAuxiliaryClient,
        _LadderStep,
        _TRANSIENT_RETRY_BACKOFF_BASE,
        _call_fallback_candidate_sync,
        _create_with_progress,
        _drive_ladder,
        _is_transient_transport_error,
        _provider_requires_stream,
        _relay_sync_completion,
        _relay_sync_stream,
        _retry_same_provider_sync,
        _transient_retry_count,
        _validate_llm_response,
        logger,
    )

    req, retry_kwargs, candidate_kwargs = _plan_aux_call(
        task, async_mode=False, provider=provider, model=model, base_url=base_url,
        api_key=api_key, main_runtime=main_runtime, messages=messages,
        temperature=temperature, max_tokens=max_tokens, tools=tools, timeout=timeout,
        extra_body=extra_body, reasoning_config=reasoning_config,
        extra_headers=extra_headers, api_mode=api_mode, route_info=route_info,
    )
    client, kwargs, request_provider = req.client, req.kwargs, req.request_provider
    # Streaming path (MoA aggregator): return the raw SDK stream, skipping validation and
    # the fallback chain (they assume a complete response); the caller owns reassembly/fallback.
    if stream:
        kwargs["stream"] = True
        if stream_options:
            kwargs["stream_options"] = stream_options
        if task == "moa_aggregator" and isinstance(client, CodexAuxiliaryClient):
            # Responses-shim clients consume the stream internally and return a completed
            # object Relay's managed stream would iterate; the MoA facade wraps it as one chunk.
            return client.chat.completions.create(**kwargs)
        return _relay_sync_stream(client, kwargs, provider=request_provider, api_mode=req.resolved_api_mode)

    def _primary(**validate_kw: Any) -> Any:
        # Retry on the same provider for a transient transport blip (connection reset / streaming-close /
        # incomplete chunked read / 5xx / 408) before the except-chain below escalates to provider/model
        # fallback. A dropped connection shouldn't abandon an otherwise-healthy provider — this especially
        # matters for pinned auxiliary calls like MoA reference advisors, where "fallback to another
        # provider" is not a meaningful recovery (the advisor is a specific model), so a transient blip that
        # isn't retried simply loses that advisor for the turn (root of the run2 double-advisor "Connection
        # error" collapse — a genuine upstream blip hitting both parallel advisors at once). Attempts are
        # bounded and use exponential backoff. Count is configurable via auxiliary.transient_retries
        # (default 2 retries → 3 total attempts); a second/third failure or any non-transient error falls
        # through to ``first_err`` and the existing fallback handling unchanged. Unified home for the
        # transient retry every auxiliary task shares. (PR #16587)
        return _validate_llm_response(
            _relay_sync_completion(
                client, kwargs, provider=request_provider, api_mode=req.resolved_api_mode,
                create=lambda request: _create_with_progress(
                    client, request, task,
                    force_stream=_provider_requires_stream(
                        request_provider, req.base_info or req.resolved_base_url),
                ),
            ),
            task, **validate_kw,
        )
    try:
        # Bounded same-provider retry (exponential backoff, auxiliary.transient_retries) for
        # transient blips before escalating to fallback — a dropped connection shouldn't
        # abandon a healthy provider (matters for pinned MoA advisors).
        try:
            return _primary(provider=request_provider, base_url=req.base_info)
        except Exception as transient_err:
            if not _should_retry_same_provider(task, transient_err, ""):
                raise
            _max_transient_retries = _transient_retry_count()
            _last_transient = transient_err
            for _attempt in range(1, _max_transient_retries + 1):
                _backoff = min(_TRANSIENT_RETRY_BACKOFF_BASE * (2.0 ** (_attempt - 1)), 8.0)
                logger.info("Auxiliary %s: transient transport error (attempt %d/%d); "
                            "retrying same provider after %.1fs before fallback: %s",
                            task or "call", _attempt, _max_transient_retries, _backoff, _last_transient)
                time.sleep(_backoff)
                try:
                    return _primary()
                except Exception as retry_transient:
                    if not _is_transient_transport_error(retry_transient):
                        raise
                    _last_transient = retry_transient
            raise _last_transient
    except Exception as first_err:
        def _perform(step: _LadderStep) -> Any:
            kind, args, kw = _ladder_step_call(step, req, retry_kwargs, candidate_kwargs)
            if kind == "call":
                return _validate_llm_response(_relay_sync_completion(*args, **kw), task)
            if kind == "retry":
                return _retry_same_provider_sync(**kw)
            return _call_fallback_candidate_sync(*args, **kw)
        return _drive_ladder(
            _start_recovery_ladder(first_err, req, retry_kwargs, task=task, async_mode=False, route_info=route_info),
            _perform)


async def async_call_llm(
    task: str | None = None, *, provider: str | None = None, model: str | None = None, base_url: str | None = None,
    api_key: str | None = None, main_runtime: Optional[dict[str, Any]] = None, messages: list,
    temperature: Optional[float] = None, max_tokens: int | None = None, tools: list | None = None,
    timeout: float | None = None, extra_body: dict | None = None, reasoning_config: Optional[dict] = None,
    route_info: Optional[dict[str, str]] = None,
) -> Any:
    """Run an asynchronous auxiliary LLM request under the configured limit."""
    from agent.auxiliary_client import (
        _acquire_async_aux_semaphore,
        scoped_runtime_main,
    )

    semaphore = _acquire_async_aux_semaphore(task)
    if semaphore is not None:
        await semaphore.acquire()
    try:
        with scoped_runtime_main(main_runtime):
            return await _async_call_llm_impl(
                task=task, provider=provider, model=model, base_url=base_url, api_key=api_key,
                main_runtime=main_runtime, messages=messages, temperature=temperature,
                max_tokens=max_tokens, tools=tools, timeout=timeout, extra_body=extra_body,
                reasoning_config=reasoning_config, route_info=route_info,
            )
    finally:
        if semaphore is not None:
            semaphore.release()


async def _async_call_llm_impl(
    task: str | None = None, *, provider: str | None = None, model: str | None = None, base_url: str | None = None,
    api_key: str | None = None, main_runtime: Optional[dict[str, Any]] = None, messages: list,
    temperature: Optional[float] = None, max_tokens: int | None = None, tools: list | None = None,
    timeout: float | None = None, extra_body: dict | None = None, reasoning_config: Optional[dict] = None,
    route_info: Optional[dict[str, str]] = None,
) -> Any:
    """Centralized asynchronous LLM call; see call_llm() for full documentation.
    No per-request header / api_mode override on the async entry point."""
    from agent.auxiliary_client import (
        _LadderStep,
        _acreate_with_progress,
        _call_fallback_candidate_async,
        _drive_ladder_async,
        _provider_requires_stream,
        _relay_async_completion,
        _retry_same_provider_async,
        _to_async_client,
        _validate_llm_response,
        logger,
    )

    req, retry_kwargs, candidate_kwargs = _plan_aux_call(
        task, async_mode=True, provider=provider, model=model, base_url=base_url,
        api_key=api_key, main_runtime=main_runtime, messages=messages,
        temperature=temperature, max_tokens=max_tokens, tools=tools, timeout=timeout,
        extra_body=extra_body, reasoning_config=reasoning_config,
        extra_headers=None, api_mode=None, route_info=route_info,
    )
    client, kwargs, request_provider = req.client, req.kwargs, req.request_provider
    try:
        # Retry ONCE on the same provider for a transient blip before fallback (see call_llm()).
        # (PR #16587)
        _force_stream_async = _provider_requires_stream(request_provider, req.base_info or req.resolved_base_url)

        async def _acreate(_kwargs: dict[str, Any]) -> Any:
            return await _acreate_with_progress(client, _kwargs, task, force_stream=_force_stream_async)

        async def _primary(**validate_kw: Any) -> Any:
            return _validate_llm_response(
                await _relay_async_completion(
                    client, kwargs, provider=request_provider, api_mode=req.resolved_api_mode,
                    create=_acreate),
                task, **validate_kw)
        try:
            return await _primary(provider=request_provider, base_url=req.base_info)
        except Exception as transient_err:
            # The async Codex adapter wraps the sync stream via to_thread: same TimeoutError here.
            if not _should_retry_same_provider(task, transient_err, " (async)"):
                raise
            logger.info("Auxiliary %s (async): transient transport error; retrying "
                        "once on the same provider before fallback: %s", task or "call", transient_err)
            return await _primary()
    except Exception as first_err:
        async def _perform(step: _LadderStep) -> Any:
            kind, args, kw = _ladder_step_call(step, req, retry_kwargs, candidate_kwargs)
            if kind == "call":
                return _validate_llm_response(await _relay_async_completion(*args, **kw), task)
            if kind == "retry":
                return await _retry_same_provider_async(**kw)
            fb_client, fb_model, fb_label = args
            fb_client, _ = _to_async_client(fb_client, fb_model or "", is_vision=(task == "vision"))
            return await _call_fallback_candidate_async(fb_client, fb_model, fb_label, **kw)
        return await _drive_ladder_async(
            _start_recovery_ladder(first_err, req, retry_kwargs, task=task, async_mode=True, route_info=route_info),
            _perform)

