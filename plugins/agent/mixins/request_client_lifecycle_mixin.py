"""Per-request wire-client lifecycle helpers (run_agent.py shard s4, c1+c2).

Extracted verbatim from run_agent.py (wave 1, shard s4, clusters c1+c2,
10 methods).  Method bodies are character-for-character copies; only this
header and the import block are new.  ``logger`` is bound to the same
logger name as run_agent's module logger so log records keep their origin.

Class attributes referenced via ``self.``/``cls.`` stay on ``AIAgent`` and
resolve through the MRO: ``_REQUEST_CLIENT_REUSE_REASONS`` (declared on
AIAgent), ``_openai_client_lock`` / ``_create_openai_client`` /
``_close_openai_client`` / ``_is_openai_client_closed`` /
``_force_close_tcp_sockets`` / ``_client_log_context`` /
``_ensure_primary_openai_client`` (shard s3), and
``_try_refresh_anthropic_client_credentials`` (credential_refresh cluster).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from hermes_cli.timeouts import get_provider_request_timeout
from utils import base_url_host_matches

logger = logging.getLogger("run_agent")


class RequestClientLifecycleMixin:
    @staticmethod
    def _api_kwargs_have_image_parts(api_kwargs: dict) -> bool:
        """Return True when the outbound request still contains native image parts."""
        if not isinstance(api_kwargs, dict):
            return False
        candidates = []
        messages = api_kwargs.get("messages")
        if isinstance(messages, list):
            candidates.extend(messages)
        # Responses API payloads use `input`; after conversion, image parts can
        # still be present there instead of in `messages`.
        response_input = api_kwargs.get("input")
        if isinstance(response_input, list):
            candidates.extend(response_input)

        def _contains_image(value: Any) -> bool:
            if isinstance(value, dict):
                ptype = value.get("type")
                if ptype in {"image_url", "input_image"}:
                    return True
                return any(_contains_image(v) for v in value.values())
            if isinstance(value, list):
                return any(_contains_image(v) for v in value)
            return False

        return any(_contains_image(item) for item in candidates)

    def _copilot_headers_for_request(self, *, is_vision: bool) -> dict:
        from hermes_cli.copilot_auth import copilot_request_headers

        return copilot_request_headers(is_agent_turn=True, is_vision=is_vision)

    def _request_client_cache_ref(self) -> dict:
        # Lazy init — tests build agents via AIAgent.__new__ without __init__.
        cache = getattr(self, "_request_client_cache", None)
        if cache is None:
            cache = {"client": None, "kwargs": None, "poisoned": False, "in_use": False}
            self._request_client_cache = cache
        return cache

    def _create_request_openai_client(self, *, reason: str, api_kwargs: Optional[dict] = None) -> Any:
        from unittest.mock import Mock

        primary_client = self._ensure_primary_openai_client(reason=reason)
        if self.provider == "moa":
            return primary_client
        if isinstance(primary_client, Mock):
            return primary_client
        with self._openai_client_lock():
            request_kwargs = dict(self._client_kwargs)
        # Per-request OpenAI-wire clients (used by both the non-streaming
        # chat-completions path and the streaming chat-completions path
        # in `_interruptible_api_call`) should not run the SDK's built-in
        # retry loop: the agent's outer loop owns retries with credential
        # rotation, provider fallback, and backoff that the SDK can't
        # see. Leaving SDK retries on (default 2) compounds with our outer
        # retries and lets a single hung provider request stretch to ~3x
        # the per-call timeout before our stale detector reports it.
        # Shared/primary clients and Anthropic / Bedrock paths are
        # unaffected (they don't go through here).
        request_kwargs["max_retries"] = 0
        if (
            base_url_host_matches(str(request_kwargs.get("base_url", "")), "githubcopilot.com")
            and self._api_kwargs_have_image_parts(api_kwargs or {})
        ):
            request_kwargs["default_headers"] = self._copilot_headers_for_request(is_vision=True)
        # Reuse the cached wire client while the effective kwargs are
        # unchanged: constructing openai.OpenAI + its httpx pool costs
        # ~19-35ms per LLM call (fresh TCP+TLS handshake), ~5x per turn.
        # The cache is a single checked-out slot: `in_use` prevents two
        # concurrent calls from sharing one pool's close/abort lifecycle
        # (a second concurrent call gets a fresh untracked client with
        # the old build-per-request behavior).
        stale = None
        with self._openai_client_lock():
            cache = self._request_client_cache_ref()
            cached = cache["client"]
            if cached is not None and not cache["in_use"]:
                if (
                    not cache["poisoned"]
                    and cache["kwargs"] == request_kwargs
                    and not self._is_openai_client_closed(cached)
                ):
                    cache["in_use"] = True
                    return cached
                # kwargs changed (credential rotation, provider failover),
                # poisoned by a cross-thread abort (#29507), or externally
                # closed — never reuse; discard and rebuild below.
                stale = cached
                cache["client"] = None
                cache["kwargs"] = None
                cache["poisoned"] = False
        if stale is not None:
            # Safe to close from this thread: in_use was False, so no
            # worker thread owns the pool's FDs (#29507 concerns clients
            # with an in-flight request on another thread).
            self._close_openai_client(stale, reason=f"reuse_evict:{reason}", shared=False)
        client = self._create_openai_client(request_kwargs, reason=reason, shared=False)
        with self._openai_client_lock():
            cache = self._request_client_cache_ref()
            if cache["client"] is None:
                cache["client"] = client
                # Snapshot nested dicts (default_headers): rotation sites
                # assign fresh inner dicts today, but an aliased inner
                # object would compare equal even after in-place mutation.
                cache["kwargs"] = {
                    k: dict(v) if isinstance(v, dict) else v
                    for k, v in request_kwargs.items()
                }
                cache["poisoned"] = False
                cache["in_use"] = True
            # else: a concurrent call holds the slot — hand this client
            # out untracked; _close_request_openai_client fully closes
            # untracked clients, preserving the per-request lifecycle.
        return client

    def _close_request_openai_client(self, client: Any, *, reason: str) -> None:
        with self._openai_client_lock():
            cache = self._request_client_cache_ref()
            if cache["client"] is client:
                if reason in self._REQUEST_CLIENT_REUSE_REASONS and not cache["poisoned"]:
                    # Clean finish on the owning thread — keep the wire client
                    # (and its warm httpx pool) for the next sequential call.
                    cache["in_use"] = False
                    return
                # Failure / kill / abort outcome: drop the slot and fall
                # through to a real close. This runs on the owning worker
                # thread, which is where the FD release belongs (#29507).
                cache["client"] = None
                cache["kwargs"] = None
                cache["poisoned"] = False
                cache["in_use"] = False
        self._close_openai_client(client, reason=reason, shared=False)

    def _close_cached_request_openai_client(self, *, reason: str) -> None:
        """Teardown hook: really close the cached per-request wire client."""
        with self._openai_client_lock():
            cache = getattr(self, "_request_client_cache", None)
            client = cache["client"] if cache else None
            in_use = bool(cache["in_use"]) if cache else False
            if cache is not None:
                cache["client"] = None
                cache["kwargs"] = None
                cache["poisoned"] = False
                cache["in_use"] = False
        if client is None:
            return
        if in_use:
            # A worker thread has this client checked out for an in-flight
            # request (workers can outlive turns — see interruptible_api_call).
            # client.close() here would release its FDs from a stranger thread,
            # the #29507 race teardown must not reintroduce. Abort the sockets
            # instead; the slot is already cleared, so the worker's own finally
            # sees an untracked client and does the real close on its thread.
            self._abort_request_openai_client(client, reason=f"{reason}_in_flight")
            return
        self._close_openai_client(client, reason=reason, shared=False)

    def _abort_request_openai_client(self, client: Any, *, reason: str) -> None:
        """Cross-thread abort: shut sockets down without releasing FDs.

        Companion to :meth:`_close_request_openai_client` for stranger-thread
        callers (interrupt-check loop, stale-call detector). Calling
        ``client.close()`` from a thread that does not own the active httpx
        connection raced the still-live SSL BIO and corrupted unrelated file
        descriptors when the kernel recycled the just-freed TCP FD (#29507).

        Here we only ``shutdown(SHUT_RDWR)`` the pool's sockets. That unblocks
        the owning worker thread's pending ``recv``/``send`` with an EOF or
        ``EPIPE`` so it can unwind and close ``client`` from its own context
        — which is where the FD release belongs.
        """
        if client is None:
            return
        # A pool whose sockets were shut down from a stranger thread must
        # never be reused: poison the cache slot so the owner-thread close
        # discards it and the next create builds a fresh client.
        with self._openai_client_lock():
            cache = self._request_client_cache_ref()
            if cache["client"] is client:
                cache["poisoned"] = True
        try:
            shutdown_count = self._force_close_tcp_sockets(client)
            # tcp_force_closed=0 means the stranger-thread abort found no
            # sockets to shut down — the worker stays blocked in recv and the
            # provider keeps the slot (#72975). Surface that as WARNING so it
            # cannot be mistaken for a successful abort in the logs.
            _log = logger.warning if shutdown_count == 0 else logger.info
            _log(
                "OpenAI client aborted (%s, shared=False, tcp_force_closed=%d, "
                "deferred_close=stranger_thread) %s%s",
                reason,
                shutdown_count,
                self._client_log_context(),
                (
                    " — no sockets found; in-flight request may keep running "
                    "until the provider finishes"
                    if shutdown_count == 0
                    else ""
                ),
            )
        except Exception as exc:
            logger.debug(
                "OpenAI client abort failed (%s, shared=False) %s error=%s",
                reason,
                self._client_log_context(),
                exc,
            )

    def _create_request_anthropic_client(self, *, reason: str) -> Any:
        """Build a request-local Anthropic client for one in-flight call.

        The shared ``_anthropic_client`` stays the long-lived primary, but the
        stale/interrupt watchdog runs on the poll thread and must never call
        ``close()`` on the client whose TLS socket a worker thread is still
        reading: releasing that FD from a stranger thread lets the kernel
        recycle it under a still-live SSL BIO, which then writes a TLS record
        into an unrelated SQLite header (#29507 / #67142). A per-request client
        lets the stranger thread ``shutdown()`` the socket while the owning
        worker performs the SDK-level close from its own context — the same
        ownership contract the OpenAI-wire path already uses.

        Mirrors ``_rebuild_anthropic_client`` construction (direct + Bedrock,
        1M-beta drop) but returns a fresh client instead of swapping the shared
        one.
        """
        if self.api_mode == "anthropic_messages":
            self._try_refresh_anthropic_client_credentials()
        _drop_1m = bool(getattr(self, "_oauth_1m_beta_disabled", False))
        if getattr(self, "provider", None) == "bedrock":
            from agent.anthropic_adapter import build_anthropic_bedrock_client
            region = getattr(self, "_bedrock_region", "us-east-1") or "us-east-1"
            client = build_anthropic_bedrock_client(region)
        else:
            from agent.anthropic_adapter import build_anthropic_client
            client = build_anthropic_client(
                self._anthropic_api_key,
                getattr(self, "_anthropic_base_url", None),
                timeout=get_provider_request_timeout(self.provider, self.model),
                drop_context_1m_beta=_drop_1m,
            )
        logger.debug(
            "Anthropic request client created (%s, shared=False) provider=%s model=%s",
            reason,
            getattr(self, "provider", None),
            getattr(self, "model", None),
        )
        return client

    def _close_request_anthropic_client(self, client: Any, *, reason: str) -> None:
        """Owner-thread full close of a request-local Anthropic client.

        Force-closes the pool's TCP sockets first (CLOSE-WAIT hygiene, parity
        with ``_close_openai_client``), then does the graceful SDK close. Safe
        because the caller owns the connection.
        """
        if client is None:
            return
        try:
            self._force_close_tcp_sockets(client)
            client.close()
            logger.info(
                "Anthropic client closed (%s, shared=False) provider=%s model=%s",
                reason,
                getattr(self, "provider", None),
                getattr(self, "model", None),
            )
        except Exception as exc:
            logger.debug(
                "Anthropic client close failed (%s, shared=False) provider=%s model=%s error=%s",
                reason,
                getattr(self, "provider", None),
                getattr(self, "model", None),
                exc,
            )

    def _abort_request_anthropic_client(self, client: Any, *, reason: str) -> None:
        """Cross-thread abort for request-local Anthropic clients.

        Stranger threads (the interrupt-check / stale-stream detector loop)
        must not call the SDK ``close()`` — that races the owning worker's live
        SSL BIO and can recycle a TLS FD into a SQLite header (#29507 /
        #67142). Only ``shutdown(SHUT_RDWR)`` the pool's sockets so the worker
        unblocks and releases the FD from its own thread.
        """
        if client is None:
            return
        try:
            shutdown_count = self._force_close_tcp_sockets(client)
            # Same visibility contract as the OpenAI abort path (#72975):
            # zero sockets shut down means the abort did not unblock the
            # worker — log WARNING, not a success-shaped INFO.
            _log = logger.warning if shutdown_count == 0 else logger.info
            _log(
                "Anthropic client aborted (%s, shared=False, tcp_force_closed=%d, "
                "deferred_close=stranger_thread) provider=%s model=%s%s",
                reason,
                shutdown_count,
                getattr(self, "provider", None),
                getattr(self, "model", None),
                (
                    " — no sockets found; in-flight request may keep running "
                    "until the provider finishes"
                    if shutdown_count == 0
                    else ""
                ),
            )
        except Exception as exc:
            logger.debug(
                "Anthropic client abort failed (%s, shared=False) provider=%s model=%s error=%s",
                reason,
                getattr(self, "provider", None),
                getattr(self, "model", None),
                exc,
            )
