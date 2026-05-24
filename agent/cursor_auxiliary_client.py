"""OpenAI-client shim that routes auxiliary LLM calls through the Cursor SDK.

When the main provider is ``cursor`` (``cursor://sdk`` / ``cursor_sdk_runtime``),
auxiliary tasks (kanban specify/decompose, title generation, compression, etc.)
must not use an OpenAI HTTP client — that URL is not a real endpoint and fails
with ``APIConnectionError``.  This module exposes ``client.chat.completions.create()``
via ``Agent.prompt()`` instead.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger(__name__)

CURSOR_AUX_BASE_URL = "cursor://sdk"
_DEFAULT_AUX_TIMEOUT = 120.0

_sdk_client: Any = None
_sdk_clients: dict[str, Any] = {}
_shared_bridge: Any = None
_shared_bridge_endpoint: Any = None
_sdk_client_lock = threading.Lock()
_bridge_launch_lock = threading.Lock()
_active_auxiliary_ops = 0
_active_auxiliary_ops_lock = threading.Lock()

# Cursor specify/decompose agents may tool-loop in the repo; HTTP timeouts are too tight.
_CURSOR_KANBAN_AUX_TIMEOUT_FLOOR = max(
    120.0,
    float(os.getenv("HERMES_KANBAN_CURSOR_AUX_TIMEOUT", "600")),
)


def _emit_aux_progress(text: str) -> None:
    if not text:
        return
    try:
        from hermes_cli.kanban_worker_log import get_task_worker_log_sink

        sink = get_task_worker_log_sink()
        if sink is not None:
            sink(text)
            return
    except ImportError:
        pass
    logger.info(text.rstrip())


def _ensure_shared_bridge(*, cwd: Optional[str] = None) -> Any:
    """Launch or reuse the process-wide cursor-sdk-bridge subprocess."""
    global _shared_bridge, _shared_bridge_endpoint

    bridge_url = os.environ.get("CURSOR_SDK_BRIDGE_URL", "").strip()
    bridge_token = (
        os.environ.get("CURSOR_SDK_BRIDGE_TOKEN")
        or os.environ.get("CURSOR_SDK_BRIDGE_AUTH_TOKEN")
        or ""
    ).strip()
    if bridge_url and bridge_token:
        from cursor_sdk import Client

        return Client(
            base_url=bridge_url,
            auth_token=bridge_token,
            allow_api_key_env_fallback=True,
        )

    with _sdk_client_lock:
        if _shared_bridge_endpoint is not None:
            return _shared_bridge_endpoint

    with _bridge_launch_lock:
        with _sdk_client_lock:
            if _shared_bridge_endpoint is not None:
                return _shared_bridge_endpoint

        from agent.transports.cursor_sdk_session import (
            _bridge_launch_needs_workaround,
            _launch_bridge_threaded,
            preflight_cursor_sdk,
        )

        _emit_aux_progress("Starting Cursor bridge…\n")
        preflight_cursor_sdk()
        workspace = cwd or _aux_cursor_cwd()

        if _bridge_launch_needs_workaround():
            bridge, _process = _launch_bridge_threaded(workspace)
        else:
            from cursor_sdk._client import Client as SdkClient

            launched = SdkClient.launch_bridge(
                workspace=workspace,
                allow_api_key_env_fallback=True,
            )
            bridge = getattr(launched, "_owned_bridge", None)
            if bridge is None:
                # External bridge config — launched client owns transport only.
                return launched

        with _sdk_client_lock:
            _shared_bridge = bridge
            _shared_bridge_endpoint = bridge.endpoint
        logger.info(
            "cursor auxiliary bridge ready url=%s pid=%s",
            getattr(_shared_bridge_endpoint, "url", "?"),
            getattr(getattr(_shared_bridge, "process", None), "pid", "?"),
        )
        return _shared_bridge_endpoint


def _client_from_shared_bridge(*, cwd: Optional[str] = None) -> Any:
    """New SDK Client transport to the shared bridge (does not own the subprocess)."""
    from cursor_sdk import Client

    bridge_or_client = _ensure_shared_bridge(cwd=cwd)
    if hasattr(bridge_or_client, "agents"):
        return bridge_or_client
    return Client(
        bridge_or_client,
        allow_api_key_env_fallback=True,
    )


def _create_cursor_sdk_client(*, cwd: Optional[str] = None) -> Any:
    """Create a Cursor SDK ``Client`` connected to the shared bridge."""
    return _client_from_shared_bridge(cwd=cwd)


def get_cursor_sdk_client(
    *,
    cwd: Optional[str] = None,
    kanban_isolation_key: Optional[str] = None,
) -> Any:
    """Return a Cursor SDK ``Client``.

    Kanban specify/decompose pass ``kanban_isolation_key`` (task id) so
    concurrent cards get separate transports to the same bridge subprocess.
    Non-kanban auxiliary calls reuse the process-global client.
    """
    if kanban_isolation_key:
        key = str(kanban_isolation_key).strip()
        if not key:
            raise ValueError("kanban_isolation_key must be non-empty")
        with _sdk_client_lock:
            existing = _sdk_clients.get(key)
            if existing is not None:
                return existing
        client = _client_from_shared_bridge(cwd=cwd)
        with _sdk_client_lock:
            return _sdk_clients.setdefault(key, client)

    global _sdk_client
    with _sdk_client_lock:
        if _sdk_client is not None:
            return _sdk_client
    client = _client_from_shared_bridge(cwd=cwd)
    with _sdk_client_lock:
        if _sdk_client is None:
            _sdk_client = client
        return _sdk_client


def release_cursor_sdk_client(kanban_isolation_key: str) -> None:
    """Drop a per-card SDK client and close its transport (not the shared bridge)."""
    key = str(kanban_isolation_key or "").strip()
    if not key:
        return
    with _sdk_client_lock:
        client = _sdk_clients.pop(key, None)
    if client is None or client is _sdk_client:
        return
    try:
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            close_fn()
    except Exception:
        pass


def effective_cursor_auxiliary_timeout(requested: float) -> float:
    """Apply the Cursor kanban auxiliary floor so tool loops can finish."""
    try:
        return max(float(requested), _CURSOR_KANBAN_AUX_TIMEOUT_FLOOR)
    except (TypeError, ValueError):
        return _CURSOR_KANBAN_AUX_TIMEOUT_FLOOR


def _aux_cursor_cwd() -> str:
    env = os.environ.get("HERMES_CURSOR_AUX_CWD", "").strip()
    if env:
        return env
    try:
        cwd = os.getcwd()
        if cwd:
            return cwd
    except Exception:
        pass
    try:
        from hermes_cli.config import get_hermes_home

        return str(get_hermes_home())
    except Exception:
        return os.path.expanduser("~")


def _messages_to_prompt(messages: list) -> str:
    """Flatten chat messages into a single user prompt for ``Agent.prompt``."""
    parts: list[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user").strip().lower()
        content = msg.get("content", "")
        if isinstance(content, list):
            text_bits: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = str(block.get("text") or "").strip()
                    if text:
                        text_bits.append(text)
            content = "\n".join(text_bits)
        else:
            content = str(content or "").strip()
        if not content:
            continue
        if role == "system":
            parts.append(content)
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}")
        else:
            parts.append(content)
    return "\n\n".join(parts).strip()


def _run_result_to_openai_response(result: Any, *, model: str) -> Any:
    status = str(getattr(result, "status", "") or "")
    text = str(getattr(result, "result", "") or "")
    if status == "error":
        raise RuntimeError(text or "Cursor SDK run failed")
    assistant_message = SimpleNamespace(content=text, tool_calls=None, reasoning=None)
    choice = SimpleNamespace(
        index=0,
        message=assistant_message,
        finish_reason="stop",
    )
    return SimpleNamespace(choices=[choice], model=model, usage=None)


def _run_cursor_prompt_streaming(
    *,
    prompt: str,
    options: Any,
    client: Any,
    timeout: float,
    progress_emit: Callable[[str], None],
) -> Any:
    from cursor_sdk import Agent
    from hermes_cli.kanban_worker_log import CursorStreamLogger

    def _worker() -> Any:
        with Agent.create(options, client=client) as agent:
            run = agent.send(prompt)
            stream_logger = CursorStreamLogger(progress_emit)
            for message in run.messages():
                stream_logger.handle(message)
            return run.wait()

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cursor-aux")
    future = pool.submit(_worker)
    try:
        return future.result(timeout=max(1.0, timeout))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _cursor_api_key(explicit: Optional[str] = None) -> str:
    """Resolve CURSOR_API_KEY for auxiliary calls.

    Long-lived gateway/dashboard processes may have an empty
    ``os.environ["CURSOR_API_KEY"]`` placeholder that blocks
    :func:`hermes_cli.config.get_env_value` from reading ``~/.hermes/.env``.
    Read the on-disk .env directly when the live env var is blank.
    """
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    env_val = os.environ.get("CURSOR_API_KEY", "")
    if env_val and env_val.strip():
        return env_val.strip()
    try:
        from hermes_cli.config import load_env

        file_val = (load_env().get("CURSOR_API_KEY") or "").strip()
        if file_val:
            return file_val
    except Exception:
        pass
    try:
        from hermes_cli.auth import resolve_api_key_provider_credentials

        creds = resolve_api_key_provider_credentials("cursor")
        return str(creds.get("api_key") or "").strip()
    except Exception:
        return ""


def reset_cursor_sdk_client() -> None:
    """Drop SDK clients and the shared bridge subprocess."""
    global _sdk_client, _shared_bridge, _shared_bridge_endpoint
    with _sdk_client_lock:
        per_card = list(_sdk_clients.values())
        _sdk_clients.clear()
        client = _sdk_client
        _sdk_client = None
    for extra in per_card:
        if extra is client:
            continue
        try:
            close_fn = getattr(extra, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass
    if client is not None:
        try:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass
    with _bridge_launch_lock:
        bridge = _shared_bridge
        _shared_bridge = None
        _shared_bridge_endpoint = None
    if bridge is not None:
        try:
            bridge.close()
        except Exception:
            pass


@contextmanager
def auxiliary_operation_in_flight() -> Iterator[None]:
    """Track concurrent kanban specify/decompose auxiliary LLM calls.

    While more than one call is active, ``prepare_cursor_auxiliary_credentials``
    reloads credentials without closing shared SDK/HTTP clients that an
    in-flight call may still be using.
    """
    global _active_auxiliary_ops
    with _active_auxiliary_ops_lock:
        _active_auxiliary_ops += 1
    try:
        yield
    finally:
        with _active_auxiliary_ops_lock:
            _active_auxiliary_ops -= 1


def prepare_cursor_auxiliary_credentials(
    *,
    force_reset: bool = False,
    reload_only: bool = False,
) -> None:
    """Reload Cursor credentials and optionally reset SDK state.

    ``reload_only=True`` refreshes ``~/.hermes/.env`` into the process without
    closing any clients — used by per-card kanban auxiliary ops that own
    their own client for the duration of the call.
    """
    try:
        from hermes_cli.config import get_hermes_home, invalidate_env_cache
        from hermes_cli.env_loader import load_hermes_dotenv

        load_hermes_dotenv(hermes_home=get_hermes_home())
        invalidate_env_cache()
    except Exception:
        pass
    if reload_only:
        return
    with _active_auxiliary_ops_lock:
        skip_reset = (not force_reset) and _active_auxiliary_ops > 1
    if skip_reset:
        return
    reset_cursor_sdk_client()
    try:
        from agent import auxiliary_client as aux

        evict_cached_auxiliary_clients = aux.evict_cached_auxiliary_clients
        evict_cached_auxiliary_clients(
            lambda client: type(client).__name__ == "CursorAuxiliaryClient"
        )
    except Exception:
        pass


class _CursorCompletionsAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        cwd: Optional[str] = None,
        kanban_isolation_key: Optional[str] = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._cwd = cwd or _aux_cursor_cwd()
        self._kanban_isolation_key = kanban_isolation_key

    def create(self, **kwargs: Any) -> Any:
        from cursor_sdk import Agent, AgentOptions, CursorAgentError, LocalAgentOptions
        from agent.transports.cursor_sdk_session import build_cursor_model_selection

        messages = kwargs.get("messages") or []
        model = str(kwargs.get("model") or self._model or "composer-2.5")
        timeout = effective_cursor_auxiliary_timeout(
            float(kwargs.get("timeout") or _DEFAULT_AUX_TIMEOUT)
        )
        prompt = _messages_to_prompt(messages)
        if not prompt:
            raise ValueError("Cursor auxiliary call requires at least one message")

        api_key = _cursor_api_key(self._api_key)
        if not api_key:
            raise CursorAgentError(
                "CURSOR_API_KEY is not set. Add it to ~/.hermes/.env or export it.",
                is_retryable=False,
            )
        # Keep the live env in sync so cursor-sdk bridge fallback matches AgentOptions.
        os.environ["CURSOR_API_KEY"] = api_key

        selection = build_cursor_model_selection(model)
        options = AgentOptions(
            model=selection,
            api_key=api_key,
            local=LocalAgentOptions(cwd=self._cwd, setting_sources=[]),
        )

        try:
            from hermes_cli.kanban_worker_log import (
                CursorStreamLogger,
                get_task_worker_log_sink,
            )

            progress_emit = get_task_worker_log_sink()
        except ImportError:
            progress_emit = None

        client = get_cursor_sdk_client(
            cwd=self._cwd,
            kanban_isolation_key=self._kanban_isolation_key,
        )

        def _prompt() -> Any:
            if progress_emit is not None:
                progress_emit("Starting Cursor agent…\n")
                return _run_cursor_prompt_streaming(
                    prompt=prompt,
                    options=options,
                    client=client,
                    timeout=timeout,
                    progress_emit=progress_emit,
                )
            from cursor_sdk import Agent

            return Agent.prompt(prompt, options, client=client)

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cursor-aux")
        future = pool.submit(_prompt)
        try:
            result = future.result(timeout=max(1.0, timeout))
        except FuturesTimeoutError as exc:
            raise TimeoutError(
                f"Cursor SDK auxiliary call timed out after {int(timeout)}s"
            ) from exc
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        return _run_result_to_openai_response(result, model=model)


class _CursorChatShim:
    def __init__(self, adapter: _CursorCompletionsAdapter) -> None:
        self.completions = adapter


class CursorAuxiliaryClient:
    """OpenAI-compatible wrapper over ``Agent.prompt`` for side tasks."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str = "",
        cwd: Optional[str] = None,
        kanban_isolation_key: Optional[str] = None,
    ) -> None:
        self._model = model
        self._kanban_isolation_key = kanban_isolation_key
        adapter = _CursorCompletionsAdapter(
            model=model,
            api_key=api_key,
            cwd=cwd,
            kanban_isolation_key=kanban_isolation_key,
        )
        self.chat = _CursorChatShim(adapter)
        self.api_key = api_key or _cursor_api_key()
        self.base_url = CURSOR_AUX_BASE_URL
        self._real_client = None


class _AsyncCursorCompletionsAdapter:
    def __init__(self, sync_adapter: _CursorCompletionsAdapter) -> None:
        self._sync = sync_adapter

    async def create(self, **kwargs: Any) -> Any:
        import asyncio

        return await asyncio.to_thread(self._sync.create, **kwargs)


class _AsyncCursorChatShim:
    def __init__(self, adapter: _AsyncCursorCompletionsAdapter) -> None:
        self.completions = adapter


class AsyncCursorAuxiliaryClient:
    def __init__(self, sync_wrapper: CursorAuxiliaryClient) -> None:
        sync_adapter = sync_wrapper.chat.completions
        self.chat = _AsyncCursorChatShim(_AsyncCursorCompletionsAdapter(sync_adapter))
        self.api_key = sync_wrapper.api_key
        self.base_url = sync_wrapper.base_url
        self._real_client = sync_wrapper._real_client


def build_cursor_auxiliary_client(
    model: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    kanban_isolation_key: Optional[str] = None,
) -> tuple[Optional[CursorAuxiliaryClient], Optional[str]]:
    """Build a Cursor SDK auxiliary client when ``CURSOR_API_KEY`` is available."""
    resolved_model = (model or "").strip() or "composer-2.5"
    key = _cursor_api_key(api_key)
    if not key:
        logger.warning(
            "Auxiliary client: cursor requested but CURSOR_API_KEY is not set"
        )
        return None, None
    logger.debug("Auxiliary client: Cursor SDK (%s)", resolved_model)
    return (
        CursorAuxiliaryClient(
            model=resolved_model,
            api_key=key,
            kanban_isolation_key=kanban_isolation_key,
        ),
        resolved_model,
    )
