"""Session adapter for the Cursor SDK runtime.

Owns one ``cursor_sdk.Agent`` per Hermes session. Drives ``agent.send()``,
consumes ``run.messages()``, projects events via :class:`CursorEventProjector`,
and returns a turn result for :func:`agent.cursor_runtime.run_cursor_sdk_turn`.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from agent.transports.cursor_event_projector import CursorEventProjector

logger = logging.getLogger(__name__)

_DEFAULT_TURN_TIMEOUT = 600.0
_STARTUP_TIMEOUT = 180.0
_BRIDGE_LAUNCH_TIMEOUT = 45.0
_POLL_INTERVAL_S = 0.2
_MIN_PYTHON = (3, 11)
_MIN_PYTHON_DEFAULT_BRIDGE = (3, 12)


@dataclass
class TurnResult:
    final_text: str = ""
    projected_messages: list[dict] = field(default_factory=list)
    tool_iterations: int = 0
    interrupted: bool = False
    error: Optional[str] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    should_retire: bool = False


def _bridge_launch_needs_workaround() -> bool:
    """cursor-sdk Bridge.launch uses os.get_blocking + select(); broken on Windows."""
    return os.name == "nt"


def preflight_cursor_sdk(
    *,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """Validate Python/deps before a Cursor turn (not subject to bridge timeouts)."""
    if sys.version_info < _MIN_PYTHON:
        raise RuntimeError(
            f"Cursor SDK requires Python {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}+ "
            f"(this interpreter is {sys.version_info.major}.{sys.version_info.minor})."
        )
    if (
        sys.version_info < _MIN_PYTHON_DEFAULT_BRIDGE
        and not _bridge_launch_needs_workaround()
    ):
        raise RuntimeError(
            f"Cursor SDK requires Python {_MIN_PYTHON_DEFAULT_BRIDGE[0]}."
            f"{_MIN_PYTHON_DEFAULT_BRIDGE[1]}+ on this platform "
            f"(found {sys.version_info.major}.{sys.version_info.minor})."
        )

    if progress_callback is not None:
        try:
            progress_callback("Installing Cursor SDK dependencies…")
        except Exception:
            pass

    try:
        from tools.lazy_deps import ensure, is_available

        if not is_available("provider.cursor"):
            # Non-interactive: Hermes agent thread is never a TTY.
            ensure("provider.cursor", prompt=False)
    except Exception as exc:
        raise ImportError(
            "Cursor provider requires the cursor-sdk package. "
            "From the Hermes venv run: python -m pip install cursor-sdk==0.1.5 "
            "or reinstall with the cursor extra."
        ) from exc


def _ensure_cursor_sdk_installed() -> None:
    preflight_cursor_sdk()


def build_hermes_tools_mcp_servers() -> dict[str, dict[str, Any]]:
    """Stdio MCP entry so Cursor can call back into Hermes tools."""
    from hermes_cli.codex_runtime_plugin_migration import _build_hermes_tools_mcp_entry

    return {"hermes-tools": _build_hermes_tools_mcp_entry()}


def _cursor_error_message(exc: BaseException) -> str:
    msg = getattr(exc, "message", None)
    if msg:
        return str(msg)
    return str(exc)


def build_cursor_model_selection(model: str) -> Any:
    """Map a Hermes model id to a Cursor SDK ``ModelSelection``.

    Cursor's ``composer-2.5`` API defaults to ``fast=true`` (usage shows as
    ``composer-2.5-fast``). Hermes treats plain ``composer-2.5`` as the
    standard (non-fast) variant unless the user picks ``composer-2.5-fast``.
    """
    from cursor_sdk import ModelParameterValue, ModelSelection

    name = (model or "composer-2.5").strip() or "composer-2.5"
    lower = name.lower()

    if lower in {"auto", "default"}:
        return name

    base_id = "composer-2.5"
    if lower == base_id:
        return ModelSelection(
            id=base_id,
            params=(ModelParameterValue(id="fast", value="false"),),
        )
    if lower == f"{base_id}-fast":
        return ModelSelection(
            id=base_id,
            params=(ModelParameterValue(id="fast", value="true"),),
        )

    return name


def _format_startup_error(exc: BaseException) -> str:
    text = str(exc)
    winerr = getattr(exc, "winerror", None)
    if winerr == 10038 or "10038" in text:
        return (
            "Cursor SDK bridge failed on Windows (socket/select incompatibility). "
            "Hermes applies a threaded bridge launcher automatically; if this "
            "persists, set CURSOR_SDK_BRIDGE_URL/CURSOR_SDK_BRIDGE_TOKEN to a "
            "manually started cursor-sdk-bridge."
        )
    if "get_blocking" in text:
        return (
            "Cursor SDK bridge requires Python 3.12+ (os.get_blocking). "
            "Upgrade the Hermes venv Python version."
        )
    return text


def _read_bridge_discovery_from_stderr(
    process: subprocess.Popen[str],
    timeout: float,
) -> Mapping[str, Any]:
    """Windows-safe bridge discovery: read stderr on a thread (no select())."""
    from cursor_sdk._bridge import parse_discovery_line
    from cursor_sdk.errors import CursorSDKError

    if process.stderr is None:
        raise CursorSDKError("Bridge process stderr is unavailable")

    holder: dict[str, Any] = {}
    stderr_lines: list[str] = []

    def _reader() -> None:
        try:
            for line in process.stderr:
                stderr_lines.append(line)
                discovery = parse_discovery_line(line)
                if discovery is not None:
                    holder["discovery"] = discovery
                    return
        except Exception as exc:
            holder["error"] = exc

    thread = threading.Thread(target=_reader, name="cursor-bridge-stderr", daemon=True)
    thread.start()
    thread.join(timeout=max(1.0, float(timeout)))
    if "discovery" in holder:
        return holder["discovery"]

    exit_code = process.poll()
    detail = "".join(stderr_lines)[-2000:]
    if holder.get("error") is not None:
        raise CursorSDKError(f"Bridge stderr reader failed: {holder['error']}") from holder["error"]
    if exit_code is not None:
        raise CursorSDKError(
            f"Bridge exited before discovery (status {exit_code}): {detail}"
        )
    raise CursorSDKError(
        f"Timed out waiting for bridge discovery after {int(timeout)}s"
        + (f": {detail}" if detail else "")
    )


def _launch_bridge_threaded(
    workspace: str,
    *,
    timeout: float = _BRIDGE_LAUNCH_TIMEOUT,
) -> tuple[Any, subprocess.Popen[str]]:
    """Launch cursor-sdk-bridge with a Windows-safe discovery reader."""
    from cursor_sdk._bridge import Bridge, BridgeEndpoint, _bridge_subprocess_env
    from cursor_sdk._vendor import resolve_bridge_path

    argv = [resolve_bridge_path(), "--workspace", os.fspath(workspace)]
    process = subprocess.Popen(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=_bridge_subprocess_env(),
    )
    try:
        discovery = _read_bridge_discovery_from_stderr(process, timeout)
        endpoint = BridgeEndpoint.from_discovery(discovery)
        bridge = Bridge(endpoint, process)
        logger.info(
            "cursor SDK bridge ready url=%s pid=%s",
            endpoint.url,
            getattr(process, "pid", "?"),
        )
        return bridge, process
    except Exception:
        try:
            process.kill()
        except Exception:
            pass
        raise


class CursorSDKSession:
    """One Cursor agent per Hermes session; not thread-safe."""

    def __init__(
        self,
        *,
        cwd: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "composer-2.5",
        on_event: Optional[Callable[[Any], None]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._cwd = cwd or os.getcwd()
        self._api_key = (api_key or os.environ.get("CURSOR_API_KEY") or "").strip()
        self._model = (model or "composer-2.5").strip() or "composer-2.5"
        self._model_selection = build_cursor_model_selection(self._model)
        self._on_event = on_event
        self._progress_callback = progress_callback
        self._agent: Any = None
        self._agent_cm: Any = None
        self._sdk_client: Any = None
        self._bridge: Any = None
        self._interrupt_event = threading.Event()
        self._active_run: Any = None
        self._closed = False

    def _notify_progress(self, text: str) -> None:
        if not text or self._progress_callback is None:
            return
        try:
            self._progress_callback(text)
        except Exception:
            pass

    def _resolve_sdk_client(self) -> Any:
        if self._sdk_client is not None:
            return self._sdk_client

        bridge_url = os.environ.get("CURSOR_SDK_BRIDGE_URL", "").strip()
        bridge_token = (
            os.environ.get("CURSOR_SDK_BRIDGE_TOKEN")
            or os.environ.get("CURSOR_SDK_BRIDGE_AUTH_TOKEN")
            or ""
        ).strip()
        from cursor_sdk import Client

        if bridge_url and bridge_token:
            self._sdk_client = Client(
                base_url=bridge_url,
                auth_token=bridge_token,
                allow_api_key_env_fallback=True,
            )
            return self._sdk_client

        if _bridge_launch_needs_workaround():
            self._notify_progress("Starting Cursor bridge…")
            self._bridge, _process = _launch_bridge_threaded(self._cwd)
            self._sdk_client = Client(
                self._bridge.endpoint,
                allow_api_key_env_fallback=True,
            )
            return self._sdk_client

        from cursor_sdk._client import Client as SdkClient

        self._sdk_client = SdkClient.launch_bridge(
            workspace=self._cwd,
            timeout=_BRIDGE_LAUNCH_TIMEOUT,
            allow_api_key_env_fallback=True,
        )
        return self._sdk_client

    def _start_agent_blocking(self) -> str:
        """Create the Cursor SDK agent (must run off the prompt_toolkit thread)."""
        from cursor_sdk import Agent, LocalAgentOptions

        if not self._api_key:
            raise RuntimeError(
                "CURSOR_API_KEY is not set. Add it to ~/.hermes/.env or export it. "
                "Create a key at https://cursor.com/dashboard/integrations"
            )

        logger.info(
            "cursor SDK Agent.create starting model=%s selection=%r cwd=%s",
            self._model,
            self._model_selection,
            self._cwd,
        )
        self._notify_progress("Starting Cursor agent…")
        client = self._resolve_sdk_client()
        self._agent_cm = Agent.create(
            model=self._model_selection,
            api_key=self._api_key,
            local=LocalAgentOptions(cwd=self._cwd),
            client=client,
        )
        self._agent = self._agent_cm.__enter__()
        agent_id = str(getattr(self._agent, "agent_id", "") or "")
        logger.info("cursor SDK agent started: id=%s cwd=%s", agent_id[:12], self._cwd)
        return agent_id

    def ensure_started(self) -> str:
        if self._agent is not None:
            return str(getattr(self._agent, "agent_id", "") or "")

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cursor-start")
        future = pool.submit(self._start_agent_blocking)
        try:
            return future.result(timeout=_STARTUP_TIMEOUT)
        except FuturesTimeoutError as exc:
            logger.error(
                "cursor SDK startup timed out after %.0fs", _STARTUP_TIMEOUT
            )
            raise RuntimeError(
                f"Cursor SDK startup timed out after {int(_STARTUP_TIMEOUT)}s "
                "(bridge launch or Agent.create). Ensure cursor-sdk is installed in "
                "the Hermes venv, CURSOR_API_KEY is set, and cursor-sdk-bridge can start."
            ) from exc
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._active_run is not None:
            try:
                if self._active_run.supports("cancel"):
                    self._active_run.cancel()
            except Exception:
                pass
            self._active_run = None
        if self._agent_cm is not None:
            try:
                self._agent_cm.__exit__(None, None, None)
            except Exception:
                pass
            self._agent_cm = None
            self._agent = None
        if self._sdk_client is not None:
            try:
                self._sdk_client.close()
            except Exception:
                pass
            self._sdk_client = None
        if self._bridge is not None:
            try:
                self._bridge.close()
            except Exception:
                pass
            self._bridge = None

    def request_interrupt(self) -> None:
        self._interrupt_event.set()
        run = self._active_run
        if run is not None:
            try:
                if run.supports("cancel"):
                    run.cancel()
            except Exception:
                pass

    def _run_turn_worker(
        self,
        user_input: str,
        *,
        mcp_servers: dict[str, Any],
        result_holder: dict[str, Any],
    ) -> None:
        """Execute the Cursor SDK turn on a worker thread."""
        from cursor_sdk import CursorAgentError, SendOptions

        result = TurnResult()
        result_holder["result"] = result
        try:
            if mcp_servers:
                send_options = SendOptions(
                    mcp_servers=mcp_servers,
                    model=self._model_selection,
                )
            else:
                send_options = SendOptions(model=self._model_selection)
            run = self._agent.send(user_input, send_options)
            self._active_run = run
            result.run_id = str(getattr(run, "id", "") or "")

            projector = CursorEventProjector()
            tool_iterations = 0
            for message in run.messages():
                if self._interrupt_event.is_set():
                    result.interrupted = True
                    break
                if self._on_event is not None:
                    try:
                        self._on_event(message)
                    except Exception:
                        pass
                msg_type = ""
                if isinstance(message, dict):
                    msg_type = str(message.get("type") or "")
                else:
                    msg_type = str(getattr(message, "type", "") or "")
                if msg_type == "tool_call":
                    name = ""
                    status = ""
                    if isinstance(message, dict):
                        name = str(message.get("name") or "tool")
                        status = str(message.get("status") or "")
                    else:
                        name = str(getattr(message, "name", "") or "tool")
                        status = str(getattr(message, "status", "") or "")
                    if status == "running":
                        self._notify_progress(f"⚕ Cursor: {name}…")
                elif msg_type == "status":
                    status_text = ""
                    if isinstance(message, dict):
                        status_text = str(message.get("message") or message.get("status") or "")
                    else:
                        status_text = str(
                            getattr(message, "message", "")
                            or getattr(message, "status", "")
                            or ""
                        )
                    if status_text:
                        self._notify_progress(f"⚕ {status_text}")

                projected = projector.project(message)
                if projected.is_tool_iteration:
                    tool_iterations += 1
                if projected.messages:
                    result.projected_messages.extend(projected.messages)
                if projected.final_text:
                    result.final_text = projected.final_text

            if result.interrupted:
                try:
                    if run.supports("cancel"):
                        run.cancel()
                except Exception:
                    pass
            else:
                terminal = run.wait()
                status = str(getattr(terminal, "status", "") or "").strip().lower()
                if status == "error":
                    result.error = str(
                        getattr(terminal, "result", "") or "Cursor run failed"
                    )
                elif status == "cancelled":
                    result.interrupted = True
                final = str(getattr(terminal, "result", "") or "").strip()
                if final:
                    result.final_text = final
            result.tool_iterations = tool_iterations
        except CursorAgentError as exc:
            retryable = getattr(exc, "is_retryable", False)
            result.error = (
                f"Cursor SDK error: {_cursor_error_message(exc)} "
                f"(retryable={retryable})"
            )
            result.should_retire = True
        except Exception as exc:
            result.error = f"Cursor SDK turn failed: {exc}"
            result.should_retire = True
        finally:
            self._active_run = None
            if not result.final_text and result.projected_messages:
                for msg in reversed(result.projected_messages):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        result.final_text = str(msg["content"])
                        break

    def _run_turn_worker_with_startup(
        self,
        user_input: str,
        *,
        mcp_servers: dict[str, Any],
        result_holder: dict[str, Any],
        startup_holder: dict[str, Any],
    ) -> None:
        try:
            startup_holder["agent_id"] = self.ensure_started()
        except Exception as exc:
            startup_holder["error"] = exc
            return
        self._run_turn_worker(user_input, mcp_servers=mcp_servers, result_holder=result_holder)

    def run_turn(
        self,
        user_input: str,
        *,
        mcp_servers: Optional[dict[str, Any]] = None,
        turn_timeout: float = _DEFAULT_TURN_TIMEOUT,
    ) -> TurnResult:
        result = TurnResult()
        self._interrupt_event.clear()
        servers = mcp_servers if mcp_servers is not None else build_hermes_tools_mcp_servers()
        holder: dict[str, Any] = {}
        startup_holder: dict[str, Any] = {}
        self._notify_progress("⚕ Cursor agent working…")

        deadline = time.monotonic() + max(5.0, float(turn_timeout or _DEFAULT_TURN_TIMEOUT))
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cursor-sdk")
        future = pool.submit(
            self._run_turn_worker_with_startup,
            user_input,
            mcp_servers=servers,
            result_holder=holder,
            startup_holder=startup_holder,
        )
        try:
            while not future.done():
                if self._interrupt_event.is_set():
                    self.request_interrupt()
                    result.interrupted = True
                    break
                if time.monotonic() >= deadline:
                    self.request_interrupt()
                    result.error = (
                        f"Cursor SDK turn timed out after {int(turn_timeout)}s"
                    )
                    result.should_retire = True
                    break
                self._notify_progress("⚕ Cursor agent working…")
                try:
                    future.result(timeout=_POLL_INTERVAL_S)
                except FuturesTimeoutError:
                    continue
                except Exception as exc:
                    result.error = f"Cursor SDK turn failed: {exc}"
                    result.should_retire = True
                    break

            if future.done():
                if startup_holder.get("error") is not None:
                    exc = startup_holder["error"]
                    result.error = (
                        f"Cursor SDK startup failed: {_format_startup_error(exc)}"
                    )
                    result.should_retire = True
                elif "result" in holder:
                    result = holder["result"]
                    result.agent_id = str(startup_holder.get("agent_id") or "")
                    if self._interrupt_event.is_set() and not result.error:
                        result.interrupted = True
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        if self._progress_callback is not None:
            try:
                self._progress_callback("")
            except Exception:
                pass
        return result
