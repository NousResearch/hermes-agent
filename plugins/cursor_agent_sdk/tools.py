"""Tool implementation for the Cursor SDK plugin."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Mapping


DEFAULT_MODEL = "composer-2.5"
DEFAULT_TIMEOUT_SECONDS = 900
MAX_TIMEOUT_SECONDS = 3600
MAX_ERROR_CHARS = 2000
SUCCESS_STATUS = "finished"

logger = logging.getLogger(__name__)


class CursorRunTimeout(TimeoutError):
    """Raised when the Cursor SDK run exceeds the configured wall-clock limit."""


CURSOR_AGENT_SCHEMA = {
    "name": "cursor_agent",
    "description": (
        "Delegate a focused coding task to a Cursor SDK agent. Cursor runs as "
        "a separate coding agent runtime; use this for implementation, review, "
        "or repository-analysis tasks that benefit from Cursor's harness."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The task or question to send to the Cursor agent.",
            },
            "cwd": {
                "type": "string",
                "description": (
                    "Working directory for local Cursor runs. Defaults to the "
                    "current Hermes process working directory."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Cursor model ID. Defaults to composer-2.5 when omitted."
                ),
            },
            "runtime": {
                "type": "string",
                "enum": ["local", "cloud"],
                "description": (
                    "Cursor runtime to use. local runs against cwd; cloud runs "
                    "against a GitHub repository or pull request."
                ),
                "default": "local",
            },
            "cloud_repo_url": {
                "type": "string",
                "description": "GitHub repository URL for cloud runs.",
            },
            "cloud_starting_ref": {
                "type": "string",
                "description": "Branch, tag, or SHA for cloud repository checkout.",
            },
            "cloud_pr_url": {
                "type": "string",
                "description": "GitHub pull request URL for cloud runs.",
            },
            "work_on_current_branch": {
                "type": "boolean",
                "description": "For cloud runs, work on the repository's current branch.",
                "default": False,
            },
            "sandbox": {
                "type": "boolean",
                "description": "For local runs, enable Cursor SDK sandboxing.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_TIMEOUT_SECONDS,
                "default": DEFAULT_TIMEOUT_SECONDS,
                "description": (
                    "Maximum time to wait for the Cursor run before cancelling it."
                ),
            },
        },
        "required": ["prompt"],
    },
}


def check_cursor_sdk_available() -> bool:
    """Return True when Cursor SDK can plausibly run."""
    return bool(os.getenv("CURSOR_API_KEY", "").strip())


def handle_cursor_agent(args: Mapping[str, Any], **kwargs: Any) -> str:
    """Run a Cursor SDK agent and return a Hermes tool JSON string."""
    prompt = str(args.get("prompt") or "").strip()
    if not prompt:
        return _json_error("Missing required argument: prompt")

    api_key = str(os.getenv("CURSOR_API_KEY") or "").strip()
    if not api_key:
        return _json_error("CURSOR_API_KEY is required to use cursor_agent")

    dependency_error = _ensure_cursor_sdk_dependency()
    if dependency_error is not None:
        return dependency_error

    try:
        from cursor_sdk import (
            AsyncAgent,
            AsyncClient,
            CloudAgentOptions,
            CloudRepository,
            LocalAgentOptions,
        )
    except Exception as exc:
        return _json_error(
            "cursor-sdk is not installed or could not be imported. Install it "
            "with `pip install cursor-sdk==0.1.5`.",
            exc,
        )

    runtime = str(args.get("runtime") or "local").strip().lower()
    model = str(args.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL

    try:
        cwd: Path | None = None
        timeout_seconds = _coerce_timeout_seconds(args.get("timeout_seconds"))
        if runtime == "local":
            _ensure_local_terminal_backend()
            cwd = _resolve_cwd(args.get("cwd"))
            sandbox = args.get("sandbox")
            local_options: dict[str, Any] = {"cwd": str(cwd)}
            if isinstance(sandbox, bool):
                local_options["sandbox_options"] = {"enabled": sandbox}
            create_kwargs = {"local": LocalAgentOptions(**local_options)}
            runtime_payload = {"runtime": "local", "cwd": str(cwd)}
        elif runtime == "cloud":
            cloud_options, runtime_payload = _build_cloud_options(
                args,
                CloudAgentOptions=CloudAgentOptions,
                CloudRepository=CloudRepository,
            )
            create_kwargs = {"cloud": cloud_options}
        else:
            return _json_error("runtime must be either 'local' or 'cloud'")
    except Exception as exc:
        return _json_error("Invalid cursor_agent arguments", exc)

    try:
        response = _run_async(
            _execute_cursor_agent(
                AsyncAgent=AsyncAgent,
                AsyncClient=AsyncClient,
                api_key=api_key,
                model=model,
                prompt=prompt,
                create_kwargs=create_kwargs,
                runtime_payload=runtime_payload,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
            )
        )
        return json.dumps(response, ensure_ascii=False)
    except Exception as exc:
        return _json_error("Cursor SDK agent run failed", exc)


async def _execute_cursor_agent(
    *,
    AsyncAgent: Any,
    AsyncClient: Any,
    api_key: str,
    model: str,
    prompt: str,
    create_kwargs: dict[str, Any],
    runtime_payload: dict[str, Any],
    cwd: Path | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    agent = None
    client = None
    run = None
    deadline = time.monotonic() + timeout_seconds
    try:
        client = await _create_cursor_client(
            AsyncClient,
            cwd=cwd,
            timeout_seconds=_remaining_timeout(deadline),
        )
        agent = await _wait_for_deadline(
            _maybe_await(
                AsyncAgent.create(
                    client=client,
                    api_key=api_key,
                    model=model,
                    **create_kwargs,
                )
            ),
            timeout_seconds=_remaining_timeout(deadline),
        )
        run = await _wait_for_deadline(
            _maybe_await(agent.send(prompt)),
            timeout_seconds=_remaining_timeout(deadline),
        )
        text, status = await _wait_for_deadline(
            _run_text_and_status(run),
            timeout_seconds=_remaining_timeout(deadline),
        )
        success = status in {"", SUCCESS_STATUS}
        response = {
            "success": success,
            "model": model,
            "agent_id": _string_attr(agent, "agent_id") or _string_attr(agent, "id"),
            "run_id": _string_attr(run, "id") or _string_attr(run, "run_id"),
            "status": status,
            "text": text,
            **runtime_payload,
        }
        if not success:
            response["error"] = f"Cursor SDK run ended with status: {status}"
        return response
    except CursorRunTimeout as exc:
        await _cancel_run(run)
        raise TimeoutError(
            f"Cursor SDK run exceeded timeout_seconds={timeout_seconds}"
        ) from exc
    except Exception:
        await _cancel_run(run)
        raise
    finally:
        await _close_agent(agent)
        await _close_client(client)


def _run_async(coro: Any) -> Any:
    try:
        from model_tools import _run_async as run_model_tools_async
    except Exception:
        return _run_async_fallback(coro)
    return run_model_tools_async(coro)


def _run_async_fallback(coro: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    if not loop.is_running():
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _wait_for_deadline(awaitable: Any, *, timeout_seconds: float) -> Any:
    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.wait_for(task, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        if task.done() and not task.cancelled():
            return await task
        raise CursorRunTimeout(
            f"Cursor SDK run exceeded timeout_seconds={timeout_seconds}"
        ) from exc


def _remaining_timeout(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise CursorRunTimeout("Cursor SDK run exceeded timeout_seconds")
    return remaining


async def _close_agent(agent: Any) -> None:
    if agent is None:
        return
    close = getattr(agent, "close", None)
    if not callable(close):
        return
    try:
        await _maybe_await(close())
    except Exception as exc:
        logger.debug("Cursor SDK agent cleanup failed: %s", exc)


async def _close_client(client: Any) -> None:
    if client is None:
        return
    close = getattr(client, "aclose", None) or getattr(client, "close", None)
    if not callable(close):
        return
    try:
        await _maybe_await(close())
    except Exception as exc:
        logger.debug("Cursor SDK client cleanup failed: %s", exc)


async def _run_text_and_status(run: Any) -> tuple[str, str]:
    wait_fn = getattr(run, "wait", None)
    if callable(wait_fn):
        result = await _maybe_await(wait_fn())
        if result is not None:
            text = await _text_from_object(result)
            if not text:
                text = await _text_from_object(run)
            status = _status_from_object(result) or _status_from_object(run)
            return text, status

    return await _text_from_object(run), _status_from_object(run)


async def _cancel_run(run: Any) -> None:
    cancel = getattr(run, "cancel", None)
    if not callable(cancel):
        return
    try:
        await _maybe_await(cancel())
    except Exception as exc:
        logger.debug("Cursor SDK run cancellation failed: %s", exc)


async def _create_cursor_client(
    AsyncClient: Any,
    *,
    cwd: Path | None,
    timeout_seconds: float,
) -> Any:
    bridge_url = str(os.getenv("CURSOR_SDK_BRIDGE_URL") or "").strip()
    bridge_token = str(
        os.getenv("CURSOR_SDK_BRIDGE_TOKEN")
        or os.getenv("CURSOR_SDK_BRIDGE_AUTH_TOKEN")
        or ""
    ).strip()

    if bridge_url and bridge_token:
        return AsyncClient(
            base_url=bridge_url,
            auth_token=bridge_token,
            timeout=timeout_seconds,
            unary_timeout=timeout_seconds,
            stream_timeout=timeout_seconds,
            max_retries=0,
            allow_api_key_env_fallback=False,
        )

    return await _maybe_await(
        AsyncClient.launch_bridge(
            workspace=str(cwd) if cwd is not None else None,
            timeout=timeout_seconds,
            client_timeout=timeout_seconds,
            max_retries=0,
            allow_api_key_env_fallback=True,
        )
    )


def _build_cloud_options(
    args: Mapping[str, Any],
    *,
    CloudAgentOptions: Any,
    CloudRepository: Any,
) -> tuple[Any, dict[str, Any]]:
    repo_url = str(args.get("cloud_repo_url") or "").strip()
    pr_url = str(args.get("cloud_pr_url") or "").strip()
    if not repo_url and not pr_url:
        raise ValueError("cloud runtime requires cloud_repo_url or cloud_pr_url")

    repositories = []
    effective_repo_url = repo_url
    if pr_url:
        effective_repo_url = repo_url or _repo_url_from_pr_url(pr_url)
        if not effective_repo_url:
            raise ValueError("cloud_pr_url must include a GitHub pull request URL")
        repositories.append(CloudRepository(url=effective_repo_url, pr_url=pr_url))
    else:
        starting_ref = str(args.get("cloud_starting_ref") or "").strip() or None
        repositories.append(CloudRepository(url=repo_url, starting_ref=starting_ref))

    options: dict[str, Any] = {"repos": repositories}
    if isinstance(args.get("work_on_current_branch"), bool):
        options["work_on_current_branch"] = args["work_on_current_branch"]

    payload = {
        "runtime": "cloud",
        "cloud_repo_url": effective_repo_url,
        "cloud_pr_url": pr_url,
        "cloud_starting_ref": str(args.get("cloud_starting_ref") or "").strip(),
        "work_on_current_branch": bool(args.get("work_on_current_branch", False)),
    }
    return CloudAgentOptions(**options), payload


def _ensure_cursor_sdk_dependency() -> str | None:
    try:
        from tools.lazy_deps import FeatureUnavailable, ensure
    except ImportError as exc:
        return _json_error("cursor-sdk dependency manager is unavailable", exc)

    try:
        ensure("tool.cursor_sdk", prompt=False)
    except FeatureUnavailable as exc:
        return _json_error("cursor-sdk is not available", exc)
    except Exception as exc:
        return _json_error("cursor-sdk dependency check failed", exc)
    return None


def _repo_url_from_pr_url(pr_url: str) -> str:
    if "/pull/" not in pr_url:
        return ""
    return pr_url.split("/pull/", 1)[0]


def _resolve_cwd(value: Any) -> Path:
    base = Path(os.getenv("TERMINAL_CWD") or os.getcwd()).expanduser()
    raw = str(value or "").strip()
    if not raw:
        return base.resolve()

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _ensure_local_terminal_backend() -> None:
    backend = str(os.getenv("TERMINAL_ENV") or "local").strip().lower() or "local"
    if backend != "local":
        raise ValueError(
            "local Cursor runtime requires Hermes terminal.backend=local "
            f"(current TERMINAL_ENV={backend!r}); use runtime='cloud' for "
            "Cursor cloud delegation"
        )


def _coerce_timeout_seconds(value: Any) -> int:
    if value in (None, ""):
        return DEFAULT_TIMEOUT_SECONDS
    try:
        timeout_seconds = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout_seconds must be an integer") from exc
    if timeout_seconds < 1 or timeout_seconds > MAX_TIMEOUT_SECONDS:
        raise ValueError(
            f"timeout_seconds must be between 1 and {MAX_TIMEOUT_SECONDS}"
        )
    return timeout_seconds


async def _text_from_object(obj: Any) -> str:
    for name in ("text", "result"):
        value = getattr(obj, name, None)
        if callable(value):
            value = value()
        value = await _maybe_await(value)
        if value is not None:
            return str(value)
    return ""


def _status_from_object(obj: Any) -> str:
    status = getattr(obj, "status", "")
    return status if isinstance(status, str) else str(status or "")


def _string_attr(obj: Any, name: str) -> str:
    value = getattr(obj, name, "")
    return value if isinstance(value, str) else ""


def _json_error(message: str, exc: Exception | None = None) -> str:
    if exc is not None:
        detail = f"{type(exc).__name__}: {exc}"
        if len(detail) > MAX_ERROR_CHARS:
            detail = detail[: MAX_ERROR_CHARS - 3] + "..."
        payload = {"success": False, "error": message, "detail": detail}
    else:
        payload = {"success": False, "error": message}
    return json.dumps(payload, ensure_ascii=False)
