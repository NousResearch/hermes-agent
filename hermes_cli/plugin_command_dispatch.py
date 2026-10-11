"""Shared invocation helpers for plugin slash commands."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import threading
from typing import Any, Callable


def invoke_plugin_command(handler: Callable, raw_args: str, **context: Any) -> Any:
    """Invoke a slash-command handler with only the context it declares."""
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        return handler(raw_args)
    parameter_list = list(signature.parameters.values())
    try:
        bound = signature.bind_partial(raw_args)
    except TypeError:
        return handler(raw_args)
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameter_list)
    accepted_names = {
        p.name for p in parameter_list
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    unbound_context = {name: value for name, value in context.items() if name not in bound.arguments}
    accepted_context = unbound_context if accepts_kwargs else {
        name: value for name, value in unbound_context.items() if name in accepted_names
    }
    try:
        signature.bind_partial(raw_args, **accepted_context)
    except TypeError:
        return handler(raw_args)
    return handler(raw_args, **accepted_context)


_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS = 30.0


def resolve_plugin_command_result(result: Any) -> Any:
    """Resolve a plugin command result, awaiting async handlers without wedging the caller."""
    if not inspect.isawaitable(result):
        return result
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(result)
    outcome: dict[str, Any] = {}
    failure: dict[str, BaseException] = {}
    done = threading.Event()

    def _runner() -> None:
        try:
            outcome["value"] = asyncio.run(result)
        except BaseException as exc:  # pragma: no cover - re-raised below
            failure["exc"] = exc
        finally:
            done.set()

    threading.Thread(
        target=contextvars.copy_context().run,
        args=(_runner,),
        name="hermes-plugin-command-await",
        daemon=True,
    ).start()
    if not done.wait(timeout=_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS):
        raise TimeoutError(
            "Plugin command async handler did not complete within "
            f"{_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS:.0f}s"
        )
    if "exc" in failure:
        raise failure["exc"]
    return outcome.get("value")
