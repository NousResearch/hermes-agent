"""Resolve synchronous and asynchronous plugin command results."""
from typing import Any

def resolve_plugin_command_result(result: Any) -> Any:
    """Resolve a plugin command result, awaiting async handlers: ``asyncio.run`` when no loop is
    running, else a helper thread with its own loop (30s bound so a hung handler cannot wedge the
    terminal)."""
    from hermes_cli.plugins import (
        _PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS, asyncio, contextvars, inspect, threading,
    )

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

    # copy_context: the helper thread must see the caller's profile/secret scope, else an
    # async hook under a running loop reads the default HERMES_HOME and get_secret raises.
    threading.Thread(target=contextvars.copy_context().run, args=(_runner,),
                     name="hermes-plugin-command-await", daemon=True).start()
    if not done.wait(timeout=_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS):
        raise TimeoutError("Plugin command async handler did not complete within "
                           f"{_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS:.0f}s")
    if "exc" in failure:
        raise failure["exc"]
    return outcome.get("value")
