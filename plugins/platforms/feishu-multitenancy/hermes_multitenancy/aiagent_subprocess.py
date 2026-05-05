"""Subprocess entry point for running AIAgent in isolation.

Avoids the gateway-async-loop ↔ AIAgent-sync deadlock that occurs when
``_run_with_aiagent`` is called via ``asyncio.to_thread`` from the gateway's
event loop. Hermes' ``agent.run_conversation`` internally uses sync HTTP /
nested loops that conflict with the parent async context.

The fix: shell out to a fresh Python process with no parent event loop.
Cost: ~0.5-1s extra startup per message.

I/O contract:
  stdin:  JSON {"event": {...}, "profile_home": "/path/to/profile", "messages": [...]}
  stdout: JSON {"result": "...", "error": null}  on success
          JSON {"result": "", "error": "..."}    on failure
  exit:   0 always (errors are reported via JSON)

The event dict must contain at minimum: text, message_id, source.* fields
(open_id, user_id, user_name, chat_id, chat_name, chat_type, platform).
"""

import json
import os
import sys
import traceback
import importlib.util
from pathlib import Path


class _ReplayedSource:
    """Reconstruct event.source from a flat dict."""
    def __init__(self, d: dict):
        for k, v in (d or {}).items():
            setattr(self, k, v)


class _ReplayedEvent:
    """Reconstruct a MessageEvent-shaped object from a flat dict."""
    def __init__(self, d: dict):
        self.text = d.get("text", "")
        self.message_id = d.get("message_id", "")
        self.sender_open_id = d.get("sender_open_id", "")
        self.source = _ReplayedSource(d.get("source") or {})


def _load_run_with_aiagent():
    """Load sibling agent_real regardless of package name used by Hermes."""
    try:
        from hermes_multitenancy.agent_real import _run_with_aiagent
        return _run_with_aiagent
    except ModuleNotFoundError as exc:
        if exc.name != "hermes_multitenancy":
            raise

    agent_real_path = Path(__file__).with_name("agent_real.py")
    spec = importlib.util.spec_from_file_location(
        "_hermes_multitenancy_agent_real",
        agent_real_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load agent_real from {agent_real_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._run_with_aiagent


def main() -> None:
    payload = json.loads(sys.stdin.read())
    event = _ReplayedEvent(payload["event"])
    profile_home = Path(payload["profile_home"])
    messages = payload.get("messages")
    if not isinstance(messages, list):
        messages = None

    # Lazy import so import errors are reported as JSON, not crash
    _run_with_aiagent = _load_run_with_aiagent()

    protocol_stdout = sys.stdout
    event_stream = os.getenv("HERMES_AIAGENT_EVENT_STREAM") == "1"

    def emit(event: str, **payload) -> None:
        protocol_stdout.write(json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n")
        protocol_stdout.flush()

    try:
        # stdout is the parent/child JSON protocol. Send any incidental prints
        # from Hermes core, providers, or tools to stderr so the parent can
        # parse stdout deterministically.
        sys.stdout = sys.stderr
        if event_stream:
            if messages is None:
                result = _run_with_aiagent(event, profile_home, event_sink=emit)
            else:
                result = _run_with_aiagent(
                    event,
                    profile_home,
                    messages=messages,
                    event_sink=emit,
                )
            out = {"event": "done", "result": result or "", "error": None}
        else:
            if messages is None:
                result = _run_with_aiagent(event, profile_home)
            else:
                result = _run_with_aiagent(event, profile_home, messages=messages)
            out = {"result": result or "", "error": None}
    except Exception as exc:
        if event_stream:
            out = {
                "event": "done",
                "result": "",
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            }
        else:
            out = {
                "result": "",
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            }
    finally:
        sys.stdout = protocol_stdout

    protocol_stdout.write(json.dumps(out, ensure_ascii=False))
    if event_stream:
        protocol_stdout.write("\n")
    protocol_stdout.flush()


if __name__ == "__main__":
    main()
