"""Task-owned process capability for Claude subscription workers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


class WorkerProcessBroker:
    """Trusted process-tool closure bound to one raw worker task identity.

    Model-provided arguments never carry the authorization identity.  That
    identity lives only in this closure, and is checked against
    ``ProcessSession.owner_task_id`` rather than the potentially collapsed
    terminal-container key in ``ProcessSession.task_id``.
    """

    def __init__(self, task_id: str, registry: Any | None = None) -> None:
        self.task_id = str(task_id)
        if registry is None:
            from tools.process_registry import process_registry

            registry = process_registry
        self.registry = registry

    def _owned_session(self, session_id: str) -> Any:
        session = self.registry.get(session_id)
        owner_task_id = str(getattr(session, "owner_task_id", "") or "")
        if session is None or owner_task_id != self.task_id:
            raise RuntimeError(
                f"Process {session_id} does not belong to worker task {self.task_id}"
            )
        return session

    def handle(self, arguments: Mapping[str, Any]) -> str:
        args = dict(arguments or {})
        action = str(args.get("action") or "")
        if action == "list":
            from tools.process_registry import _redact_process_result

            sessions = [
                _redact_process_result(item)
                for item in self.registry.list_sessions(owner_task_id=self.task_id)
            ]
            return json.dumps({"processes": sessions}, ensure_ascii=False)

        session_id = str(args.get("session_id") or "")
        if not session_id:
            raise RuntimeError(f"session_id is required for {action or 'process action'}")
        self._owned_session(session_id)

        if action == "poll":
            result = self.registry.poll(session_id)
        elif action == "log":
            result = self.registry.read_log(
                session_id, offset=args.get("offset", 0), limit=args.get("limit", 200)
            )
        elif action == "wait":
            result = self.registry.wait(session_id, timeout=args.get("timeout"))
        elif action == "kill":
            result = self.registry.kill_process(session_id)
        elif action == "write":
            result = self.registry.write_stdin(session_id, str(args.get("data", "")))
        elif action == "submit":
            result = self.registry.submit_stdin(session_id, str(args.get("data", "")))
        elif action == "close":
            result = self.registry.close_stdin(session_id)
        else:
            raise RuntimeError(
                f"Unknown process action: {action}. Use: list, poll, log, wait, "
                "kill, write, submit, close"
            )
        if action in {"poll", "log", "wait"}:
            from tools.process_registry import _redact_process_result

            result = _redact_process_result(result)
        return json.dumps(result, ensure_ascii=False)


__all__ = ["WorkerProcessBroker"]
