"""Request-scoped MCP access to Hermes' authoritative approval queue."""

from contextlib import contextmanager
import threading


class RegistryApprovals:
    def __init__(self, publish):
        self._publish = publish
        self._lock = threading.RLock()
        self._sessions = set()
        self._closed = False

    @contextmanager
    def invocation(self, session_key):
        from gateway.session_context import clear_session_vars, set_session_vars
        from tools import approval
        from tools.approval_context import (
            reset_current_session_key,
            reset_hermes_interactive_context,
            set_current_session_key,
            set_hermes_interactive_context,
        )

        def notify(data):
            with self._lock:
                if self._closed:
                    raise RuntimeError("MCP approval bridge stopped")
                self._publish(
                    "approval_requested",
                    session_key,
                    {**data, "id": data["request_id"], "session_key": session_key},
                )

        with self._lock:
            if self._closed:
                raise RuntimeError("MCP approval bridge stopped")
            self._sessions.add(session_key)
            approval.register_gateway_notify(session_key, notify)
        session_tokens = set_session_vars(
            platform="mcp",
            session_key=session_key,
            session_id=session_key,
            cron_session="",
            async_delivery=False,
        )
        key_token = set_current_session_key(session_key)
        interactive_token = set_hermes_interactive_context(False)
        try:
            yield
        finally:
            approval.unregister_gateway_notify(session_key)
            approval.clear_session(session_key)
            with self._lock:
                self._sessions.discard(session_key)
            reset_hermes_interactive_context(interactive_token)
            reset_current_session_key(key_token)
            clear_session_vars(session_tokens)

    def pending(self):
        from tools.approval import list_gateway_approvals

        with self._lock:
            return [
                {**data, "id": data["request_id"], "session_key": key}
                for key in sorted(self._sessions)
                for data in list_gateway_approvals(key)
            ]

    def respond(self, approval_id, decision):
        from tools.approval import resolve_gateway_approval

        choices = {"allow-once": "once", "allow-always": "always", "deny": "deny"}
        if decision not in choices:
            return {"error": f"Invalid decision: {decision}"}
        with self._lock:
            data = next(
                (item for item in self.pending() if item["id"] == approval_id), None
            )
            if data is None:
                return {"error": f"Approval not found: {approval_id}"}
            if decision == "allow-always" and not data.get("allow_permanent", False):
                return {"error": "This approval does not permit permanent consent"}
            key = data["session_key"]
            if not resolve_gateway_approval(
                key, choices[decision], request_id=approval_id
            ):
                return {"error": f"Approval expired: {approval_id}"}
            self._publish(
                "approval_resolved",
                key,
                {"approval_id": approval_id, "decision": decision},
            )
        return {"resolved": True, "approval_id": approval_id, "decision": decision}

    def stop(self):
        from tools.approval import unregister_gateway_notify

        with self._lock:
            self._closed = True
            for key in self._sessions:
                unregister_gateway_notify(key)
