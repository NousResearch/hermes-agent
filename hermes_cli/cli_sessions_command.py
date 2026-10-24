"""Small grammar and owner forwarding seam for classic /sessions only."""
import json
import logging
import math

from hermes_cli.gateway_client import GatewayClientError
from hermes_cli.gateway_session_store import GatewaySessionStore

logger = logging.getLogger(__name__)


class SessionsModal:
    """Delegate composer state, but cancel every raw-stdin fallback of the modal."""
    def __init__(self, cli):
        object.__setattr__(self, "cli", cli)

    def __getattr__(self, name):
        value = getattr(self.cli, name)
        if name == "_prompt_text_input_modal" and hasattr(value, "__func__"):
            return value.__func__.__get__(self)
        return value

    def __setattr__(self, name, value):
        setattr(self.cli, name, value)

    def _prompt_text_input(self, *args, **kwargs):
        return None


class SessionCommandAction(GatewaySessionStore):
    """Reuse the facade's mutation engine without adopting an off-view CAS attachment.

    Only this short-lived forwarding facade owns cleanup; the real facade's view state
    is never changed. Check held subscriptions again at release, since a real switch
    may acquire the same owner while the RPC is in flight.
    """
    def __init__(self, cli, operation):
        self.store, self.cli, self.operation = cli._session_db, cli, operation
        self.view_id = cli.session_id
        super().__init__(self.store.connection)
        self.attachment = None
        self.receipt = None
        self.detached = False

    def held(self, sid):
        current = self.store.current or {}
        state = self.connection.states.get(sid)
        return sid in {self.cli.session_id, current.get("session_id"), current.get("stored_session_id")} or (
            state is not None and state.subscription_id is not None)

    def call(self, method, **params):
        if method == "session.detach":
            if self.held(params["session_id"]):
                return None
            self.detached = True
        result = self.store.call(method, **params)
        if method == "session.resume":
            self.attachment = result
        elif method == "session.mutate":
            if not isinstance(result, dict) or not result:
                raise GatewayClientError("missing owner receipt; completion unknown")
            self.receipt = result
        return result

    def _mutate(self, session_id, operation, payload, snapshot=None):
        snapshot = snapshot or self.call("session.resume", session_id=session_id)
        resolved = snapshot.get("session_id") or session_id
        if operation == "delete" and self.cli.session_id != self.view_id:
            raise GatewayClientError("Session view changed; deletion cancelled")
        if operation == "delete" and self.held(resolved):
            raise GatewayClientError("Cannot delete a current/live conversation; close/switch first.")
        return super()._mutate(session_id, operation, payload, snapshot)

    def _title_holder(self, session_id, title):
        # Preserve the actual owner error; a transport loss is not a title conflict,
        # and must never initiate a local lookup or alternate writer.
        return None

    def run(self, target, title=None):
        retired = False
        try:
            if self.operation == "delete":
                self.delete_session(target)
                retired = True
            else:
                self.set_session_title(target, title)
            return self.receipt or {}
        finally:
            snapshot = self.attachment or {}
            resolved = snapshot.get("session_id") or target
            if not retired and not self.detached and snapshot.get("subscription_id") and not self.held(resolved):
                try:
                    self.call("session.detach", session_id=resolved, subscription_id=snapshot["subscription_id"])
                except GatewayClientError as exc:
                    logger.debug("sessions action attachment cleanup failed: %s", exc)


USAGE = "Usage: /sessions [list|ls|browse|delete <exactID> [--yes]|rename <exactID> <title>|prune [--older-than <duration>] [--dry-run] [--yes]]"


def parse_sessions(arg):
    """Return (verb, target/remainder, title/consent); never resolve a selector."""
    words = arg.split(None, 1)
    verb = words[0].lower() if words else "list"
    rest = words[1].strip() if len(words) > 1 else ""
    if any(c in arg for c in "\r\n\0"):
        raise ValueError(USAGE)
    if verb in {"list", "ls", "browse"}:
        if rest:
            raise ValueError(USAGE)
        return "list", "", None
    if verb == "delete":
        parts = rest.split()
        if not parts or len(parts) > 2 or (len(parts) == 2 and parts[1] != "--yes") or parts[0].startswith("--"):
            raise ValueError(USAGE)
        return verb, parts[0], len(parts) == 2
    if verb == "rename":
        parts = rest.split(None, 1)
        if len(parts) != 2 or parts[0].startswith("--"):
            raise ValueError(USAGE)
        title = parts[1].strip()
        if title.startswith('"'):
            try:
                title = json.loads(title)
            except ValueError:
                raise ValueError(USAGE) from None
        if not isinstance(title, str) or not title.strip() or any(c in title for c in "\r\n\0"):
            raise ValueError(USAGE)
        return verb, parts[0], title
    if verb == "prune":
        from hermes_cli.session_filters import parse_duration_seconds
        parts, seen, duration = rest.split(), set(), None
        while parts:
            flag = parts.pop(0)
            if flag in seen or flag not in {"--older-than", "--dry-run", "--yes"}:
                raise ValueError(USAGE)
            seen.add(flag)
            if flag == "--older-than":
                if not parts:
                    raise ValueError(USAGE)
                duration = parts.pop(0)
                seconds = parse_duration_seconds(duration)
                if seconds is None or not math.isfinite(seconds):
                    raise ValueError(USAGE)
        return verb, duration, None
    return "resume", arg, None
