"""Session capabilities for native, in-process plugin commands.

Adapters supply the authorized, canonical session; plugins receive no database
or agent handle. Parsing precedes resolution and all writes precede live changes.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
from dataclasses import dataclass
from pathlib import Path
import os
from uuid import uuid4
from hermes_state_pidns import holder_namespace_token
from typing import Any, Callable



@dataclass(frozen=True)
class SessionCommandRequest:
    write: bool
    data: Any


class SessionCommandHandler:
    def __init__(self, name: str, prepare: Callable, handler: Callable):
        if not callable(prepare) or not callable(handler) or any(
            inspect.iscoroutinefunction(fn) for fn in (prepare, handler)
        ):
            raise TypeError("session command prepare and handler must be synchronous callables")
        self.name, self.prepare, self.handler = name, prepare, handler


def is_session_command(handler) -> bool:
    return isinstance(handler, SessionCommandHandler)


def prepare_session_command(handler, raw_args: str) -> SessionCommandRequest:
    request = handler.prepare(raw_args)
    if not isinstance(request, SessionCommandRequest) or type(request.write) is not bool:
        raise TypeError("session command prepare must return SessionCommandRequest")
    return request


def run_session_command(handler, request, context):
    context._write_allowed = request.write
    context._active = True
    try:
        return handler.handler(request, context)
    finally:
        context._active = False
        context._write_allowed = False


class SessionCommandContext:
    """Exact session capability, valid only during its host-owned command lease."""

    def __init__(self, *, agent, session_db, session_id, ensure_session=None,
                 config_loader=None, profile_home=None):
        from hermes_constants import get_hermes_home

        self.session_id = str(session_id or "")
        self.profile_home = str(Path(profile_home or get_hermes_home()).resolve())
        if agent is not None and str(getattr(agent, "session_id", "")) != self.session_id:
            raise ValueError("live session identity mismatch; refusing to mutate")
        db_path = getattr(session_db, "db_path", None)
        if db_path and Path(db_path).resolve().parent != Path(self.profile_home):
            raise ValueError("session database profile mismatch; refusing to mutate")
        self._agent, self._db = agent, session_db
        self._ensure = ensure_session or getattr(agent, "_ensure_db_session", None)
        self._config_loader = config_loader or load_session_command_config
        self._write_allowed = False
        self._active = False
        self.writes: dict[str, bool | None] = {}

    @property
    def live(self) -> bool:
        return self._agent is not None

    def get_config(self) -> dict:
        self._require_active()
        config = self._config_loader()
        if not isinstance(config, dict):
            raise ValueError("could not read global configuration")
        return deepcopy(config)

    def get_override(self, key: str) -> bool | None:
        self._require_active()
        from agent.session_policies import policy_entries
        from hermes_state import SessionDB

        if key not in policy_entries():
            raise ValueError(f"unregistered session policy: {key}")
        if not self.session_id or self._db is None:
            return None
        return SessionDB.session_policy_override(self._db.get_session(self.session_id), key)

    def set_override(self, key: str, override: bool | None) -> bool:
        self._require_active()
        from agent.session_policies import apply_session_policy, policy_entries

        if not self._write_allowed:
            raise ValueError("read-only session command cannot write an override")
        entry = policy_entries().get(key)
        if entry is None:
            raise ValueError(f"unregistered session policy: {key}")
        if override is not None and type(override) is not bool:
            raise ValueError("session policy override must be a literal bool or None")
        if not self.session_id.strip() or self._db is None or not callable(self._ensure):
            raise ValueError("canonical session persistence is not available")
        # Configuration is fallible; validate it before creating even a lazy row.
        config = self.get_config()
        holder = f"pid={os.getpid()}{holder_namespace_token()}:session-command={uuid4().hex}"
        if not self._db.try_acquire_session_turn_lease(self.session_id, holder):
            raise ValueError("session busy; interrupt the current turn first")
        try:
            self._validate_identity()
            self._ensure()
            self._validate_identity()
            if self._db.get_session(self.session_id) is None:
                raise ValueError(f"Session row not found: {self.session_id}")
            self._db.set_session_policy_override(self.session_id, key, override)
            self.writes[key] = override
            if self._agent is None:
                return False
            return apply_session_policy(self._agent, key, entry, override, config)
        finally:
            self._db.release_session_turn_lease(self.session_id, holder)

    def _require_active(self):
        if not self._active:
            raise ValueError("session command capability is no longer active")

    def _validate_identity(self):
        if self._agent is not None and str(getattr(self._agent, "session_id", "")) != self.session_id:
            raise ValueError("live session identity changed; refusing to mutate")



def make_session_context(**kwargs) -> SessionCommandContext:
    return SessionCommandContext(**kwargs)


def load_session_command_config() -> dict:
    """Read effective profile settings without initialization, recovery, or backup writes."""
    from hermes_cli.config_effective import load_user_config_effective
    return load_user_config_effective(fail_closed=True, persist_backup=False)
