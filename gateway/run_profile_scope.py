"""Separate blocking profile-policy preparation from task-local binding."""
import asyncio
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional

from tools.terminal_scope import (
    TerminalPolicyRefusal,
    prepare_profile_terminal_scope,
    reset_terminal_scope,
    set_terminal_scope,
)


@dataclass(frozen=True)
class PreparedProfileRuntimeScope:
    home: str
    secrets: Mapping[str, str] = field(repr=False)
    terminal: Mapping[str, str] = field(repr=False)


async def run_off_loop(func, *args):
    """Run blocking preparation on the live gateway's fenced executor (``_run_in_executor_with_context``,
    joined by ``_shutdown_executor``), so it cannot outlive the SessionDB close (#94736). Hosts with no
    gateway runner (``hermes serve``, scripts) use ``asyncio.to_thread``; both copy the caller's context."""
    ref = getattr(sys.modules.get("gateway.run"), "_gateway_runner_ref", None)
    runner = ref() if callable(ref) else None
    if runner is not None:
        return await runner._run_in_executor_with_context(func, *args)
    return await asyncio.to_thread(func, *args)


def prepare_profile_runtime_scope(
    profile_home: Path, prepared_secret_scope: Optional[Mapping[str, str]] = None, *,
    hydrate_secrets: bool = True,
    env_overlay: Optional[dict[str, str]] = None,
) -> PreparedProfileRuntimeScope:
    """Blocking preparation phase; async callers must offload it, not the ContextVar binding."""
    from gateway.run import _load_profile_secret_scope
    from tui_gateway.launch_profile_policy import owner_secret_scope  # upstream 70872e7fa6: launch home keeps env-only keys
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    token = set_hermes_home_override(str(profile_home))
    try:
        secrets = prepared_secret_scope
        if secrets is None:
            secrets = (_load_profile_secret_scope(profile_home) if hydrate_secrets
                       else owner_secret_scope(profile_home))
        terminal = prepare_profile_terminal_scope(profile_home, env_overlay=env_overlay)
        return PreparedProfileRuntimeScope(
            str(profile_home), MappingProxyType(dict(secrets)),
            terminal if isinstance(terminal, TerminalPolicyRefusal) else MappingProxyType(terminal))
    finally:
        reset_hermes_home_override(token)


@contextmanager
def bind_profile_runtime_scope(prepared: PreparedProfileRuntimeScope):
    """I/O-free binding in the caller's context (never bind only in a worker)."""
    from agent.secret_scope import set_secret_scope, reset_secret_scope
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    home_token = set_hermes_home_override(prepared.home)
    secret_token = terminal_token = None
    try:
        # Pairing/plugin hydration refreshes the installed dict, never the prepared snapshot.
        secret_token = set_secret_scope(dict(prepared.secrets), profile_home=prepared.home)
        terminal_token = set_terminal_scope(prepared.terminal)
        yield
    finally:
        if terminal_token is not None:
            reset_terminal_scope(terminal_token)
        if secret_token is not None:
            reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
