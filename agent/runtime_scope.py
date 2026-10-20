"""Bind a runtime scope and the profile state derived from it.

The value types and the context-local slot live in ``hermes_scope`` (stdlib-only). This module is
the one place that turns a bound scope into profile state: the ``HERMES_HOME`` override, the secret
scope and the terminal policy. Code that needs another profile's state uses
:func:`bind_profile_scope`, not the legacy setters (``set_hermes_home_override``,
``set_secret_scope``, ``install_profile_terminal_scope``); code-health rule HX014 blocks new direct
calls to them outside this module. Inside a bound scope those setters cannot split it: a call that
names another profile rebinds the whole scope (:func:`legacy_switch`) on a local install and
refuses in a hosted scope.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Optional

from hermes_scope import (
    LOCAL_OWNER,
    LOCAL_TENANT,
    Scope,
    ScopeSwitchRefused,
    ScopeTokens,
    TurnScope,
    current_scope,
    reset_scope,
    set_scope,
)

__all__ = ["bind_profile_scope", "bind_state", "legacy_switch", "scope_for_profile"]


def _same_home(a: str | Path, b: str | Path) -> bool:
    from hermes_constants import hermes_home_key
    return hermes_home_key(a) == hermes_home_key(b)


def scope_for_profile(home: str | Path, *, turn: Optional[tuple[str, str]] = None) -> Scope:
    """The scope a local activity for profile *home* runs under.

    Re-entering the bound profile without naming a new turn returns the bound scope, so a turn's
    session and turn id survive the nested profile binds the gateway makes inside a turn. The
    actor is the bound scope's, else the local owner (a local install has one person)."""
    current = current_scope()
    if turn is None and current is not None and current.is_local and _same_home(current.agent, home):
        return current
    actor = current.actor if current is not None else LOCAL_OWNER
    if turn is not None:
        return TurnScope(LOCAL_TENANT, str(home), actor, turn[0], turn[1])
    return Scope(LOCAL_TENANT, str(home), actor)


def bind_state(
    scope: Scope,
    *,
    secrets: Optional[Mapping[str, str]] = None,
    hydrate_secrets: bool = True,
    stamp_home: bool = True,
    env_overlay: Optional[dict[str, str]] = None,
    home_override: bool = True,
    terminal: bool = True,
) -> ScopeTokens:
    """Bind *scope*, then the profile state derived from its home; return the reset tokens.

    The scope slot is set first, so the setters below see their own home and pass straight
    through. A hosted scope binds the slot alone: its state lives behind the storage ports, not
    in a directory. For a local scope: the home override names the agent's home; the secret scope
    is *secrets*, or a build of the profile's ``.env`` plus external sources (hydrated first unless
    ``hydrate_secrets=False``), stamped with the home unless ``stamp_home=False`` (unstamped marks
    the process's own profile); the terminal policy is built from the profile's config over
    *env_overlay* (a refusal scope if it cannot be read). ``home_override`` / ``terminal`` skip a
    piece for the hosts that deliberately bind less (a single-profile launch turn binds secrets
    only). Whatever was bound is released if a later piece raises."""
    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.terminal_scope import install_profile_terminal_scope, reset_terminal_scope

    tokens = ScopeTokens()
    try:
        tokens.scope = tokens.push(reset_scope, set_scope(scope))
        home = scope.home()
        if home is None:
            return tokens
        if home_override:
            tokens.home = tokens.push(reset_hermes_home_override, set_hermes_home_override(home))
        if secrets is None:
            if hydrate_secrets:
                from hermes_cli.env_loader import hydrate_profile_secret_sources
                hydrate_profile_secret_sources(Path(home))
            secrets = build_profile_secret_scope(Path(home))
        tokens.secret = tokens.push(
            reset_secret_scope, set_secret_scope(secrets, profile_home=home if stamp_home else None))
        if terminal:
            tokens.terminal = tokens.push(
                reset_terminal_scope, install_profile_terminal_scope(Path(home), env_overlay=env_overlay))
        return tokens
    except BaseException:
        tokens.reset()
        raise


def _bind_whole_profile(
    scope: Scope, *, secrets: Optional[Mapping[str, str]] = None, hydrate_secrets: bool = True,
) -> ScopeTokens:
    """Home, secrets and terminal policy for local *scope*, together. The process's own home keeps
    its launch semantics (environment over ``.env``, the frozen launch ``TERMINAL_*`` overlay,
    unstamped), so systemd / ``op run`` injection still resolves; any other home gets its own
    ``.env`` and sources only, stamped as routed."""
    from hermes_constants import get_process_hermes_home
    if _same_home(scope.agent, get_process_hermes_home()):
        from tui_gateway.launch_profile_policy import launch_secret_scope, launch_terminal_env
        return bind_state(scope, secrets=launch_secret_scope(Path(scope.agent)) if secrets is None else secrets,
                          stamp_home=False, env_overlay=launch_terminal_env())
    return bind_state(scope, secrets=secrets, hydrate_secrets=hydrate_secrets)


@contextmanager
def bind_profile_scope(
    home: str | Path, *, turn: Optional[tuple[str, str]] = None, hydrate_secrets: bool = True,
) -> Iterator[Scope]:
    """Run a body for local profile *home*: home, secrets and terminal policy bound together.

    The replacement for calling the legacy setters directly. ``turn=(session_id, turn_id)`` binds
    a :class:`~hermes_scope.TurnScope`."""
    scope = scope_for_profile(home, turn=turn)
    tokens = _bind_whole_profile(scope, hydrate_secrets=hydrate_secrets)
    try:
        yield scope
    finally:
        tokens.reset()


@contextmanager
def bind_turn(session_id: str, turn_id: str) -> Iterator[Scope | None]:
    """Mark the body as one turn of the bound local scope: rebinds the slot as a
    :class:`~hermes_scope.TurnScope` with the same tenant, agent and actor. The profile state is
    already bound by the enclosing scope, so nothing else changes. With no scope bound (a
    single-profile gateway keeps its ambient scope) this binds nothing."""
    current = current_scope()
    if current is None:
        yield None
        return
    turn = TurnScope(current.tenant, current.agent, current.actor, session_id, turn_id)
    token = set_scope(turn)
    try:
        yield turn
    finally:
        reset_scope(token)


def legacy_switch(home: str | Path | None, *, secrets: Optional[Mapping[str, str]] = None) -> ScopeTokens | None:
    """What a legacy setter does with a scope bound. ``home=None`` means the process home.

    Returns ``None`` to pass through: no scope is bound, or the call names the bound profile.
    Otherwise, on a local install, binds the whole scope for *home* (keeping the actor) and
    returns its tokens; the caller's *secrets* win when it passed some, else the profile's own
    are built, not hydrated (a legacy caller that needs external sources hydrates them itself, as
    it always did). In a hosted scope it raises :class:`~hermes_scope.ScopeSwitchRefused`: a
    hosted activity's agent is fixed by its signed inbound frame."""
    current = current_scope()
    if current is None:
        return None
    if not current.is_local:
        raise ScopeSwitchRefused(
            "a legacy profile setter tried to switch profiles inside a hosted scope "
            f"(tenant={current.tenant!r}); bind a new scope at the activity's entry point")
    from hermes_constants import get_process_hermes_home
    target = home if home is not None else get_process_hermes_home()
    if _same_home(current.agent, target):
        return None
    return _bind_whole_profile(Scope(LOCAL_TENANT, str(target), current.actor), secrets=secrets,
                               hydrate_secrets=False)
