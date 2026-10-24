"""Bind a runtime scope and the profile state derived from it.

The value types and the state slot live in ``hermes_scope`` (stdlib-only). This module is the one
place that turns a scope into profile state: the ``HERMES_HOME`` override, the secret scope and
the terminal policy, installed as ONE change to the scope state so they cannot describe different
profiles. Code that needs another profile's state uses :func:`bind_profile_scope`, not the legacy
setters (``set_hermes_home_override``, ``set_secret_scope``, ``install_profile_terminal_scope``);
code-health rule HX014 blocks new direct calls to them outside this module. With a scope bound
those setters cannot split it: a call that names another profile rebinds the whole scope
(:func:`legacy_switch`) on a local install and refuses in a hosted scope.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from hermes_scope import (
    LOCAL_OWNER,
    LOCAL_TENANT,
    BoundSecrets,
    Scope,
    ScopeSwitchRefused,
    StateToken,
    TurnScope,
    current_scope,
    current_state,
    push_state,
)

__all__ = ["bind_profile_scope", "bind_state", "bind_terminal_policy", "bind_turn", "legacy_switch", "scope_for_profile"]


def _same_home(a: str | Path, b: str | Path) -> bool:
    from hermes_constants import hermes_home_key
    return hermes_home_key(a) == hermes_home_key(b)


def _refuse_inside_hosted(what: str, *, unless: Scope | None = None) -> None:
    """Refuse inside a hosted scope, unless *unless* is the same tenant and agent (a turn of it)."""
    current = current_scope()
    if current is None or current.is_local:
        return
    if unless is not None and (unless.tenant, unless.agent) == (current.tenant, current.agent):
        return
    raise ScopeSwitchRefused(
        f"{what} inside a hosted scope (tenant={current.tenant!r}, agent={current.agent!r}); "
        "a hosted activity's agent is fixed by its signed inbound frame")


def scope_for_profile(home: str | Path, *, turn: Optional[tuple[str, str]] = None) -> Scope:
    """The scope a local activity for profile *home* runs under.

    Re-entering the bound profile without naming a new turn returns the bound scope, so a turn's
    session and turn id survive the nested profile binds the gateway makes inside a turn. The
    actor is the bound scope's, else the local owner (a local install has one person). Raises
    :class:`~hermes_scope.ScopeSwitchRefused` inside a hosted scope."""
    _refuse_inside_hosted("a local profile bind")
    current = current_scope()
    if turn is None and current is not None and _same_home(current.agent, home):
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
) -> StateToken:
    """Bind *scope* and the profile state derived from its home as one change; return its token
    (``reset()`` or use it as a context manager).

    A hosted scope binds the scope with no home, secrets or terminal policy: its state lives
    behind the storage ports, not in a directory. For a local scope: the home override names the
    agent's home; the secret scope is *secrets*, or ``owner_secret_scope`` (the launch home's
    frozen env plus ``.env``, any other home's own files) with external sources hydrated first
    unless ``hydrate_secrets=False``, stamped with the home unless ``stamp_home=False`` (unstamped marks the process's own profile); the terminal policy
    is built from the profile's config over *env_overlay* (the refusal scope if it cannot be
    read). ``home_override=False`` / ``terminal=False`` keep that piece as currently bound, for the
    one host that deliberately binds less (a single-profile launch turn keeps a routed request's
    home override, as it always has). The kept value is copied into this bind, so the bind is whole
    on its own: removing the earlier change it was copied from cannot take the piece away. Every
    piece is computed before anything is installed, so a failure leaves nothing bound.

    Inside a hosted scope, only a bind of the same tenant and agent is allowed; anything else
    raises :class:`~hermes_scope.ScopeSwitchRefused`."""
    _refuse_inside_hosted("a bind of another scope", unless=scope)
    home = scope.home()
    if home is None:
        return push_state(scope=scope, home=None, secrets=None, terminal=None)
    from tools.terminal_scope import profile_terminal_policy
    from tui_gateway.launch_profile_policy import owner_secret_scope

    if secrets is None:
        if hydrate_secrets:
            from hermes_cli.env_loader import hydrate_profile_secret_sources
            hydrate_profile_secret_sources(Path(home))
        secrets = owner_secret_scope(Path(home))
    kept = current_state()
    policy = profile_terminal_policy(Path(home), env_overlay=env_overlay) if terminal else kept.terminal
    return push_state(scope=scope, home=home if home_override else kept.home,
                      secrets=BoundSecrets(secrets, home if stamp_home else None), terminal=policy)


def _bind_whole_profile(
    scope: Scope, *, secrets: Optional[Mapping[str, str]] = None, hydrate_secrets: bool = True,
    env_overlay: Optional[dict[str, str]] = None,
) -> StateToken:
    """Home, secrets and terminal policy for local *scope*, together. The process's own home keeps
    its launch semantics (environment over ``.env``, the frozen launch ``TERMINAL_*`` overlay,
    unstamped), so systemd / ``op run`` injection still resolves; any other home gets its own
    ``.env`` and sources only, stamped as routed. A caller's *env_overlay* replaces the default."""
    from agent.secret_scope import _is_process_home
    if _is_process_home(Path(scope.agent)):  # the pinned launch identity, not a mirrored HERMES_HOME
        from tui_gateway.launch_profile_policy import launch_secret_scope, launch_terminal_env
        return bind_state(scope, secrets=launch_secret_scope(Path(scope.agent)) if secrets is None else secrets,
                          stamp_home=False, env_overlay=launch_terminal_env() if env_overlay is None else env_overlay)
    return bind_state(scope, secrets=secrets, hydrate_secrets=hydrate_secrets, env_overlay=env_overlay)


@contextmanager
def bind_profile_scope(
    home: str | Path, *, turn: Optional[tuple[str, str]] = None, hydrate_secrets: bool = True,
) -> Iterator[Scope]:
    """Run a body for local profile *home*: home, secrets and terminal policy bound together.

    The replacement for calling the legacy setters directly. ``turn=(session_id, turn_id)`` binds
    a :class:`~hermes_scope.TurnScope`. Refuses inside a hosted scope."""
    scope = scope_for_profile(home, turn=turn)
    with _bind_whole_profile(scope, hydrate_secrets=hydrate_secrets):
        yield scope


@contextmanager
def bind_turn(session_id: str, turn_id: str) -> Iterator[Scope | None]:
    """Mark the body as one turn of the bound scope: the scope becomes a
    :class:`~hermes_scope.TurnScope` with the same tenant, agent and actor; nothing else changes.
    With no scope bound (a single-profile gateway keeps its ambient scope) this binds nothing."""
    current = current_scope()
    if current is None:
        yield None
        return
    turn = TurnScope(current.tenant, current.agent, current.actor, session_id, turn_id)
    with push_state(scope=turn):
        yield turn


def legacy_switch(
    home: str | Path | None, *, secrets: Optional[Mapping[str, str]] = None,
    env_overlay: Optional[dict[str, str]] = None,
) -> StateToken | None:
    """What a legacy setter does with a scope bound. ``home=None`` means the process home.

    Returns ``None`` to pass through: no scope is bound, or the call names the bound profile.
    Otherwise, on a local install, binds the whole scope for *home* (keeping the actor) and
    returns its token; the caller's *secrets* win when it passed some, else the profile's own are
    built, not hydrated (a legacy caller that needs external sources hydrates them itself, as it
    always did). In a hosted scope it raises :class:`~hermes_scope.ScopeSwitchRefused`."""
    current = current_scope()
    if current is None:
        return None
    _refuse_inside_hosted("a legacy profile setter tried to switch profiles")
    from hermes_constants import get_process_hermes_home
    target = home if home is not None else get_process_hermes_home()
    if _same_home(current.agent, target):
        return None
    return _bind_whole_profile(Scope(LOCAL_TENANT, str(target), current.actor), secrets=secrets,
                               hydrate_secrets=False, env_overlay=env_overlay)


def owns_bound_scope(home: str | Path | None) -> bool:
    """Whether *home* is the bound local scope's agent (a raw terminal policy's declared owner)."""
    current = current_scope()
    return current is not None and current.is_local and home is not None and _same_home(current.agent, home)


def bind_terminal_policy(policy: Mapping[str, str], *, profile_home: str | Path | None) -> StateToken:
    """Run under *policy*, a terminal policy owned by *profile_home* (a session's frozen policy, or one
    built from a profile's config); returns its token. With no scope bound it just installs the
    policy. Inside a bound scope the policy must belong to the bound agent: anything else raises
    :class:`~hermes_scope.ScopeSwitchRefused`, since the scope's home and secrets would stay behind
    with another profile's backend. To run under another profile's policy, bind that profile."""
    from tools.terminal_scope import set_terminal_scope
    return set_terminal_scope(dict(policy), profile_home=profile_home)
