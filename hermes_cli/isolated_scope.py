"""Which profile an operator's ``-p <name> serve|dashboard --isolated`` backend lists.

UX only, NOT a security boundary (SECURITY.md §2.2/§2.6): profiles under one OS user can read
each other, and every ``?profile=`` route keeps resolving siblings. A dedicated per-profile server
just stops *advertising* them — the profile list, ``/api/status`` names, the unified session
fan-outs and the gateway ``profiles.list`` roster — so a Desktop or dashboard pointed at it shows
the profile it was launched for (#76932, #91330). Tenant isolation needs separate OS users or
containers.

Captured once at launch as a canonical path (never a name string: ``get_active_profile_name()``
answers ``default``/``custom`` sentinels that are also valid profile ids), because a
multi-profile host mirrors a turn's profile into ``HERMES_HOME`` later. The root home and
Desktop-owned backends (its SSH ``serve --isolated`` is one backend per connection serving every
profile by ``?profile=``) stay unscoped.
"""

from __future__ import annotations

from typing import Iterable, Optional, TypeVar

T = TypeVar("T")

_LAUNCH_PROFILE: Optional[str] = None


def record_launch_scope(*, isolated: bool, desktop_owned: bool) -> Optional[str]:
    """Pin the listing scope for this process from its launch home; returns the profile or None."""
    global _LAUNCH_PROFILE
    _LAUNCH_PROFILE = None
    if isolated and not desktop_owned:
        from hermes_constants import get_default_hermes_root, get_process_hermes_home, profile_name_for_home

        home = get_process_hermes_home().resolve()
        if home != get_default_hermes_root().resolve():
            _LAUNCH_PROFILE = profile_name_for_home(home)
    return _LAUNCH_PROFILE


def isolated_launch_profile() -> Optional[str]:
    """The profile an isolated named-profile backend lists, or None (list every profile)."""
    return _LAUNCH_PROFILE


def scope_to_launch_profile(items: Iterable[T], name_of=lambda item: item) -> list[T]:
    """``items`` narrowed to the launch profile on an isolated backend; unchanged otherwise."""
    scope = _LAUNCH_PROFILE
    return [item for item in items if scope is None or name_of(item) == scope]
