"""Which daemon a named-profile client must start when nothing serves its home.

A ``<root>/profiles/<name>`` home that the default multiplexer serves has no daemon of its own.
When the multiplexer is down (restart, update, crash) the profile looks ``absent`` and the ensure
loop would spawn a per-profile gateway for it — a second owner that answers this client and then
blocks the multiplexer's next all-or-nothing reserve (``Cannot reserve gateway profiles``). The
client must start the multiplexer instead, so the start target is decided here from the same facts
the multiplexer boots on (``gateway_multiplex_mode.host_serves_named_profile``): a profile's own
``gateway.standalone: true``, the default profile's explicit ``true``, or — when the flag is unset or
the retired ``false`` that boot settles like unset — the served set the last multiplexer recorded in
the root's ``gateway_state.json`` (a standalone default gateway clears it).
"""
from __future__ import annotations

import os
from pathlib import Path


def profile_entry(home: Path) -> Path | None:
    """The ``<root>/profiles/<name>`` entry that is *home* itself: its own folder, or the link a linked
    profile is served through. Never the profile enclosing a home nested inside it, which
    ``named_profile_home`` answers for any path within."""
    from hermes_constants import named_profile_home
    named = named_profile_home(home)
    if named is None or named == home:
        return named
    same = os.path.normcase(os.path.realpath(named)) == os.path.normcase(os.path.realpath(home))
    return named if same else None


def multiplexer_root_for(home: Path) -> Path | None:
    """Default root that may multiplex *home*, when *home* is a named profile under it. A linked
    profile's resolved home lies outside ``profiles/``: its root is its entry's, never the target's."""
    named = profile_entry(home)
    if named is None:
        return None
    root = named.parent.parent
    return root if root != home else None


def implied_host_root(home: Path) -> Path | None:
    """The root whose gateway must serve named profile *home* when nothing records a verdict.

    ``hermes gateway run`` refuses a ``profiles/<name>`` home a gateway of its own unless it authored
    ``gateway.standalone: true`` (or passes ``--force``); the retired ``multiplex_profiles: false``
    no longer opts out, so only a standalone profile keeps its own daemon.
    """
    root = multiplexer_root_for(home)
    if root is None:
        return None
    from hermes_cli.gateway_multiplex_mode import host_serves_named_profile
    return None if host_serves_named_profile(root, home) is False else root


def multiplexer_serves_home(home: Path) -> Path | None:
    """The root whose (possibly stopped) multiplexer serves *home*, else None.

    ``hermes_cli.gateway.named_profile_served_by_running_multiplexer`` answers the live question;
    this is its offline twin for the moment no gateway runs, deciding from the boot policy
    (:func:`hermes_cli.gateway_multiplex_mode.host_serves_named_profile`): a standalone secondary is
    never served, an explicit ``true`` serves the rest, and an unset or retired ``false`` flag falls
    back to the ``served_profiles`` the last multiplexer recorded. No evidence keeps it unanswered.
    """
    root = multiplexer_root_for(home)
    if root is None:
        return None
    from hermes_cli.gateway_multiplex_mode import host_serves_named_profile
    verdict = host_serves_named_profile(root, home)
    if verdict is not None:
        return root if verdict else None
    from gateway.status import read_runtime_status
    from hermes_cli.profiles import normalize_profile_name
    served = (read_runtime_status(root / "gateway_state.json") or {}).get("served_profiles")
    if not isinstance(served, list):
        return None
    names = {normalize_profile_name(str(p)) for p in served if p}
    # The entry's name: a junction's target folder can be called anything.
    return root if normalize_profile_name((profile_entry(home) or home).name) in names else None
