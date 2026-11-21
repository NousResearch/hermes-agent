"""Canonical dashboard deep links for Kanban tasks.

The board UI can already *receive* a card deep link (``/kanban?board=<slug>&task=<id>``)
once an external caller hands one over, but nothing in Hermes ever *emits* one: the CLI,
the completion notifier and the plugin API all hand out a bare task id. Without an emitter
the "cite the card" workflow degrades to pasting ``t_abc123`` into prose, which a reader
cannot click.

This module is the single place that builds that URL, so the CLI, the notifier and any
future caller agree on one contract instead of each inventing a format.

Contract: ``<dashboard-base>/kanban?board=<slug>&task=<task-id>``

``<dashboard-base>`` is the URL the dashboard is *reachable* at — which is emphatically
not always a public one. A single-machine install reaches it at ``http://localhost:9119``
(WSL forwarding it to the Windows host, say), an on-prem install over an RFC1918 address
or an internal DNS name, and only an internet-facing deploy at a genuinely public URL.
The base is resolved in this order:

1. ``dashboard.public_url`` / ``HERMES_DASHBOARD_PUBLIC_URL`` — the operator's explicit
   declaration, via ``dashboard_auth.prefix.resolve_public_url()`` (the same resolver the
   web server trusts for its own absolute links). Highest precedence because it is the
   only source that survives a reverse proxy: the path prefix is preserved.
2. The local host record (``gateway.host_rendezvous``) — the same ``(pid, host, port)``
   the dashboard itself uses when it tells you where it is already running. This is what
   makes a link work on a plain single-machine install, where nothing is configured and
   the dashboard is simply on loopback. Wildcard binds are reached over loopback.
3. Otherwise ``None``, and callers print nothing rather than a fabricated link.

* ``board`` is always explicit, including ``default``: omitting it makes the browser fall
  back to its own localStorage pin, which is how a link silently opens the wrong board.
* ``profile`` is included only when a local dashboard actually serves it, so the link
  lands in the same scope the command ran from without naming a profile the server lacks.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import quote

__all__ = ["resolve_dashboard_base", "task_url"]

#: Query keys the browser board reads. Kept here so the emitter and any test agree.
BOARD_PARAM = "board"
TASK_PARAM = "task"
PROFILE_PARAM = "profile"

#: Loopback spellings a dashboard bind can take; rendered as ``localhost`` for browsers.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "[::1]", ""})


def resolve_dashboard_base() -> Optional[str]:
    """URL the dashboard is reachable at (no trailing slash), or ``None``.

    Prefers the operator's declared URL, then falls back to the local host record so an
    unconfigured single-machine install still gets a working link.
    """
    return _declared_public_url() or _local_dashboard_base()


def _declared_public_url() -> Optional[str]:
    """``dashboard.public_url`` / ``HERMES_DASHBOARD_PUBLIC_URL``, or ``None``.

    An unset or blank setting is an ordinary "not configured" here, not an error worth
    logging on every card render.
    """
    try:
        from hermes_cli.dashboard_auth.prefix import resolve_public_url
    except ImportError:
        # Stripped/minimal installs may not ship the dashboard auth package.
        return None
    raw = (resolve_public_url() or "").strip()
    if not raw:
        return None
    return raw.rstrip("/") or None


def _local_dashboard_base() -> Optional[str]:
    """``http://<host>:<port>`` of the dashboard on this machine, or ``None``.

    Reads the same host record ``hermes dashboard`` uses to answer "already running on
    this host" (``gateway.host_rendezvous``), so a loopback- or LAN-bound dashboard is
    found with no configuration at all.

    A loopback bind is rendered as ``localhost`` rather than the literal ``127.0.0.1``:
    the link is meant for a *browser*, and under WSL / a container NAT the loopback
    literal reaches the guest, not the host machine where the browser runs. ``localhost``
    follows the port forward both sides agree on.
    """
    record = _host_record()
    if record is None or not record.port:
        return None
    try:
        from gateway import host_rendezvous as hr
        from hermes_cli.url_utils import format_url_host

        host = format_url_host(hr.dial_host(record))
    except ImportError:
        return None
    if host in _LOOPBACK_HOSTS:
        host = "localhost"
    return f"http://{host}:{record.port}"


def _host_record():
    """Live local dashboard record, or ``None`` (deferred import: this stays cheap)."""
    try:
        from gateway import host_rendezvous as hr
    except ImportError:
        return None
    return hr.read_record(hr.ROLE_SERVE)


def _served_profiles() -> tuple[str, ...]:
    """Profiles the local dashboard serves, or ``()`` when it cannot be determined."""
    record = _host_record()
    if record is None:
        return ()
    return tuple(record.profiles or ())


def task_url(task_id: str, board: Optional[str] = None, base: Optional[str] = None) -> Optional[str]:
    """Absolute deep link to one card, or ``None`` when no dashboard URL is known.

    ``board=None`` resolves the current board (never guessed), so a link always names the
    board it was built on even if the reader's browser has a different board pinned.
    """
    tid = (task_id or "").strip()
    if not tid:
        return None
    origin = (base or "").strip().rstrip("/") or resolve_dashboard_base()
    if not origin:
        return None
    params = []
    slug = (board or "").strip() or _current_board_slug()
    if slug:
        params.append(f"{BOARD_PARAM}={quote(slug, safe='')}")
    params.append(f"{TASK_PARAM}={quote(tid, safe='')}")
    profile = _link_profile()
    if profile:
        params.append(f"{PROFILE_PARAM}={quote(profile, safe='')}")
    return f"{origin}/kanban?{'&'.join(params)}"


def _current_board_slug() -> str:
    """Current board slug, or ``""`` when it cannot be resolved (the link still works)."""
    from hermes_cli import kanban_db as kb

    return (kb.get_current_board() or "").strip()


def _link_profile() -> str:
    """Active profile to scope the link to, or ``""``.

    Included only when a local dashboard is running and lists that profile: a link naming
    a profile the server does not serve is worse than one that omits it.
    """
    served = _served_profiles()
    if not served:
        return ""
    try:
        from hermes_cli.profiles import get_active_profile_name

        active = (get_active_profile_name() or "").strip()
    except ImportError:
        # A stripped install may not ship the profile registry; omit the param.
        return ""
    return active if active in served else ""
