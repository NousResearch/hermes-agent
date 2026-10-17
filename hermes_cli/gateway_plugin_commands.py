"""Plugin slash commands of the classic CLI on the shared gateway (``slash.exec``).

On main a plugin's ``/command`` ran in this process. Rewired, the plugin's hooks, tools and MCP servers
live in the gateway, so its command runs there too (``gateway/session_commands.py``, the messaging
gateway's own plugin dispatch): ``ctx.call_mcp`` reaches the servers the gateway connected, and this
process never imports the plugin to run it. The names come from the gateway's ``commands.catalog``,
read once per session and gateway life (``/new``, resume and a redial read again) and again once when
a typed name misses, so a plugin the gateway enabled since still dispatches; a cached name a plugin
reload dropped since is re-read once when the gateway refuses it, then dispatched again (:data:`GONE`).
A catalog whose ``plugin_discovery_error`` is set lists no plugins because the gateway could not load
them, not because there are none: it is treated as a failed read and never cached. Off the gateway
(tests, the TUI slash worker's shadow CLI) main's dispatch applies.
"""
from __future__ import annotations

import logging
import threading

from hermes_cli.gateway_client import GatewayClientError

logger = logging.getLogger(__name__)

_CATEGORY = "Plugin commands"  # tui_gateway/command_discovery.py
GONE = object()  # run(): the gateway no longer has the command; the caller dispatches the name again
_GONE_LINE = "  /{}: the gateway no longer has this plugin command; nothing was run."
_STOPPED_LINE = "  /{}: stopped waiting. The command keeps running in the gateway and its reply will not be shown."


class CatalogUnavailable(Exception):
    """The gateway's command list could not be read (already printed): run nothing for this input."""


def _boundary(cli, connection):
    return connection, getattr(connection, "epoch", None), getattr(cli, "session_id", None)


def _read(cli, connection):
    """Fetch the catalog's plugin commands and cache them for the session and gateway life that served
    them (raises GatewayClientError, also when the gateway reports its plugin discovery failed). The
    result is published only if neither moved during the call: a reply from epoch E that arrives after
    a concurrent redial is never relabelled E+1 (``call_tagged``)."""
    session = getattr(cli, "session_id", None)
    catalog, epoch = _catalog(cli, connection, session)
    catalog = catalog or {}
    if catalog.get("plugin_discovery_error"):  # absent on older gateways: today's behaviour
        raise GatewayClientError(f"plugin command discovery failed: {catalog['plugin_discovery_error']}")
    listed = catalog.get("plugin_commands")
    if isinstance(listed, dict):  # every plugin command, also one a display row (Ink's /logs) hides
        found = {str(name).lstrip("/"): str(desc or "") for name, desc in listed.items()}
    else:  # older gateways: the display category
        found = {}
        for category in catalog.get("categories") or []:
            if isinstance(category, dict) and category.get("name") == _CATEGORY:
                for row in category.get("pairs") or []:
                    if isinstance(row, (list, tuple)) and len(row) == 2 and isinstance(row[0], str):
                        found[row[0].lstrip("/")] = str(row[1] or "")
    # The session's skill commands as the gateway resolved them (its plugin skills included): /help,
    # completion and dispatch read these, so this process never scans plugin skills to list them.
    descriptions = {row[0]: str(row[1] or "") for row in catalog.get("pairs") or []
                    if isinstance(row, (list, tuple)) and len(row) == 2 and isinstance(row[0], str)}
    skills = {key: {"name": key.lstrip("/"), "description": descriptions.get(key) or "Skill"}
              for key in catalog.get("skills") or {} if isinstance(key, str)}
    served = (connection, epoch, session)
    if _boundary(cli, connection) == served:
        cli._gateway_plugin_commands = (served, found)
        cli._gateway_skills = (served, skills, str(catalog.get("warning") or ""))
    return found


def _catalog(cli, connection, session):
    """``(commands.catalog, epoch)`` for the open gateway session, so its skills come from the session's
    folder (its trusted project skills), not the gateway's. A gateway older than this CLI refuses the
    ``session_id`` (``invalid_params``): it is asked again without it, as before."""
    from hermes_cli.gateway_client import GatewayRPCError
    if session and getattr(cli, "_gateway_session_open", False):
        try:
            return connection.call_tagged("commands.catalog", session_id=session)
        except GatewayRPCError as exc:
            if str(exc) != "invalid_params":
                raise
    return connection.call_tagged("commands.catalog")


def invalidate(cli):
    """Drop ``cli``'s copy of the gateway's catalog after a skill rescan (``/reload-skills``, ``/skills
    install``): the next read asks the gateway again."""
    cli._gateway_plugin_commands = cli._gateway_skills = None


def _hung_up(connection):
    """A closed connection keeps its epoch until the next call redials, maybe to another life: no
    cache it labelled is served."""
    return getattr(connection, "closed", False)


def _cached(cli, connection):
    if _hung_up(connection):
        return None
    boundary, found = getattr(cli, "_gateway_plugin_commands", None) or (None, None)
    return found if boundary == _boundary(cli, connection) else None


def skills(cli, *, notes=False):
    """``{"/key": {name, description}}`` of the session's skill commands from the gateway's catalog
    (``notes``: its one-line notice instead, e.g. skills a built-in command shadows), or None off the
    gateway. A failed read is printed once per input and lists none; nothing is scanned here."""
    from hermes_cli.gateway_side import connection_for
    connection = connection_for(cli)
    if connection is None:
        return None
    boundary, found, warning = getattr(cli, "_gateway_skills", None) or (None, None, "")
    if _hung_up(connection) or boundary != _boundary(cli, connection):
        try:
            _read(cli, connection)
        except GatewayClientError as exc:
            logger.debug("commands.catalog unavailable: %s", exc)
            return "" if notes else {}
        _seen, found, warning = getattr(cli, "_gateway_skills", None) or (None, {}, "")
    return warning if notes else found


def skill_notes(cli):
    """``/help skills`` notes (skills a built-in command shadows): the gateway's notice on its session,
    else this process's own scan."""
    notice = skills(cli, notes=True)
    if notice is not None:
        return [notice] if notice else []
    from agent.skill_commands import skill_command_collision_note
    from tools.skills_tool import _find_all_skills
    return list(filter(None, (skill_command_collision_note(s["name"]) for s in _find_all_skills())))


def run_skill(cli, base_cmd, rest):
    """``/skill ...`` on the gateway session: the gateway resolves and loads the skill (its plugin
    skills and stacked skills too) and returns the message, which is queued as main queued it; True
    when answered there (False off the gateway: main's local load)."""
    from hermes_cli.gateway_side import connection_for
    connection = connection_for(cli)
    if connection is None:
        return False
    from agent.i18n import t
    from cli import _cprint
    command = base_cmd.lstrip("/") + (" " + rest if rest else "")
    try:
        reply = connection.call("slash.exec", session_id=cli.session_id, command=command) or {}
    except GatewayClientError as exc:
        _cprint(f"  {base_cmd}: {exc}")
        return True
    if reply.get("type") == "skill" and reply.get("message"):
        cli._queue_loaded_skills(reply["message"], t("cli.skills.loading_skill", name=reply.get("name") or command), None)
    if reply.get("notice") or reply.get("output"):
        _cprint(f"  {reply.get('notice') or reply.get('output')}")
    return True


def toolsets(cli):
    """``([(key, label, description)], {portable MCP server})`` the gateway's plugins provide for this
    profile (``plugins.list``), or None off the gateway (this process's own registry applies). An older
    gateway or a failed read gives none, never a local scan. Read once per gateway life."""
    from hermes_cli.gateway_side import connection_for
    connection = connection_for(cli)
    if connection is None:
        return None
    boundary = (connection, getattr(connection, "epoch", None))
    cached = getattr(cli, "_gateway_plugin_toolsets", None)
    if cached and cached[0] == boundary and not _hung_up(connection):
        return cached[1]
    try:
        reply = connection.call("plugins.list") or {}
    except GatewayClientError as exc:
        logger.debug("plugins.list unavailable: %s", exc)
        return [], set()
    rows = [(str(r.get("key")), str(r.get("label") or r.get("key")), str(r.get("description") or ""))
            for r in reply.get("toolsets") or [] if isinstance(r, dict) and isinstance(r.get("key"), str)]
    found = (rows, {str(n) for n in reply.get("portable_mcp_servers") or [] if isinstance(n, str)})
    cli._gateway_plugin_toolsets = (boundary, found)
    return found


def names(cli):
    """``{name: description}`` of the gateway's plugin commands for completion, or None off the
    gateway. A failed read degrades quietly to none; dispatch uses :func:`lookup`."""
    from hermes_cli.gateway_side import connection_for
    connection = connection_for(cli)
    if connection is None:
        return None
    found = _cached(cli, connection)
    if found is None:
        try:
            found = _read(cli, connection)
        except GatewayClientError:
            return {}  # not cached: the next read asks again
    return found


def lookup(cli, name):
    """Whether ``/name`` is one of the gateway's plugin commands; None off the gateway. A name the
    cached list lacks re-reads it once. A failed read is not evidence the name is no plugin: it is
    printed and :class:`CatalogUnavailable` raised, so no skill, bundle or expansion stands in."""
    from hermes_cli.gateway_side import connection_for
    connection = connection_for(cli)
    if connection is None:
        return None
    found = _cached(cli, connection)
    if found is not None and name in found:
        return True
    try:
        return name in _read(cli, connection)
    except GatewayClientError as exc:
        from cli import _cprint
        _cprint(f"  /{name}: the gateway's command list is unavailable ({exc}); nothing was run.")
        raise CatalogUnavailable(name) from exc


def run(cli, name, arg):
    """True when the gateway answered ``/name arg`` (the local handler must not run); :data:`GONE`
    when it no longer has that plugin command and a fresh catalog read agrees (a plugin reload, e.g.
    ``hermes plugins disable``, dropped it in this session and gateway life, which no cache boundary
    sees): nothing ran, and the caller dispatches the name again (bundle, skill, prefix)."""
    from hermes_cli.gateway_side import connection_for
    connection = connection_for(cli)
    if connection is None:
        return False
    from cli import _cprint
    from hermes_cli.gateway_client import GatewayRPCError
    from hermes_cli.gateway_connection import CallStopped
    # No answer deadline: main's CLI waited for the handler, which may wait minutes (ctx.call_mcp). On
    # a socket of its own, so the session's prompts and interrupts never queue behind it.
    wake = cli._plugin_command_wait = threading.Event()  # set by Ctrl+C (cli_tui_mixin)
    try:
        reply = connection.call_until("slash.exec", wake, session_id=cli.session_id,
                                      command=name + (" " + arg if arg else ""))
    except (CallStopped, KeyboardInterrupt) as exc:
        _cprint(_STOPPED_LINE.format(name))
        if isinstance(exc, KeyboardInterrupt):
            raise
        return True
    except GatewayRPCError as exc:
        if str(exc) != "unsupported_command":
            _cprint(f"  /{name}: {exc}")
            return True
        reply = {"type": "unsupported"}
    except GatewayClientError as exc:
        _cprint(f"  /{name}: {exc}")
        return True
    finally:
        cli._plugin_command_wait = None
    reply = reply or {}
    if reply.get("type", "exec") != "exec":  # it lost the plugin since lookup: a refusal or skill reply
        if _dropped(cli, connection, name):
            return GONE
        _cprint(_GONE_LINE.format(name))
        return True
    output = str(reply.get("output") or "")
    if output:
        _cprint(output)
    return True


def _dropped(cli, connection, name):
    """Whether the gateway's catalog, read again with the cached copy dropped, no longer lists
    ``name`` (a failed read: not known, so False)."""
    cli._gateway_plugin_commands = None
    try:
        return name not in _read(cli, connection)
    except GatewayClientError:
        return False


def report_gone(name, *known):
    """Print that ``/name`` is no longer a plugin command, unless one of the built-in commands or a
    ``known`` skill or bundle name starts with it (then prefix expansion decides); True when printed."""
    from hermes_cli.commands import COMMANDS
    if any(key.startswith("/" + name) for group in (COMMANDS, *known) for key in group):
        return False
    from cli import _cprint
    _cprint(_GONE_LINE.format(name))
    return True


def observe(cli, name, arg, *, runs=True):
    """Tell the gateway about a recognized command this process runs itself (``command.observe``), so
    ``pre_command`` fires there, in the session's profile, and never here; False off the gateway (main's
    in-process hook applies). A command that ``runs`` nothing here (a refusal) is not reported. The
    gateway replies before any plugin runs. One older than this CLI (no such method) gets one printed
    line, and is not asked again on that connection life: local commands then reach no ``pre_command``
    observer, and this process still fires none itself."""
    from hermes_cli.gateway_side import connection_for
    connection = connection_for(cli)
    if connection is None:
        return False
    if not runs:
        return True
    epoch = getattr(connection, "epoch", None)
    if getattr(cli, "_observe_unsupported", None) == (connection, epoch):
        return True
    from hermes_cli.gateway_client import GatewayRPCError
    try:
        connection.call("command.observe", session_id=cli.session_id, name=name, arg=arg)
    except GatewayRPCError as exc:
        if exc.code != -32601 and str(exc) != "unknown_method":
            logger.debug("command.observe /%s refused: %s", name, exc)
            return True
        cli._observe_unsupported = (connection, epoch)
        if not getattr(cli, "_observe_notice_shown", False):
            cli._observe_notice_shown = True
            from cli import _cprint
            _cprint("  This gateway predates this CLI: plugin pre_command hooks don't see the commands the CLI "
                    "runs itself until it restarts.")
    except GatewayClientError as exc:
        logger.debug("command.observe /%s not delivered: %s", name, exc)
    return True


def loaded(cli):
    """``{name: row}`` of the plugins the gateway loaded (tool/hook/command counts, load error) for
    ``/plugins``, or None off the gateway. A gateway without ``plugins.list`` on this socket (older
    than this CLI) or one that can't answer gets one printed line and ``{}``: the listing then shows
    no load details, and this process never loads plugins to fill them in."""
    from hermes_cli.gateway_side import connection_for
    connection = connection_for(cli)
    if connection is None:
        return None
    from cli import _cprint
    from hermes_cli.gateway_client import GatewayRPCError
    try:
        rows = (connection.call("plugins.list") or {}).get("plugins") or []
    except GatewayRPCError as exc:
        older = exc.code == -32601 or str(exc) == "unknown_method"
        _cprint("  This gateway doesn't report its loaded plugins (it predates this CLI); restart it for load details."
                if older else f"  The gateway's plugin list is unavailable ({exc}).")
        return {}
    except GatewayClientError as exc:
        _cprint(f"  The gateway's plugin list is unavailable ({exc}).")
        return {}
    return {row["name"]: row for row in rows if isinstance(row, dict) and isinstance(row.get("name"), str)}
