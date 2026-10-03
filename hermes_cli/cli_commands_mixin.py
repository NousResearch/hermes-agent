"""Slash-command handlers for the interactive CLI (``HermesCLI`` inherits ``CLICommandsMixin``).

cli.py-internal symbols (``_cprint``/``_ACCENT``/``save_config_value``…) are imported LAZILY inside
the helpers/handlers via ``from cli import ...`` — cli.py imports this module (cycle otherwise).
"""

from __future__ import annotations

import argparse
import atexit
import importlib
import io
import json
import os
import shlex
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import redirect_stdout, suppress
from io import StringIO
from datetime import datetime
from urllib.parse import urlparse

from rich import box as rich_box
from rich.markup import escape as _escape
from rich.panel import Panel

from hermes_constants import display_hermes_home
from hermes_state_ids import new_session_id as mint_session_id
from agent.i18n import t
from agent.message_metadata import message_identity
from agent.turn_context import extract_api_content_sidecar
from hermes_cli.cli_agent_setup_mixin import _retire_agent
from hermes_cli.browser_connect import (
    DEFAULT_BROWSER_CDP_URL, discover_local_cdp_url, find_free_debug_port, is_browser_debug_ready,
    launch_chrome_debug, local_port_in_use, manual_chrome_debug_command)


# Output helpers. Slash-command text is user-visible: every literal below is load-bearing.
def _cp(*lines: str) -> None:
    """``_cprint`` each line (lazy import: cli.py imports this module)."""
    from cli import _cprint
    for line in lines:
        _cprint(line)


def _pr(*lines: str) -> None:
    """print() each line."""
    for line in lines:
        print(line)


def _save(key: str, value) -> bool:
    """cli.save_config_value, resolved lazily (cli.py imports this module)."""
    from cli import save_config_value
    return save_config_value(key, value)


def _dim(text: str) -> str:
    """Wrap ``text`` in the dim ANSI escape."""
    from cli import _DIM, _RST
    return f"{_DIM}{text}{_RST}"


def _dim_line(text: str) -> str:
    """Two-space indented dim line (the standard slash-command hint shape)."""
    return "  " + _dim(text)


def _accent(text: str) -> str:
    """Wrap ``text`` in the skin accent ANSI escape."""
    from cli import _ACCENT, _RST
    return f"{_ACCENT}{text}{_RST}"


def _accent_line(text: str) -> str:
    """Two-space indented accent line (the standard slash-command headline shape)."""
    return f"  {_accent(text)}"


def _t(key: str, **kwargs) -> str:
    """Catalog text for this module: ``cli.commands.<key>`` in the active language."""
    return t(f"cli.commands.{key}", **kwargs)


def _tn(key: str, count: int, **kwargs) -> str:
    """Plural-aware ``_t``: ``<key>_one`` when ``count == 1``, else ``<key>_other``."""
    return _t(f"{key}_{'one' if count == 1 else 'other'}", count=count, **kwargs)


def _gt(key: str, **kwargs) -> str:
    """A ``gateway.<key>`` catalog entry the CLI shares verbatim with the messaging gateway."""
    return t(f"gateway.{key}", **kwargs)


def _lines(text: str, pad: str = "  ") -> list:
    """Each line of a (possibly multi-line) catalog value prefixed with ``pad`` (blank lines kept)."""
    return [f"{pad}{line}" if line else "" for line in text.splitlines()]


def _probe(module: str, name: str, default, *args):
    """``<module>.<name>(*args)`` or ``default`` when the import or the call fails (optional
    subsystems: browser backends, async delegations, wake word, ...)."""
    try:
        return getattr(importlib.import_module(module), name)(*args)
    except Exception:
        return default


class _TTYBuf(StringIO):
    """StringIO that claims to be a TTY so ``hermes_cli.colors.color()`` still emits ANSI escapes."""
    def isatty(self) -> bool:
        return True


_FAILED = object()


def _attempt(label: str, errors, fn, *args, **kwargs):
    """``fn(*args, **kwargs)``, or ``_FAILED`` after printing ``  <label>: <exc>`` for ``errors``."""
    try:
        return fn(*args, **kwargs)
    except errors as exc:
        _cp(f"  {label}: {exc}")
        return _FAILED


def _say_block(*lines: str) -> None:
    """print() the lines framed by a blank line above and below (the /browser output style)."""
    _pr("", *lines, "")


def _command_arg(cmd: str, *, lower: bool = False) -> str:
    """Everything after the slash-command word, stripped (optionally lowercased)."""
    parts = (cmd or "").strip().split(None, 1)
    arg = parts[1].strip() if len(parts) > 1 else ""
    return arg.lower() if lower else arg


def _shlex_args(cmd: str) -> list:
    """Tokens after the command word; falls back to whitespace split on unbalanced quotes."""
    try:
        return shlex.split(cmd)[1:] if cmd else []
    except ValueError:
        return (cmd or "").split()[1:]


def _take_flag(parts: list, flag: str):
    """Pop ``flag VALUE`` out of ``parts``: ``(rest, value, ok)``; ok=False when the value is
    missing (caller prints usage)."""
    if flag not in parts:
        return parts, None, True
    idx = parts.index(flag)
    if idx + 1 >= len(parts):
        return parts, None, False
    return parts[:idx] + parts[idx + 2:], parts[idx + 1], True


def _summarize_paths(paths, limit: int = 5) -> str:
    """``a, b, c (+N more)`` for a list of paths."""
    more = _t("shared.more_suffix", count=len(paths) - limit) if len(paths) > limit else ""
    return ", ".join(paths[:limit]) + more


def _ellipsize(text: str, limit: int) -> str:
    """``text[:limit]`` plus ``...`` when truncated."""
    return f"{text[:limit]}{'...' if len(text) > limit else ''}"


# Small data tables.

# /cron flag tables: flag -> opts key. Order-sensitive in _parse_cron_flags: bool flags never
# consume a value; --repeat is int-validated separately.
_CRON_BOOL_FLAGS = {"--clear-skills": "clear_skills", "--all": "all"}
_CRON_LIST_FLAGS = {"--skill": "skills", "--add-skill": "add_skills", "--remove-skill": "remove_skills"}
_CRON_VALUE_FLAGS = {"--name": "name", "--deliver": "deliver", "--prompt": "prompt", "--schedule": "schedule"}
# /cron subcommand -> CLICommandsMixin method name.
_CRON_SUBCOMMANDS = {
    "list": "_cron_list", "add": "_cron_add", "create": "_cron_add", "edit": "_cron_edit",
    **{k: "_cron_job_action" for k in ("pause", "resume", "run", "remove", "rm", "delete")}}

_ON_WORDS = {"on", "enable", "true", "1"}
_OFF_WORDS = {"off", "disable", "false", "0"}

# /busy modes; what Enter does while Hermes is working lives in the catalog as
# ``cli.commands.busy.short_<mode>`` (status line) / ``long_<mode>`` (post-set explanation).
_BUSY_MODES = ("queue", "steer", "interrupt")

# /fast argument -> (service_tier value, persisted config value)
_FAST_TIERS = {
    "fast": ("priority", "fast"), "on": ("priority", "fast"), "normal": (None, "normal"),
    "off": (None, "normal"), "auto": ("auto", "auto"), "cold": ("cold", "cold"),
    "ultrafast": ("ultrafast", "ultrafast")}

# /reasoning display toggles: arg -> (attr, value, headline key, follow-up note key or None);
# the keys resolve under ``cli.commands.reasoning.*`` at call time.
_REASONING_TOGGLES = {
    **dict.fromkeys(("show", "on"), ("show_reasoning", True, "headline_on", "note_show")),
    **dict.fromkeys(("hide", "off"), ("show_reasoning", False, "headline_off", None)),
    **dict.fromkeys(("full", "all"), ("reasoning_full", True, "headline_full", "note_full")),
    **dict.fromkeys(("clamp", "collapse", "short"), ("reasoning_full", False, "headline_clamped", None)),
}

# /bg AIAgent provider-routing kwargs -> HermesCLI attribute carrying the value.
_BG_PROVIDER_KWARGS = {
    "providers_allowed": "_providers_only", "providers_ignored": "_providers_ignore",
    "providers_order": "_providers_order", "provider_sort": "_provider_sort",
    "provider_require_parameters": "_provider_require_params",
    "provider_data_collection": "_provider_data_collection",
    "openrouter_min_coding_score": "_openrouter_min_coding_score", "fallback_model": "_fallback_model"}

# /worktree subcommand -> CLICommandsMixin method name (all need a repo root).
_WORKTREE_SUBCOMMANDS = {
    **dict.fromkeys(("prune", "gc", "clean"), "_worktree_prune"),
    **dict.fromkeys(("list", "ls"), "_worktree_list"),
    **dict.fromkeys(("new", "add", "create"), "_worktree_new")}

# Message fields copied verbatim onto a /branch row (plus role / tool_name / api_content).
_BRANCH_COPY_KEYS = ("content", "tool_calls", "tool_call_id", "reasoning", "reasoning_details",
                     "codex_reasoning_items", "codex_message_items", "timestamp")

# /hatch progress event -> catalog key (``cli.commands.hatch.*``).
_HATCH_PROGRESS = {"compose": "hatch.progress_compose", "save": "hatch.progress_save"}

# /diff argument -> mode (anything else is a path; --stat/stat is the stat flag).
_DIFF_MODES = {
    "staged": "staged", "--staged": "staged", "cached": "staged", "--cached": "staged",
    "all": "all", "--all": "all", "head": "all", "session": "session"}
# /diff mode -> stat headline catalog key (``cli.commands.diff.*``).
_DIFF_LABELS = {"working": "diff.label_working", "staged": "diff.label_staged", "all": "diff.label_all"}


def _persist_display_choice(key: str, value: str, label: str, note: str) -> None:
    """Save a /busy-style choice to config and report saved vs session-only."""
    if _save(key, value):
        _cp(_accent_line(_t("shared.set_saved", label=label, value=value)), _dim_line(note))
    else:
        _cp(_accent_line(_t("shared.set_session_only", label=label, value=value)))


def _split_scope_flags(raw: str):
    """``(arg, explicit_global)`` for /reasoning + /fast: session scope by default, ``--global``
    persists to config.yaml, ``--session`` is an explicit no-op (parity with /model)."""
    tokens = raw.strip().lower().split()
    return " ".join(t for t in tokens if t not in ("--global", "--session")), "--global" in tokens


def _scope_outcome(explicit_global: bool, saved: bool) -> str:
    """Parenthetical tail for a scoped setting change."""
    if saved:
        return _t("shared.scope_saved")
    if explicit_global:
        return _t("shared.scope_save_failed")
    return _t("shared.scope_session")


def _toggle_target(arg: str, current: bool):
    """Resolve a ``/x [on|off|status]`` argument: "status" for a status query, a bool for the
    new state (bare arg toggles), or None when the argument is unrecognized."""
    if arg in {"status", "?"}:
        return "status"
    if arg in _ON_WORDS:
        return True
    if arg in _OFF_WORDS:
        return False
    if arg == "":
        return not current
    return None


def _cron_api(**kwargs) -> dict:
    """Call the cronjob model tool and decode its JSON reply."""
    from tools.cronjob_tools import cronjob as cronjob_tool
    return json.loads(cronjob_tool(**kwargs))


def _normalize_skills(values) -> list:
    """Strip, drop empties, and dedupe (order-preserving)."""
    normalized = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _parse_cron_flags(tokens):
    """Parse /cron flags into an opts dict (None after printing an error for a bad --repeat)."""
    opts = {
        "name": None, "deliver": None, "repeat": None, "prompt": None, "schedule": None,
        "skills": [], "add_skills": [], "remove_skills": [],
        "clear_skills": False, "all": False, "positionals": []}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        has_value = i + 1 < len(tokens)
        if token in _CRON_BOOL_FLAGS:
            opts[_CRON_BOOL_FLAGS[token]] = True
            i += 1
        elif token in _CRON_LIST_FLAGS and has_value:
            opts[_CRON_LIST_FLAGS[token]].append(tokens[i + 1])
            i += 2
        elif token == "--repeat" and has_value:
            try:
                opts["repeat"] = int(tokens[i + 1])
            except ValueError:
                return print(_t("cron.repeat_must_be_integer"))
            i += 2
        elif token in _CRON_VALUE_FLAGS and has_value:
            opts[_CRON_VALUE_FLAGS[token]] = tokens[i + 1]
            i += 2
        else:
            opts["positionals"].append(token)
            i += 1
    return opts


# Session-switch plumbing shared by /resume and /branch.
def _end_current_session(cli, reason: str) -> None:
    """Flush un-persisted messages, then end the current session row with ``reason``.
    Best-effort on both steps (the switch proceeds even if the DB write fails)."""
    if cli.agent:
        with suppress(Exception):
            cli.agent._flush_messages_to_session_db(
                cli.conversation_history, conversation_history=cli.conversation_history)
    with suppress(Exception):
        cli._session_db.end_session(cli.session_id, reason)


def _sync_agent_to_session(cli, session_id: str, *, parent_session_id: str, reason: str) -> None:
    """Point an already-built agent at ``session_id`` after a /resume or /branch switch: reset
    per-session state, re-anchor the DB flush index, and notify memory providers with
    reset=False (their state stays valid and just targets the new id; parent keeps lineage)."""
    if not cli.agent:
        return
    cli.agent.session_id = session_id
    cli.agent.reset_session_state()
    if hasattr(cli.agent, "_last_flushed_db_idx"):
        cli.agent._last_flushed_db_idx = len(cli.conversation_history)
    if hasattr(cli.agent, "_todo_store"):
        with suppress(Exception):
            from tools.todo_tool import TodoStore
            cli.agent._todo_store = TodoStore()
    if hasattr(cli.agent, "_invalidate_system_prompt"):
        cli.agent._invalidate_system_prompt()
    with suppress(Exception):
        _mm = getattr(cli.agent, "_memory_manager", None)
        # Notify memory providers that session_id rotated to a fresh conversation. reset=True signals
        # providers to flush accumulated per-session state (_session_turns, _turn_counter, _document_id).
        # Fires BEFORE the plugin on_session_reset hook (shell hooks only see the new id; Python providers
        # see the transition). See #6672. When the old session has history, end-of-session extraction
        # (LLM-bound, seconds) and this switch are queued as ONE task on the memory manager's serialized
        # worker — end strictly before switch, without blocking /new (#16454). With no history there is
        # nothing to extract; switch inline as before.
        # Notify memory providers that session_id rotated to a resumed session. reset=False — the provider's
        # accumulated state is still valid; it just needs to target the new session_id for subsequent
        # writes. See #6672.
        # Notify memory providers that session_id forked to a new branch. reset=False — the branched session
        # carries the transcript forward, so provider state tracks the lineage. parent_session_id links the
        # branch back to the original. See #6672.
        if _mm is not None:
            _mm.on_session_switch(
                session_id, parent_session_id=parent_session_id or "", reset=False, reason=reason)


def _without_session_meta(messages) -> list:
    return [m for m in (messages or []) if m.get("role") != "session_meta"]


def _db_unavailable_line() -> str:
    from hermes_state import format_session_db_unavailable
    return f"  {format_session_db_unavailable(details=True)}"


def _print_side_result_panel(cli, *, header_lines, body, title_suffix, empty_note, console=None) -> None:
    """Print a worker-thread result (/bg, /btw, /login) into the scrollback: accent rules around
    ``header_lines``, then ``body`` in a skinned Rich panel (or ``empty_note``).
    Forces a TUI refresh first so the spinner/status bar don't overlap the output."""
    from cli import ChatConsole, _accent_hex, _maybe_remap_for_light_mode, _render_final_assistant_content
    _refresh_tui_before_print(cli)
    rich_console = console or ChatConsole()
    rich_console.print(f"[{_accent_hex()}]{'─' * 40}[/]")
    if console is None:
        _cp(*header_lines)
    else:
        for line in header_lines:
            console.print(line)
    rich_console.print(f"[{_accent_hex()}]{'─' * 40}[/]")
    if not body:
        return _cp(empty_note) if console is None else console.print(empty_note)
    try:
        from hermes_cli.skin_engine import get_active_skin
        _skin = get_active_skin()
        label = _skin.get_branding("response_label", "☤ Hermes")
        _resp_color = _maybe_remap_for_light_mode(_skin.get_color("response_border", "#CD7F32"))
        _resp_text = _maybe_remap_for_light_mode(_skin.get_color("banner_text", "#FFF8DC"))
    except Exception:
        label, _resp_color, _resp_text = "☤ Hermes", "#CD7F32", "#FFF8DC"
    rich_console.print(Panel(
        _render_final_assistant_content(body, mode=cli.final_response_markdown),
        title=f"[{_resp_color} bold]{label} {title_suffix}[/]", title_align="left",
        border_style=_resp_color, style=_resp_text, box=rich_box.HORIZONTALS, padding=(1, 4),
        width=cli._scrollback_box_width()))


def _refresh_tui_before_print(cli) -> None:
    """Invalidate the running TUI (brief pause for the redraw) then print a blank separator, so
    worker-thread output doesn't overlap the spinner/status bar."""
    if cli._app:
        cli._app.invalidate()
        time.sleep(0.05)
    print()


# /browser sub-handlers.
def _print_lightpanda_engine_status() -> None:
    """``/browser status`` line(s) about ``browser.engine: lightpanda`` — silent unless set;
    says whether it is in use or which higher-precedence setting shadows it."""
    if not _probe("tools.browser_tool_lightpanda_fallback", "_using_lightpanda_engine", False):
        return
    used, reason = _probe("tools.browser_tool_lightpanda_fallback", "lightpanda_engine_status", (None, None))
    if reason is None:
        return
    if not used:
        return print(f"   {_t('browser.engine_not_in_use', reason=reason)}")
    print(f"   {_t('browser.engine_lightpanda', reason=reason)}")
    try:
        from tools.browser_lightpanda import LIGHTPANDA_INSTALL_HINT, find_lightpanda_binary
        lightpanda_bin = find_lightpanda_binary()
    except Exception:
        return
    print(f"   {_t('browser.binary', path=lightpanda_bin)}" if lightpanda_bin
          else f"   {_t('browser.binary_not_found', hint=LIGHTPANDA_INSTALL_HINT)}")


def _browser_use(cli, arg: str) -> None:
    """/browser use [off] — toggle Browser Use mode (browser.backend); resets the session."""
    from tools.browser_use_cli import set_browser_use_mode
    if arg not in {"on", "off"}:
        return _say_block(
            _t("browser.use_usage"),
            f"   {_t('browser.use_on_hint')}", f"   {_t('browser.use_off_hint')}")
    set_browser_use_mode(arg == "on")
    headline = _t("browser.use_enabled" if arg == "on" else "browser.use_disabled")
    cli.new_session()
    _say_block(headline, f"   {_t('browser.session_reset')}")


def _normalize_cdp_url(cdp_url: str):
    """Validate a /browser connect URL: ``(cdp_url, port)`` or None after printing the error.
    A ``/devtools/browser/<id>`` path is kept verbatim; anything else is reduced to the origin."""
    parsed = urlparse(cdp_url if "://" in cdp_url else f"http://{cdp_url}")
    if parsed.scheme not in {"http", "https", "ws", "wss"}:
        _say_block(f"   {_t('browser.unsupported_scheme', scheme=parsed.scheme or _t('browser.scheme_missing'))}")
        return None
    try:
        port = parsed.port or (443 if parsed.scheme in {"https", "wss"} else 80)
    except ValueError:
        _say_block(f"   {_t('browser.invalid_port', url=cdp_url)}")
        return None
    if not parsed.hostname:
        _say_block(f"   {_t('browser.missing_host', url=cdp_url)}")
        return None
    if parsed.path.startswith("/devtools/browser/"):
        return parsed.geturl(), port
    return parsed._replace(path="", params="", query="", fragment="").geturl(), port


def _launch_default_cdp_browser(port: int):
    """Launch a local debug browser for the default CDP URL; returns the discovered CDP url or
    None (after printing what went wrong / how to launch manually)."""
    import platform as _plat
    launch_port = port
    if local_port_in_use(port):
        launch_port = find_free_debug_port(port)
        _pr(f"   {_t('browser.port_occupied', port=port)}",
            f"     {_t('browser.port_occupied_detail', launch_port=launch_port)}")
    else:
        print(f"   {_t('browser.launching')}")
    launch = launch_chrome_debug(launch_port, _plat.system())
    if not launch.launched:
        print(f"   {_t('browser.launch_failed')}")
        if launch.hint:
            print(f"     {launch.hint}")
        chrome_cmd = manual_chrome_debug_command(launch_port, _plat.system())
        if chrome_cmd:
            _pr(f"     {_t('browser.launch_manually')}", f"     {chrome_cmd}")
        else:
            print(f"     {_t('browser.no_executable')}")
        return None
    for _wait in range(10):  # wait for the DevTools discovery endpoint to come up
        found = discover_local_cdp_url(launch_port, timeout=1.0)
        if found:
            print(f"   {_t('browser.launched', port=launch_port)}")
            return found
        time.sleep(0.5)
    _pr(f"   {_t('browser.not_responding', port=launch_port)}", f"     {_t('browser.try_again')}")
    return None


def _browser_connect(cli, cdp_url: str) -> None:
    """/browser connect [url] — validate the CDP URL, find or launch a debug browser, then
    point the browser tools at it (BROWSER_CDP_URL) and tell the model."""
    normalized = _normalize_cdp_url(cdp_url)
    if normalized is None:
        return
    cdp_url, port = normalized
    # Clear any existing browser sessions so the next tool call uses the new backend
    _probe("tools.browser_tool_lifecycle", "cleanup_all_browsers", None)
    print()
    # Already serving CDP? For the default-local URL probe both loopbacks: a squatter
    # on 127.0.0.1:<port> (e.g. an IDE debugger) can push the browser to bind [::1] only.
    is_default = cdp_url == DEFAULT_BROWSER_CDP_URL
    if is_default:
        found = discover_local_cdp_url(port, timeout=1.0)
    else:
        found = cdp_url if is_browser_debug_ready(cdp_url, timeout=1.0) else None
    if found:
        print(f"   {_t('browser.already_listening', url=found)}")
    elif is_default:
        found = _launch_default_cdp_browser(port)
    else:
        print(f"   {_t('browser.port_unreachable', port=port, url=cdp_url)}")
    if not found:
        return _say_block(_t("browser.not_connected"))
    os.environ["BROWSER_CDP_URL"] = found
    # Eagerly start the CDP supervisor so pending_dialogs + frame_tree show up in the next snapshot.
    _probe("tools.browser_tool_cdp", "_ensure_cdp_supervisor", None, "default")
    _say_block(_t("browser.connected"), f"   {_t('browser.endpoint', url=found)}")
    # Tell the model the CDP browser was made available on purpose.
    if hasattr(cli, '_pending_input'):
        cli._pending_input.put(
            "[System note: The user invoked /browser connect and connected your browser tools to "
            "a Chromium-family dev/debug browser via Chrome DevTools Protocol. "
            "Your browser_navigate, browser_snapshot, browser_click, and other browser tools now "
            "control that CDP browser. The command itself is a signal that using browser tools for "
            "their current browser-related request is expected; do not wait for separate permission "
            "just because CDP is connected. This is typically a Hermes-managed isolated debug "
            "profile, not the user's main everyday browser. It is still user-visible and may contain "
            "pages, logged-in sessions, or cookies in that debug profile, so avoid destructive actions, "
            "closing tabs, or navigating away unless the user's task calls for it.]")


def _browser_disconnect(cli) -> None:
    if not os.environ.get("BROWSER_CDP_URL", "").strip():
        return _say_block(_t("browser.already_default"))
    os.environ.pop("BROWSER_CDP_URL", None)
    with suppress(Exception):
        from tools.browser_tool_lifecycle import cleanup_all_browsers
        from tools.browser_tool_cdp import _stop_cdp_supervisor
        _stop_cdp_supervisor("default")
        cleanup_all_browsers()
    _say_block(_t("browser.disconnected"), f"   {_t('browser.reverted')}")
    if hasattr(cli, '_pending_input'):
        cli._pending_input.put(
            "[System note: The user has disconnected the browser tools from their live Chromium-family browser. "
            "Browser tools are back to default mode (headless local browser or cloud provider).]")


# /browser status headline per local browser.engine value: (headline key, *detail keys) under
# ``cli.commands.browser.*``; detail lines are indented at print time.
_LOCAL_ENGINE_LINES = {
    "lightpanda": ("status_lightpanda", "status_lightpanda_note", "status_lightpanda_fallback"),
    "chrome": ("status_chrome",),
    "auto": ("status_auto",)}


def _local_engine_lines(engine: str) -> list:
    head, *details = _LOCAL_ENGINE_LINES.get(engine, _LOCAL_ENGINE_LINES["auto"])
    return [_t(f"browser.{head}"), *(f"   {_t(f'browser.{key}')}" for key in details)]


def _browser_status() -> None:
    current = os.environ.get("BROWSER_CDP_URL", "").strip()
    print()
    if _probe("tools.browser_use_cli", "is_browser_use_cli_mode", False):
        _pr(_t("browser.status_browser_use"), f"   {_t('browser.status_browser_use_note')}")
        _print_lightpanda_engine_status()
        return _say_block(f"   {_t('browser.status_use_off_hint')}")
    if current:
        _pr(_t("browser.status_connected"), f"   {_t('browser.endpoint', url=current)}")
        _print_lightpanda_engine_status()
        _port = 9222
        with suppress(ValueError, IndexError):
            _port = int(current.rsplit(":", 1)[-1].split("/")[0])
        try:
            import socket
            socket.create_connection(("127.0.0.1", _port), timeout=1).close()
            print(f"   {_t('browser.status_reachable')}")
        except Exception:
            print(f"   {_t('browser.status_unreachable')}")
    else:
        provider = _probe("tools.browser_tool_cloud", "_get_cloud_provider", None)
        if provider is not None:
            print(_t("browser.status_cloud", provider=provider.display_name))
            _print_lightpanda_engine_status()
        else:
            engine = _probe("tools.browser_tool_cloud", "_get_browser_engine", "auto")
            _pr(*_local_engine_lines(engine))
            if engine == "lightpanda":
                _print_lightpanda_engine_status()
    _say_block(f"   {_t('browser.status_connect_hint')}", f"   {_t('browser.status_disconnect_hint')}")


# /browser subcommand word → handler(cli, rest); ``rest`` is the raw (case-preserved)
# remainder of the line. Adding a subcommand is one row here plus a usage line.
_BROWSER_SUBCOMMANDS = {
    "use": lambda cli, rest: _browser_use(cli, rest.lower() or "on"),
    "connect": lambda cli, rest: _browser_connect(cli, rest or DEFAULT_BROWSER_CDP_URL),
    "disconnect": lambda cli, rest: _browser_disconnect(cli),
    "status": lambda cli, rest: _browser_status(),
}


class CLICommandsMixin:
    """Mixin holding the interactive-CLI slash-command handlers."""

    # ---- /rollback ------------------------------------------------------------------------
    def _checkpoint_manager(self, disabled_lines):
        """The agent's checkpoint manager, or None after printing why it is unavailable."""
        if not hasattr(self, 'agent') or not self.agent:
            return print(f"  {_t('shared.no_active_agent_session')}")
        mgr = self.agent._checkpoint_mgr
        if not mgr.enabled:
            return _pr(*disabled_lines)
        return mgr

    def _handle_rollback_command(self, command: str):
        """Handle /rollback [diff] <N> [<file>|--all] — list, diff, or restore checkpoints.
        A restore also undoes the last chat turn; ``--all`` overwrites user hand-edits too."""
        from tools.checkpoint_manager import format_checkpoint_list
        mgr = self._checkpoint_manager(_lines(_t("rollback.not_enabled")))
        if mgr is None:
            return
        cwd = os.getenv("TERMINAL_CWD", os.getcwd())
        args = command.split()[1:]
        # --all / --force: classic full restore, overwriting user edits too.
        restore_all = any(a.lower() in ("--all", "--force") for a in args)
        args = [a for a in args if a.lower() not in ("--all", "--force")]
        if reason := mgr.unsupported_backend_reason():  # CLI: no session key, the "default" container
            # Container-backed session: any host checkpoint listed here belongs to another tree,
            # so diff/restore are refused; the list stays visible for local administration.
            print(f"  {reason}")
            if args:
                return
        if not args:
            # No checkpoints for this dir → cross-project view (writes may sit under the session cwd).
            checkpoints = mgr.list_checkpoints(cwd)
            if not checkpoints:
                # List checkpoints — fall back to the cross-project view when the current directory has none
                # (#10505, reapply of PR #10633 by @nightq). The Aug 2026 QA sweep hit this live: writes
                # landed checkpoints under the session cwd (/tmp/qa-repo) while bare /rollback searched only
                # TERMINAL_CWD's project and reported "No checkpoints found" despite fresh checkpoints
                # existing.
                all_checkpoints = mgr.list_all_checkpoints()
                if all_checkpoints:
                    print(f"  {_t('rollback.none_here_showing_all', cwd=cwd)}")
                    return print(format_checkpoint_list(all_checkpoints, _t("rollback.all_directories")))
            return print(format_checkpoint_list(checkpoints, cwd))
        is_diff = args[0].lower() == "diff"
        if is_diff and len(args) < 2:
            return print(f"  {_t('rollback.usage_diff')}")
        checkpoints = mgr.list_checkpoints(cwd)
        if not checkpoints:
            return print(f"  {_gt('rollback.none_found', cwd=cwd)}")
        target_hash = self._resolve_checkpoint_ref(args[1 if is_diff else 0], checkpoints)
        if not target_hash:
            return
        if is_diff:
            self._rollback_diff(mgr, cwd, target_hash)
        else:
            file_path = args[1] if len(args) > 1 else None
            self._rollback_restore(mgr, cwd, target_hash, file_path, restore_all)

    def _rollback_diff(self, mgr, cwd: str, target_hash: str) -> None:
        result = mgr.diff(cwd, target_hash)
        if not result["success"]:
            return print(f"  ❌ {result['error']}")
        stat, diff = result.get("stat", ""), result.get("diff", "")
        if not stat and not diff:
            return print(f"  {_t('rollback.no_changes_since')}")
        if stat:
            print(f"\n{stat}")
        if diff:
            # Limit diff output to avoid terminal flood
            diff_lines = diff.splitlines()
            if len(diff_lines) > 80:
                _pr("\n".join(diff_lines[:80]),
                    f"\n  {_t('rollback.more_lines', count=len(diff_lines) - 80)}")
            else:
                print(f"\n{diff}")

    def _rollback_restore(self, mgr, cwd, target_hash: str, file_path, restore_all: bool) -> None:
        result = mgr.restore(cwd, target_hash, file_path=file_path,
                             safe=not restore_all and not file_path)
        if not result["success"]:
            return print(f"  ❌ {result['error']}")
        restored = dict(hash=result["restored_to"], reason=result["reason"])
        print(f"  {_t('rollback.restored_file', file=file_path, **restored)}" if file_path
              else f"  {_t('rollback.restored_all', **restored)}")
        skipped = result.get("skipped_user_edits") or []
        if skipped:
            _pr(*_lines(_gt("rollback.kept_user_edits", files=_summarize_paths(skipped))))
        oversize = result.get("skipped_oversize") or []
        if oversize:
            print(f"  {_gt('rollback.kept_oversize', files=_summarize_paths(oversize))}")
        failed = result.get("failed_deletes") or []
        if failed:
            print(f"  {_gt('rollback.failed_deletes', files=_summarize_paths(failed))}")
        print(f"  {_t('rollback.snapshot_saved')}")
        # Also undo the last conversation turn so the agent's context matches the restored files.
        if self.conversation_history:
            self.undo_last(prefill=False)
            print(f"  {_t('rollback.chat_turn_undone')}")

    # ---- /diff ----------------------------------------------------------------------------
    def _handle_diff_command(self, command: str):
        """Handle /diff [working|staged|all|session] [--stat] [<path>...] — git changes in the
        cwd; ``session`` is everything Hermes changed since the checkpoint baseline."""
        stat_only = False
        mode = "working"
        paths: list[str] = []
        for arg in _shlex_args(command):  # shlex preserves quoted paths
            low = arg.lower()
            if low in ("--stat", "stat"):
                stat_only = True
            elif low in _DIFF_MODES:
                mode = _DIFF_MODES[low]
            else:
                paths.append(arg)
        cwd = os.getenv("TERMINAL_CWD", os.getcwd())
        if mode == "session":
            return self._print_session_diff(cwd, stat_only)
        from tools.working_diff import collect_working_diff
        result = collect_working_diff(cwd, mode=mode, paths=paths or None)
        if not result.get("success"):
            return print(f"  {result.get('error') or _t('shared.could_not_generate_diff')}")
        stat, diff = result.get("stat", ""), result.get("diff", "")
        untracked = result.get("untracked", [])
        if result.get("empty") or (not stat and not diff and not untracked):
            return print(f"  {_gt('diff.no_changes')}")
        if stat:
            print(f"\n  {_t(_DIFF_LABELS[mode])}:")
            self._print_diff_text(stat)
        if untracked and mode in ("working", "all"):
            _pr(f"\n  {_t('diff.untracked')}", *(f"    + {rel}" for rel in untracked[:20]))
            if len(untracked) > 20:
                print(f"    {_t('diff.and_more', count=len(untracked) - 20)}")
        if diff and not stat_only:
            self._print_diff_body(diff, _t("diff.stat_hint"))

    def _print_diff_body(self, diff: str, stat_hint: str, limit: int = 400) -> None:
        """Print a diff, capped at ``limit`` lines with a pointer to the --stat form."""
        print("")
        diff_lines = diff.splitlines()
        if len(diff_lines) > limit:
            self._print_diff_text("\n".join(diff_lines[:limit]))
            print(f"\n  {_t('diff.more_lines', count=len(diff_lines) - limit, hint=stat_hint)}")
        else:
            self._print_diff_text(diff)

    def _print_session_diff(self, cwd: str, stat_only: bool):
        """Print the cumulative checkpoint-baseline diff (/diff session)."""
        mgr = self._checkpoint_manager(_lines(_t("diff.not_enabled")))
        if mgr is None:
            return
        if reason := mgr.unsupported_backend_reason():  # host baseline is not this session's tree
            return print(f"  {reason}")
        result = mgr.session_diff(cwd)
        if not result.get("success"):
            return print(f"  {result.get('error') or _t('shared.could_not_generate_diff')}")
        stat, diff = result.get("stat", ""), result.get("diff", "")
        if result.get("empty") or (not stat and not diff):
            return print(f"  {_t('diff.no_session_changes')}")
        if stat:
            self._print_diff_text(f"\n{stat}")
        if diff and not stat_only:
            self._print_diff_body(diff, _t("diff.stat_hint_session"))

    def _print_diff_text(self, text: str) -> None:
        """Render diff/stat text with color when a rich console is present; plain print otherwise
        (e.g. unit tests instantiating the mixin standalone)."""
        console = getattr(self, "console", None)
        if console is not None:
            try:
                from cli import _rich_text_from_ansi
                console.print(_rich_text_from_ansi(text))
                return
            except Exception:
                pass
        print(text)

    # ---- /snapshot ------------------------------------------------------------------------
    def _handle_snapshot_command(self, command: str):
        """Handle /snapshot [list|create [label]|restore <id>|prune [N]] — state snapshots."""
        parts = command.split()
        subcmd = parts[1].lower() if len(parts) > 1 else "list"
        handler = {
            "list": self._snapshot_list, "ls": self._snapshot_list, "create": self._snapshot_create,
            "restore": self._snapshot_restore, "rewind": self._snapshot_restore,
            "prune": self._snapshot_prune}.get(subcmd)
        if handler is None:
            return _pr(f"  {_t('snapshot.unknown_subcommand', subcommand=subcmd)}",
                       f"  {_t('snapshot.usage')}")
        handler(parts)

    def _snapshot_list(self, parts) -> None:
        from hermes_cli.backup import list_quick_snapshots
        snaps = list_quick_snapshots()
        if not snaps:
            return _pr(f"  {_t('snapshot.none_yet')}", f"  {_t('snapshot.create_hint')}")
        print(f"  {_t('snapshot.list_header', dir=display_hermes_home())}\n")
        # Column labels are translated values re-padded here (never padded in the catalog).
        _pr(f"  {_t('snapshot.col_num'):>3}  {_t('snapshot.col_id'):<35} {_t('snapshot.col_files'):>5} "
            f"{_t('snapshot.col_size'):>10} {_t('snapshot.col_label')}",
            f"  {'─'*3}  {'─'*35} {'─'*5} {'─'*10} {'─'*20}")
        for i, s in enumerate(snaps, 1):
            size = s.get("total_size", 0)
            size_str = (f"{size} B" if size < 1024 else f"{size / 1024:.0f} KB"
                        if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB")
            label = s.get("label") or ""
            print(f"  {i:3}  {s['id']:<35} {s.get('file_count', 0):>5} {size_str:>10} {label}")

    def _snapshot_create(self, parts) -> None:
        from hermes_cli.backup import create_quick_snapshot
        snap_id = create_quick_snapshot(label=" ".join(parts[2:]) if len(parts) > 2 else None)
        print(f"  {_t('snapshot.created', snapshot_id=snap_id)}" if snap_id
              else f"  {_t('snapshot.nothing_to_snapshot')}")

    def _snapshot_restore(self, parts) -> None:
        from hermes_cli.backup import list_quick_snapshots, restore_quick_snapshot
        if len(parts) < 3:
            print(f"  {_t('snapshot.usage_restore')}")
            snaps = list_quick_snapshots(limit=1)
            if snaps:
                print(f"  {_t('snapshot.most_recent', snapshot_id=snaps[0]['id'])}")
            return
        snap_id = parts[2]
        try:
            idx = int(snap_id)  # restore by number (1-indexed)
        except ValueError:
            idx = None
        if idx is not None:
            snaps = list_quick_snapshots()
            if not 1 <= idx <= len(snaps):
                return print(f"  {_t('snapshot.invalid_number', max=len(snaps))}")
            snap_id = snaps[idx - 1]["id"]
        # Close our SessionDB first so the restore doesn't contend with this process's live connection.
        local_session_db = getattr(self, "_session_db", None)
        if local_session_db is not None:
            with suppress(Exception):
                local_session_db.close()
                self._session_db = None
        if restore_quick_snapshot(snap_id):
            _pr(f"  {_t('snapshot.restored', snapshot_id=snap_id)}",
                f"  {_t('snapshot.restart_recommended')}")
        elif snap_id in {s.get("id") for s in list_quick_snapshots(limit=10**6)}:
            # False also means the auth.json merge was refused; don't call an existing snapshot missing.
            print(f"  {_t('snapshot.restore_incomplete', snapshot_id=snap_id)}")
        else:
            print(f"  {_t('snapshot.not_found', snapshot_id=snap_id)}")

    def _snapshot_prune(self, parts) -> None:
        from hermes_cli.backup import prune_quick_snapshots
        keep = 20
        if len(parts) > 2:
            # isdecimal() also rejects "-1": a negative keep would slice away the
            # newest snapshots instead of the oldest.
            if not parts[2].isdecimal():
                return print(f"  {_t('snapshot.usage_prune')}")
            keep = int(parts[2])
        deleted = prune_quick_snapshots(keep=keep)
        print(f"  {_t('snapshot.pruned', deleted=deleted, keep=keep)}")

    # ---- /export, /import -----------------------------------------------------------------
    def _handle_export_command(self, command: str):
        """Handle /export [profile] [-o path] — export a profile to a shareable .tar.gz archive."""
        from hermes_cli.profiles import export_profile, get_active_profile_name, get_profile_export_path
        parts, output, ok = _take_flag(command.split()[1:], "-o")
        if not ok:
            return print(f"  {_t('export.usage')}")
        name = parts[0] if parts else (get_active_profile_name() or "default")
        try:
            result = export_profile(name, output or str(get_profile_export_path(name)))
            _pr(f"  {_t('export.exported', name=name, path=result)}", f"  {_t('export.share_hint')}")
        except (ValueError, FileNotFoundError, OSError) as e:
            print(f"  {_t('shared.error', error=e)}")

    def _handle_import_command(self, command: str):
        """Handle /import <archive.tar.gz> [--name <name>] — import a shared profile archive as a
        new profile."""
        from hermes_cli.profiles import check_alias_collision, create_wrapper_script, import_profile
        parts, name, ok = _take_flag(command.split()[1:], "--name")
        if not ok or not parts:
            return print(f"  {_t('import.usage')}")
        try:
            profile_dir = import_profile(" ".join(parts), name=name)  # paths may contain spaces
        except (ValueError, FileExistsError, FileNotFoundError) as e:
            return print(f"  {_t('shared.error', error=e)}")
        imported = profile_dir.name
        print(f"  {_t('import.imported', name=imported, path=profile_dir)}")
        with suppress(Exception):
            if not check_alias_collision(imported):
                wrapper_path = create_wrapper_script(imported)
                if wrapper_path:
                    print(f"  {_t('import.wrapper_created', path=wrapper_path)}")
        print(f"  {_t('import.use_it', name=imported)}")

    # ---- /stop, /agents -------------------------------------------------------------------
    def _handle_stop_command(self):
        """Handle /stop — kill all running background processes and background (async) delegations.
        Separate from interrupt (stop the current turn), as in Codex.

        See #14602.
        """
        from tools.process_registry import process_registry
        running = [p for p in process_registry.list_sessions() if p.get("status") == "running"]
        # Background subagents live in their own registry, not the process registry.
        n_async = _probe("tools.async_delegation", "active_count", 0)
        if not running and not n_async:
            return print(f"  {_t('stop.none_running')}")
        if running:
            print(f"  {_t('stop.stopping', count=len(running))}")
            print(f"  {_t('stop.stopped', count=process_registry.kill_all(source='cli.stop'))}")
        if n_async:
            from tools.async_delegation import interrupt_all
            print(f"  {_t('stop.interrupted_delegations', count=interrupt_all(reason='/stop'))}")

    def _handle_agents_command(self):
        """Handle /agents — show background processes and agent status."""
        from tools.process_registry import format_uptime_short, process_registry
        processes = process_registry.list_sessions()
        running = [p for p in processes if p.get("status") == "running"]
        finished = [p for p in processes if p.get("status") != "running"]
        _cp(f"  {_t('agents.running_processes', count=len(running))}")
        for p in running:
            up = format_uptime_short(p.get("uptime_seconds", 0))
            _cp(f"    {p.get('session_id', '?')} · {up} · {p.get('command', '')[:80]}")
        if finished:
            _cp(f"  {_t('agents.recently_finished', count=len(finished))}")
        # Background (async) delegations — delegate_task(background=true)
        delegations = _probe("tools.async_delegation", "list_async_delegations", [])
        if delegations:
            running_d = [d for d in delegations if d.get("status") in ("running", "stalling")]
            _cp(f"  {_t('agents.background_delegations', count=len(running_d))}")
            for d in delegations:
                status = d.get("status", "?")
                line = f"    {d.get('delegation_id', '?')} · {status} · {(d.get('goal') or '')[:60]}"
                # Live-status detail for in-flight delegations.
                # See #51690.
                if status == "stalling":
                    quiet = d.get("stalled_after_quiet_seconds")
                    if quiet is not None:
                        line += _t("agents.no_progress", seconds=f"{quiet:.0f}")
                elif status == "running":
                    quiet = d.get("seconds_since_progress")
                    if quiet is not None and quiet >= 60:
                        line += _t("agents.quiet", seconds=f"{quiet:.0f}")
                _cp(line)
                for i, child in enumerate(d.get("children_activity") or []):
                    if not isinstance(child, dict):
                        continue
                    tool = child.get("current_tool")
                    doing = _t("agents.in_tool", tool=tool) if tool else _t("agents.between_turns")
                    part = "      " + _t("agents.child_line", index=i + 1,
                                         api_calls=child.get("api_calls", "?"), doing=doing)
                    idle = child.get("seconds_since_activity")
                    if idle is not None:
                        part += _t("agents.last_activity", seconds=f"{idle:.0f}")
                    _cp(part)
        agent_running = getattr(self, "_agent_running", False)
        _cp(f"  {_t('agents.agent_running') if agent_running else _t('agents.agent_idle')}")

    # ---- /journey, /paste, /copy, /image --------------------------------------------------
    def _handle_journey_command(self, cmd_original: str) -> None:
        """Handle /journey — the learning timeline (see `hermes journey`). Read-only views render
        Rich color that patch_stdout would swallow, so capture with forced ANSI and re-emit via
        ``_cprint``; ``delete``/``edit`` are interactive and keep real stdio."""
        from hermes_cli.journey import register_cli
        parser = argparse.ArgumentParser(prog="/journey", add_help=False)
        register_cli(parser)
        try:
            args = parser.parse_args(shlex.split(cmd_original)[1:])
        except SystemExit:
            return
        try:
            if getattr(args, "journey_action", None) in ("delete", "edit"):
                args.func(args)
                return
            args.force_color = True
            buf = io.StringIO()
            with redirect_stdout(buf):
                args.func(args)
            _cp(buf.getvalue().rstrip("\n"))
        except Exception as exc:
            _cp(f"  {_t('journey.failed', error=exc)}")

    def _handle_paste_command(self):
        """Handle /paste — explicitly check clipboard for an image.

        This is the reliable fallback for terminals where BracketedPaste
        doesn't fire for image-only clipboard content (e.g., VSCode terminal,
        Windows Terminal with WSL2).
        """
        from hermes_cli.clipboard import has_clipboard_image
        if not has_clipboard_image():
            _cp(_dim_line(_t("paste.no_image")))
        elif self._try_attach_clipboard_image():
            _cp(f"  {_t('paste.attached', index=len(self._attached_images))}")
        else:
            _cp(_dim_line(_t("paste.extract_failed")))

    def _handle_copy_command(self, cmd_original: str) -> None:
        """Handle /copy [number] — copy assistant output to clipboard."""
        from cli import _assistant_copy_text
        arg = _command_arg(cmd_original)
        assistant = [m for m in self.conversation_history if m.get("role") == "assistant"]
        if not assistant:
            return _cp(f"  {_t('copy.nothing_yet')}")
        if arg:
            try:
                idx = int(arg) - 1
            except ValueError:
                return _cp(f"  {_t('copy.usage')}")
            if idx < 0 or idx >= len(assistant):
                return _cp(f"  {_t('copy.invalid_number', max=len(assistant))}")
        else:  # latest response that has copyable text
            idx = next((i for i in range(len(assistant) - 1, -1, -1)
                        if _assistant_copy_text(assistant[i].get("content"))), -1)
            if idx < 0:
                return _cp(f"  {_t('copy.nothing_in_responses')}")
        text = _assistant_copy_text(assistant[idx].get("content"))
        if not text:
            return _cp(f"  {_t('copy.nothing_in_response')}")
        try:
            from hermes_cli.clipboard import is_remote_shell_session, write_clipboard_text
            # Over SSH native tools write the REMOTE clipboard; OSC 52 reaches the user's terminal.
            # Locally, OSC 52 is the fallback when native tools are unavailable/fail (SSH/tmux).
            if is_remote_shell_session() or not write_clipboard_text(text):
                # Fixes #31528.
                self._write_osc52_clipboard(text)
                _cp(f"  {_t('copy.copied_osc52', index=idx + 1)}")
            else:
                _cp(f"  {_t('copy.copied', index=idx + 1)}")
        except Exception as e:
            _cp(f"  {_t('copy.failed', error=e)}")

    def _handle_image_command(self, cmd_original: str):
        """Handle /image <path> — attach a local image file for the next prompt."""
        from cli import _DIM, _IMAGE_EXTENSIONS, _RST, _cprint, _resolve_attachment_path, _split_path_input
        raw_args = (cmd_original.split(None, 1)[1].strip() if " " in cmd_original else "")
        if not raw_args:
            hint = "/path/to/image.png"
            _cprint(f"  {_DIM}{_t('image.usage', example=hint)}{_RST}")
            return

        path_token, _remainder = _split_path_input(raw_args)
        image_path = _resolve_attachment_path(path_token)
        if image_path is None:
            return _cp(_dim_line(_t("image.not_found", path=path_token)))
        if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            return _cp(_dim_line(_t("image.unsupported", name=image_path.name)))
        self._attached_images.append(image_path)
        _cp(f"  {_t('image.attached', name=image_path.name)}")
        if _remainder:
            _cprint(f"  {_DIM}{_t('image.now_type_prompt', text=_remainder)}{_RST}")

    # ---- /tools, /profile -----------------------------------------------------------------
    def _handle_tools_command(self, cmd: str):
        """Handle /tools [list|disable|enable]. Bare shows the tool list; ``list`` shows per-toolset
        status; disable/enable save to config and reset the session so the new tool set takes
        effect cleanly (no prompt-cache breakage mid-conversation)."""
        parts = _shlex_args(cmd)
        subcommand = parts[0] if parts else ""
        if subcommand not in {"list", "disable", "enable"}:
            return self.show_tools()
        if subcommand == "list":
            return self._run_tools_config(tools_action="list", platform="cli")
        names = parts[1:]
        if not names:
            return _pr(_t("tools.usage", subcommand=subcommand),
                       f"  {_t('tools.example_toolset', subcommand=subcommand)}",
                       f"  {_t('tools.example_mcp', subcommand=subcommand)}")
        # Typing the command is consent. Do NOT use input() — it hangs in prompt_toolkit's loop.
        _cp(_accent(_t("tools.disabling" if subcommand == "disable" else "tools.enabling",
                       names=", ".join(names))))
        self._run_tools_config(tools_action=subcommand, names=names, platform="cli")
        from hermes_cli.tools_config import _get_platform_tools
        from hermes_cli.config import load_config
        self.enabled_toolsets = _get_platform_tools(load_config(), "cli")
        self.new_session()
        _cp(_dim(_t("tools.session_reset")))

    def _run_tools_config(self, **ns) -> None:
        """Run ``tools_disable_enable_command``. Inside the interactive TUI its ANSI print() output
        is captured (isatty=True so colors still render) and re-emitted through _cprint so
        patch_stdout's StdoutProxy doesn't garble the escapes; standalone/tests call straight through."""
        from argparse import Namespace
        from hermes_cli.tools_config import tools_disable_enable_command
        if getattr(self, "_app", None) is None:
            return tools_disable_enable_command(Namespace(**ns))
        buf = _TTYBuf()
        with redirect_stdout(buf):
            tools_disable_enable_command(Namespace(**ns))
        _cp(*buf.getvalue().splitlines())

    def _handle_profile_command(self):
        """Display active profile name and home directory."""
        from hermes_cli.slash_exec import CommandContext, execute_command
        reply = execute_command("profile", CommandContext(surface="cli"))
        _say_block(f"  {_t('profile.profile', profile=reply.data['profile'])}",
                   f"  {_t('profile.home', home=reply.data['home'])}")

    # ---- /handoff -------------------------------------------------------------------------

    _HANDOFF_PENDING_TIMEOUT = 60.0
    _HANDOFF_RUNNING_TIMEOUT = 900.0  # full synthetic agent turn + delivery
    _HANDOFF_HEARTBEAT_EVERY = 30.0

    @staticmethod
    def _handoff_keep(*lines: str) -> bool:
        """Print ``lines`` and keep the CLI session (the True verdict of /handoff)."""
        _cp(*lines)
        return True

    def _handle_handoff_command(self, cmd_original: str) -> bool:
        """Handle ``/handoff <platform>`` — transfer this CLI session to a gateway platform.

        Validate target → prepare session row → mark pending → block-poll (see ``_handoff_wait``).
        Returns False only on ``completed`` (caller exits like /quit); True keeps the session."""
        platform_name = _command_arg(cmd_original).lower()
        if not platform_name:
            return self._handoff_keep(
                f"  {_t('handoff.usage')}", f"  {_t('handoff.usage_detail_1')}",
                f"  {_t('handoff.usage_detail_2')}")
        home = self._handoff_validate_target(platform_name)
        if home is None:
            return True
        session_title = self._handoff_prepare_session()
        if session_title is None:
            return True
        if not self._session_db.request_handoff(self.session_id, platform_name):
            return self._handoff_keep(f"  {_t('handoff.already_in_flight')}")
        _cp(f"  {_t('handoff.queued', title=session_title, platform=platform_name, home=home.name)}",
            f"  {_t('handoff.waiting')}")
        return self._handoff_wait(platform_name, session_title)

    def _handoff_validate_target(self, platform_name: str):
        """Resolve the destination home channel via the live gateway config; None (after printing
        the reason) when the platform is unknown, disabled, or has no home channel."""
        try:
            from gateway.config import load_gateway_config, Platform
        except Exception as exc:  # pragma: no cover — gateway pkg always shipped
            return _cp(f"  {_t('handoff.config_load_failed', error=exc)}")
        try:
            platform = Platform(platform_name)
        except (ValueError, KeyError):
            return _cp(f"  {_t('handoff.unknown_platform', platform=platform_name)}")
        try:
            gw_config = load_gateway_config()
        except Exception as exc:
            return _cp(f"  {_t('handoff.config_load_failed', error=exc)}")
        pcfg = gw_config.platforms.get(platform)
        if not pcfg or not pcfg.enabled:
            # Relay aliasing: a relay-fronted gateway has only a RELAY block yet /handoff discord
            # is deliverable. UX pre-check only — the gateway watcher re-checks before dispatch.
            relay_fronts = False
            with suppress(Exception):
                from gateway.relay import relay_platform_identities
                relay_cfg = gw_config.platforms.get(Platform.RELAY)
                if relay_cfg and relay_cfg.enabled:
                    relay_fronts = platform_name in {p for p, _ in relay_platform_identities()}
            if not relay_fronts:
                return _cp(f"  {_t('handoff.platform_not_enabled', platform=platform_name)}")
        home = gw_config.get_home_channel(platform)
        if not home or not home.chat_id:
            return _cp(f"  {_t('handoff.no_home_channel', platform=platform_name)}",
                       f"  {_t('handoff.set_home_hint')}")
        return home

    def _handoff_prepare_session(self):
        """Refuse mid-turn, make sure a SessionDB handle + session row exist, and return the
        display title (None after printing why the handoff cannot start)."""
        # An in-flight agent run would race the gateway's switch_session and the synthetic turn.
        if getattr(self, "_agent_running", False):
            return _cp(f"  {_t('shared.agent_busy', command='/handoff')}")
        if not self._session_db:
            with suppress(Exception):
                from hermes_state_registry import acquire
                self._session_db = acquire()
        if not self._session_db:
            return _cp(_db_unavailable_line())
        # Ensure the session row exists (an empty session has flushed nothing yet): the gateway
        # needs a row to switch_session onto; set_session_title's INSERT OR IGNORE creates it.
        try:
            if not self._session_db.get_session(self.session_id):
                self._session_db.set_session_title(self.session_id, f"handoff-{self.session_id[:8]}")
        except Exception as exc:
            return _cp(f"  {_t('handoff.session_row_failed', error=exc)}")
        session_title = ""
        with suppress(Exception):
            session_title = (self._session_db.get_session(self.session_id) or {}).get("title") or ""
        return session_title or self.session_id[:8]

    def _handoff_wait(self, platform_name: str, session_title: str) -> bool:
        """Two-phase 0.5s poll. PENDING (unclaimed): 60s, then CAS-fail the row so the user can
        retry (a claim racing this instant wins). RUNNING (claimed): the gateway replays the
        transcript via a synthetic turn (routinely >60s) — wait 15 min with heartbeats and on
        timeout do NOT touch the row; failing it here was the split-brain bug."""
        pending_deadline = time.time() + self._HANDOFF_PENDING_TIMEOUT
        running_deadline = None
        next_heartbeat = None
        last_state = "pending"
        while True:
            try:
                state_row = self._session_db.get_handoff_state(self.session_id)
            except Exception:
                state_row = None
            current = (state_row or {}).get("state") or "pending"
            if current != last_state:
                if current == "running":
                    _cp(f"  {_t('handoff.picked_up')}")
                    running_deadline = time.time() + self._HANDOFF_RUNNING_TIMEOUT
                    next_heartbeat = time.time() + self._HANDOFF_HEARTBEAT_EVERY
                last_state = current
            if current == "completed":
                _cp("", f"  {_t('handoff.complete', platform=platform_name)}",
                    f"  {_t('handoff.resume_hint', title=session_title)}", "")
                # _run_cleanup must NOT finalize the row on exit: the gateway owns it now, and an
                # end_reason set under it would drop the handoff leg from session history/search.
                # See #88234.
                from cli import _handed_off_session_ids
                _handed_off_session_ids.add(self.session_id)
                self._should_exit = True  # same exit semantics as /quit
                return False
            if current == "failed":
                err = (state_row or {}).get("error") or _t("handoff.unknown_error")
                return self._handoff_keep(
                    f"  {_t('handoff.failed', error=err)}", f"  {_t('handoff.session_intact_retry')}")
            now = time.time()
            if current == "pending":
                if now >= pending_deadline:
                    break
            else:  # running
                if next_heartbeat is not None and now >= next_heartbeat:
                    _cp(f"  {_t('handoff.still_transferring')}")
                    next_heartbeat = now + self._HANDOFF_HEARTBEAT_EVERY
                if running_deadline is not None and now >= running_deadline:
                    # Do NOT fail the row: the gateway owns it (split-brain bug otherwise).
                    return self._handoff_keep(
                        f"  {_t('handoff.taking_long')}",
                        f"  {_t('handoff.check_platform', platform=platform_name)}",
                        f"  {_t('handoff.no_longer_waiting')}", f"  {_t('handoff.retry_when_settled')}")
            time.sleep(0.5)
        try:  # pending timed out: CAS-clear so the user can retry
            self._session_db.fail_handoff(
                self.session_id, "timed out waiting for gateway", only_states=("pending",))
        except TypeError:
            # Older SessionDB without only_states (mixed installs): legacy unconditional fail.
            with suppress(Exception):
                self._session_db.fail_handoff(self.session_id, "timed out waiting for gateway")
        except Exception:
            pass
        return self._handoff_keep(f"  {_t('handoff.timed_out')}", f"  {_t('handoff.session_intact')}")

    # ---- /resume, /sessions, /branch ------------------------------------------------------
    def _handle_resume_command(self, cmd_original: str) -> None:
        """Handle /resume <session_id_or_title> — switch to a previous session mid-conversation."""
        if getattr(self, "_agent_running", False):
            return _cp(f"  {_t('shared.agent_busy', command='/resume')}")
        from cli import _sync_process_session_id
        target = _command_arg(cmd_original)
        # Users copy the help text's placeholder brackets/quotes verbatim (``/resume <abc123>``).
        if len(target) >= 2 and target[0] + target[-1] in {"<>", "[]", '""', "''"}:
            target = target[1:-1].strip()
        if not target:
            _cp(f"  {_t('resume.usage')}")
            if self._show_recent_sessions(reason="resume"):
                # Arm a one-shot pending-resume selection so the user can type just the number (`3`) on the
                # next line instead of having to retype `/resume 3`. The list here must match the one shown
                # by _show_recent_sessions and used for index resolution below — all three go through
                # _list_recent_sessions() sized by display.resume_list_limit. See #34584.
                self._pending_resume_sessions = self._list_recent_sessions()
                return
            return _cp(f"  {_t('resume.tip_find_sessions')}")
        # Any explicit /resume <target> supersedes a previously-armed bare numbered prompt.
        self._pending_resume_sessions = None
        if not self._session_db:
            return _cp(_db_unavailable_line())
        resolved = self._resolve_resume_target(target)
        if resolved is None:
            return
        target_id, session_meta = resolved
        if target_id == self.session_id:
            return _cp(f"  {_t('resume.already_on')}")
        old_session_id = self.session_id
        _end_current_session(self, "resumed_other")
        self.session_id, self._resumed, self._pending_title = target_id, True, None
        _sync_process_session_id(target_id)
        # One lineage SELECT, two projections: model_history is alternation-repaired for live
        # replay (heals a durable user;user once); display_history is verbatim (as startup --resume).
        model_history, display_history = self._session_db.get_resume_conversations(target_id)
        self.conversation_history = _without_session_meta(model_history)
        self._resume_display_history = _without_session_meta(display_history)
        with suppress(Exception):  # re-open the target session so it's not marked as ended
            self._session_db.reopen_session(target_id)
        _sync_agent_to_session(self, target_id, parent_session_id=old_session_id, reason="resume")
        title_part = f" \"{session_meta['title']}\"" if session_meta.get("title") else ""
        from agent.context_compressor import is_user_originated_turn
        # Count only user-originated turns: legacy compaction handoffs are durable role=user rows
        # without display_kind.
        msg_count = len([m for m in self._resume_display_history if is_user_originated_turn(m)])
        if self.conversation_history:
            _cp("  " + _tn("resume.resumed", msg_count, session_id=target_id, title=title_part,
                           total=len(self.conversation_history)))
            self._display_resumed_history()
        else:
            _cp(f"  {_t('resume.resumed_empty', session_id=target_id, title=title_part)}")
        # Same contract as startup --resume: retarget the tool cwd, restore the persisted YOLO
        # bypass (approval session key changed) and the model/provider (else config default).
        # Retarget the process + tool cwd to where the session was started, so a mid-chat /resume (and
        # /sessions <id>, which delegates here) lands in the same directory as a startup `hermes
        # -c`/`--resume`. The startup resume paths already call this; without it, the terminal/code-exec
        # tools and relative-path resolution keep operating in the wrong repo. Idempotent and a no-op when
        # the session recorded no cwd. See #38562.
        self._restore_session_cwd(session_meta)
        self._restore_session_yolo(session_meta)
        self._restore_session_model(session_meta)

    def _resolve_resume_target(self, target: str):
        """``(session_id, meta)`` for a numbered selection, title, or id; None after printing why
        it could not be resolved. An empty compression-chain head redirects to the descendant
        that actually holds the transcript."""
        if target.isdigit():
            sessions = self._list_recent_sessions()
            index = int(target)
            if index < 1 or index > len(sessions):
                return _cp(*_lines(_gt("resume.out_of_range", index=index)))
            target_id = sessions[index - 1]["id"]
        else:
            from hermes_cli.main import _resolve_session_by_name_or_id
            target_id = _resolve_session_by_name_or_id(target) or target
        session_meta = self._session_db.get_session(target_id)
        if not session_meta:
            return _cp(f"  {_t('resume.not_found', target=target)}", f"  {_t('resume.not_found_hint')}")
        try:
            # If the target is the empty head of a compression chain, redirect to the descendant that
            # actually holds the transcript. See #15000.
            resolved_id = self._session_db.resolve_resume_session_id(target_id)
        except Exception:
            resolved_id = target_id
        if resolved_id and resolved_id != target_id:
            _cp(f"  {_t('resume.compressed_redirect', session_id=target_id, resolved_id=resolved_id)}")
            target_id = resolved_id
            session_meta = self._session_db.get_session(target_id) or session_meta
        return target_id, session_meta

    def _handle_sessions_command(self, cmd_original: str) -> None:
        """Handle /sessions [list|<id_or_title>] — bare/``list`` prints the recent-sessions table;
        an explicit target delegates to /resume so both spellings behave identically."""
        arg = _command_arg(cmd_original)
        if arg and arg.lower() not in {"list", "ls", "browse"}:
            self._handle_resume_command(f"/resume {arg}")
        elif not self._session_db:
            _cp(_db_unavailable_line())
        elif not self._show_recent_sessions(reason="sessions"):
            _cp(f"  {_t('sessions.none_yet')}")

    def _handle_branch_command(self, cmd_original: str) -> None:
        """Handle /branch [name] — fork the current session into a new independent copy of the
        full history so a different approach can be explored without losing the original."""
        # An in-flight agent run would flush through the rotating session identity: the branch
        # ends the parent row and repoints agent.session_id (_sync_agent_to_session), so the
        # turn's remaining messages land on the branch. Refuse mid-turn like /handoff does.
        if getattr(self, "_agent_running", False):
            return _cp(f"  {_t('shared.agent_busy', command='/branch')}")
        from cli import _sync_process_session_id
        if not self.conversation_history:
            return _cp(f"  {_gt('branch.no_conversation')}")
        if not self._session_db:
            return _cp(_db_unavailable_line())
        # CLI has no threads: always in place; strip the gateway's ``--here`` so it is never a title.
        from gateway.slash_commands_branch_thread import parse_branch_args
        _, branch_name = parse_branch_args(_command_arg(cmd_original))
        now = datetime.now()
        new_session_id = mint_session_id(now)
        branch_title = branch_name or self._session_db.get_next_title_in_lineage(
            self._session_db.get_session_title(self.session_id) or "branch")
        parent_session_id = self.session_id
        # Create the child BEFORE ending the parent: a failed create_session must leave the session the
        # user is still on open, not ended with end_reason="branched" and no branch (#11030).
        # The stable ``_branched_from`` marker keeps the branch visible in /resume + /sessions
        # even after the parent is re-ended with a different end_reason.
        # The child sends the parent's exact system prompt: a row without one makes the branch's first
        # turn rebuild (re-probing the workspace), so the warm cache the copied transcript buys is
        # lost at byte 0 whenever the repo moved since the parent's session start.
        parent_prompt = getattr(self.agent, "_cached_system_prompt", None)
        if not isinstance(parent_prompt, str) or not parent_prompt:
            with suppress(Exception):
                parent_prompt = (self._session_db.get_session(parent_session_id) or {}).get("system_prompt")
        try:
            self._session_db.create_session(
                session_id=new_session_id, source=os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                model=self.model, parent_session_id=parent_session_id, system_prompt=parent_prompt or None,
                model_config={"max_iterations": self.max_turns, "reasoning_config": self.reasoning_config,
                              "_branched_from": parent_session_id})
        except Exception as e:
            return _cp(f"  {_gt('branch.create_failed', error=e)}")
        _end_current_session(self, "branched")
        # Best-effort chunked copy (a failed copy still yields a usable branch); the api_content
        # sidecar lets the branch's first turn replay the parent's exact wire bytes (warm cache).
        with suppress(Exception):
            self._session_db.append_messages_batch(new_session_id, [
                {"role": msg.get("role", "user"), "tool_name": msg.get("tool_name") or msg.get("name"),
                 "api_content": extract_api_content_sidecar(msg),
                 **{k: msg.get(k) for k in _BRANCH_COPY_KEYS}, **message_identity(msg, with_tool_uids=True)}
                for msg in self.conversation_history], chunk_rows=500)
        with suppress(Exception):
            self._session_db.set_session_title(new_session_id, branch_title)
        # Switch to the new session
        self._transfer_session_yolo(self.session_id, new_session_id)
        self.session_id, self.session_start, self._pending_title = new_session_id, now, None
        self._resumed = True  # Prevents auto-title generation
        _sync_process_session_id(new_session_id)
        if self.agent:
            self.agent.session_start = now
        _sync_agent_to_session(self, new_session_id, parent_session_id=parent_session_id, reason="branch")
        msg_count = len([m for m in self.conversation_history if m.get("role") == "user"])
        _cp("  " + _tn("branch.branched", msg_count, title=branch_title),
            f"  {_t('branch.original_session', session_id=parent_session_id)}",
            f"  {_t('branch.branch_session', session_id=new_session_id)}")

    # ---- /worktree ------------------------------------------------------------------------
    def _handle_worktree_command(self, cmd_original: str) -> None:
        """Handle /worktree [new [name]|list|prune [--dry-run]] — isolated git worktrees.
        ``new`` moves this session into the tree (as ``hermes -w``: kept on exit only with
        unpushed commits); ``prune`` never deletes tracked changes, unique commits, or in-use trees."""
        import cli as _cli
        parts = cmd_original.split(None, 2)
        sub = parts[1].lower() if len(parts) > 1 else ""
        rest = parts[2].strip() if len(parts) > 2 else ""
        repo_root = _cli._git_repo_root()
        if not sub or sub in {"status", "show"}:
            active = _cli._active_worktree
            if active:
                _pr(f"  {_t('worktree.active', path=active['path'])}",
                    f"  {_t('worktree.branch', branch=active['branch'])}")
            else:
                print(f"  {_t('worktree.none_active')}")
            if repo_root:
                _pr(f"  {_t('worktree.hint_new')}", f"  {_t('worktree.hint_prune')}")
            else:
                print(f"  {_t('worktree.not_in_repo_note')}")
            return
        handler = _WORKTREE_SUBCOMMANDS.get(sub)
        if handler is None:
            return _pr(f"  {_t('worktree.unknown_subcommand', subcommand=sub)}", f"  {_t('worktree.usage')}")
        if not repo_root:
            print(f"  {_t('worktree.new_requires_repo')}" if handler == "_worktree_new"
                  else f"  {_t('worktree.not_in_repo')}")
            return
        getattr(self, handler)(repo_root, rest)

    def _worktree_prune(self, repo_root: str, rest: str) -> None:
        import cli as _cli
        from hermes_cli import worktree_gc
        rest = rest.lower()
        dry_run = "--dry-run" in rest or "-n" in rest.split()
        active = _cli._active_worktree
        tree_records = worktree_gc.audit_worktrees(repo_root, with_sizes=False)
        if active:
            # Never reap the tree this session is sitting in, even if judged clean+merged.
            active_path = str(active.get("path") or "")
            tree_records = [record for record in tree_records if record.path != active_path]
        actions = worktree_gc.reclaim_worktrees(repo_root, dry_run=dry_run, records=tree_records)
        actions += worktree_gc.reclaim_branches(repo_root, dry_run=dry_run)
        if actions:
            _pr(*(f"  {line}" for line in actions),
                "  " + _t("worktree.actions_planned" if dry_run else "worktree.actions_done", count=len(actions)))
        else:
            print(f"  {_t('worktree.nothing_to_reclaim')}")
        kept = [r for r in tree_records
                if r.verdict == "keep" and "kanban" not in r.reason and "in use" not in r.reason]
        if kept:
            _pr(f"  {_t('worktree.preserved', count=len(kept))}",
                *(f"    {record.name}: {record.reason}" for record in kept))

    def _worktree_list(self, repo_root: str, rest: str) -> None:
        try:
            result = subprocess.run(
                ["git", "worktree", "list"], capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=10, cwd=repo_root)
            out = result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            out = ""
        _pr(*(f"  {line}" for line in out.splitlines()) if out else (f"  {_t('worktree.list_failed')}",))

    def _worktree_new(self, repo_root: str, rest: str) -> None:
        import cli as _cli
        from hermes_cli.config import load_config
        try:
            sync_base = bool(load_config().get("worktree_sync", True))
        except Exception:
            sync_base = True
        wt_info = _cli._setup_worktree(repo_root=repo_root, sync_base=sync_base, name=rest or None)
        if not wt_info:
            return  # _setup_worktree already printed the failure
        # Retarget the session's terminal/file tools at the new tree (as `hermes -w` does).
        try:
            os.chdir(wt_info["path"])
        except OSError as e:
            print(f"  {_t('worktree.enter_failed', error=e)}")
        os.environ["TERMINAL_CWD"] = wt_info["path"]
        # Same keep-if-unpushed cleanup as `hermes -w`. Only one tree is "active" per process;
        # an earlier one keeps its own atexit registration (explicit info arg).
        _cli._active_worktree = wt_info
        atexit.register(_cli._cleanup_worktree, wt_info)
        _pr(f"  {_t('worktree.ready', path=wt_info['path'])}",
            f"  {_t('worktree.branch', branch=wt_info['branch'])}", f"  {_t('worktree.tools_in_worktree')}")

    # ---- /personality, /pet, /hatch -------------------------------------------------------
    def _handle_personality_command(self, cmd: str):
        """Handle /personality [name] — list or set a predefined personality. All resolution and
        persistence goes through hermes_cli.personality, the single owner of personality state."""
        from hermes_cli.personality import (
            describe_personality, normalize_personality_name, persist_personality, prompt_text,
            resolve_personality)
        personality_name = _command_arg(cmd)
        if not personality_name:
            try:
                from hermes_cli.config import read_raw_config
                current = normalize_personality_name(
                    (read_raw_config().get("display") or {}).get("personality", ""))
            except Exception:
                current = ""
            _pr("", "+" + "-" * 50 + "+", "|" + _t("personality.title").center(50) + "|",
                "+" + "-" * 50 + "+", "",
                f" {' *' if not current else '  '}{'none':<12} - {_t('personality.none_option')}")
            for name, prompt in self.personalities.items():
                marker = " *" if name == current else "  "
                print(f" {marker}{name:<12} - {describe_personality(prompt)}")
            return _pr("", f"  {_t('personality.usage')}", "")
        try:
            name, personality_prompt = resolve_personality(personality_name, getattr(self, "config", None))
        except ValueError:
            print(_t("personality.unknown", name=personality_name.lower()))
            return print(f"  {_t('personality.available', available=', '.join(self.personalities.keys()))}")
        saved = persist_personality(name)
        scope = _t("personality.scope_saved") if saved else _t("personality.scope_session")
        face = "(^_^)b" if saved else "(^_^)"
        if not name:
            # Neutral reset — fall back to the user-owned manual prompt.
            try:
                from hermes_cli.config import cfg_get, read_raw_config
                self.system_prompt = prompt_text(
                    cfg_get(read_raw_config(), "agent", "system_prompt", default=""))
            except Exception:
                self.system_prompt = ""
            _retire_agent(self)  # Force re-init
            _pr(_t("personality.cleared", face=face, scope=scope), f"  {_t('personality.cleared_note')}")
        else:
            self.system_prompt = personality_prompt
            _retire_agent(self)  # Force re-init
            _pr(_t("personality.set_to", face=face, name=name, scope=scope),
                f"  \"{_ellipsize(personality_prompt, 60)}\"")

    def _handle_pet_command(self, cmd: str):
        """Handle /pet [toggle|list|scale <n>|off|<slug>] — the petdex mascot. Writes
        ``display.pet.*`` to config; pet surfaces pick it up on their next poll."""
        from agent.pet import store
        from agent.pet.manifest import ManifestError
        from hermes_cli.pets import (
            _set_active, _set_enabled, print_pet_gallery, set_pet_scale, toggle_pet_display)
        arg = _command_arg(cmd)
        low = arg.lower()
        if not arg or low == "toggle":
            enabled, name, err = toggle_pet_display()
            print(_t("pet.error", error=err) if err else _t("pet.is_out", name=name) if enabled
                  else _t("pet.put_away_named", name=name) if name else _t("pet.put_away"))
        elif low in ("list", "gallery", "browse", "all"):
            print_pet_gallery()
        elif low == "scale" or low.startswith("scale "):
            value = arg[len("scale"):].strip()
            if not value:
                return print(_t("pet.usage_scale"))
            scale, err = set_pet_scale(value)
            print(_t("pet.error", error=err) if err else _t("pet.scale_set", scale=f"{scale:g}"))
        elif low == "off":
            _set_enabled(False)
            print(_t("pet.put_away"))
        else:
            print(_t("pet.fetching", name=arg))
            try:
                pet = store.install_pet(arg)
            except (store.PetStoreError, ManifestError) as exc:
                return print(_t("pet.adopt_failed", name=arg, error=exc))
            _set_active(arg)
            print(_t("pet.is_out", name=pet.display_name))

    def _handle_hatch_command(self, cmd: str):
        """Generate ("hatch") a new petdex pet from a description: base look, one animation row
        per state, spritesheet, then adopt. Progress streams inline (~a minute of image calls).
        The desktop app opens a richer overlay for this command instead."""
        from agent.pet import store
        from agent.pet.generate import orchestrate
        from agent.pet.generate.imagegen import GenerationError
        from hermes_cli.pets import _set_active
        concept = _command_arg(cmd)
        if not concept:
            # prompt_toolkit owns stdin on this daemon thread — raw input() never renders and eats
            # keystrokes; prefer the thread-aware helper (None when prompting isn't safe).
            # Bare /hatch is dispatched from the process_loop daemon thread while prompt_toolkit owns stdin
            # — a raw input() here types into a prompt that never renders and swallows the next keystrokes
            # (same class as #23185; found in the Aug 2026 full-surface CLI QA sweep: bare /hatch left the
            # session eating input until Ctrl+C). Route through the thread-aware prompt helper, which uses
            # run_in_terminal on the main thread and cancels cleanly (None) when prompting isn't safe.
            prompt_helper = getattr(self, "_prompt_text_input", None)
            try:
                concept = ((prompt_helper or input)(_t("hatch.describe_prompt")) or "").strip()
            except (EOFError, KeyboardInterrupt):
                return print()
        if not concept:
            return print(_t("hatch.usage"))
        # A short, friendly display name from the first few words of the concept.
        display_name = (" ".join(w.capitalize() for w in concept.split()[:3])[:28].strip()
                        or _t("hatch.default_name"))
        slug = store.slugify(display_name) or store.slugify(concept) or "pet"
        print(_t("hatch.designing", concept=concept))
        try:
            drafts = orchestrate.generate_base_drafts(concept, n=1)
        except GenerationError as exc:
            return print(_t("hatch.base_failed", error=exc))
        if not drafts:
            return print(_t("hatch.no_base_draft"))

        def _progress(event: str, detail: str) -> None:
            if event == "row":  # detail is "<state>:<done>:<total>"; show the state name.
                print(f"  {_t('hatch.progress_drawing', state=detail.split(':', 1)[0])}")
            elif event in _HATCH_PROGRESS:
                print(f"  {_t(_HATCH_PROGRESS[event])}")

        try:
            result = orchestrate.hatch_pet(
                base_image=drafts[0], slug=slug, display_name=display_name, concept=concept,
                on_progress=_progress)
        except GenerationError as exc:
            return print(_t("hatch.failed", error=exc))
        _set_active(result.slug)
        print(_t("hatch.hatched", name=result.display_name))

    # ---- /cron ----------------------------------------------------------------------------
    def _handle_cron_command(self, cmd: str):
        """Handle the /cron command to manage scheduled tasks."""
        tokens = shlex.split(cmd)
        if len(tokens) == 1:
            return self._cron_overview()
        subcommand = tokens[1].lower()
        opts = _parse_cron_flags(tokens[2:])
        if opts is None:
            return
        handler = _CRON_SUBCOMMANDS.get(subcommand)
        if handler is None:
            return _pr(_t("cron.unknown_command", subcommand=subcommand), f"  {_t('cron.available')}")
        getattr(self, handler)(subcommand, opts)

    def _cron_overview(self) -> None:
        _pr("", "+" + "-" * 68 + "+", "|" + _t("cron.overview_title").center(68) + "|",
            "+" + "-" * 68 + "+", "", f"  {_t('cron.overview_commands')}",
            *_lines(_t("cron.overview_usage"), pad="    "), "")
        result = _cron_api(action="list")
        jobs = result.get("jobs", []) if result.get("success") else []
        if jobs:
            from hermes_cli.cron import _next_run_row
            _pr(f"  {_t('cron.current_jobs')}", "  " + "-" * 63)
            for job in jobs:
                print(f"    {job['job_id'][:12]:<12} | {job['schedule']:<15} | {job.get('repeat', '?'):<8}")
                if job.get("skills"):
                    print(f"      {_t('cron.skills', skills=', '.join(job['skills']))}")
                print(f"      {job.get('prompt_preview', '')}")
                if job.get("next_run_at"):
                    # A stamp parked past the scheduler grace must not read as upcoming (#114309).
                    label, value = _next_run_row(job)
                    print(f"      {_t('cron.next_short') if label == 'Next run' else label}: {value}")
                print()
        else:
            print(f"  {_t('cron.no_jobs_hint')}")
        print()

    def _cron_list(self, subcommand: str, opts: dict) -> None:
        result = _cron_api(action="list", include_disabled=opts["all"])
        jobs = result.get("jobs", []) if result.get("success") else []
        if not jobs:
            return print(_t("cron.no_jobs"))
        from hermes_cli.cron import _next_run_row
        print()
        _pr(_t("cron.list_header"), "-" * 80)
        for job in jobs:
            _pr(f"  {_t('cron.id', job_id=job['job_id'])}", f"  {_t('cron.name', name=job['name'])}",
                f"  {_t('cron.state', state=job.get('state', '?'))}",
                f"  {_t('cron.schedule_repeat', schedule=job['schedule'], repeat=job.get('repeat', '?'))}",
                "  %s: %s" % _next_run_row(job) if job.get("next_run_at") else f"  {_t('cron.next_run_na')}")
            if job.get("skills"):
                print(f"  {_t('cron.skills', skills=', '.join(job['skills']))}")
            print(f"  {_t('cron.prompt', prompt=job.get('prompt_preview', ''))}")
            if job.get("last_run_at"):
                status = job.get("last_status") or "?"
                # delivery_failed: the run succeeded but delivery didn't — the reason lives
                # in last_delivery_error (last_error is None).
                if status == "delivery_failed" and job.get("last_delivery_error"):
                    status = _t("cron.status_delivery_failed", error=job["last_delivery_error"])
                elif status == "error" and job.get("last_error"):
                    status = _t("cron.status_error", error=job["last_error"])
                print(f"  {_t('cron.last_run', when=job['last_run_at'], status=status)}")
            print()

    def _cron_add(self, subcommand: str, opts: dict) -> None:
        positionals = opts["positionals"]
        if not positionals:
            return print(_t("cron.usage_add"))
        schedule = opts["schedule"] or positionals[0]
        prompt = opts["prompt"] or " ".join(positionals[1:])
        skills = _normalize_skills(opts["skills"])
        if not prompt and not skills:
            return print(_t("cron.need_prompt_or_skill"))
        result = _cron_api(
            action="create", schedule=schedule, prompt=prompt or None, name=opts["name"],
            deliver=opts["deliver"], repeat=opts["repeat"], skills=skills or None)
        if not result.get("success"):
            return print(_t("cron.create_failed", error=result.get("error")))
        _pr(_t("cron.created", job_id=result["job_id"]),
            f"  {_t('cron.schedule', schedule=result['schedule'])}")
        if result.get("skills"):
            print(f"  {_t('cron.skills', skills=', '.join(result['skills']))}")
        print(f"  {_t('cron.next_run', value=result['next_run_at'])}")

    def _cron_edit(self, subcommand: str, opts: dict) -> None:
        from cron import get_job
        positionals = opts["positionals"]
        if not positionals:
            return print(_t("cron.usage_edit"))
        job_id = positionals[0]
        existing = get_job(job_id)
        if not existing:
            return print(_t("cron.job_not_found", job_id=job_id))
        # Skill edit precedence: --clear-skills > --skill (replace) > --add/--remove (merge) > untouched.
        final_skills = None
        replacement_skills = _normalize_skills(opts["skills"])
        add_skills = _normalize_skills(opts["add_skills"])
        remove_skills = set(_normalize_skills(opts["remove_skills"]))
        if opts["clear_skills"]:
            final_skills = []
        elif replacement_skills:
            final_skills = replacement_skills
        elif add_skills or remove_skills:
            existing_skills = list(
                existing.get("skills") or ([existing["skill"]] if existing.get("skill") else []))
            final_skills = [skill for skill in existing_skills if skill not in remove_skills]
            final_skills += [skill for skill in add_skills if skill not in final_skills]
        result = _cron_api(
            action="update", job_id=job_id, schedule=opts["schedule"], prompt=opts["prompt"],
            name=opts["name"], deliver=opts["deliver"], repeat=opts["repeat"], skills=final_skills)
        if not result.get("success"):
            return print(_t("cron.update_failed", error=result.get("error")))
        job = result["job"]
        _pr(_t("cron.updated", job_id=job["job_id"]), f"  {_t('cron.schedule', schedule=job['schedule'])}",
            f"  {_t('cron.skills', skills=', '.join(job['skills']))}" if job.get("skills")
            else f"  {_t('cron.skills_none')}")

    def _cron_job_action(self, subcommand: str, opts: dict) -> None:
        """pause / resume / run / remove (aliases rm, delete) on one job id."""
        positionals = opts["positionals"]
        if not positionals:
            return print(_t("cron.usage_action", subcommand=subcommand))
        job_id = positionals[0]
        action = "remove" if subcommand in {"remove", "rm", "delete"} else subcommand
        result = _cron_api(action=action, job_id=job_id,
                           reason="paused from /cron" if action == "pause" else None)
        if not result.get("success"):
            return print(_t(f"cron.{action}_failed", error=result.get("error")))
        if action == "remove":
            removed = result.get("removed_job", {})
            return print(_t("cron.removed", name=removed.get("name", job_id), job_id=job_id))
        job = result["job"]
        if action == "run" and job.get("execution_skipped"):
            # A refused run-now (claim lost, paused, gone) must not read as accepted.
            return print(_t("cron.did_not_run", name=job["name"], job_id=job_id, reason=job["execution_skipped"]))
        # One full template per action verb (cron.paused / cron.resumed / cron.triggered).
        print(_t({"pause": "cron.paused", "resume": "cron.resumed", "run": "cron.triggered"}[action],
                 name=job["name"], job_id=job_id))
        if action == "resume":
            print(f"  {_t('cron.next_run', value=job.get('next_run_at'))}")
        elif action == "run":
            from hermes_cli.cron import _run_outcome
            print(f"  {_run_outcome(job)}")

    # ---- delegating handlers: /suggestions, /blueprint, /curator, /kanban, /skills, /memory --
    def _handle_suggestions_command(self, cmd: str):
        """Handle /suggestions — review/accept/dismiss suggested automations via the shared handler.
        CLI origin is the local platform so an accepted job's "origin" delivery resolves to a home channel."""
        args = " ".join(_shlex_args(cmd))
        try:
            from hermes_cli.suggestions_cmd import handle_suggestions_command
            output = handle_suggestions_command(args)
        except Exception as e:
            output = _t("suggestions.failed", error=e)
        self._console_print(output)

    def _handle_blueprint_command(self, cmd: str):
        """Handle /blueprint — set up an automation from a blueprint template (shared handler).
        Bare lists the catalog; ``<name>`` seeds the agent to ask for each value conversationally
        (``agent_seed``, run as the next turn); ``<name> slot=val …`` creates the job directly."""
        args = " ".join(shlex.quote(t) for t in _shlex_args(cmd))
        try:
            from hermes_cli.blueprint_cmd import handle_blueprint_command
            result = handle_blueprint_command(args)
        except Exception as e:
            self._console_print(_t("blueprint.failed", error=e))
            return
        self._console_print(result.text)
        seed = getattr(result, "agent_seed", None)
        if seed:
            # One-shot: the interactive loop picks this up right after the slash command
            # returns and runs it as a normal agent turn.
            self._pending_agent_seed = seed

    def _handle_curator_command(self, cmd: str):
        """Handle /curator — delegates to hermes_cli.curator so the CLI and the `hermes curator`
        subcommand share the same handler set."""
        tokens = shlex.split(cmd)[1:] if cmd else []
        try:
            from hermes_cli.curator import cli_main
            cli_main(tokens or ["status"])
        except SystemExit:
            pass  # argparse exits on --help/errors; don't kill the interactive session
        except Exception as exc:
            print(_t("curator.failed", error=exc))

    def _handle_kanban_command(self, cmd: str):
        """Handle /kanban — strip the leading ``/kanban`` and hand the rest to ``kanban.run_slash``."""
        from hermes_cli.kanban import run_slash
        rest = cmd.strip().lstrip("/")
        if rest.startswith("kanban"):
            rest = rest[len("kanban"):].lstrip()
        try:
            output = run_slash(rest)
        except Exception as exc:  # pragma: no cover - defensive
            output = _t("kanban.failed", error=exc)
        if output:
            print(output)

    def _handle_skills_command(self, cmd: str):
        """Handle /skills slash command — delegates to hermes_cli.skills_hub, after intercepting the
        write-approval review subcommands (pending/approve/reject/diff/mode)."""
        from cli import ChatConsole
        args = cmd.strip().split()[1:]
        review_words = {"pending", "approve", "apply", "reject", "deny", "drop", "diff", "approval", "mode"}
        if args and args[0].lower() in review_words:
            from hermes_cli.write_approval_commands import handle_pending_subcommand
            from tools import write_approval as wa
            out = handle_pending_subcommand(
                wa.SKILLS, args, set_mode_fn=lambda enabled: self._save_write_approval("skills", enabled),
            )
            if out is not None:
                return print(out)
        from hermes_cli.skills_hub import handle_skills_slash
        handle_skills_slash(cmd, ChatConsole())

    def _handle_memory_command(self, cmd: str):
        """Handle /memory slash command — pending review + approval-gate toggle."""
        from hermes_cli.write_approval_commands import handle_pending_subcommand
        from tools import write_approval as wa
        args = cmd.strip().split()[1:]
        store = getattr(self.agent, "_memory_store", None) if getattr(self, "agent", None) else None
        if store is None:
            # No live agent store (e.g. Desktop GUI): use a fresh on-disk store, as the gateway
            # does — same MEMORY/USER.md, same configured char limits.
            # Apply against a freshly loaded on-disk store, mirroring the gateway path
            # (gateway/slash_commands.py): it persists to the same MEMORY/USER.md and creates MEMORY.md on
            # the first approved write. Without this the shared handler returns "memory store unavailable".
            # See #46783.
            from tools.memory_tool import load_on_disk_store
            store = load_on_disk_store()
        out = handle_pending_subcommand(
            wa.MEMORY, args, memory_store=store,
            set_mode_fn=lambda enabled: self._save_write_approval("memory", enabled))
        print(out if out is not None else _t("memory.unknown_subcommand"))

    def _save_write_approval(self, subsystem: str, enabled: bool):
        """Persist <subsystem>.write_approval to config (for /memory|/skills approval)."""
        _save(f"{subsystem}.write_approval", bool(enabled))

    # ---- prompt-queueing handlers: /learn, /plan, /init -----------------------------------
    def _queue_prompt_turn(self, msg: str, command: str) -> None:
        """Inject ``msg`` onto the agent's input queue as the next normal user turn (the
        /learn, /plan, /init pattern: no engine, no model-tool footprint, prompt-cache safe)."""
        if hasattr(self, "_pending_input"):
            self._pending_input.put(msg)
        else:  # pragma: no cover - defensive (no live input loop)
            print(f"  {_t('shared.needs_active_session', command=command)}")

    def _handle_learn_command(self, cmd: str):
        """Handle /learn — distill a reusable skill from anything the user describes (a directory,
        a URL, "what we just did", pasted notes). The live agent gathers the material with the
        tools it already has and authors the skill via ``skill_manage``."""
        from agent.learn_prompt import build_learn_prompt
        user_request = _command_arg(cmd)
        print("\n" + _t("learn.from_description" if user_request else "learn.from_conversation"))
        self._queue_prompt_turn(build_learn_prompt(user_request), "/learn")

    def _handle_plan_command(self, cmd: str):
        """Handle /plan — write a markdown implementation plan, no execution. The live agent
        inspects the workspace read-only and saves the plan under ``.hermes/plans/``."""
        from agent.plan_prompt import build_plan_prompt
        task = _command_arg(cmd)  # optional — empty infers the task from conversation context
        print("\n" + (_t("plan.planning", task=_ellipsize(task, 80)) if task else _t("plan.from_conversation")))
        self._queue_prompt_turn(build_plan_prompt(task), "/plan")

    def _handle_init_command(self, cmd: str):
        """Handle /init — generate or update AGENTS.md from a project scan performed by the
        live agent with its own read-only tools."""
        from hermes_cli.init_command import build_init_prompt_for_cwd
        # session_key="" targets the single-session CLI's "default" cwd record, which tracks
        # `cd` and workspace switches, so /init follows the directory the user works in.
        msg = build_init_prompt_for_cwd(extra=_command_arg(cmd), session_key="")  # optional user emphasis
        print("\n" + _t("init.updating" if "UPDATE the existing AGENTS.md" in msg else "init.generating"))
        self._queue_prompt_turn(msg, "/init")

    # ---- side-session handlers: /bg, /btw -------------------------------------------------
    def _handle_background_command(self, cmd: str):
        """Handle /bg <prompt> — run a prompt in a separate background session (its own AIAgent
        on a thread); the result prints here without touching the active history."""
        from cli import set_approval_callback, set_secret_capture_callback, set_sudo_password_callback
        from run_agent import AIAgent
        prompt = _command_arg(cmd)
        if not prompt:
            return _cp(*_lines(_t("background.usage")))
        self._background_task_counter += 1
        task_num = self._background_task_counter
        task_id = f"bg_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"
        if not self._ensure_runtime_credentials():
            return _cp(f"  {_t('background.no_credentials')}")
        preview = _ellipsize(prompt, 60)
        _cp(f"  {_t('background.started', number=task_num, preview=preview)}",
            f"  {_t('background.task_id', task_id=task_id)}", f"  {_t('background.keep_chatting')}\n")
        turn_route = self._resolve_turn_agent_config(prompt)
        runtime = turn_route["runtime"]

        def produce():
            from agent.vault_backends.unlock import set_code_prompt_callback, set_save_login_prompt_callback, set_unlock_prompt_callback
            set_sudo_password_callback(self._sudo_password_callback)
            set_approval_callback(self._approval_callback)
            set_unlock_prompt_callback(self._vault_unlock_callback)
            set_save_login_prompt_callback(self._vault_save_login_callback)
            set_code_prompt_callback(self._vault_code_callback)
            with suppress(Exception):
                set_secret_capture_callback(self._secret_capture_callback)
            try:
                bg_agent = AIAgent(
                    model=turn_route["model"], acp_command=runtime.get("command"),
                    acp_args=runtime.get("args"), max_iterations=self.max_turns,
                    **{k: runtime.get(k) for k in ("api_key", "base_url", "provider", "api_mode",
                                                   "max_tokens")}, enabled_toolsets=self.enabled_toolsets,
                    quiet_mode=True, verbose_logging=False, session_id=task_id, platform="cli",
                    side_agent=True,
                    session_db=self._session_db, reasoning_config=self.reasoning_config,
                    service_tier=self.service_tier,
                    request_overrides=turn_route.get("request_overrides"),
                    **{kw: getattr(self, attr) for kw, attr in _BG_PROVIDER_KWARGS.items()})
                # Silence raw spinner; route thinking through TUI widget when no foreground agent is active.
                bg_agent._print_fn = lambda *_a, **_kw: None

                def _bg_thinking(text: str) -> None:
                    # Concurrent bg tasks may race on _spinner_text; acceptable for best-effort UI.
                    if not self._agent_running:
                        self._spinner_text = text
                        if self._app:
                            # Display result in the CLI (thread-safe via patch_stdout). Force a TUI refresh
                            # first so spinner/status bar don't overlap with the output (fixes #2718).
                            # Same TUI refresh pattern as success path (#2718)
                            self._app.invalidate()

                bg_agent.thinking_callback = _bg_thinking
                try:
                    result = bg_agent.run_conversation(user_message=prompt, task_id=task_id)
                    response = result.get("final_response", "") if result else ""
                    if not response and result and result.get("error"):
                        response = _gt("model.error_prefix", error=result["error"])
                    return response
                finally:
                    # One agent per /bg task in a long-lived CLI process: close()
                    # is the owner boundary (memory shutdown, tool subprocesses,
                    # httpx clients); an unclosed side agent leaks all of them
                    # until the CLI exits (#50197).
                    with suppress(Exception):
                        bg_agent.close()
            finally:
                with suppress(Exception):
                    set_sudo_password_callback(None)
                    set_approval_callback(None)
                    set_secret_capture_callback(None)
                    set_unlock_prompt_callback(None)
                    set_save_login_prompt_callback(None)
                    set_code_prompt_callback(None)

        def done():
            self._background_tasks.pop(task_id, None)
            if not self._agent_running:  # clear spinner only if no foreground agent owns it
                self._spinner_text = ""

        thread = self._side_worker(
            produce, name=f"bg-task-{task_id}", fail_label=_t("background.label", number=task_num),
            header_lines=[f"  {_t('background.complete', number=task_num)}",
                          f"  {_t('background.prompt', preview=preview)}"],
            title_suffix=_t("background.title_suffix", number=task_num),
            empty_note=f"  {_t('background.no_response')}", bell=True, on_done=done)
        self._background_tasks[task_id] = thread
        thread.start()

    def _side_worker(self, produce, *, name, fail_label, header_lines, title_suffix, empty_note,
                     bell=False, on_done=None, console=None) -> threading.Thread:
        """Daemon thread for /bg, /btw and /login: ``produce()`` returns the body to print in a side-result
        panel; failures print ``fail_label`` failed; the TUI is always re-invalidated afterwards."""
        def run():
            try:
                body = produce()
                _print_side_result_panel(self, header_lines=header_lines, body=body,
                                         title_suffix=title_suffix, empty_note=empty_note,
                                         console=console)
                if bell:
                    self._ring_bell(context=_t("side_result.complete", label=fail_label))
            except Exception as e:
                _refresh_tui_before_print(self)
                line = f"  {_t('side_result.failed', label=fail_label, error=e)}"
                # Same console the caller captured, so a late failure can't splice into a later command.
                if console is not None:
                    console.print(line, markup=False)
                else:
                    _cp(line)
            finally:
                if on_done is not None:
                    on_done()
                if self._app:
                    self._invalidate(min_interval=0)

        return threading.Thread(target=run, daemon=True, name=name)

    def _handle_login_command(self, cmd_original: str) -> None:
        """Start an in-chat sign-in without blocking the input loop while approval is pending."""
        from hermes_cli import anon_auth
        # Pin the output target now. Under the live TUI ``self.console`` writes straight to
        # patch_stdout's StdoutProxy, which mangles Rich's escapes — there ``None`` keeps the
        # panel on the ``_cprint`` path. Only the slash worker (``_app`` is None) swaps the console.
        console = None if getattr(self, "_app", None) else getattr(self, "console", None)
        _cp(f"  {anon_auth.LOGIN_STARTING}")
        gen = anon_auth.run_sign_in(timeout_seconds=8.0)
        try:
            first = next(gen, None)
        except KeyboardInterrupt:
            with suppress(Exception):
                gen.close()
            return _cp(anon_auth.UPGRADE_CANCELLED)
        if first is None:
            return
        if first.terminal:
            return _cp(f"  {first.copy}")
        anon_auth.render_sign_in_cli_code(first, chat=True, printer=_cp)

        def _settle_session_model(state) -> None:
            """A completed sign-in moved this profile onto the account: the welcome host is gone and
            the portal serves ``nous/welcome`` as a paid model, so a session still carrying it must
            move too — the CLI counterpart of the gateway's on-``Completed`` sweep. Only the free
            tier's own model is replaced; a model the user picked while the sign-in was pending
            stands. Writing ``self.model`` is enough: ``chat()`` compares the turn-route signature
            and rebuilds the agent on the next turn, so a turn already in flight keeps the agent it
            started with. ``getattr``: tests drive this handler with minimal shells.
            """
            if state.kind != "completed" or not getattr(state, "model_changed", False):
                return
            if str(getattr(self, "model", "") or "") == anon_auth.GUEST_MODEL:
                # "" when the settle cleared the default: _ensure_runtime_credentials then applies
                # the provider's silent default, which is what settle_after_upgrade documents.
                self.model = state.model or ""

        thread = self._side_worker(
            lambda: anon_auth.drain_sign_in_copy(gen, chat=True, on_terminal=_settle_session_model),
            name="login", fail_label=_t("login.label"), header_lines=[f"  {_t('login.label')}"],
            title_suffix=_t("login.title_suffix"), empty_note=f"  {_t('login.no_result')}", console=console)
        thread.start()

    def _handle_btw_command(self, cmd: str):
        """Handle /btw <question> — answer a side question about this conversation from a
        history snapshot via a one-shot auxiliary call. The live session is never touched
        (no history mutation, no role-alternation risk, no cache invalidation)."""
        question = _command_arg(cmd)
        if not question:
            return _cp(*_lines(_t("btw.usage")))
        if not self._ensure_runtime_credentials():
            return _cp(f"  {_t('btw.no_credentials')}")
        # Snapshot NOW, on the UI thread — the foreground turn keeps appending to
        # conversation_history while the worker runs.
        history_snapshot = list(self.conversation_history or [])
        # Live agent → cache-parity fork (full context, warm cache reads).
        parent_agent = self.agent
        turn_route = self._resolve_turn_agent_config(question)
        runtime = turn_route["runtime"]
        main_runtime = {
            "model": turn_route["model"],
            **{k: runtime.get(k) for k in ("provider", "base_url", "api_key", "api_mode")},
            "session_id": getattr(parent_agent, "session_id", None),
        }
        preview = _ellipsize(question, 60)
        _cp(*_lines(_gt("btw.started", preview=preview)), "")

        def produce():
            from agent.side_question import answer_side_question
            return answer_side_question(
                question, history_snapshot, parent_agent=parent_agent, main_runtime=main_runtime)

        self._side_worker(produce, name="btw-side-question", fail_label="/btw",
                          header_lines=[f"  {_t('btw.header', preview=preview)}"],
                          title_suffix=_t("btw.title_suffix"),
                          empty_note=f"  {_t('btw.no_answer')}").start()

    # ---- /bundles, /browser ---------------------------------------------------------------
    def _handle_bundles_command(self, cmd: str) -> None:
        """In-session ``/bundles`` — show installed skill bundles (``hermes bundles list`` rendered
        inside the running CLI). Bundles are loaded via ``/<bundle-name>``."""
        from cli import ChatConsole, _BOLD, _RST, _accent_hex
        from hermes_cli.slash_exec import CommandContext, execute_command
        reply = execute_command("bundles", CommandContext(surface="cli"))
        if "error" in reply.data:
            return _cp(f"\033[1;31m{_t('bundles.unavailable', error=reply.data['error'])}{_RST}")
        bundles = reply.data["bundles"]
        if not bundles:
            return _cp(f"  {_t('bundles.none_installed')}", _dim_line(_t("bundles.create_hint")),
                       _dim_line(_t("bundles.directory", dir=reply.data["dir"])))
        _cp(f"\n  ▣ {_BOLD}{_t('bundles.title')}{_RST} {_t('bundles.installed_count', count=len(bundles))}")
        for info in bundles:
            skill_count = len(info.get("skills", []))
            desc = info.get("description") or _t("bundles.default_desc", count=skill_count)
            ChatConsole().print(
                f"    [bold {_accent_hex()}]/{info['slug']:<20}[/] "
                f"[dim]-[/] {_escape(desc)} [dim]{_escape(_t('bundles.skills_count', count=skill_count))}[/]")
            for s in info.get("skills", []):
                ChatConsole().print(f"        [dim]· {_escape(s)}[/]")
        _cp("\n" + _dim_line(_t("bundles.invoke_hint")))

    def _handle_browser_command(self, cmd: str):
        """Handle /browser connect|disconnect|status|use — manage the live Chromium-family CDP connection."""
        # The subcommand word is matched case-insensitively; the raw argument keeps
        # its case because a CDP URL's path segment is case-sensitive.
        parts = _command_arg(cmd).split(None, 1)
        word = parts[0].lower() if parts else "status"
        rest = parts[1].strip() if len(parts) > 1 else ""
        handler = _BROWSER_SUBCOMMANDS.get(word)
        if handler is None:
            _say_block(
                _t("browser.usage"), "",
                *(f"   {_t(f'browser.usage_{sub}')}" for sub in ("connect", "disconnect", "status", "use")))
            return
        handler(self, rest.strip())

    # ---- /heartbeat, /refine, /review -----------------------------------------------------
    def _session_manager(self, getter, label: str):
        """The session-scoped manager from ``getter()``, or None after the standard dim
        "<label> unavailable (no active session)." line."""
        mgr = getter()
        if mgr is None:
            _cp(_dim_line(_t("shared.unavailable_no_session", label=label)))
        return mgr

    def _handle_heartbeat_command(self, cmd: str) -> None:
        """Dispatch /heartbeat: set / status / pause / resume / clear. ``/heartbeat every 10m <prompt>``
        sets the session's one recurring instruction, injected as a normal user turn when due.
        Session-scoped and in-process — use `hermes cron` for durable schedules."""
        from hermes_cli.heartbeat import format_interval
        arg = _command_arg(cmd)
        lower = arg.lower()
        mgr = self._session_manager(self._get_heartbeat_manager, _t("heartbeat.label"))
        if mgr is None:
            return
        if not arg or lower == "status":
            _cp(f"  {mgr.status_line()}")
        elif lower == "pause":
            state = mgr.pause()
            _cp(f"  {_t('heartbeat.paused', prompt=state.prompt)}" if state
                else _dim_line(_t("heartbeat.none_set")))
        elif lower == "resume":
            state = mgr.resume()
            if state is None:
                _cp(_dim_line(_t("heartbeat.none_to_resume")))
            else:
                self._start_heartbeat_watchdog()
                _cp(f"  {_t('heartbeat.resumed', interval=format_interval(state.interval_seconds), prompt=state.prompt)}")
        elif lower in {"clear", "stop", "off"}:
            _cp(f"  {_t('heartbeat.cleared')}" if mgr.clear() else _dim_line(_t("heartbeat.none_set")))
        else:
            self._heartbeat_set(mgr, arg)

    def _heartbeat_set(self, mgr, arg: str) -> None:
        """Set: ``/heartbeat every 10m <prompt>`` (also accepts ``10m <prompt>``)."""
        from hermes_cli.heartbeat import parse_interval, format_interval
        tokens = arg.split(None, 2)
        interval = None
        prompt = ""
        if tokens and tokens[0].lower() == "every" and len(tokens) >= 2:
            interval = parse_interval(f"every {tokens[1]}")
            prompt = tokens[2] if len(tokens) > 2 else ""
        elif tokens:
            interval = parse_interval(tokens[0])
            prompt = arg[len(tokens[0]):].strip() if interval and interval > 0 else ""
        if interval is None:
            return _cp(f"  {_t('heartbeat.usage')}", _dim_line(_t("heartbeat.usage_also")))
        if interval < 0:
            from hermes_cli.heartbeat import MIN_INTERVAL_SECONDS
            return _cp(f"  {_t('heartbeat.interval_too_small', min_seconds=MIN_INTERVAL_SECONDS)}")
        if not prompt.strip():
            return _cp(f"  {_t('heartbeat.prompt_required')}")
        state = _attempt(_t("heartbeat.invalid_label"), ValueError, mgr.set, prompt, interval)
        if state is _FAILED:
            return
        self._start_heartbeat_watchdog()
        _cp(f"  {_t('heartbeat.set', interval=format_interval(state.interval_seconds), prompt=state.prompt)}",
            _dim_line(_t("heartbeat.set_note")))

    def _handle_refine_command(self, cmd: str) -> None:
        """Dispatch /refine — run the memory/skill review fork on demand (same machinery as the
        automatic post-turn ``_spawn_background_review``), with optional focus text. Background
        fork; the live conversation and prompt cache are never touched."""
        focus = _command_arg(cmd)
        agent = getattr(self, "agent", None)
        if agent is None:
            return _cp(_dim_line(_t("refine.nothing_yet")))
        snapshot = list(getattr(self, "conversation_history", None) or [])
        if not snapshot:
            return _cp(_dim_line(_t("refine.empty")))
        try:
            agent._spawn_background_review(
                messages_snapshot=snapshot, review_memory=True,
                review_skills="skill_manage" in getattr(agent, "valid_tool_names", set()),
                focus=focus or None, explicit=True)
        except Exception as exc:
            return _cp(f"  {_t('refine.failed', error=exc)}")
        tail = _t("refine.focus_suffix", focus=focus) if focus else ""
        _cp(f"  {_t('refine.started', focus=tail)}")

    def _handle_review_command(self, cmd: str) -> None:
        """Dispatch /review — snapshot the last N messages (+ argument text as instructions) and
        spawn an independent reviewer subagent via async delegation; the review re-enters this
        session as a normal delegation completion."""
        prompt = _command_arg(cmd)
        agent = getattr(self, "agent", None)
        if agent is None:
            return _cp(_dim_line(_t("review.nothing_yet")))
        snapshot = list(getattr(self, "conversation_history", None) or [])
        try:
            from agent.review_engine import format_dispatch_note, start_review
            result = start_review(agent, snapshot, prompt)
        except ValueError as exc:
            return _cp(_dim_line(str(exc)))
        except Exception as exc:
            return _cp(f"  {_t('review.failed', error=exc)}")
        _cp(f"  {format_dispatch_note(result, prompt)}")

    # ---- /goal, /loop, /subgoal -----------------------------------------------------------
    def _handle_goal_command(self, cmd: str) -> None:
        from hermes_cli.goal_command import dispatch_goal_command
        from hermes_cli.goals import last_user_message_content

        mgr = self._session_manager(self._get_goal_manager, _t("goal.label"))
        if mgr is None:
            return
        result = dispatch_goal_command(
            mgr, _command_arg(cmd), authorize_gate=lambda: None,
            progress=lambda text: _cp(_dim_line(text)),
            last_user_message=last_user_message_content(getattr(self, "conversation_history", None)),
        )
        for line in result.output.splitlines():
            _cp(f"  {line}")
        if result.prompt:
            queued = self._kick_goal(result.prompt)
            if not result.kickoff:
                _cp(_dim_line(_t("goal.continuing") if queued else _t("goal.send_to_kick")))

    def _kick_goal(self, prompt: str) -> bool:
        """Queue the next turn without mutating cached conversation history."""
        try:
            self._pending_input.put(prompt)
            return True
        except Exception:
            return False

    def _handle_loop_command(self, cmd: str) -> None:
        """Dispatch /loop — recurring in-session wakeups: ``/loop [interval] <prompt> [--times N]
        [--until <cond>]`` starts one; ``status | pause | resume | stop`` control it."""
        arg = _command_arg(cmd)
        mgr = self._session_manager(self._get_loop_manager, _t("loop.label"))
        if mgr is None:
            return
        from hermes_cli.loops import dispatch_loop_command
        result = dispatch_loop_command(mgr, arg)
        for line in (result.get("output") or "").splitlines():
            _cp(f"  {line}")
        if result.get("created"):
            with suppress(Exception):
                from hermes_cli.loops import goal_blocks_loop_tick
                if goal_blocks_loop_tick(mgr.session_id):
                    _cp(_dim_line(_t("loop.goal_defers_note")))

    def _handle_subgoal_command(self, cmd: str) -> None:
        """Dispatch /subgoal: bare → show, ``<text>`` → append, ``remove <n>`` (1-based), ``clear``.
        Subgoals join the judge + continuation prompts at the next turn boundary (no kick)."""
        parts = (cmd or "").strip().split(None, 2)
        arg = " ".join(parts[1:]).strip() if len(parts) > 1 else ""
        mgr = self._session_manager(self._get_goal_manager, _t("goal.label"))
        if mgr is None:
            return
        if not mgr.has_goal():
            return _cp(_dim_line(_t("subgoal.no_goal")))
        if not arg:  # list current subgoals
            _cp(f"  {mgr.status_line()}")
            return _cp(f"  {mgr.render_subgoals()}")
        tokens = arg.split(None, 1)
        verb = tokens[0].lower()
        rest = tokens[1].strip() if len(tokens) > 1 else ""
        if verb == "remove":
            if not rest:
                return _cp(f"  {_t('subgoal.usage_remove')}")
            try:
                idx = int(rest.split()[0])
            except ValueError:
                return _cp(f"  {_t('subgoal.remove_needs_int')}")
            removed = _attempt("/subgoal remove", (IndexError, RuntimeError), mgr.remove_subgoal, idx)
            if removed is not _FAILED:
                _cp(f"  {_t('subgoal.removed', index=idx, text=removed)}")
        elif verb == "clear":
            prev = _attempt("/subgoal clear", RuntimeError, mgr.clear_subgoals)
            if prev is not _FAILED:
                _cp("  " + _tn("subgoal.cleared", prev) if prev else _dim_line(_t("subgoal.none_to_clear")))
        else:  # append the whole arg as a new subgoal
            text = _attempt("/subgoal", (ValueError, RuntimeError), mgr.add_subgoal, arg)
            if text is not _FAILED:
                idx = len(mgr.state.subgoals) if mgr.state else 0
                _cp(f"  {_t('subgoal.added', index=idx, text=text)}")

    # ---- /skin, /prompt -------------------------------------------------------------------
    def _handle_skin_command(self, cmd: str):
        """Handle /skin [name] — show or change the display skin."""
        from cli import _ACCENT
        try:
            from hermes_cli.skin_engine import list_skins, set_active_skin, get_active_skin_name
        except ImportError:
            return print(_t("skin.unavailable"))
        new_skin = _command_arg(cmd).lower()
        if not new_skin:  # show current skin and list available
            current = get_active_skin_name()
            _pr(f"\n  {_t('skin.current', name=current)}", f"  {_t('skin.available_header')}")
            for s in list_skins():
                marker = " ●" if s["name"] == current else "  "
                source = f" ({s['source']})" if s["source"] == "user" else ""
                print(f"   {marker} {s['name']}{source} — {s['description']}")
            return _pr(f"\n  {_t('skin.usage')}",
                       f"  {_t('skin.custom_hint', dir=display_hermes_home())}\n")
        available = {s["name"] for s in list_skins()}
        if new_skin not in available:
            return _pr(f"  {_t('skin.unknown', name=new_skin)}",
                       f"  {_t('skin.available', names=', '.join(sorted(available)))}")
        set_active_skin(new_skin)
        _ACCENT.reset()  # re-resolve ANSI color for the new skin (_DIM is a fixed escape)
        saved = _t("skin.saved_suffix") if _save("display.skin", new_skin) else ""
        _pr(f"  {_t('skin.set_to', name=new_skin, saved=saved)}", f"  {_t('skin.banner_note')}")
        if self._apply_tui_skin_style():
            print(f"  {_t('skin.colors_updated')}")

    def _compose_in_editor(self, initial_text: str = "") -> str:
        """Open ``$VISUAL``/``$EDITOR`` on a temp markdown file and return the saved buffer with
        ``#!`` comment lines stripped; "" if the editor failed or the buffer was left empty.
        Factored out so the read-back/strip logic is unit-testable."""
        editor = (os.environ.get("VISUAL") or os.environ.get("EDITOR")
                  or ("notepad" if os.name == "nt" else "nano"))
        fd, path = tempfile.mkstemp(suffix=".md", prefix="hermes_prompt_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(_t("prompt_compose.editor_header") + "\n\n")
                if initial_text:
                    fh.write(initial_text)
            try:
                editor_argv = [*shlex.split(editor), path]
            except ValueError:
                return ""  # unbalanced quotes in $EDITOR: cancel, never retry through a shell
            try:
                status = subprocess.call(editor_argv)
            except OSError:
                return ""  # editor not runnable: cancel the compose (#81364)
            # A failed editor may leave seeded or abandoned text in the buffer.
            if status != 0:
                return ""
            with open(path, "r", encoding="utf-8-sig") as fh:
                raw = fh.read()
        finally:
            with suppress(OSError):
                os.unlink(path)
        return "\n".join(ln for ln in raw.splitlines() if not ln.startswith("#!")).strip()

    def _handle_prompt_compose_command(self, cmd_original: str) -> None:
        """Handle /prompt — compose the next prompt in $EDITOR (optionally seeded with the argument)
        and queue it as the next agent turn via the one-shot ``_pending_agent_seed`` (as /blueprint)."""
        parts = (cmd_original or "").strip().split(None, 1)
        try:
            composed = self._compose_in_editor(parts[1] if len(parts) > 1 else "")
        except Exception as exc:
            return _cp(_dim_line(_t("prompt_compose.editor_failed", error=exc)))
        if not composed:
            return _cp(_dim_line(_t("prompt_compose.empty")))
        # One-shot seed: the interactive loop runs this as the next agent turn right after
        # process_command() returns (see cli.py main loop).
        self._pending_agent_seed = composed

    # ---- /focus and its display hooks -----------------------------------------------------
    def _handle_focus_command(self, cmd_original: str) -> None:
        """``/focus [on|off|status]`` — DISPLAY-ONLY reduced output. Reuses the ``/verbose``
        suppression path: on stashes tool_progress_mode and snaps it to "off" (what
        ``agent/tool_executor.py`` gates on); off restores the stash verbatim. Never touches
        history, system prompt, or request payloads."""
        from hermes_cli.colors import Colors as _Colors
        from hermes_cli.focus_view import (
            FOCUS_CONFIG_KEY, FOCUS_TOOL_PROGRESS_MODE, format_focus_status,
            format_focus_toggle_message, normalize_tool_progress_mode, resolve_focus_arg)
        current = bool(getattr(self, "_focus_view_enabled", False))
        action, target = resolve_focus_arg(_command_arg(cmd_original), current)
        if action == "usage":
            return _cp(f"  {_t('focus.usage')}")
        # The mode /focus off restores: while focus is ON the live mode is "off", so use the stash.
        restore_mode = normalize_tool_progress_mode(
            getattr(self, "_focus_saved_tool_progress", None) if current
            else getattr(self, "tool_progress_mode", "all"))
        if action == "status":
            head, _, tail = format_focus_status(current, restore_mode).partition("\n")
            label, _, rest = head.partition(":")
            state_color = _Colors.GREEN if current else _Colors.DIM
            return _cp(f"  {_Colors.BOLD}{label}:{_Colors.RESET}{state_color}{rest}{_Colors.RESET}"
                       + (f"\n{_Colors.DIM}  {tail.strip()}{_Colors.RESET}" if tail else ""))
        if target == current:  # idempotent explicit set — report without rewriting config
            return _cp(f"  {format_focus_toggle_message(current, restore_mode)}")
        if target:
            # Stash the user's configured mode, then reuse the EXISTING suppression path by
            # snapping to "off".
            self._focus_saved_tool_progress = restore_mode
            self._set_tool_progress_mode(FOCUS_TOOL_PROGRESS_MODE)
        else:
            self._set_tool_progress_mode(restore_mode)
            self._focus_saved_tool_progress = None
        self._focus_view_enabled = bool(target)
        self._focus_hidden_lines = 0
        _save(FOCUS_CONFIG_KEY, bool(target))
        message = format_focus_toggle_message(bool(target), restore_mode)
        # Re-colour just the enabled/disabled word so the line matches siblings.
        for word in ("enabled", "disabled"):
            if word in message:
                colour = _Colors.GREEN if target else _Colors.DIM
                message = message.replace(word, f"{colour}{word}{_Colors.RESET}", 1)
                break
        _cp(f"  {message}")

    def _set_tool_progress_mode(self, mode: str) -> None:
        """Set the live tool-progress mode on both the CLI and the agent (one write path for
        /focus and /verbose — the agent copy is what ``agent/tool_executor.py`` gates on)."""
        from hermes_cli.focus_view import normalize_tool_progress_mode
        normalized = normalize_tool_progress_mode(mode)
        self.tool_progress_mode = normalized
        agent = getattr(self, "agent", None)
        if agent is not None:
            with suppress(Exception):
                agent.tool_progress_mode = normalized

    def _note_focus_hidden_line(self, function_name: str) -> None:
        """Count one tool line focus view is suppressing this turn — against the mode the user had
        BEFORE focus snapped to "off", so a prior ``/verbose off`` user is never told focus hid lines."""
        if not getattr(self, "_focus_view_enabled", False):
            return
        from hermes_cli.focus_view import would_display_tool_line
        saved = getattr(self, "_focus_saved_tool_progress", None)
        last = getattr(self, "_focus_last_counted_tool", None)
        if not would_display_tool_line(saved, function_name, last):
            return
        self._focus_last_counted_tool = function_name
        self._focus_hidden_lines = int(getattr(self, "_focus_hidden_lines", 0)) + 1

    def _emit_focus_recovery_line(self) -> None:
        """Print the dim post-turn recovery line and reset the counter."""
        count = int(getattr(self, "_focus_hidden_lines", 0) or 0)
        self._focus_hidden_lines = 0
        self._focus_last_counted_tool = None
        if not getattr(self, "_focus_view_enabled", False):
            return
        from hermes_cli.focus_view import format_hidden_line
        line = format_hidden_line(count)
        if line:
            with suppress(Exception):
                _cp(_dim_line(line))

    # ---- persisted display toggles: /approvals, /footer, /timestamps ----------------------
    def _handle_approvals_command(self, cmd_original: str) -> None:
        """Show or persist the profile-wide dangerous-command approval mode."""
        from hermes_cli.approval_mode import run_approval_mode_command
        parts = (cmd_original or "").strip().split(None, 1)
        result = run_approval_mode_command(parts[1] if len(parts) > 1 else None)
        _cp(f"  {result.message}")

    def _toggle_setting(self, arg: str, current: bool, *, usage: str, status_line: str,
                        config_key: str, label: str, failed: str):
        """Shared /footer + /timestamps flow: status query, usage error, or save + report.
        Returns the new bool state when it was saved (None otherwise)."""
        from hermes_cli.colors import Colors as _Colors
        new_state = _toggle_target(arg, current)
        if new_state == "status":
            state = _t("shared.state_on") if current else _t("shared.state_off")
            return _cp(f"  {_Colors.BOLD}{label}:{_Colors.RESET} {state}{status_line}")
        if new_state is None:
            return _cp(f"  {_t('shared.usage', usage=usage)}")
        if _save(config_key, new_state):
            colour = _Colors.GREEN if new_state else _Colors.DIM
            state = _t("shared.state_on") if new_state else _t("shared.state_off")
            _cp(f"  {label}: {colour}{state}{_Colors.RESET}")
        else:
            _cp(f"  {_t('toggle.save_failed', setting=failed)}")
        return new_state

    def _handle_footer_command(self, cmd_original: str) -> None:
        """Toggle or inspect ``display.runtime_footer.enabled`` (``/footer [on|off|status]``)."""
        from hermes_cli.config import load_config
        footer_cfg = (((load_config() or {}).get("display") or {}).get("runtime_footer") or {})
        fields = footer_cfg.get("fields") or ["model", "context_pct", "cwd"]
        self._toggle_setting(
            _command_arg(cmd_original, lower=True), bool(footer_cfg.get("enabled", False)),
            usage=_t("footer.usage"), status_line=f"\n  {_t('footer.fields', fields=', '.join(fields))}",
            config_key="display.runtime_footer.enabled", label=_t("footer.label"), failed="runtime_footer",
        )

    def _handle_timestamps_command(self, cmd_original: str) -> None:
        """Toggle or inspect ``display.timestamps`` (``/timestamps [on|off|status]``). When on,
        message labels carry an ``[HH:MM]`` suffix and ``/history`` prefixes stored-timestamp turns."""
        arg = _command_arg(cmd_original, lower=True)
        current = bool(getattr(self, "show_timestamps", False))
        new_state = _toggle_target(arg, current)
        if isinstance(new_state, bool):
            self.show_timestamps = new_state
        self._toggle_setting(
            arg, current, usage=_t("timestamps.usage"), status_line="",
            config_key="display.timestamps", label=_t("timestamps.label"), failed="timestamps")

    # ---- model-behaviour settings: /reasoning, /busy, /indicator, /fast -------------------
    def _handle_reasoning_command(self, cmd: str):
        """Handle /reasoning [<level> [--global]|show|hide|full|clamp] — effort level (session
        scope unless --global) and thinking display toggles (always saved)."""
        from cli import CLI_CONFIG, _parse_reasoning_config
        from agent.reasoning_effort import effort_display_label
        raw = _command_arg(cmd)
        from hermes_cli.codex_runtime_switch import get_current_runtime
        # The live agent's api_mode, else the configured runtime: ``ultra`` is verbatim on the Codex app-server.
        _route = (getattr(self, "provider", None), getattr(self, "model", None),
                  getattr(getattr(self, "agent", None), "api_mode", None) or get_current_runtime(CLI_CONFIG))
        if not raw:  # show current state
            rc = self.reasoning_config
            level = (_gt("reasoning.level_default") if rc is None else _gt("reasoning.level_disabled")
                     if rc.get("enabled") is False else effort_display_label(rc.get("effort", "medium"), *_route))
            display_state = _t("reasoning.display_on") if self.show_reasoning else _t("reasoning.display_off")
            full_state = (_t("reasoning.full_state_full") if getattr(self, "reasoning_full", False)
                          else _t("reasoning.full_state_clamped"))
            return _cp(_accent_line(_t("reasoning.effort_line", level=level)),
                       _accent_line(_t("reasoning.display_line", display=display_state, full=full_state)),
                       _dim_line(_t("reasoning.usage")))
        arg, explicit_global = _split_scope_flags(raw)
        toggle = _REASONING_TOGGLES.get(arg)
        if toggle is not None:  # display show/hide or full/clamp recap toggle
            attr, value, headline_key, note_key = toggle
            setattr(self, attr, value)
            if attr == "show_reasoning" and self.agent:
                self.agent.reasoning_callback = self._current_reasoning_callback()
            _save(f"display.{attr}", value)
            _cp(_accent_line(_t("reasoning.display_saved", headline=_t(f"reasoning.{headline_key}"))))
            if note_key:
                _cp(_dim_line(f"  {_t(f'reasoning.{note_key}')}"))
            if attr == "reasoning_full" and value and not self.show_reasoning:
                _cp(_dim_line(f"  {_t('reasoning.display_off_note')}"))
            return
        # Effort level change
        parsed = _parse_reasoning_config(arg)
        if parsed is None:
            return _cp(_dim_line(_t("shared.unknown_argument", arg=arg)),
                       _dim_line(_t("reasoning.valid_levels")), _dim_line(_t("reasoning.valid_display")),
                       _dim_line(_t("reasoning.valid_scope")))
        self.reasoning_config = parsed
        _retire_agent(self)  # Force agent re-init with new reasoning config
        saved = explicit_global and _save("agent.reasoning_effort", arg)
        if saved:
            if not isinstance(CLI_CONFIG.get("agent"), dict):
                CLI_CONFIG["agent"] = {}
            CLI_CONFIG["agent"]["reasoning_effort"] = arg
        _cp(_accent_line(_t("reasoning.effort_set", effort=effort_display_label(arg, *_route),
                            scope=_scope_outcome(explicit_global, saved))))

    def _handle_busy_command(self, cmd: str):
        """Handle /busy [status|queue|steer|interrupt] — what Enter does while Hermes is working."""
        arg = _command_arg(cmd, lower=True)
        usage = _dim_line(_t("busy.usage"))
        if not arg or arg == "status":
            mode = self.busy_input_mode if self.busy_input_mode in _BUSY_MODES else "interrupt"
            return _cp(_accent_line(_t("busy.status", mode=self.busy_input_mode)),
                       _dim_line(_t("busy.enter_while_busy", behavior=_t(f"busy.short_{mode}"))), usage)
        if arg not in _BUSY_MODES:
            return _cp(_dim_line(_t("shared.unknown_argument", arg=arg)), usage)
        self.busy_input_mode = arg
        _persist_display_choice("display.busy_input_mode", arg, _t("busy.label"), _t(f"busy.long_{arg}"))

    def _handle_indicator_command(self, cmd: str):
        """Handle /indicator [status|kaomoji|emoji|unicode|ascii] — pick the TUI busy-indicator style.
        Persists to ``display.tui_status_indicator`` (the key the TUI reads) for its next render."""
        from hermes_constants import DEFAULT_INDICATOR_STYLE, INDICATOR_STYLES
        current = (self.config.get("display") or {}).get("tui_status_indicator", DEFAULT_INDICATOR_STYLE)
        arg = _command_arg(cmd, lower=True)
        usage = _dim_line(_t("indicator.usage", styles="|".join(INDICATOR_STYLES)))
        if not arg or arg == "status":
            return _cp(_accent_line(_t("indicator.status", style=current)), usage)
        if arg not in INDICATOR_STYLES:
            return _cp(_dim_line(_t("indicator.unknown_style", arg=arg)), usage)
        self.config.setdefault("display", {})["tui_status_indicator"] = arg
        _persist_display_choice("display.tui_status_indicator", arg, _t("indicator.label"), _t("indicator.note"))

    def _handle_fast_command(self, cmd: str):
        """Handle /fast — toggle fast mode (OpenAI Priority Processing / Anthropic Fast Mode).
        Session-scoped by default; ``--global`` persists agent.service_tier to config.yaml
        (parity with /model and /reasoning)."""
        if not self._fast_command_available():
            return _cp(f"  {_t('fast.not_supported')}")
        # Determine the branding for the current model
        model = getattr(getattr(self, "agent", None), "model", None) or getattr(self, "model", None)
        anthropic = _probe("hermes_cli.models", "_is_anthropic_fast_model", None, model)
        feature_name = _t("fast.feature_generic" if anthropic is None
                          else "fast.feature_anthropic" if anthropic else "fast.feature_openai")
        raw = _command_arg(cmd)
        usage = _dim_line(_t("fast.usage"))
        if not raw or raw.lower() == "status":
            from agent.fast_mode import service_tier_word
            status = service_tier_word(self.service_tier)
            return _cp(_accent_line(_t("fast.status", feature=feature_name, status=status)), usage)
        arg, explicit_global = _split_scope_flags(raw)
        if arg not in _FAST_TIERS:
            return _cp(_dim_line(_t("shared.unknown_argument", arg=arg)), usage)
        if arg == "ultrafast":
            if not _probe("hermes_cli.models", "model_supports_ultrafast", False, model):
                return _cp(_dim_line(_t("fast.ultrafast_not_supported", model=model or "?")), usage)
            feature_name = _t("fast.feature_ultrafast")
        self.service_tier, saved_value = _FAST_TIERS[arg]
        _retire_agent(self)  # Force agent re-init with new service-tier config
        saved = explicit_global and _save("agent.service_tier", saved_value)
        outcome = _scope_outcome(explicit_global, saved)
        _cp(_accent_line(_t("fast.set_to", feature=feature_name, value=saved_value.upper(), scope=outcome)))

    # ---- /debug, /update, /voice, /wake ---------------------------------------------------
    def _handle_debug_command(self, cmd_original: str = ""):
        """Handle /debug [nous|local] — upload debug report + logs and print share URLs.
        Default: public paste service; ``nous``: Nous-internal (staff-only); ``local``: render to
        stdout, no upload. ``local`` wins if both are given (never touches the network)."""
        from hermes_cli.debug import run_debug_share
        from types import SimpleNamespace
        words = {w.lower() for w in cmd_original.split()[1:]}
        local = "local" in words
        # Typing /debug is the upload consent (yes=True); input() would hang in prompt_toolkit anyway.
        run_debug_share(SimpleNamespace(
            lines=200, expire=7, local=local, nous="nous" in words and not local, yes=True))

    def _handle_update_command(self) -> bool:
        """Handle /update — exit the session and relaunch as ``hermes update``. Returns True when
        confirmed (the caller exits the app; the relaunch runs on the main thread after
        prompt_toolkit restores terminal modes), False when cancelled."""
        from hermes_cli.config import is_managed, format_managed_message
        if is_managed():
            print(f"  ✗ {format_managed_message(_t('update.managed_action'))}")
            return False
        # prompt_toolkit-native modal: renders above the composer, no raw input() races.
        choices = [("once", _t("update.choice_update"), _t("update.choice_update_desc")),
                   ("cancel", _t("update.choice_cancel"), _t("update.choice_cancel_desc"))]
        raw = self._prompt_text_input_modal(
            title=_t("update.title"), detail=_t("update.detail"), choices=choices)
        if raw is None or self._normalize_slash_confirm_choice(raw, choices) != "once":
            print(f"  {_t('update.cancelled')}")
            return False
        _say_block(f"  {_t('update.launching')}")
        # run() execs this on the main thread after prompt_toolkit restores terminal modes;
        # relaunching from this daemon thread would skip cleanup (POSIX) / only end the thread (Windows).
        self._pending_relaunch = ["update"]
        return True

    def _handle_voice_command(self, command: str):
        """Handle /voice [on|off|tts|status] command."""
        subcommand = _command_arg(command, lower=True) or ("off" if self._voice_mode else "on")
        actions = {"on": self._enable_voice_mode, "off": self._disable_voice_mode,
                   "tts": self._toggle_voice_tts, "status": self._show_voice_status}
        if subcommand in actions:
            actions[subcommand]()
        else:
            _cp(_t("voice.unknown_subcommand", subcommand=subcommand), _t("voice.usage"))

    def _handle_wake_command(self, command: str):
        """Handle /wake [on|off|status] — the 'Hey Hermes' hotword listener. The toggle IS the
        config: on/off also writes ``wake_word.enabled`` so the choice persists; startup
        auto-arm only reads it."""
        subcommand = _command_arg(command, lower=True) or (
            "off" if getattr(self, "_wake_word_active", False) else "on")  # bare /wake toggles
        if subcommand == "on":
            if self._start_wake_word_listener(announce=True):
                self._persist_wake_word_enabled(True)
        elif subcommand == "off":
            self._stop_wake_word_listener(announce=True)
            self._persist_wake_word_enabled(False)
        elif subcommand == "status":
            self._show_wake_word_status()
        else:
            _cp(_t("wake.unknown_subcommand", subcommand=subcommand), _t("wake.usage"))

    def _persist_wake_word_enabled(self, enabled: bool):
        """Save ``wake_word.enabled`` so the /wake toggle sticks for future sessions."""
        persisted = _probe("tools.wake_word", "load_wake_word_config", None)
        if isinstance(persisted, dict) and bool(persisted.get("enabled")) == enabled:
            return  # already persisted — don't rewrite config or re-announce
        if _save("wake_word.enabled", enabled):
            _cp(_dim(_t("wake.enabled_in_config" if enabled else "wake.disabled_in_config")))
