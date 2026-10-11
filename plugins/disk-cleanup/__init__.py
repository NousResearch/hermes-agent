"""disk-cleanup plugin — auto-cleanup of ephemeral Hermes session files.

``pre_tool_call`` + ``post_tool_call`` silently track test/temp paths CREATED by write_file/patch/terminal;
``on_session_end`` follows the configured cleanup/report-only mode when any test file was tracked;
``/disk-cleanup`` exposes status / dry-run / quick / deep / track / forget.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

from . import disk_cleanup as dg

logger = logging.getLogger(__name__)


# Test files newly tracked this turn, keyed by task_id (or session_id) so on_session_end can
# decide whether to run cleanup. Locked: post_tool_call fires concurrently on parallel calls.
_recent_test_tracks: dict[str, set[str]] = {}
_lock = threading.Lock()
_SESSION_END_MODES = frozenset({"report_only", "cleanup", "disabled"})
_MAX_REPORTED_CANDIDATES = 100


def _extract_path_arg(args: dict[str, Any]) -> set[str]:
    """write_file/patch: the single ``path`` arg."""
    path = args.get("path")
    return {path} if isinstance(path, str) and path else set()


def _extract_paths_from_terminal(args: dict[str, Any]) -> set[str]:
    """Candidate paths named in a terminal command; guess_category/is_safe_path filter later."""
    paths: set[str] = set()
    cmd = args.get("command") or ""
    if isinstance(cmd, str) and cmd:
        # Tokenise the command — catches `touch /tmp/hermes-x/test_foo.py`.
        # ``posix`` follows the host so Windows backslash paths survive
        # (``shlex.split(posix=True)`` would eat them as escapes).
        # Non-posix mode keeps quote characters in tokens, so strip a fully
        # wrapping pair (`"C:\file"` → `C:\file`) before matching.
        try:
            for tok in shlex.split(cmd, posix=os.name == "posix"):
                if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ("'", '"'):
                    tok = tok[1:-1]
                if tok.startswith(("/", "~")) or re.match(r"^[A-Za-z]:[\\/]", tok):
                    paths.add(tok)
        except ValueError:
            pass
    return paths


_PATH_EXTRACTORS: dict[str, Callable[[dict[str, Any]], set[str]]] = {
    "write_file": _extract_path_arg,
    "patch": _extract_path_arg,
    "terminal": _extract_paths_from_terminal}


# (owner, tool_call_id) -> (snapshot time, argument paths absent before the call). Post tracks only
# these, so pre-existing files and paths swapped in after the snapshot (another hook's modify) fail closed.
_pre_call: dict[tuple[str, str], tuple[float, frozenset[str]]] = {}
_PRE_CALL_TTL_S = 3600.0  # a call whose post hook never fires (blocked) must not leak its entry


def _owner(task_id: str, session_id: str) -> str:
    return task_id or session_id or "default"


def _pre_call_key(task_id: str, session_id: str, tool_call_id: str) -> tuple[str, str]:
    # tool_call_id alone is not unique (llama.cpp sends one constant id), so scope it by owner.
    return (_owner(task_id, session_id), tool_call_id)


def _on_pre_tool_call(tool_name: str = "", args: Optional[dict[str, Any]] = None, task_id: str = "",
                      session_id: str = "", tool_call_id: str = "", **_: Any) -> None:
    """Snapshot which argument paths are absent before the call. Never raises: pre_tool_call hooks
    fail CLOSED (a raise would block the tool), so an error records no absent path."""
    extractor = _PATH_EXTRACTORS.get(tool_name)
    if not tool_call_id or extractor is None or not isinstance(args, dict):
        return
    now = time.time()
    try:
        absent = frozenset(str(p) for p in (Path(s).expanduser() for s in extractor(args)) if not p.exists())
    except Exception:
        absent = frozenset()
    with _lock:
        for key in [k for k, (taken, _a) in _pre_call.items() if now - taken > _PRE_CALL_TTL_S]:
            del _pre_call[key]
        _pre_call[_pre_call_key(task_id, session_id, tool_call_id)] = (now, absent)
    return


def _on_post_tool_call(tool_name: str = "", args: Optional[dict[str, Any]] = None,
                       task_id: str = "", session_id: str = "", tool_call_id: str = "", **_: Any) -> None:
    """Auto-track ephemeral files THIS call created: a path from the call's final arguments that
    this call's own pre-call snapshot saw absent and that exists now. Paths seen only in terminal
    OUTPUT (a `find` or `ls` listing), or absent from the snapshot, carry no such proof and are
    never tracked. Best-effort, never raises."""
    extractor = _PATH_EXTRACTORS.get(tool_name)
    if not isinstance(args, dict) or extractor is None or not tool_call_id:
        return
    with _lock:
        absent = _pre_call.pop(_pre_call_key(task_id, session_id, tool_call_id), (0.0, frozenset()))[1]
    if not absent:
        return
    for path_str in extractor(args):
        try:
            p = Path(path_str).expanduser()
            created = str(p) in absent and p.exists()
        except Exception:
            continue
        category = dg.guess_category(p) if created else None
        if category is not None and dg.track(str(p), category, silent=True) and category == "test":
            with _lock:
                _recent_test_tracks.setdefault(_owner(task_id, session_id), set()).add(str(p))


def _session_end_mode() -> str:
    """Read the legacy plugin config; absent preserves the historical cleanup default.

    A present but malformed value is deliberately disabled rather than silently becoming
    destructive. Before using the canonical merged config, validate the user and managed files themselves:
    the general loader deliberately recovers from invalid YAML with defaults, which would
    otherwise make an ambiguous configured state look like an absent setting. The strict
    check establishes only whether the files are unambiguous; the final mode still comes
    from ``load_config_readonly`` so managed overlays retain their normal precedence.
    """
    try:
        from hermes_cli.config import get_config_path, load_config_readonly
        from utils import fast_safe_load

        from hermes_cli.managed_scope import get_managed_dir

        config_paths = [get_config_path()]
        managed_dir = get_managed_dir()
        if managed_dir is not None:
            config_paths.append(managed_dir / "config.yaml")
        for config_path in config_paths:
            try:
                with open(config_path, encoding="utf-8-sig") as f:
                    user_config = fast_safe_load(f)
            except FileNotFoundError:
                pass  # No user config is the historical cleanup-default state.
            except Exception:
                return "disabled"
            else:
                if not isinstance(user_config, dict):
                    return "disabled"
                # Validate the relevant raw shape without reading its value for precedence.
                # ``_deep_merge`` intentionally fills a null plugin block from defaults, but this
                # destructive lifecycle hook must treat such an explicit ambiguity as disabled.
                raw_plugins = user_config.get("plugins", {})
                if not isinstance(raw_plugins, dict):
                    return "disabled"
                raw_plugin_config = raw_plugins.get("config", {})
                if not isinstance(raw_plugin_config, dict):
                    return "disabled"
                if "disk-cleanup" in raw_plugin_config and not isinstance(
                    raw_plugin_config["disk-cleanup"], dict,
                ):
                    return "disabled"
        plugins = (load_config_readonly() or {}).get("plugins", {})
        if not isinstance(plugins, dict):
            return "disabled"
        config = plugins.get("config", {})
        if config is None:
            return "disabled"
        if not isinstance(config, dict):
            return "disabled"
        if "disk-cleanup" not in config:
            return "cleanup"
        disk_cleanup = config["disk-cleanup"]
        if not isinstance(disk_cleanup, dict):
            return "disabled"
        if "session_end_mode" not in disk_cleanup:
            return "cleanup"
        mode = disk_cleanup["session_end_mode"]
    except Exception:
        return "disabled"
    return mode if isinstance(mode, str) and mode in _SESSION_END_MODES else "disabled"


def _report_session_end_candidates() -> dict[str, Any]:
    """Return bounded candidate counts only; dry-run never mutates tracked files or paths."""
    try:
        auto, prompt = dg.dry_run()
        auto_count, prompt_count = len(auto), len(prompt)
    except Exception as exc:
        logger.debug("disk-cleanup report-only candidate query failed: %s", type(exc).__name__)
        return {"mode": "report_only", "status": "unavailable"}
    report = {
        "mode": "report_only",
        "auto_candidates": min(auto_count, _MAX_REPORTED_CANDIDATES),
        "prompt_candidates": min(prompt_count, _MAX_REPORTED_CANDIDATES),
        "truncated": auto_count > _MAX_REPORTED_CANDIDATES or prompt_count > _MAX_REPORTED_CANDIDATES,
    }
    logger.info("disk-cleanup AUTO_REPORT (session_end): auto=%d prompt=%d truncated=%s",
                report["auto_candidates"], report["prompt_candidates"], report["truncated"])
    return report


def _on_session_end(
    session_id: str = "", completed: bool = True, interrupted: bool = False, **_: Any) -> Optional[dict[str, Any]]:
    """Apply the configured automatic lifecycle mode after draining tracked test paths."""
    # Drain the session bucket plus every task-scoped bucket (subagents record into their own).
    with _lock:
        had_tracks = bool(_recent_test_tracks.pop(session_id or "default", None) or _recent_test_tracks)
        _recent_test_tracks.clear()
    if not had_tracks:
        return
    mode = _session_end_mode()
    if mode == "disabled":
        return {"mode": "disabled"}
    if mode == "report_only":
        return _report_session_end_candidates()
    try:
        summary = dg.quick()
    except Exception as exc:
        logger.debug("disk-cleanup quick cleanup failed: %s", exc)
        return
    if summary["deleted"] or summary["empty_dirs"]:
        dg._log(f"AUTO_QUICK (session_end): deleted={summary['deleted']} "
                f"dirs={summary['empty_dirs']} freed={dg.fmt_size(summary['freed'])}")


_HELP_TEXT = """\
/disk-cleanup — ephemeral-file cleanup

Subcommands:
  status                     Per-category breakdown + top-10 largest
  dry-run                    Preview what quick/deep would delete
  quick                      Run safe cleanup now (no prompts)
  deep                       Run quick, then list items that need prompts
  track <path> <category>    Manually add a path to tracking
  forget <path>              Stop tracking a path (does not delete)

Categories: temp | test | research | download | chrome-profile | cron-output | other

All operations are scoped to HERMES_HOME and /tmp/hermes-*.  # no-tmp: ok — legacy scratch scope this plugin cleans up
Test files are auto-tracked on write_file / terminal and auto-cleaned at session end.
"""


def _fmt_summary(summary: dict[str, Any]) -> str:
    base = (f"[disk-cleanup] Cleaned {summary['deleted']} files + "
            f"{summary['empty_dirs']} empty dirs, freed {dg.fmt_size(summary['freed'])}.")
    if summary.get("errors"):
        base += f"\n  {len(summary['errors'])} error(s); see cleanup.log."
    return base


def _item_block(header: str, items: list[dict], indent: str) -> list[str]:
    """``header`` formatted with ``{n}`` / ``{size}``, followed by one line per item."""
    size = dg.fmt_size(sum(i["size"] for i in items))
    return [header.format(n=len(items), size=size)] + [f"{indent}[{item['category']}] {item['path']}" for item in items]


def _cmd_dry_run(argv: list[str]) -> str:
    auto, prompt = dg.dry_run()
    lines = ["Dry-run preview (nothing deleted):"]
    lines += _item_block("  Auto-delete : {n} files ({size})", auto, "    ")
    lines += _item_block("  Needs prompt: {n} files ({size})", prompt, "    ")
    lines.append(f"\n  Total potential: {dg.fmt_size(sum(i['size'] for i in auto + prompt))}")
    return "\n".join(lines)


def _cmd_deep(argv: list[str]) -> str:
    # In-session deep can't prompt — show what quick cleaned plus items needing confirmation.
    quick_summary = dg.quick()
    _auto, prompt_items = dg.dry_run()
    lines = [_fmt_summary(quick_summary)]
    if prompt_items:
        lines += _item_block("\n{n} item(s) need confirmation ({size}):", prompt_items, "  ")
        lines.append("\nRun `/disk-cleanup forget <path>` to skip, or delete manually via terminal.")
    return "\n".join(lines)


def _cmd_track(argv: list[str]) -> str:
    if len(argv) < 3:
        return "Usage: /disk-cleanup track <path> <category>"
    path_arg, category = argv[1], argv[2]
    if category not in dg.ALLOWED_CATEGORIES:
        return f"Unknown category '{category}'. Allowed: {sorted(dg.ALLOWED_CATEGORIES)}"
    if dg.track(path_arg, category, silent=True):
        return f"Tracked {path_arg} as '{category}'."
    return f"Not tracked (already present, missing, or outside HERMES_HOME): {path_arg}"


def _cmd_forget(argv: list[str]) -> str:
    if len(argv) < 2:
        return "Usage: /disk-cleanup forget <path>"
    n = dg.forget(argv[1])
    return (f"Removed {n} tracking entr{'y' if n == 1 else 'ies'} for {argv[1]}." if n
            else f"Not found in tracking: {argv[1]}")


_SUBCOMMANDS: dict[str, Callable[[list[str]], str]] = {
    "status": lambda argv: dg.format_status(dg.status()),
    "dry-run": _cmd_dry_run,
    "quick": lambda argv: _fmt_summary(dg.quick()),
    "deep": _cmd_deep,
    "track": _cmd_track,
    "forget": _cmd_forget}


def _handle_slash(raw_args: str) -> Optional[str]:
    argv = raw_args.strip().split()
    if not argv or argv[0] in {"help", "-h", "--help"}:
        return _HELP_TEXT
    handler = _SUBCOMMANDS.get(argv[0])
    if handler is None:
        return f"Unknown subcommand: {argv[0]}\n\n{_HELP_TEXT}"
    return handler(argv)


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_command("disk-cleanup", handler=_handle_slash,
                         description="Track and clean up ephemeral Hermes session files.")
