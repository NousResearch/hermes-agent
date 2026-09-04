"""Shell-script hooks bridge: ``hooks:`` config → first-use consent per ``(event, command)`` →
callbacks on the plugin hook manager, so every ``invoke_hook()`` site dispatches to the scripts.
Wire: stdin JSON ``{hook_event_name, tool_name, tool_input, session_id, cwd, extra}``; optional stdout
JSON ``{"decision"|"action": "block"|"modify", ...}`` / ``{"context": ...}`` via ``_parse_response``.
Exit code 2 blocks a ``pre_tool_call`` even without JSON (Claude-Code / Cursor). Fail open unless ``fail_closed``."""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

# split_command_line, not shlex: shlex eats Windows path backslashes.
from hermes_cli._subprocess_compat import IS_WINDOWS, kill_process_tree, split_command_line, windows_hide_flags

try:
    import fcntl  # POSIX only; Windows falls back to best-effort without flock.
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from hermes_constants import get_hermes_home
from utils import atomic_replace

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 300
ALLOWLIST_FILENAME = "shell-hooks-allowlist.json"
_DEFAULT_BLOCK_MESSAGE = "Blocked by shell hook."
# Exit code that signals "block this action" independent of stdout (Claude Code / Cursor).
BLOCK_EXIT_CODE = 2
# Events whose block directive is honored downstream; exit-2 blocking and fail_closed only apply here.
_BLOCKING_EVENTS = frozenset({"pre_tool_call"})
_TOOL_EVENTS = frozenset({"pre_tool_call", "post_tool_call"})
_STDERR_MESSAGE_LIMIT = 400
_TRUTHY = {"1", "true", "yes", "on"}
# kwargs promoted to top-level payload keys; everything else lands under ``extra``.
_TOP_LEVEL_PAYLOAD_KEYS = {"tool_name", "args", "session_id", "parent_session_id"}

# (home, event, matcher, command) wired in this process: matcher in the key (one script may register
# per-tool under one event), home so multiplexed-gateway profiles can register identical triples.
_registered: Set[Tuple[str, str, Optional[str], str]] = set()
_registered_lock = threading.Lock()
# Non-POSIX fallback for allowlist read-modify-write. Must be separate from _registered_lock, which
# register_from_config already holds when it triggers _record_approval (Lock is non-reentrant).
_allowlist_write_lock = threading.Lock()


def _home_key() -> str:
    return str(get_hermes_home().expanduser().resolve())


def _forget_home_registrations(registry: Set[tuple], lock: threading.Lock) -> None:
    """Drop the current home's keys only (shared with outbound webhooks): profile A's reload must not drop B."""
    home_key = _home_key()
    with lock:
        registry.difference_update({k for k in registry if k[0] == home_key})


def _entry_matches(e: Any, event: Optional[str], command: str) -> bool:
    return isinstance(e, dict) and (event is None or e.get("event") == event) and e.get("command") == command


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _payload_fields(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Common stdin/POST payload fields (shared with outbound webhooks); key order is wire order."""
    try:
        cwd = str(Path.cwd())
    except OSError:
        cwd = ""
    return {
        "tool_name": kwargs.get("tool_name"),
        "tool_input": kwargs.get("args") if isinstance(kwargs.get("args"), dict) else None,
        "session_id": kwargs.get("session_id") or kwargs.get("parent_session_id") or "",
        "cwd": cwd,
        "extra": {k: v for k, v in kwargs.items() if k not in _TOP_LEVEL_PAYLOAD_KEYS},
    }


class _ToolMatcherMixin:
    """``matcher`` regex handling shared by shell-hook specs and outbound webhook targets."""

    _MATCHER_KIND = "shell hook"
    matcher: Optional[str]
    compiled_matcher: Optional[re.Pattern]

    def __post_init__(self) -> None:
        # Strip YAML folding whitespace — " terminal" would silently never match.
        if isinstance(self.matcher, str):
            self.matcher = self.matcher.strip() or None
        if self.matcher:
            try:
                self.compiled_matcher = re.compile(self.matcher)
            except re.error as exc:
                logger.warning(
                    "%s matcher %r is invalid (%s) — treating as literal equality", self._MATCHER_KIND, self.matcher, exc,
                )
                self.compiled_matcher = None

    def matches_tool(self, tool_name: Optional[str]) -> bool:
        if not self.matcher:
            return True
        if tool_name is None:
            return False
        if self.compiled_matcher is None:  # regex failed to compile: literal fallback
            return tool_name == self.matcher
        return self.compiled_matcher.fullmatch(tool_name) is not None


@dataclass
class ShellHookSpec(_ToolMatcherMixin):
    """Parsed and validated representation of a single ``hooks:`` entry."""

    event: str
    command: str
    matcher: Optional[str] = None
    timeout: int = DEFAULT_TIMEOUT_SECONDS
    fail_closed: bool = False
    compiled_matcher: Optional[re.Pattern] = field(default=None, repr=False)


# --- Public API ---

def register_from_config(cfg: Optional[Dict[str, Any]], *, accept_hooks: bool = False) -> List[ShellHookSpec]:
    """Register every configured shell hook (idempotent); returns the newly wired specs. Skipped
    entries (unknown, malformed, not allowlisted, already registered) are logged only."""
    if not isinstance(cfg, dict):
        return []
    from utils import env_var_enabled
    if env_var_enabled("HERMES_SAFE_MODE"):  # hooks are user customizations too — fire zero user-configured code
        logger.info("HERMES_SAFE_MODE=1 — shell-hook registration skipped")
        return []
    effective_accept = _resolve_effective_accept(cfg, accept_hooks)
    specs = _parse_hooks_block(cfg.get("hooks"))
    if not specs:
        return []
    from hermes_cli.plugins import get_plugin_manager  # lazy: avoids import cycle
    manager, home_key, registered = get_plugin_manager(), _home_key(), []
    # Idempotence + allowlist read under the lock; TTY prompt outside it; mutation re-takes the lock and re-checks.
    for spec in specs:
        key = (home_key, spec.event, spec.matcher, spec.command)
        with _registered_lock:
            if key in _registered:
                continue
            already_allowlisted = _is_allowlisted(spec.event, spec.command)
        if not already_allowlisted and not _prompt_and_record(spec.event, spec.command, accept_hooks=effective_accept):
            logger.warning("shell hook for %s (%s) not allowlisted — skipped. Use --accept-hooks / "
                           "HERMES_ACCEPT_HOOKS=1 / hooks_auto_accept: true, or approve at the TTY prompt next run.",
                           spec.event, spec.command)
            continue
        with _registered_lock:
            if key in _registered:
                continue
            manager._hooks.setdefault(spec.event, []).append(_make_callback(spec))
            _registered.add(key)
            registered.append(spec)
            logger.info("shell hook registered: %s -> %s (matcher=%s, timeout=%ds, fail_closed=%s)",
                        spec.event, spec.command, spec.matcher, spec.timeout, spec.fail_closed)
    return registered


def iter_configured_hooks(cfg: Optional[Dict[str, Any]]) -> List[ShellHookSpec]:
    """Parse config hooks without registering (``hermes hooks list`` / doctor)."""
    return _parse_hooks_block(cfg.get("hooks")) if isinstance(cfg, dict) else []


def re_register_config_hooks() -> None:
    """Re-register after a plugin force-reload cleared the manager's hooks; only this home's keys
    are cleared (profile A's reload never drops B), never re-prompts.

    ``PluginManager.discover_and_load(force=True)`` unloads via the ownership ledger and clears the
    manager's ``_hooks`` dict, which silently drops shell hooks that were registered from ``config.yaml`` at
    startup (they are config-owned, not plugin-owned, so the ledger cannot restore them). Clear the
    idempotence set and re-run ``register_from_config()`` so hooks are wired again (#60036 / PR #60267;
    tracking #64178 — salvaged from PR #64188).
    Only the idempotence keys for the *current* Hermes home are cleared — ``discover_and_load(force=True)``
    only unloads the manager scoped to that one home, so clearing every home's keys would make a
    force-reload in profile A drop profile B's still-live registration from the ledger and duplicate it on
    B's next registration call (#92682 review).
    """
    _forget_home_registrations(_registered, _registered_lock)
    from hermes_cli.config import load_config
    register_from_config(load_config())


def reset_for_tests() -> None:
    """Test-only: clear the idempotence set."""
    with _registered_lock:
        _registered.clear()


# --- Config parsing ---

def _parse_hooks_block(hooks_cfg: Any) -> List[ShellHookSpec]:
    """Normalise ``hooks:`` into specs; malformed entries warn-and-skip, never raise."""
    from hermes_cli.plugins import SHELL_UNSUPPORTED_HOOKS, VALID_HOOKS
    if not isinstance(hooks_cfg, dict):
        return []
    specs: List[ShellHookSpec] = []
    for event_name, entries in hooks_cfg.items():
        if event_name in ("output_spill", "outbound"):  # reserved non-event sub-sections under `hooks:`
            continue
        if event_name in SHELL_UNSUPPORTED_HOOKS:  # _parse_response has no channel for these directives — refuse loudly
            logger.warning("hook event %r is Python-plugin-only: shell hooks cannot return its directive, "
                           "so this registration is refused rather than silently ignored", event_name)
            continue
        if event_name not in VALID_HOOKS:
            suggestion = difflib.get_close_matches(str(event_name), VALID_HOOKS, n=1, cutoff=0.6)
            if suggestion:
                logger.warning("unknown hook event %r in hooks: config — did you mean %r?", event_name, suggestion[0])
            else:
                logger.warning("unknown hook event %r in hooks: config (valid: %s)", event_name, ", ".join(sorted(VALID_HOOKS)))
            continue
        if entries is None:
            continue
        if not isinstance(entries, list):
            logger.warning("hooks.%s must be a list of hook definitions; got %s", event_name, type(entries).__name__)
            continue
        specs.extend(filter(None, (_parse_single_entry(event_name, i, raw) for i, raw in enumerate(entries))))
    return specs


def _parse_single_entry(event: str, index: int, raw: Any) -> Optional[ShellHookSpec]:
    def warn(msg: str, *args: Any) -> None:
        logger.warning("hooks.%s[%d]" + msg, event, index, *args)

    if not isinstance(raw, dict):
        warn(" must be a mapping with a 'command' key; got %s", type(raw).__name__)
        return None
    command = raw.get("command")
    if not isinstance(command, str) or not command.strip():
        warn(" is missing a non-empty 'command' field")
        return None
    matcher = raw.get("matcher")
    if matcher is not None and not isinstance(matcher, str):
        warn(".matcher must be a string regex; ignoring")
        matcher = None
    if matcher is not None and event not in _TOOL_EVENTS:
        warn(".matcher=%r will be ignored at runtime — the matcher field is only honored for "
             "pre_tool_call / post_tool_call.  The hook will fire on every %s event.", matcher, event)
        matcher = None
    try:
        timeout = int(raw.get("timeout", DEFAULT_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        warn(".timeout must be an int (got %r); using default %ds", raw.get("timeout"), DEFAULT_TIMEOUT_SECONDS)
        timeout = DEFAULT_TIMEOUT_SECONDS
    if timeout < 1:
        warn(".timeout must be >=1; using default %ds", DEFAULT_TIMEOUT_SECONDS)
        timeout = DEFAULT_TIMEOUT_SECONDS
    elif timeout > MAX_TIMEOUT_SECONDS:
        warn(".timeout=%ds exceeds max %ds; clamping", timeout, MAX_TIMEOUT_SECONDS)
        timeout = MAX_TIMEOUT_SECONDS
    # ``fail_closed`` (canonical) wins over ``failClosed`` (Cursor/Claude-Code compat).
    fail_closed = raw.get("fail_closed", raw.get("failClosed", False))
    if not isinstance(fail_closed, bool):
        warn(".fail_closed must be a boolean (got %r); using default false (fail open)", fail_closed)
        fail_closed = False
    if fail_closed and event not in _BLOCKING_EVENTS:
        warn(".fail_closed=true will be ignored at runtime — fail_closed only applies to blocking-capable "
             "events (%s).  The hook will fail open on %s like any other hook.", ", ".join(sorted(_BLOCKING_EVENTS)), event)
        fail_closed = False
    return ShellHookSpec(event=event, command=command.strip(), matcher=matcher, timeout=timeout, fail_closed=fail_closed)


# --- Subprocess callback ---

# Popen failure -> diagnostic; anything else is reported as str(exc).
_POPEN_ERRORS = ((FileNotFoundError, "command not found"), (PermissionError, "command not executable"))


def _spawn(spec: ShellHookSpec, stdin_json: str) -> Dict[str, Any]:
    """The single subprocess site: run ``spec.command`` with ``stdin_json`` on stdin. Same result keys for every outcome."""
    result: Dict[str, Any] = {"returncode": None, "stdout": "", "stderr": "", "timed_out": False, "elapsed_seconds": 0.0, "error": None}

    def failed(error: str) -> Dict[str, Any]:
        result["error"] = error
        return result

    try:
        argv = split_command_line(os.path.expanduser(spec.command))
    except ValueError as exc:
        return failed(f"command {spec.command!r} cannot be parsed: {exc}")
    if not argv:
        return failed("empty command")
    t0 = time.monotonic()
    # Own process group on POSIX so a timed-out hook's descendants are reaped with it (Windows: kill_process_tree
    # / taskkill /T). Hooks that finish in time keep detached helpers alive.
    popen_kwargs: Dict[str, Any] = {"creationflags": windows_hide_flags()} if IS_WINDOWS else {"process_group": 0}
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding='utf-8', errors='replace', shell=False, **popen_kwargs)
    except Exception as exc:
        return failed(next((msg for cls, msg in _POPEN_ERRORS if isinstance(exc, cls)), str(exc)))
    try:
        stdout, stderr = proc.communicate(input=stdin_json, timeout=spec.timeout)
    except Exception as exc:
        kill_process_tree(proc)  # the whole tree — forked helpers holding the pipes would stall the drain
        with suppress(Exception):
            proc.communicate(timeout=1)
        if not isinstance(exc, subprocess.TimeoutExpired):  # pragma: no cover — defensive
            return failed(str(exc))
        result.update(timed_out=True, elapsed_seconds=round(time.monotonic() - t0, 3))
        return result
    result.update(returncode=proc.returncode, stdout=stdout or "", stderr=stderr or "", elapsed_seconds=round(time.monotonic() - t0, 3))
    return result


def _make_callback(spec: ShellHookSpec) -> Callable[..., Optional[Dict[str, Any]]]:
    """Build the closure that ``invoke_hook()`` will call per firing."""

    def _callback(**kwargs: Any) -> Optional[Dict[str, Any]]:
        if spec.event in _TOOL_EVENTS and not spec.matches_tool(kwargs.get("tool_name")):
            return None
        return _evaluate_result(spec, _spawn(spec, _serialize_payload(spec.event, kwargs)))

    _callback.__name__ = _callback.__qualname__ = f"shell_hook[{spec.event}:{spec.command}]"
    return _callback


def _fail_closed_block(spec: ShellHookSpec, reason: str) -> Dict[str, Any]:
    return {"action": "block", "message": f"hook {spec.command} failed closed: {reason}"}


def _evaluate_result(spec: ShellHookSpec, r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """``_spawn`` result → hook contribution (live callback and ``run_once``). Spawn error/timeout fail
    open unless fail_closed; exit 2 on a blocking event blocks (message: stdout JSON, then stderr, then
    default); other non-zero exits warn then parse stdout; unparseable stdout on a fail_closed hook blocks."""
    blocking_event = spec.event in _BLOCKING_EVENTS
    fail_closed = spec.fail_closed and blocking_event
    if r["error"]:
        logger.warning("shell hook failed (event=%s command=%s): %s", spec.event, spec.command, r["error"])
    elif r["timed_out"]:
        logger.warning("shell hook timed out after %.2fs (event=%s command=%s)", r["elapsed_seconds"], spec.event, spec.command)
    if r["error"] or r["timed_out"]:
        return _fail_closed_block(spec, r["error"] or f"timed out after {spec.timeout}s") if fail_closed else None
    stderr = r["stderr"].strip()
    if stderr:
        logger.debug("shell hook stderr (event=%s command=%s): %s", spec.event, spec.command, stderr[:_STDERR_MESSAGE_LIMIT])
    if r["returncode"] == BLOCK_EXIT_CODE and blocking_event:
        parsed = _parse_response(spec.event, r["stdout"])
        if isinstance(parsed, dict) and parsed.get("action") == "block":
            return parsed
        message = stderr[:_STDERR_MESSAGE_LIMIT] or _DEFAULT_BLOCK_MESSAGE
        logger.info("shell hook exited %d — blocking (event=%s command=%s): %s", BLOCK_EXIT_CODE, spec.event, spec.command, message)
        return {"action": "block", "message": message}
    # Other non-zero exits: still parse stdout so exit-code failures can carry a block directive.
    if r["returncode"] != 0:
        logger.warning("shell hook exited %d (event=%s command=%s); stderr=%s",
                       r["returncode"], spec.event, spec.command, stderr[:_STDERR_MESSAGE_LIMIT])
    stdout = (r["stdout"] or "").strip()
    parsed = _parse_response(spec.event, stdout)
    if parsed is None and fail_closed and stdout and not _is_json_object(stdout):
        # A fail-closed gate must not silently allow on garbage stdout (e.g. a stack trace).
        return _fail_closed_block(spec, "unparseable stdout (expected a JSON object)")
    return parsed


def _is_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


def _serialize_payload(event: str, kwargs: Dict[str, Any]) -> str:
    """Render the stdin JSON payload; unserialisable values are stringified."""
    return json.dumps({"hook_event_name": event, **_payload_fields(kwargs)}, ensure_ascii=False, default=str)


def _block_message(primary: Any, secondary: Any) -> str:
    """Validated string block message (primary wins), falling back to the default."""
    raw = primary or secondary
    return raw if isinstance(raw, str) and raw else _DEFAULT_BLOCK_MESSAGE


# pre_tool_call dialects in check order — Hermes ``action`` then Claude-Code ``decision`` — as (verb key,
# block-message primary, secondary, modify payload key); both translate to the canonical Hermes shape.
_PRE_TOOL_DIALECTS = (("action", "message", "reason", "args"), ("decision", "reason", "message", "tool_input"))


def _parse_pre_tool_call(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for verb, primary, secondary, _ in _PRE_TOOL_DIALECTS:
        if data.get(verb) == "block":
            return {"action": "block", "message": _block_message(data.get(primary), data.get(secondary))}
    for verb, _, _, payload in _PRE_TOOL_DIALECTS:
        if data.get(verb) == "modify" and isinstance(data.get(payload), dict):
            return {"action": "modify", "args": data[payload]}
    return None


def _parse_pre_verify(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # "continue" (Hermes) / "block" (Claude-Code Stop) both mean keep going; no message is a no-op.
    action = str(data.get("action") or data.get("decision") or "").strip().lower()
    message = data.get("message") or data.get("reason")
    if action in {"continue", "block"} and isinstance(message, str) and message.strip():
        return {"action": "continue", "message": message.strip()}
    return None


def _parse_context(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    context = data.get("context")
    return {"context": context} if isinstance(context, str) and context.strip() else None


_RESPONSE_PARSERS: Dict[str, Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = {"pre_tool_call": _parse_pre_tool_call, "pre_verify": _parse_pre_verify}


def _parse_response(event: str, stdout: str) -> Optional[Dict[str, Any]]:
    """Translate stdout JSON into a Hermes wire-shape dict, or ``None``."""
    stdout = (stdout or "").strip()
    if not stdout:
        return None
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        logger.warning("shell hook stdout was not valid JSON (event=%s): %s", event, stdout[:200])
        return None
    return _RESPONSE_PARSERS.get(event, _parse_context)(data) if isinstance(data, dict) else None


# --- Allowlist / consent ---

def allowlist_path() -> Path:
    """Path to the per-user shell-hook allowlist file."""
    return get_hermes_home() / ALLOWLIST_FILENAME


def load_allowlist() -> Dict[str, Any]:
    """Return the parsed allowlist, or an empty skeleton if absent."""
    try:
        raw = json.loads(allowlist_path().read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        raw = None
    if not isinstance(raw, dict):
        return {"approvals": []}
    if not isinstance(raw.get("approvals"), list):
        raw["approvals"] = []
    return raw


def save_allowlist(data: Dict[str, Any]) -> None:
    """Atomic write; on OSError log and keep the in-process approval."""
    p = allowlist_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f"{p.name}.", suffix=".tmp", dir=str(p.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(data, indent=2, sort_keys=True))
            atomic_replace(tmp_path, p)
        except Exception:
            with suppress(OSError):
                os.unlink(tmp_path)
            raise
    except OSError as exc:
        logger.warning("Failed to persist shell hook allowlist to %s: %s. The approval is in-memory for this run, "
                       "but the next startup will re-prompt (or skip registration on non-TTY runs without "
                       "--accept-hooks / HERMES_ACCEPT_HOOKS).", p, exc)


def _is_allowlisted(event: str, command: str) -> bool:
    return allowlist_entry_for(event, command) is not None


@contextmanager
def _locked_update_approvals() -> Iterator[Dict[str, Any]]:
    """Serialise allowlist read-modify-write across processes via a sibling flock file."""
    p = allowlist_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        if fcntl is None:  # pragma: no cover — non-POSIX fallback
            stack.enter_context(_allowlist_write_lock)
        else:
            lock_fh = stack.enter_context(open(p.with_suffix(p.suffix + ".lock"), "a+", encoding="utf-8"))
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            stack.callback(_flock_unlock, lock_fh)
        data = load_allowlist()
        yield data
        save_allowlist(data)


def _flock_unlock(lock_fh: Any) -> None:
    with suppress(OSError):
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def _prompt_and_record(event: str, command: str, *, accept_hooks: bool) -> bool:
    """Approve an unseen ``(event, command)`` pair; True iff granted and recorded."""
    if accept_hooks:
        _record_approval(event, command)
        logger.info("shell hook auto-approved via --accept-hooks / env / config: %s -> %s", event, command)
        return True
    if not sys.stdin.isatty():
        return False
    print(
        f"\n⚠ Hermes is about to register a shell hook that will run a\n  command on your behalf.\n\n"
        f"    Event:   {event}\n    Command: {command}\n\n"
        f"  Commands run with your full user credentials.  Only approve\n  commands you trust."
    )
    try:
        answer = input("Allow this hook to run? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()  # keep the terminal tidy after ^C
        return False
    if answer in {"y", "yes"}:
        _record_approval(event, command)
    return answer in {"y", "yes"}


def _record_approval(event: str, command: str) -> None:
    entry = {"event": event, "command": command, "approved_at": _utc_now_iso(), "script_mtime_at_approval": script_mtime_iso(command)}
    with _locked_update_approvals() as data:
        data["approvals"] = [e for e in data.get("approvals", []) if not _entry_matches(e, event, command)] + [entry]


def revoke(command: str) -> int:
    """Remove every allowlist entry for ``command``; returns the count. Live callbacks stay registered until restart."""
    with _locked_update_approvals() as data:
        before = len(data.get("approvals", []))
        data["approvals"] = [e for e in data.get("approvals", []) if not _entry_matches(e, None, command)]
        return before - len(data["approvals"])


_SCRIPT_EXTENSIONS: Tuple[str, ...] = (".sh", ".bash", ".zsh", ".fish", ".py", ".pyw", ".rb", ".pl", ".lua", ".js", ".mjs", ".cjs", ".ts")


def _command_script_path(command: str) -> str:
    """First token with a script extension, else the first path-like token, else the first token."""
    try:
        parts = split_command_line(command) or [command]
    except ValueError:
        return command
    return (next((p for p in parts if p.lower().endswith(_SCRIPT_EXTENSIONS)), None)
            or next((p for p in parts if "/" in p or p.startswith("~")), None) or parts[0])


def _resolve_effective_accept(cfg: Dict[str, Any], accept_hooks_arg: bool) -> bool:
    """Any truthy opt-in channel wins: explicit arg, HERMES_ACCEPT_HOOKS, hooks_auto_accept."""
    if accept_hooks_arg or os.environ.get("HERMES_ACCEPT_HOOKS", "").strip().lower() in _TRUTHY:
        return True
    cfg_val = cfg.get("hooks_auto_accept", False)
    return cfg_val if isinstance(cfg_val, bool) else isinstance(cfg_val, str) and cfg_val.strip().lower() in _TRUTHY


# --- Introspection (used by `hermes hooks` CLI) ---

def allowlist_entry_for(event: str, command: str) -> Optional[Dict[str, Any]]:
    """Return the allowlist record for this pair, if any."""
    return next((e for e in load_allowlist().get("approvals", []) if _entry_matches(e, event, command)), None)


def script_mtime_iso(command: str) -> Optional[str]:
    """ISO-8601 mtime of the resolved script path, or ``None`` if missing."""
    path = _command_script_path(command)
    try:
        mtime = os.path.getmtime(os.path.expanduser(path)) if path else None
    except OSError:
        return None
    return None if mtime is None else datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def script_is_executable(command: str) -> bool:
    """Runnable as configured: a bare script needs X_OK, an interpreter-prefixed one only R_OK (as ``_spawn`` does)."""
    path = _command_script_path(command)
    expanded = os.path.expanduser(path)
    try:
        argv = split_command_line(command) if path and os.path.isfile(expanded) else None
    except ValueError:
        return False
    return argv is not None and os.access(expanded, os.X_OK if argv and argv[0] == path else os.R_OK)


def run_once(spec: ShellHookSpec, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Fire one hook with a synthetic payload (``hermes hooks test`` / doctor) through the production path."""
    result = _spawn(spec, _serialize_payload(spec.event, kwargs))
    result["parsed"] = _evaluate_result(spec, result)
    return result


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import shlex  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
