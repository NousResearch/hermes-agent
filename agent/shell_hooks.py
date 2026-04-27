"""
Shell-script hooks bridge.

Reads the ``hooks:`` block from ``cli-config.yaml``, prompts the user for
consent on first use of each ``(event, command)`` pair, and registers
callbacks on the existing plugin hook manager so every existing
``invoke_hook()`` site dispatches to the configured shell scripts — with
zero changes to call sites.

Design notes
------------
* Python plugins and shell hooks compose naturally: both flow through
  :func:`hermes_cli.plugins.invoke_hook` and its aggregators.  Python
  plugins are registered first (via ``discover_and_load()``) so their
  block decisions win ties over shell-hook blocks.
* Subprocess execution uses ``shlex.split(os.path.expanduser(command))``
  with ``shell=False`` — no shell injection footguns.  Users that need
  pipes/redirection wrap their logic in a script.
* First-use consent is gated by the allowlist under
  ``~/.hermes/shell-hooks-allowlist.json``.  Non-TTY callers must pass
  ``accept_hooks=True`` (resolved from ``--accept-hooks``,
  ``HERMES_ACCEPT_HOOKS``, or ``hooks_auto_accept: true`` in config)
  for registration to succeed without a prompt.
* Registration is idempotent — safe to invoke from both the CLI entry
  point (``hermes_cli/main.py``) and the gateway entry point
  (``gateway/run.py``).

Wire protocol
-------------
**stdin** (JSON, piped to the script)::

    {
        "hook_event_name": "pre_tool_call",
        "tool_name":       "terminal",
        "tool_input":      {"command": "rm -rf /"},
        "session_id":      "sess_abc123",
        "cwd":             "/home/user/project",
        "extra":           {...}   # event-specific kwargs
    }

**stdout** (JSON, optional — anything else is ignored)::

    # Block a pre_tool_call (either shape accepted; normalised internally):
    {"decision": "block", "reason":  "Forbidden command"}   # Claude-Code-style
    {"action":   "block", "message": "Forbidden command"}   # Hermes-canonical

    # Inject context for pre_llm_call:
    {"context": "Today is Friday"}

    # Silent no-op:
    <empty or any non-matching JSON object>
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

try:
    import fcntl  # POSIX only; Windows falls back to best-effort without flock.
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 300
ALLOWLIST_FILENAME = "shell-hooks-allowlist.json"

# (event, matcher, command) triples that have been wired to the plugin
# manager in the current process.  Matcher is part of the key because
# the same script can legitimately register for different matchers under
# the same event (e.g. one entry per tool the user wants to gate).
# Second registration attempts for the exact same triple become no-ops
# so the CLI and gateway can both call register_from_config() safely.
_registered: Set[Tuple[str, Optional[str], str]] = set()
_registered_lock = threading.Lock()

# Intra-process lock for allowlist read-modify-write on platforms that
# lack ``fcntl`` (non-POSIX).  Kept separate from ``_registered_lock``
# because ``register_from_config`` already holds ``_registered_lock`` when
# it triggers ``_record_approval`` — reusing it here would self-deadlock
# (``threading.Lock`` is non-reentrant).  POSIX callers use the sibling
# ``.lock`` file via ``fcntl.flock`` and bypass this.
_allowlist_write_lock = threading.Lock()


@dataclass
class ShellHookSpec:
    """Parsed and validated representation of a single ``hooks:`` entry."""

    event: str
    command: str
    matcher: Optional[str] = None
    timeout: int = DEFAULT_TIMEOUT_SECONDS
    compiled_matcher: Optional[re.Pattern] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Strip whitespace introduced by YAML quirks (e.g. multi-line string
        # folding) — a matcher of " terminal" would otherwise silently fail
        # to match "terminal" without any diagnostic.
        if isinstance(self.matcher, str):
            stripped = self.matcher.strip()
            self.matcher = stripped if stripped else None
        if self.matcher:
            try:
                self.compiled_matcher = re.compile(self.matcher)
            except re.error as exc:
                logger.warning(
                    "shell hook matcher %r is invalid (%s) — treating as "
                    "literal equality", self.matcher, exc,
                )
                self.compiled_matcher = None

    def matches_tool(self, tool_name: Optional[str]) -> bool:
        if not self.matcher:
            return True
        if tool_name is None:
            return False
        if self.compiled_matcher is not None:
            return self.compiled_matcher.fullmatch(tool_name) is not None
        # compiled_matcher is None only when the regex failed to compile,
        # in which case we already warned and fall back to literal equality.
        return tool_name == self.matcher


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_from_config(
    cfg: Optional[Dict[str, Any]],
    *,
    accept_hooks: bool = False,
) -> List[ShellHookSpec]:
    """Register every configured shell hook on the plugin manager.

    ``cfg`` is the full parsed config dict (``hermes_cli.config.load_config``
    output).  The ``hooks:`` key is read out of it.  Missing, empty, or
    non-dict ``hooks`` is treated as zero configured hooks.

    ``accept_hooks=True`` skips the TTY consent prompt — the caller is
    promising that the user has opted in via a flag, env var, or config
    setting.  ``HERMES_ACCEPT_HOOKS=1`` and ``hooks_auto_accept: true`` are
    also honored inside this function so either CLI or gateway call sites
    pick them up.

    Returns the list of :class:`ShellHookSpec` entries that ended up wired
    up on the plugin manager.  Skipped entries (unknown events, malformed,
    not allowlisted, already registered) are logged but not returned.
    """
    if not isinstance(cfg, dict):
        return []

    effective_accept = _resolve_effective_accept(cfg, accept_hooks)

    specs = _parse_hooks_block(cfg.get("hooks"))
    if not specs:
        return []

    registered: List[ShellHookSpec] = []

    # Import lazily — avoids circular imports at module-load time.
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()

    # Idempotence + allowlist read happen under the lock; the TTY
    # prompt runs outside so other threads aren't parked on a blocking
    # input().  Mutation re-takes the lock with a defensive idempotence
    # re-check in case two callers ever race through the prompt.
    for spec in specs:
        key = (spec.event, spec.matcher, spec.command)
        with _registered_lock:
            if key in _registered:
                continue
            already_allowlisted = _is_allowlisted(spec.event, spec.command)

        if not already_allowlisted:
            if not _prompt_and_record(
                spec.event, spec.command, accept_hooks=effective_accept,
            ):
                logger.warning(
                    "shell hook for %s (%s) not allowlisted — skipped. "
                    "Use --accept-hooks / HERMES_ACCEPT_HOOKS=1 / "
                    "hooks_auto_accept: true, or approve at the TTY "
                    "prompt next run.",
                    spec.event, spec.command,
                )
                continue

        with _registered_lock:
            if key in _registered:
                continue
            manager._hooks.setdefault(spec.event, []).append(_make_callback(spec))
            _registered.add(key)
            registered.append(spec)
            logger.info(
                "shell hook registered: %s -> %s (matcher=%s, timeout=%ds)",
                spec.event, spec.command, spec.matcher, spec.timeout,
            )

    return registered


def iter_configured_hooks(cfg: Optional[Dict[str, Any]]) -> List[ShellHookSpec]:
    """Return the parsed ``ShellHookSpec`` entries from config without
    registering anything.  Used by ``hermes hooks list`` and ``doctor``."""
    if not isinstance(cfg, dict):
        return []
    return _parse_hooks_block(cfg.get("hooks"))


def reset_for_tests() -> None:
    """Clear the idempotence set.  Test-only helper."""
    with _registered_lock:
        _registered.clear()


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def _parse_hooks_block(hooks_cfg: Any) -> List[ShellHookSpec]:
    """Normalise the ``hooks:`` dict into a flat list of ``ShellHookSpec``.

    Malformed entries warn-and-skip — we never raise from config parsing
    because a broken hook must not crash the agent.
    """
    from hermes_cli.plugins import VALID_HOOKS

    if not isinstance(hooks_cfg, dict):
        return []

    specs: List[ShellHookSpec] = []

    for event_name, entries in hooks_cfg.items():
        if event_name not in VALID_HOOKS:
            suggestion = difflib.get_close_matches(
                str(event_name), VALID_HOOKS, n=1, cutoff=0.6,
            )
            if suggestion:
                logger.warning(
                    "unknown hook event %r in hooks: config — did you mean %r?",
                    event_name, suggestion[0],
                )
            else:
                logger.warning(
                    "unknown hook event %r in hooks: config (valid: %s)",
                    event_name, ", ".join(sorted(VALID_HOOKS)),
                )
            continue

        if entries is None:
            continue

        if not isinstance(entries, list):
            logger.warning(
                "hooks.%s must be a list of hook definitions; got %s",
                event_name, type(entries).__name__,
            )
            continue

        for i, raw in enumerate(entries):
            spec = _parse_single_entry(event_name, i, raw)
            if spec is not None:
                specs.append(spec)

    return specs


def _parse_single_entry(
    event: str, index: int, raw: Any,
) -> Optional[ShellHookSpec]:
    if not isinstance(raw, dict):
        logger.warning(
            "hooks.%s[%d] must be a mapping with a 'command' key; got %s",
            event, index, type(raw).__name__,
        )
        return None

    command = raw.get("command")
    if not isinstance(command, str) or not command.strip():
        logger.warning(
            "hooks.%s[%d] is missing a non-empty 'command' field",
            event, index,
        )
        return None

    matcher = raw.get("matcher")
    if matcher is not None and not isinstance(matcher, str):
        logger.warning(
            "hooks.%s[%d].matcher must be a string regex; ignoring",
            event, index,
        )
        matcher = None

    if matcher is not None and event not in ("pre_tool_call", "post_tool_call"):
        logger.warning(
            "hooks.%s[%d].matcher=%r will be ignored at runtime — the "
            "matcher field is only honored for pre_tool_call / "
            "post_tool_call.  The hook will fire on every %s event.",
            event, index, matcher, event,
        )
        matcher = None

    timeout_raw = raw.get("timeout", DEFAULT_TIMEOUT_SECONDS)
    try:
        timeout = int(timeout_raw)
    except (TypeError, ValueError):
        logger.warning(
            "hooks.%s[%d].timeout must be an int (got %r); using default %ds",
            event, index, timeout_raw, DEFAULT_TIMEOUT_SECONDS,
        )
        timeout = DEFAULT_TIMEOUT_SECONDS

    if timeout < 1:
        logger.warning(
            "hooks.%s[%d].timeout must be >=1; using default %ds",
            event, index, DEFAULT_TIMEOUT_SECONDS,
        )
        timeout = DEFAULT_TIMEOUT_SECONDS

    if timeout > MAX_TIMEOUT_SECONDS:
        logger.warning(
            "hooks.%s[%d].timeout=%ds exceeds max %ds; clamping",
            event, index, timeout, MAX_TIMEOUT_SECONDS,
        )
        timeout = MAX_TIMEOUT_SECONDS

    return ShellHookSpec(
        event=event,
        command=command.strip(),
        matcher=matcher,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Subprocess callback
# ---------------------------------------------------------------------------

_TOP_LEVEL_PAYLOAD_KEYS = {"tool_name", "args", "session_id", "parent_session_id"}


def _spawn(spec: ShellHookSpec, stdin_json: str) -> Dict[str, Any]:
    """Run ``spec.command`` as a subprocess with ``stdin_json`` on stdin.

    Returns a diagnostic dict with the same keys for every outcome
    (``returncode``, ``stdout``, ``stderr``, ``timed_out``,
    ``elapsed_seconds``, ``error``).  This is the single place the
    subprocess is actually invoked — both the live callback path
    (:func:`_make_callback`) and the CLI test helper (:func:`run_once`)
    go through it.
    """
    result: Dict[str, Any] = {
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
        "elapsed_seconds": 0.0,
        "error": None,
    }
    try:
        argv = shlex.split(os.path.expanduser(spec.command))
    except ValueError as exc:
        result["error"] = f"command {spec.command!r} cannot be parsed: {exc}"
        return result
    if not argv:
        result["error"] = "empty command"
        return result

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            input=stdin_json,
            capture_output=True,
            timeout=spec.timeout,
            text=True,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        result["timed_out"] = True
        result["elapsed_seconds"] = round(time.monotonic() - t0, 3)
        return result
    except FileNotFoundError:
        result["error"] = "command not found"
        return result
    except PermissionError:
        result["error"] = "command not executable"
        return result
    except Exception as exc:  # pragma: no cover — defensive
        result["error"] = str(exc)
        return result

    result["returncode"] = proc.returncode
    result["stdout"] = proc.stdout or ""
    result["stderr"] = proc.stderr or ""
    result["elapsed_seconds"] = round(time.monotonic() - t0, 3)
    return result


def _make_callback(spec: ShellHookSpec) -> Callable[..., Optional[Dict[str, Any]]]:
    """Build the closure that ``invoke_hook()`` will call per firing."""

    def _callback(**kwargs: Any) -> Optional[Dict[str, Any]]:
        # Matcher gate — only meaningful for tool-scoped events.
        if spec.event in ("pre_tool_call", "post_tool_call"):
            if not spec.matches_tool(kwargs.get("tool_name")):
                return None

        r = _spawn(spec, _serialize_payload(spec.event, kwargs))

        if r["error"]:
            logger.warning(
                "shell hook failed (event=%s command=%s): %s",
                spec.event, spec.command, r["error"],
            )
            return None
        if r["timed_out"]:
            logger.warning(
                "shell hook timed out after %.2fs (event=%s command=%s)",
                r["elapsed_seconds"], spec.event, spec.command,
            )
            return None

        stderr = r["stderr"].strip()
        if stderr:
            logger.debug(
                "shell hook stderr (event=%s command=%s): %s",
                spec.event, spec.command, stderr[:400],
            )
        # Non-zero exits: log but still parse stdout so scripts that
        # signal failure via exit code can also return a block directive.
        if r["returncode"] != 0:
            logger.warning(
                "shell hook exited %d (event=%s command=%s); stderr=%s",
                r["returncode"], spec.event, spec.command, stderr[:400],
            )
        return _parse_response(spec.event, r["stdout"])

    _callback.__name__ = f"shell_hook[{spec.event}:{spec.command}]"
    _callback.__qualname__ = _callback.__name__
    return _callback


def _serialize_payload(event: str, kwargs: Dict[str, Any]) -> str:
    """Render the stdin JSON payload.  Unserialisable values are
    stringified via ``default=str`` rather than dropped."""
    extras = {k: v for k, v in kwargs.items() if k not in _TOP_LEVEL_PAYLOAD_KEYS}
    try:
        cwd = str(Path.cwd())
    except OSError:
        cwd = ""
    payload = {
        "hook_event_name": event,
        "tool_name": kwargs.get("tool_name"),
        "tool_input": kwargs.get("args") if isinstance(kwargs.get("args"), dict) else None,
        "session_id": kwargs.get("session_id") or kwargs.get("parent_session_id") or "",
        "cwd": cwd,
        "extra": extras,
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def _parse_response(event: str, stdout: str) -> Optional[Dict[str, Any]]:
    """Translate stdout JSON into a Hermes wire-shape dict.

    For ``pre_tool_call`` the Claude-Code-style ``{"decision": "block",
    "reason": "..."}`` payload is translated into the canonical Hermes
    ``{"action": "block", "message": "..."}`` shape expected by
    :func:`hermes_cli.plugins.get_pre_tool_call_block_message`.  This is
    the single most important correctness invariant in this module —
    skipping the translation silently breaks every ``pre_tool_call``
    block directive.

    For ``pre_llm_call``, ``{"context": "..."}`` is passed through
    unchanged to match the existing plugin-hook contract.

    Anything else returns ``None``.
    """
    stdout = (stdout or "").strip()
    if not stdout:
        return None

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        logger.warning(
            "shell hook stdout was not valid JSON (event=%s): %s",
            event, stdout[:200],
        )
        return None

    if not isinstance(data, dict):
        return None

    if event == "pre_tool_call":
        if data.get("action") == "block":
            message = data.get("message") or data.get("reason") or ""
            if isinstance(message, str) and message:
                return {"action": "block", "message": message}
        if data.get("decision") == "block":
            message = data.get("reason") or data.get("message") or ""
            if isinstance(message, str) and message:
                return {"action": "block", "message": message}
        return None

    context = data.get("context")
    if isinstance(context, str) and context.strip():
        return {"context": context}

    return None


# ---------------------------------------------------------------------------
# Allowlist / consent
# ---------------------------------------------------------------------------

def allowlist_path() -> Path:
    """Path to the per-user shell-hook allowlist file."""
    return get_hermes_home() / ALLOWLIST_FILENAME


def load_allowlist() -> Dict[str, Any]:
    """Return the parsed allowlist, or an empty skeleton if absent."""
    try:
        raw = json.loads(allowlist_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"approvals": []}
    if not isinstance(raw, dict):
        return {"approvals": []}
    approvals = raw.get("approvals")
    if not isinstance(approvals, list):
        raw["approvals"] = []
    return raw


def save_allowlist(data: Dict[str, Any]) -> None:
    """Atomically persist the allowlist via per-process ``mkstemp`` +
    ``os.replace``.  Cross-process read-modify-write races are handled
    by :func:`_locked_update_approvals` (``fcntl.flock``).  On OSError
    the failure is logged; the in-process hook still registers but
    the approval won't survive across runs."""
    p = allowlist_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f"{p.name}.", suffix=".tmp", dir=str(p.parent),
        )
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(json.dumps(data, indent=2, sort_keys=True))
            os.replace(tmp_path, p)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except OSError as exc:
        logger.warning(
            "Failed to persist shell hook allowlist to %s: %s. "
            "The approval is in-memory for this run, but the next "
            "startup will re-prompt (or skip registration on non-TTY "
            "runs without --accept-hooks / HERMES_ACCEPT_HOOKS).",
            p, exc,
        )


def _is_allowlisted(event: str, command: str) -> bool:
    data = load_allowlist()
    return any(
        isinstance(e, dict)
        and e.get("event") == event
        and e.get("command") == command
        for e in data.get("approvals", [])
    )


@contextmanager
def _locked_update_approvals() -> Iterator[Dict[str, Any]]:
    """Serialise read-modify-write on the allowlist across processes.

    Holds an exclusive ``flock`` on a sibling lock file for the duration
    of the update so concurrent ``_record_approval``/``revoke`` callers
    cannot clobber each other's changes (the race Codex reproduced with
    20–50 simultaneous writers).  Falls back to an in-process lock on
    platforms without ``fcntl``.
    """
    p = allowlist_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    lock_path = p.with_suffix(p.suffix + ".lock")

    if fcntl is None:  # pragma: no cover — non-POSIX fallback
        with _allowlist_write_lock:
            data = load_allowlist()
            yield data
            save_allowlist(data)
        return

    with open(lock_path, "a+") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            data = load_allowlist()
            yield data
            save_allowlist(data)
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def _prompt_and_record(
    event: str, command: str, *, accept_hooks: bool,
) -> bool:
    """Decide whether to approve an unseen ``(event, command)`` pair.
    Returns ``True`` iff the approval was granted and recorded.
    """
    if accept_hooks:
        _record_approval(event, command)
        logger.info(
            "shell hook auto-approved via --accept-hooks / env / config: "
            "%s -> %s", event, command,
        )
        return True

    if not sys.stdin.isatty():
        return False

    print(
        f"\n⚠ Hermes is about to register a shell hook that will run a\n"
        f"  command on your behalf.\n\n"
        f"    Event:   {event}\n"
        f"    Command: {command}\n\n"
        f"  Commands run with your full user credentials.  Only approve\n"
        f"  commands you trust."
    )
    try:
        answer = input("Allow this hook to run? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()  # keep the terminal tidy after ^C
        return False

    if answer in ("y", "yes"):
        _record_approval(event, command)
        return True

    return False


def _record_approval(event: str, command: str) -> None:
    entry = {
        "event": event,
        "command": command,
        "approved_at": _utc_now_iso(),
        "script_mtime_at_approval": script_mtime_iso(command),
    }
    with _locked_update_approvals() as data:
        data["approvals"] = [
            e for e in data.get("approvals", [])
            if not (
                isinstance(e, dict)
                and e.get("event") == event
                and e.get("command") == command
            )
        ] + [entry]


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def revoke(command: str) -> int:
    """Remove every allowlist entry matching ``command``.

    Returns the number of entries removed.  Does not unregister any
    callbacks that are already live on the plugin manager in the current
    process — restart the CLI / gateway to drop them.
    """
    with _locked_update_approvals() as data:
        before = len(data.get("approvals", []))
        data["approvals"] = [
            e for e in data.get("approvals", [])
            if not (isinstance(e, dict) and e.get("command") == command)
        ]
        after = len(data["approvals"])
    return before - after


_SCRIPT_EXTENSIONS: Tuple[str, ...] = (
    ".sh", ".bash", ".zsh", ".fish",
    ".py", ".pyw",
    ".rb", ".pl", ".lua",
    ".js", ".mjs", ".cjs", ".ts",
)


def _command_script_path(command: str) -> str:
    """Return the script path from ``command`` for doctor / drift checks.

    Prefers a token ending in a known script extension, then a token
    containing ``/`` or leading ``~``, then the first token.  Handles
    ``python3 /path/hook.py``, ``/usr/bin/env bash hook.sh``, and the
    common bare-path form.
    """
    try:
        parts = shlex.split(command)
    except ValueError:
        return command
    if not parts:
        return command
    for part in parts:
        if part.lower().endswith(_SCRIPT_EXTENSIONS):
            return part
    for part in parts:
        if "/" in part or part.startswith("~"):
            return part
    return parts[0]


# ---------------------------------------------------------------------------
# Helpers for accept-hooks resolution
# ---------------------------------------------------------------------------

def _resolve_effective_accept(
    cfg: Dict[str, Any], accept_hooks_arg: bool,
) -> bool:
    """Combine all three opt-in channels into a single boolean.

    Precedence (any truthy source flips us on):
      1. ``--accept-hooks`` flag (CLI) / explicit argument
      2. ``HERMES_ACCEPT_HOOKS`` env var
      3. ``hooks_auto_accept: true`` in ``cli-config.yaml``
    """
    if accept_hooks_arg:
        return True
    env = os.environ.get("HERMES_ACCEPT_HOOKS", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    cfg_val = cfg.get("hooks_auto_accept", False)
    if isinstance(cfg_val, bool):
        return cfg_val
    if isinstance(cfg_val, str):
        return cfg_val.strip().lower() in ("1", "true", "yes", "on")
    return False


# ---------------------------------------------------------------------------
# Introspection (used by `hermes hooks` CLI)
# ---------------------------------------------------------------------------

def allowlist_entry_for(event: str, command: str) -> Optional[Dict[str, Any]]:
    """Return the allowlist record for this pair, if any."""
    for e in load_allowlist().get("approvals", []):
        if (
            isinstance(e, dict)
            and e.get("event") == event
            and e.get("command") == command
        ):
            return e
    return None


def script_mtime_iso(command: str) -> Optional[str]:
    """ISO-8601 mtime of the resolved script path, or ``None`` if the
    script is missing."""
    path = _command_script_path(command)
    if not path:
        return None
    try:
        expanded = os.path.expanduser(path)
        return datetime.fromtimestamp(
            os.path.getmtime(expanded), tz=timezone.utc,
        ).isoformat().replace("+00:00", "Z")
    except OSError:
        return None


def script_is_executable(command: str) -> bool:
    """Return ``True`` iff ``command`` is runnable as configured.

    For a bare invocation (``/path/hook.sh``) the script itself must be
    executable.  For interpreter-prefixed commands (``python3
    /path/hook.py``, ``/usr/bin/env bash hook.sh``) the script just has
    to be readable — the interpreter doesn't care about the ``X_OK``
    bit.  Mirrors what ``_spawn`` would actually do at runtime."""
    path = _command_script_path(command)
    if not path:
        return False
    expanded = os.path.expanduser(path)
    if not os.path.isfile(expanded):
        return False
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    is_bare_invocation = bool(argv) and argv[0] == path
    required = os.X_OK if is_bare_invocation else os.R_OK
    return os.access(expanded, required)


def run_once(
    spec: ShellHookSpec, kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Fire a single shell-hook invocation with a synthetic payload.
    Used by ``hermes hooks test`` and ``hermes hooks doctor``.

    ``kwargs`` is the same dict that :func:`hermes_cli.plugins.invoke_hook`
    would pass at runtime.  It is routed through :func:`_serialize_payload`
    so the synthetic stdin exactly matches what a real hook firing would
    produce — otherwise scripts tested via ``hermes hooks test`` could
    diverge silently from production behaviour.

    Returns the :func:`_spawn` diagnostic dict plus a ``parsed`` field
    holding the canonical Hermes-wire-shape response."""
    stdin_json = _serialize_payload(spec.event, kwargs)
    result = _spawn(spec, stdin_json)
    result["parsed"] = _parse_response(spec.event, result["stdout"])
    return result
