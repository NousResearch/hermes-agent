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
import threading
import time
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

# split_command_line, not shlex: shlex eats Windows path backslashes.
from hermes_cli._subprocess_compat import IS_WINDOWS, kill_process_tree, split_command_line, windows_hide_flags

try:
    import fcntl  # POSIX only; Windows falls back to best-effort without flock.
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from hermes_constants import get_hermes_home
from utils import atomic_json_write

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
_registered: set[tuple[str, str, Optional[str], str]] = set()
_registered_lock = threading.Lock()
# Non-POSIX fallback for allowlist read-modify-write; separate from _registered_lock so an allowlist
# write never waits on (or nests inside) a registry update.
_allowlist_write_lock = threading.Lock()
# sync_from_config: read + apply as one step, so an older snapshot never lands after a newer one.
_sync_lock = threading.Lock()
# Consent that lives only in this process (guarded by _sync_lock): (home, event, command) a launch's
# --accept-hooks approved but the allowlist does not hold yet, this process's own --accept-hooks, and
# the skipped hooks already logged.
_launch_consent: set[tuple[str, str, str]] = set()
_process_accept = False
_skip_logged: set[tuple[str, str, str]] = set()


def _home_key() -> str:
    return str(get_hermes_home().expanduser().resolve())


def _forget_home_registrations(registry: set[tuple], lock: threading.Lock) -> None:
    """Drop the current home's keys only (shared with outbound webhooks): profile A's reload must not drop B."""
    home_key = _home_key()
    with lock:
        registry.difference_update({k for k in registry if k[0] == home_key})


def _entry_matches(e: Any, event: Optional[str], command: str) -> bool:
    return isinstance(e, dict) and (event is None or e.get("event") == event) and e.get("command") == command


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _payload_fields(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Common stdin/POST payload fields (shared with outbound webhooks); key order is wire order."""
    try:
        cwd = str(Path.cwd())
    except OSError:
        cwd = ""
    from hermes_cli.profiles import get_active_profile_name
    return {
        "tool_name": kwargs.get("tool_name"),
        "tool_input": kwargs.get("args") if isinstance(kwargs.get("args"), dict) else None,
        "session_id": kwargs.get("session_id") or kwargs.get("parent_session_id") or "",
        "cwd": cwd,
        # Resolved at fire time: a multiplexed gateway's hook script must know which profile fired it.
        "profile": get_active_profile_name(),
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

def register_from_config(cfg: Optional[dict[str, Any]], *, accept_hooks: bool = False,
                         interactive: bool = False) -> list[ShellHookSpec]:
    """Register every configured shell hook (idempotent); returns the newly wired specs. Skipped
    entries (unknown, malformed, not allowlisted, already registered) are logged only.

    ``interactive=True`` lets a hook that is not yet allowlisted ask at the terminal; only a
    foreground CLI command passes it. Every other caller (gateway startup and profile loads, cron
    and managed workers, tui_gateway, force-reload re-registration) never prompts: ``isatty()``
    cannot tell a hidden console or ``NUL`` from a person, and a prompt there blocks the event loop
    until the liveness watchdog kills the gateway (#120356, #127822)."""
    specs, _complete = _hook_specs(cfg)
    if not specs:
        return []
    from hermes_cli.plugins import get_plugin_manager  # lazy: avoids import cycle
    manager, home_key = get_plugin_manager(), _home_key()
    with _registered_lock:  # an already-wired hook is never re-asked for (main checked this first)
        specs = [spec for spec in specs if _key(home_key, spec) not in _registered]
    accept, allowed = _resolve_effective_accept(cfg, accept_hooks), allowlisted_pairs()
    # TTY prompt outside the lock; _wire takes the lock and re-checks.
    consented = []
    for spec in specs:
        if _consented(spec, accept, allowed, prompt=interactive):
            consented.append(spec)
        else:
            logger.warning("shell hook for %s (%s) not allowlisted in %s — skipped. Use --accept-hooks / "
                           "HERMES_ACCEPT_HOOKS=1 / hooks_auto_accept: true, or approve it at a CLI prompt "
                           "for this profile.", spec.event, spec.command, allowlist_path())
    with _registered_lock:
        return _wire(manager, home_key, consented)


def sync_from_config(load_cfg: Callable[[], dict[str, Any]]) -> None:
    """Bring this home's registered shell hooks in step with its config (``load_cfg()``) and
    consent, for a long-lived process (the gateway) that must behave as a new process would: hooks
    configured and consented since start register, hooks removed from config or revoked are
    unregistered. Consent is the allowlist, an opt-in in force (``hooks_auto_accept`` /
    ``HERMES_ACCEPT_HOOKS`` / this process's own ``--accept-hooks``), or a launch's ``--accept-hooks``
    whose allowlist write failed (``accept_for_launch``). Never prompts or writes. An empty config
    (a loader that failed open), a malformed ``hooks:`` entry or an unreadable allowlist must never
    disarm a ``fail_closed`` guard, so each adds but never removes. Config and allowlist are read
    under the same lock as the apply, so the last sync to run always applies the newest state."""
    from hermes_cli.plugins import get_plugin_manager  # lazy: avoids import cycle
    with _sync_lock:
        cfg = load_cfg()
        specs, complete = _hook_specs(cfg)
        if specs is None or not cfg:
            return
        allowlist, readable = _read_allowlist()
        allowed = _approved_pairs(allowlist["approvals"])
        home_key = _home_key()
        # A launch approval the allowlist now holds is the allowlist's to keep (or revoke).
        _launch_consent.difference_update({(home_key, *pair) for pair in allowed})
        allowed |= {(event, command) for home, event, command in _launch_consent if home == home_key}
        accept = _resolve_effective_accept(cfg, False)
        wanted = []
        for spec in specs:
            if accept or (spec.event, spec.command) in allowed:
                wanted.append(spec)
            elif (home_key, spec.event, spec.command) not in _skip_logged:  # once per process, not per open
                _skip_logged.add((home_key, spec.event, spec.command))
                logger.warning("shell hook for %s (%s) not allowlisted — skipped. Approve it at a CLI prompt, "
                               "or use --accept-hooks / HERMES_ACCEPT_HOOKS=1 / hooks_auto_accept: true.",
                               spec.event, spec.command)
        manager = get_plugin_manager()
        with _registered_lock:
            _wire(manager, home_key, wanted)
            if not (readable and complete):
                suspended = (home_key, "", "removal")
                if suspended not in _skip_logged:
                    _skip_logged.add(suspended)
                    logger.warning("shell hooks: %s, so removed or revoked hooks stay registered until it is "
                                   "fixed", "the allowlist is unreadable" if not readable
                                   else "a hooks: entry in config.yaml is malformed")
                return
            # In-memory consent ends with the hook's config entry, so a removed-and-re-added hook asks again.
            configured = {(home_key, spec.event, spec.command) for spec in specs}
            _launch_consent.difference_update({c for c in _launch_consent if c[0] == home_key and c not in configured})
            keep = {_key(home_key, spec) for spec in wanted}
            stale = {k for k in _registered if k[0] == home_key and k not in keep}
            _registered.difference_update(stale)
            for event in {k[1] for k in stale}:
                # Replaced, never edited in place: a concurrent invoke_hook keeps iterating the old list.
                manager._hooks[event] = [cb for cb in manager._hooks.get(event, [])
                                         if getattr(cb, "shell_hook_key", None) not in stale]
    for _home, event, _matcher, command in stale:
        logger.info("shell hook unregistered (removed from config or allowlist): %s -> %s", event, command)


def record_consent_from_config(cfg: Optional[dict[str, Any]], *, accept_hooks: bool = False) -> list[ShellHookSpec]:
    """First-use consent without registering anything, for a client whose turns run in a gateway that
    has no TTY: the prompt (or an opt-in) for each configured hook not yet allowlisted. The gateway
    registers from the same allowlist when the client opens its session. Returns the consented specs."""
    specs, _complete = _hook_specs(cfg)
    if not specs:
        return []
    accept, allowed = _resolve_effective_accept(cfg, accept_hooks), allowlisted_pairs()
    return [spec for spec in specs if _consented(spec, accept, allowed, prompt=True)]


def accept_for_launch(cfg: Optional[dict[str, Any]]) -> None:
    """A launch's ``--accept-hooks`` on the gateway: record every configured hook in the allowlist,
    and keep the consent in memory when that write fails, so it still registers
    the hook at the next ``sync_from_config`` (main registered it in memory). Only an unsaved approval
    is held, so a later ``hermes hooks revoke`` of a saved one takes effect."""
    specs = record_consent_from_config(cfg, accept_hooks=True)
    if not specs:
        return
    saved, home_key = allowlisted_pairs(), _home_key()
    with _sync_lock:
        _launch_consent.update((home_key, spec.event, spec.command) for spec in specs
                               if (spec.event, spec.command) not in saved)


def accept_for_process() -> None:
    """This process's own ``--accept-hooks`` (``hermes gateway run --accept-hooks``): an opt-in in
    force for its lifetime, as ``HERMES_ACCEPT_HOOKS`` would be, without exporting that variable to
    every tool subprocess."""
    global _process_accept
    _process_accept = True


def _hook_specs(cfg: Optional[dict[str, Any]]) -> tuple[Optional[list[ShellHookSpec]], bool]:
    """``(specs, complete)``: the configured hooks, or ``None`` when this process must run none
    (safe worker, safe mode, no config) — the one guard every registrar and the consent path share;
    ``complete`` is False when a configured entry was dropped as malformed."""
    from agent.safe_worker_policy import safe_worker_enabled
    from utils import env_var_enabled
    if safe_worker_enabled() or not isinstance(cfg, dict):
        return None, True
    if env_var_enabled("HERMES_SAFE_MODE"):  # hooks are user customizations too — fire zero user-configured code
        logger.info("HERMES_SAFE_MODE=1 — shell-hook registration skipped")
        return None, True
    return _parse_hooks(cfg.get("hooks"))


def _key(home_key: str, spec: ShellHookSpec) -> tuple[str, str, Optional[str], str]:
    return (home_key, spec.event, spec.matcher, spec.command)


def _wire(manager: Any, home_key: str, specs: list[ShellHookSpec]) -> list[ShellHookSpec]:
    """Register ``specs`` not yet wired for this home; caller holds ``_registered_lock``."""
    registered = []
    for spec in specs:
        key = _key(home_key, spec)
        if key in _registered:
            continue
        callback = _make_callback(spec)
        callback.shell_hook_key = key  # type: ignore[attr-defined]  # identifies it for sync_from_config
        manager._hooks.setdefault(spec.event, []).append(callback)
        _registered.add(key)
        registered.append(spec)
        logger.info("shell hook registered: %s -> %s (matcher=%s, timeout=%ds, fail_closed=%s)",
                    spec.event, spec.command, spec.matcher, spec.timeout, spec.fail_closed)
    return registered


def _approved_pairs(approvals: list[Any]) -> set[tuple[str, str]]:
    return {(e["event"], e["command"]) for e in approvals
            if isinstance(e, dict) and isinstance(e.get("event"), str) and isinstance(e.get("command"), str)}


def allowlisted_pairs() -> set[tuple[str, str]]:
    return _approved_pairs(load_allowlist()["approvals"])


def _consented(spec: ShellHookSpec, accept: bool, allowed: set[tuple[str, str]], *, prompt: bool) -> bool:
    """Allowlisted, or approved now (opt-in, or a TTY prompt when ``prompt``; recorded); ``allowed``
    caches one read."""
    pair = (spec.event, spec.command)
    if pair in allowed:
        return True
    if not (accept or prompt):
        return False
    if not _prompt_and_record(spec.event, spec.command, accept_hooks=accept):
        return False
    allowed.add(pair)
    return True


def iter_configured_hooks(cfg: Optional[dict[str, Any]]) -> list[ShellHookSpec]:
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
    global _process_accept
    with _registered_lock:
        _registered.clear()
    _launch_consent.clear()
    _skip_logged.clear()
    _process_accept = False


# --- Config parsing ---

def _parse_hooks_block(hooks_cfg: Any) -> list[ShellHookSpec]:
    """Normalise ``hooks:`` into specs; malformed entries warn-and-skip, never raise."""
    return _parse_hooks(hooks_cfg)[0]


def _parse_hooks(hooks_cfg: Any) -> tuple[list[ShellHookSpec], bool]:
    """``(specs, complete)``: ``complete`` is False when an entry was dropped as malformed (a typo
    must not read as a deliberate removal); an absent or empty ``hooks:`` is complete."""
    from hermes_cli.plugins import SHELL_UNSUPPORTED_HOOKS, VALID_HOOKS
    if hooks_cfg is None:
        return [], True
    if not isinstance(hooks_cfg, dict):
        return [], False
    specs: list[ShellHookSpec] = []
    complete = True
    for event_name, entries in hooks_cfg.items():
        if event_name in ("output_spill", "outbound"):  # reserved non-event sub-sections under `hooks:`
            continue
        if event_name in SHELL_UNSUPPORTED_HOOKS:  # _parse_response has no channel for these directives — refuse loudly
            logger.warning("hook event %r is Python-plugin-only: shell hooks cannot return its directive, "
                           "so this registration is refused rather than silently ignored", event_name)
            continue  # well-formed and deliberate: refused, but not a typo that should suspend removal
        if event_name not in VALID_HOOKS:
            suggestion = difflib.get_close_matches(str(event_name), VALID_HOOKS, n=1, cutoff=0.6)
            if suggestion:
                logger.warning("unknown hook event %r in hooks: config — did you mean %r?", event_name, suggestion[0])
            else:
                logger.warning("unknown hook event %r in hooks: config (valid: %s)", event_name, ", ".join(sorted(VALID_HOOKS)))
            complete = False
            continue
        if entries is None:
            continue
        if not isinstance(entries, list):
            logger.warning("hooks.%s must be a list of hook definitions; got %s", event_name, type(entries).__name__)
            complete = False
            continue
        parsed = [_parse_single_entry(event_name, i, raw) for i, raw in enumerate(entries)]
        complete = complete and None not in parsed
        specs.extend(spec for spec in parsed if spec is not None)
    return specs, complete


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

# Popen failure -> diagnostic; WinError 193 gets its own below, everything else is str(exc).
_POPEN_ERRORS = ((FileNotFoundError, "command not found"), (PermissionError, "command not executable"))

# A hook configured as a bare script path runs on POSIX because the kernel reads its shebang.
# CreateProcess has no such mechanism and answers WinError 193 ("%1 is not a valid Win32
# application") for a text file, so every ``command: "~/.hermes/agent-hooks/x.sh"`` example in
# the hooks docs — the canonical shape — fails on Windows while the same config works everywhere
# else. Map the suffixes that shape uses to their interpreter; unmapped suffixes keep the OS
# failure so a typo still reads as "command not found" rather than a mystery interpreter error.
_WINDOWS_SCRIPT_INTERPRETERS = {".sh": "bash", ".bash": "bash", ".py": "python"}
# WinError 193 raised for a suffix we deliberately do not map.
_NOT_DIRECTLY_EXECUTABLE = "cannot be run directly on Windows (there is no shebang support): start it with its interpreter, e.g. 'bash <path>'"


def _windows_script_argv(argv: list[str]) -> list[str]:
    """``argv`` with the interpreter prepended when element 0 is an existing script we can name an
    interpreter for; unchanged otherwise, including on POSIX, where the shebang already works."""
    suffix = os.path.splitext(argv[0])[1].lower()
    kind = _WINDOWS_SCRIPT_INTERPRETERS.get(suffix)
    if kind is None or not os.path.isfile(argv[0]):
        return argv
    if kind == "python":
        return [sys.executable, *argv]
    # Resolved inside the caller's try: no Git for Windows raises RuntimeError carrying the
    # installer's own actionable guidance, which is a better diagnostic than any we could add.
    from tools.environments.local import _find_bash
    return [_find_bash(), *argv]


def _spawn(spec: ShellHookSpec, stdin_json: str) -> dict[str, Any]:
    """The single subprocess site: run ``spec.command`` with ``stdin_json`` on stdin. Same result keys for every outcome."""
    result: dict[str, Any] = {"returncode": None, "stdout": "", "stderr": "", "timed_out": False, "elapsed_seconds": 0.0, "error": None}

    def failed(error: str) -> dict[str, Any]:
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
    popen_kwargs: dict[str, Any] = {"creationflags": windows_hide_flags()} if IS_WINDOWS else {"process_group": 0}
    # HERMES_HOME follows the routed profile (the import-time environ holds the launch profile's), and
    # under multiplexing os.environ carries the DEFAULT profile's secrets, which a secondary's hook
    # script must not inherit; single-profile runs keep the process env byte-for-byte as before.
    from agent.secret_scope import is_multiplex_active
    from tools.environments.local import build_subprocess_env
    try:
        if IS_WINDOWS:
            argv = _windows_script_argv(argv)
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding='utf-8', errors='replace', shell=False,
                                env=build_subprocess_env(scrub_secrets=is_multiplex_active()), **popen_kwargs)
    except Exception as exc:
        for cls, msg in _POPEN_ERRORS:
            if isinstance(exc, cls):
                return failed(msg)
        if getattr(exc, "winerror", None) == 193:
            # Unmapped suffix (.zsh, .fish, .rb, …) — the raw WinError text is localized, so an
            # operator on a non-English Windows could not act on it at all.
            return failed(f"{argv[0]!r} {_NOT_DIRECTLY_EXECUTABLE}")
        return failed(str(exc))
    try:
        stdout, stderr = proc.communicate(input=stdin_json, timeout=spec.timeout)
    except BaseException as exc:
        # BaseException: the hook leads its own process group, so Ctrl+C's SIGINT never reaches it — only we can.
        kill_process_tree(proc)  # the whole tree — forked helpers holding the pipes would stall the drain
        with suppress(Exception):
            proc.communicate(timeout=1)
        if not isinstance(exc, Exception):
            raise
        if not isinstance(exc, subprocess.TimeoutExpired):  # pragma: no cover — defensive
            return failed(str(exc))
        result.update(timed_out=True, elapsed_seconds=round(time.monotonic() - t0, 3))
        return result
    result.update(returncode=proc.returncode, stdout=stdout or "", stderr=stderr or "", elapsed_seconds=round(time.monotonic() - t0, 3))
    return result


def _make_callback(spec: ShellHookSpec) -> Callable[..., Optional[dict[str, Any]]]:
    """Build the closure that ``invoke_hook()`` will call per firing."""

    def _callback(**kwargs: Any) -> Optional[dict[str, Any]]:
        from agent.safe_worker_policy import safe_worker_enabled

        if safe_worker_enabled():
            return None
        if spec.event in _TOOL_EVENTS and not spec.matches_tool(kwargs.get("tool_name")):
            return None
        return _evaluate_result(spec, _spawn(spec, _serialize_payload(spec.event, kwargs)))

    _callback.__name__ = _callback.__qualname__ = f"shell_hook[{spec.event}:{spec.command}]"
    return _callback


def _fail_closed_block(spec: ShellHookSpec, reason: str) -> dict[str, Any]:
    return {"action": "block", "message": f"hook {spec.command} failed closed: {reason}"}


def _evaluate_result(
    spec: ShellHookSpec, r: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Turn a :func:`_spawn` diagnostic dict into the hook's contribution.

    Single place that encodes the failure semantics:

    * spawn error / timeout — fail open (log + ``None``) unless the spec
      is ``fail_closed`` on a blocking-capable event, in which case a
      canonical block shape is returned;
    * exit code 2 on a blocking-capable event — block, with the message
      taken from stdout block JSON, then stderr, then a default
      (Claude-Code / Cursor compatible);
    * other non-zero exits — warn, then parse stdout normally; a
      ``fail_closed`` hook blocks if no directive was produced;
    * non-JSON / unparseable stdout on a ``fail_closed`` blocking hook —
      block instead of silently contributing nothing.

    Shared by the live callback path (:func:`_make_callback`) and the CLI
    test helper (:func:`run_once`) so ``hermes hooks test`` reflects
    production behaviour exactly.
    """
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
    if parsed is None and fail_closed and r["returncode"] != 0:
        return _fail_closed_block(
            spec, f"hook exited {r['returncode']} with no directive",
        )

    if parsed is None and fail_closed and stdout and not _is_json_object(stdout):
        # The hook produced output we could not turn into a directive.
        # A fail-closed gate must not silently allow the action on
        # garbage output (e.g. a stack trace on stdout).
        return _fail_closed_block(spec, "unparseable stdout (expected a JSON object)")
    return parsed


def _is_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


def _serialize_payload(event: str, kwargs: dict[str, Any]) -> str:
    """Render the stdin JSON payload; unserialisable values are stringified."""
    return json.dumps({"hook_event_name": event, **_payload_fields(kwargs)}, ensure_ascii=False, default=str)


def _block_message(primary: Any, secondary: Any) -> str:
    """Validated string block message (primary wins), falling back to the default."""
    raw = primary or secondary
    return raw if isinstance(raw, str) and raw else _DEFAULT_BLOCK_MESSAGE


# pre_tool_call dialects in check order — Hermes ``action`` then Claude-Code ``decision`` — as (verb key,
# block-message primary, secondary, modify payload key); both translate to the canonical Hermes shape.
_PRE_TOOL_DIALECTS = (("action", "message", "reason", "args"), ("decision", "reason", "message", "tool_input"))


def _parse_pre_tool_call(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    for verb, primary, secondary, _ in _PRE_TOOL_DIALECTS:
        if data.get(verb) == "block":
            return {"action": "block", "message": _block_message(data.get(primary), data.get(secondary))}
    for verb, _, _, payload in _PRE_TOOL_DIALECTS:
        if data.get(verb) == "modify" and isinstance(data.get(payload), dict):
            return {"action": "modify", "args": data[payload]}
    # Hermes-only escalation to the human-approval gate (#92553). Claude-Code's ``decision:
    # approve`` means auto-ALLOW, so it is deliberately not mapped onto this.
    if data.get("action") == "approve":
        directive: dict[str, Any] = {"action": "approve"}
        for key in ("message", "rule_key"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                directive[key] = value.strip()
        return directive
    return None


def _parse_pre_verify(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    # "continue" (Hermes) / "block" (Claude-Code Stop) both mean keep going; no message is a no-op.
    action = str(data.get("action") or data.get("decision") or "").strip().lower()
    message = data.get("message") or data.get("reason")
    if action in {"continue", "block"} and isinstance(message, str) and message.strip():
        return {"action": "continue", "message": message.strip()}
    return None


def _parse_context(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    context = data.get("context")
    return {"context": context} if isinstance(context, str) and context.strip() else None


_RESPONSE_PARSERS: dict[str, Callable[[dict[str, Any]], Optional[dict[str, Any]]]] = {"pre_tool_call": _parse_pre_tool_call, "pre_verify": _parse_pre_verify}


def _parse_response(event: str, stdout: str) -> Optional[dict[str, Any]]:
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


def _read_allowlist() -> tuple[dict[str, Any], bool]:
    """``(allowlist, readable)``: the parsed allowlist (an empty skeleton when absent or broken) and
    whether it could be trusted; a missing file is readable (nothing approved yet), an unreadable or
    malformed one is not."""
    try:
        raw = json.loads(allowlist_path().read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {"approvals": []}, True
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {"approvals": []}, False
    if not isinstance(raw, dict):
        return {"approvals": []}, False
    readable = isinstance(raw.get("approvals"), list)
    if not readable:
        raw["approvals"] = []
    return raw, readable


def load_allowlist() -> dict[str, Any]:
    """Return the parsed allowlist, or an empty skeleton if absent."""
    return _read_allowlist()[0]


def save_allowlist(data: dict[str, Any]) -> None:
    """Atomic write; on OSError log and carry on (the approval is not saved)."""
    p = allowlist_path()
    try:
        atomic_json_write(p, data, sort_keys=True, mode=0o600)
    except OSError as exc:
        logger.warning("Failed to persist shell hook allowlist to %s: %s. The approval holds for this process "
                       "only (a gateway keeps a launch's --accept-hooks until it restarts); a gateway session "
                       "opened from a CLI prompt skips the hook, and the next run re-prompts.", p, exc)


@contextmanager
def _locked_update_approvals() -> Iterator[dict[str, Any]]:
    """Serialise allowlist read-modify-write across processes via a sibling flock file."""
    p = allowlist_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        if fcntl is None:  # pragma: no cover — non-POSIX fallback
            stack.enter_context(_allowlist_write_lock)
        else:
            lock_fh = stack.enter_context(open(p.with_suffix(p.suffix + ".lock"), "a+", encoding="utf-8"))  # windows-footgun: ok (write/append mode, not a read)
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
    """Remove every allowlist entry for ``command``; returns the count. A gateway unregisters it at its
    next local session open (``sync_from_config``) unless an opt-in approves every hook; other live
    processes keep it until restart."""
    with _locked_update_approvals() as data:
        before = len(data.get("approvals", []))
        data["approvals"] = [e for e in data.get("approvals", []) if not _entry_matches(e, None, command)]
        return before - len(data["approvals"])


_SCRIPT_EXTENSIONS: tuple[str, ...] = (".sh", ".bash", ".zsh", ".fish", ".py", ".pyw", ".rb", ".pl", ".lua", ".js", ".mjs", ".cjs", ".ts")


def _command_script_path(command: str) -> str:
    """First token with a script extension, else the first path-like token, else the first token."""
    try:
        parts = split_command_line(command) or [command]
    except ValueError:
        return command
    return (next((p for p in parts if p.lower().endswith(_SCRIPT_EXTENSIONS)), None)
            or next((p for p in parts if "/" in p or p.startswith("~")), None) or parts[0])


def _resolve_effective_accept(cfg: dict[str, Any], accept_hooks_arg: bool) -> bool:
    """Any truthy opt-in channel wins: explicit arg, this process's --accept-hooks, HERMES_ACCEPT_HOOKS,
    hooks_auto_accept."""
    from utils import env_var_enabled
    if accept_hooks_arg or _process_accept or env_var_enabled("HERMES_ACCEPT_HOOKS"):
        return True
    cfg_val = cfg.get("hooks_auto_accept", False)
    return cfg_val if isinstance(cfg_val, bool) else isinstance(cfg_val, str) and cfg_val.strip().lower() in _TRUTHY


# --- Introspection (used by `hermes hooks` CLI) ---

def allowlist_entry_for(event: str, command: str) -> Optional[dict[str, Any]]:
    """Return the allowlist record for this pair, if any."""
    return next((e for e in load_allowlist().get("approvals", []) if _entry_matches(e, event, command)), None)


def script_mtime_iso(command: str) -> Optional[str]:
    """ISO-8601 mtime of the resolved script path, or ``None`` if missing."""
    path = _command_script_path(command)
    try:
        mtime = os.path.getmtime(os.path.expanduser(path)) if path else None
    except OSError:
        return None
    return None if mtime is None else datetime.fromtimestamp(mtime, tz=UTC).isoformat().replace("+00:00", "Z")


def script_is_executable(command: str) -> bool:
    """Runnable as configured: a bare script needs X_OK, an interpreter-prefixed one only R_OK (as ``_spawn`` does)."""
    path = _command_script_path(command)
    expanded = os.path.expanduser(path)
    try:
        argv = split_command_line(command) if path and os.path.isfile(expanded) else None
    except ValueError:
        return False
    return argv is not None and os.access(expanded, os.X_OK if argv and argv[0] == path else os.R_OK)


def run_once(spec: ShellHookSpec, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Fire one hook with a synthetic payload (``hermes hooks test`` / doctor) through the production path."""
    result = _spawn(spec, _serialize_payload(spec.event, kwargs))
    result["parsed"] = _evaluate_result(spec, result)
    return result
