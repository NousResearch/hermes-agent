"""The ``hooks:`` config entry: ``ShellHookSpec`` and the parser that turns a raw block into specs
(malformed entries warn and are skipped, never raise)."""

from __future__ import annotations

import difflib
import enum
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 60

MAX_TIMEOUT_SECONDS = 300
# async_rewake hooks wait for something to happen (a message, a CI run), so their bounds are their own.
DEFAULT_REWAKE_TIMEOUT_SECONDS = 3600
MAX_REWAKE_TIMEOUT_SECONDS = 86400
DEFAULT_MAX_CONSECUTIVE_WAKES = 20
# Events that fire inside a turn of the session they name, where its route is bound: the only ones an
# async_rewake hook can wake from (agent/shell_hooks/rewake.py).
REWAKE_EVENTS = frozenset({"on_session_start", "post_tool_call", "post_llm_call", "on_session_end", "subagent_stop"})

# Events whose block directive is honored downstream; exit-2 blocking and fail_closed only apply here.
_BLOCKING_EVENTS = frozenset({"pre_tool_call"})

_TOOL_EVENTS = frozenset({"pre_tool_call", "post_tool_call"})


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


class EntryRefused(enum.Enum):
    """A well-formed ``hooks:`` entry refused on purpose: unlike a malformed one (a likely typo), it
    leaves removal of the other hooks armed."""

    REFUSED = "refused"


@dataclass
class ShellHookSpec(_ToolMatcherMixin):
    """Parsed and validated representation of a single ``hooks:`` entry."""

    event: str
    command: str
    matcher: Optional[str] = None
    timeout: int = DEFAULT_TIMEOUT_SECONDS
    fail_closed: bool = False
    async_rewake: bool = False
    max_consecutive_wakes: int = DEFAULT_MAX_CONSECUTIVE_WAKES
    compiled_matcher: Optional[re.Pattern] = field(default=None, repr=False)


def iter_configured_hooks(cfg: Optional[dict[str, Any]]) -> list[ShellHookSpec]:
    """Parse config hooks without registering (``hermes hooks list`` / doctor)."""
    return _parse_hooks_block(cfg.get("hooks")) if isinstance(cfg, dict) else []


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
        specs.extend(spec for spec in parsed if isinstance(spec, ShellHookSpec))
    return specs, complete


def _parse_single_entry(event: str, index: int, raw: Any) -> Optional[ShellHookSpec | EntryRefused]:
    """The entry's spec; None when malformed (a typo: removal stays suspended), ``EntryRefused``
    when well-formed but refused on purpose (nothing to suspend)."""
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
    rewake = _parse_rewake(event, raw, warn)
    if rewake is None:
        return EntryRefused.REFUSED
    default_timeout, max_timeout = ((DEFAULT_REWAKE_TIMEOUT_SECONDS, MAX_REWAKE_TIMEOUT_SECONDS) if rewake[0]
                                    else (DEFAULT_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS))
    try:
        timeout = int(raw.get("timeout", default_timeout))
    except (TypeError, ValueError):
        warn(".timeout must be an int (got %r); using default %ds", raw.get("timeout"), default_timeout)
        timeout = default_timeout
    if timeout < 1:
        warn(".timeout must be >=1; using default %ds", default_timeout)
        timeout = default_timeout
    elif timeout > max_timeout:
        warn(".timeout=%ds exceeds max %ds; clamping", timeout, max_timeout)
        timeout = max_timeout
    # ``fail_closed`` (canonical) wins over ``failClosed`` (Cursor/Claude-Code compat).
    fail_closed = raw.get("fail_closed", raw.get("failClosed", False))
    if not isinstance(fail_closed, bool):
        warn(".fail_closed must be a boolean (got %r); using default false (fail open)", fail_closed)
        fail_closed = False
    if fail_closed and event not in _BLOCKING_EVENTS:
        warn(".fail_closed=true will be ignored at runtime — fail_closed only applies to blocking-capable "
             "events (%s).  The hook will fail open on %s like any other hook.", ", ".join(sorted(_BLOCKING_EVENTS)), event)
        fail_closed = False
    if fail_closed and rewake[0]:
        warn(".fail_closed=true is ignored with async_rewake: a background hook cannot block what fired it")
        fail_closed = False
    return ShellHookSpec(event=event, command=command.strip(), matcher=matcher, timeout=timeout,
                         fail_closed=fail_closed, async_rewake=rewake[0], max_consecutive_wakes=rewake[1])


def _parse_rewake(event: str, raw: dict, warn: Callable[..., None]) -> Optional[tuple[bool, int]]:
    """``(async_rewake, max_consecutive_wakes)`` of one entry, or None when it is refused. ``async_rewake``
    (canonical) wins over ``asyncRewake`` (Claude Code compat), as for ``fail_closed``."""
    async_rewake = raw.get("async_rewake", raw.get("asyncRewake", False))
    if not isinstance(async_rewake, bool):
        warn(".async_rewake must be a boolean (got %r); running the hook synchronously", async_rewake)
        async_rewake = False
    if async_rewake and event not in REWAKE_EVENTS:
        warn(" is refused: async_rewake only works on events fired inside a turn of the session it wakes (%s)",
             ", ".join(sorted(REWAKE_EVENTS)))
        return None
    limit = raw.get("max_consecutive_wakes", DEFAULT_MAX_CONSECUTIVE_WAKES)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        warn(".max_consecutive_wakes must be an int >= 0 (got %r); using %d", limit, DEFAULT_MAX_CONSECUTIVE_WAKES)
        limit = DEFAULT_MAX_CONSECUTIVE_WAKES
    return async_rewake, limit
