"""Hermes Plugin System — discovers, loads, and manages plugins.

Sources, later overriding earlier on key collision: bundled ``<repo>/plugins/<name>/`` (``memory/``
and ``context_engine/`` have their own discovery), user ``~/.hermes/plugins/<name>/``, project
``./.hermes/plugins/<name>/`` (opt-in via ``HERMES_ENABLE_PROJECT_PLUGINS``), and pip packages in
the ``hermes_agent.plugins`` entry-point group. A directory plugin needs a ``plugin.yaml`` manifest
and an ``__init__.py`` exposing ``register(ctx)``. Plugins register callbacks for ``VALID_HOOKS``
(core fires ``invoke_hook(name, **kwargs)``) and tools via ``PluginContext.register_tool()``.
"""

from __future__ import annotations

import asyncio
import contextvars
import importlib.metadata
import inspect
import json
import logging
import os
import queue
import re
import sys
import threading
import types
from contextlib import suppress
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union

from hermes_constants import get_hermes_home, hermes_home_key
from registration_lifecycle import replacement_coordinator
from utils import env_var_enabled
from hermes_cli.config import load_config_readonly
from hermes_cli.middleware import VALID_MIDDLEWARE
from hermes_cli.plugin_capabilities import plugin_capability_granted
from hermes_cli.relay_plugin_cutover import RELAY_PLUGINS_CONFIG_ENV, legacy_relay_plugin_keys
# Sibling modules' names are re-exported here (origin) so plugins and tests keep one import path.
from hermes_cli.plugins_manifest import (  # noqa: F401 — re-exported
    _CONFIG_SCHEMA_TYPES, SUPPORTED_MANIFEST_VERSION, PluginManifest, _portable_skill_namespace,
    manifest_key, parse_manifest_file, resolve_module_origin, resolve_plugin_load_order,
    validate_config_schema,
)
from hermes_cli.plugins_discovery import (  # noqa: F401 — re-exported
    ENTRY_POINTS_GROUP, _get_disabled_plugins, _get_enabled_plugins, collect_directory_manifests,
    discover_entrypoint_manifests, gate_manifest, scan_directory,
)
from hermes_cli.plugins_loader import (
    PluginLoaderMixin, _BARE_MODULE_SCOPE, _MODULE_NAMESPACE_LOCK, _NS_PARENT, _evict_modules,
    _plugin_home_scope, _serialized_replacement,
)
from hermes_cli.plugins_dispatch import (  # noqa: F401 — re-exported
    DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS, HERMES_EVENT_NAMESPACE, MAX_SYSTEM_PROMPT_SECTION_CHARS,
    MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS, PLUGIN_SECTIONS_END, PLUGIN_SECTIONS_START,
    SYSTEM_PROMPT_SECTION_POSITIONS, _EVENT_EMIT_DEPTH_CAP, _EVENT_PENDING_CAP,
    _HOOK_CALLBACK_TIMEOUT_SECS, _HOOK_TIMEOUT_SUPPRESSION_SECONDS, _MAX_HOOK_CALLBACK_TIMEOUT_SECS,
    _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE, PluginDispatchMixin, PluginSystemPromptSection,
    RenderedPluginSystemPromptSection, _EventSubscription, format_system_prompt_sections,
    is_valid_system_prompt_section_id,
)
from hermes_cli.plugins_ledger import PluginLedgerMixin, PluginRegistration
from hermes_cli.plugins_state import (
    PluginState, _locked_plugin_state, _nested_plugin_mapping, _nested_plugin_value,
    _plugin_relative_segments, _plugin_settings_entry,
)
from hermes_cli.relay_plugin_cutover import (
    LEGACY_RELAY_PLUGIN_KEYS,
    RELAY_PLUGINS_CONFIG_ENV,
    legacy_relay_plugin_keys,
)


def get_bundled_plugins_dir() -> Path:
    """Bundled ``plugins/`` dir: ``HERMES_BUNDLED_PLUGINS`` (Nix wrapper / packaged installs, read-only
    store paths) first, else the in-repo path."""
    env_override = os.getenv("HERMES_BUNDLED_PLUGINS")
    if env_override:
        return Path(env_override)
    return Path(__file__).resolve().parent.parent / "plugins"


class PluginToolOverrideError(PermissionError):
    """Plugin tried to override a built-in tool without ``plugins.entries.<id>.allow_tool_override``."""


logger = logging.getLogger(__name__)

# ``HERMES_PLUGINS_DEBUG=1`` tees verbose discovery logs to stderr in addition to agent.log. Read
# once at import; tests flip it mid-process via ``_install_plugin_debug_handler(force=True)``.
_PLUGINS_DEBUG = env_var_enabled("HERMES_PLUGINS_DEBUG")
_DEBUG_HANDLER_INSTALLED = False


def _install_plugin_debug_handler(force: bool = False) -> None:
    """When HERMES_PLUGINS_DEBUG is on, tee plugin logs to stderr at DEBUG (once per process)."""
    global _DEBUG_HANDLER_INSTALLED, _PLUGINS_DEBUG
    if force:
        _PLUGINS_DEBUG = env_var_enabled("HERMES_PLUGINS_DEBUG")
    if not _PLUGINS_DEBUG or _DEBUG_HANDLER_INSTALLED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("[plugins] %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    _DEBUG_HANDLER_INSTALLED = True
    logger.debug("HERMES_PLUGINS_DEBUG=1 — verbose plugin discovery logging enabled")


_install_plugin_debug_handler()

VALID_HOOKS: Set[str] = {
    "pre_tool_call", "post_tool_call", "transform_terminal_output", "transform_tool_result",
    # transform_llm_output: return a replacement string (first non-None wins) or None.
    "transform_llm_output", "pre_llm_call", "post_llm_call",
    # Streaming observers (agent.plugin_stream_hooks), off the token path; payloads are immutable
    # normalized text/lifecycle and cannot transform the stream.
    "on_stream_start", "on_stream_delta", "on_stream_end", "on_interim_message",
    # pre_verify: once per turn when the agent edited code and is about to verify/finish. Return
    # {"action": "continue", "message"} (or Claude-Code Stop {"decision": "block", "reason"}) to keep
    # going; anything else finishes. Bounded by agent.max_verify_nudges.
    "pre_verify", "pre_api_request", "post_api_request", "api_request_error",
    # transform_api_error_classification: once per failed API call BEFORE
    # agent/error_classifier.classify_api_error(). Kwargs: provider, model, status_code, error_type,
    # error_code, error_message, error_body, error, approx_tokens, context_length, num_messages.
    # Return None or {"reason": <FailoverReason name> (required), "retryable"/"should_compress"/
    # "should_rotate_credential"/"should_fallback": bool, "message": str, "error_context": dict}.
    # Run-all-then-pick-first (see get_plugin_error_classification). Privacy: error_message/
    # error_body may be unredacted.
    "transform_api_error_classification", "on_session_start", "on_session_end",
    "on_session_finalize", "on_session_reset",
    # on_skill_lifecycle: successful skill lifecycle facts (local skill name visible to plugins).
    "on_skill_lifecycle", "subagent_start", "subagent_stop",
    # pre_gateway_dispatch: once per incoming MessageEvent, after the internal-event guard, BEFORE
    # auth/pairing and dispatch. Kwargs: event, gateway, session_store. Return {"action": "skip",
    # "reason"} -> drop; {"action": "rewrite", "text"} -> replace event.text; "allow"/None -> normal.
    "pre_gateway_dispatch",
    # agent_loop_stopped: an agent turn was interrupted mid-run (/stop, or the running-agent
    # fast-path of /new; see gateway/run.py::_interrupt_and_clear_session). Kwargs: session_key,
    # platform, reason, invalidation_reason. Return values are ignored.
    "agent_loop_stopped",
    # Approval observers (tools/approval.py); returns ignored — plugins cannot veto or pre-answer
    # (use pre_tool_call). Kwargs: command, description, pattern_key, pattern_keys, session_key,
    # surface: "cli"|"gateway"|"smart"; post_approval_response adds choice ("once"|"session"|
    # "always"|"deny"|"timeout"|"smart_approve"|"smart_deny") and decided_by.
    "pre_approval_request", "post_approval_response",
    # on_room_member_activity: a hosted Group Chat member's live runtime events (tool.started/completed,
    # request.opened, message.delta, reasoning.delta, turn.error, ...) stamped with room_id, thread_id,
    # member_id, turn_id, task_id, execution_generation. Observer, queued per consumer off the token
    # path (agent.plugin_stream_hooks); never written to the durable room log. Kwargs: those
    # coordinates + kind, seq, payload (the client-safe session event payload, approvals redacted).
    "on_room_member_activity",
    # pre_transcription: after provider resolution, BEFORE any backend runs. Kwargs: file_path,
    # provider, model, language, prompt, source. Return None or a dict mutating prompt/language/
    # model (registration order, last-writer-wins; file_path is read-only).
    "pre_transcription",
    # Kanban task observers (hermes_cli.kanban_db), fired AFTER the DB commit so a slow plugin never
    # holds the SQLite write lock; returns ignored. claimed fires in the DISPATCHER right before
    # spawn; completed/blocked fire in the WORKER (or whichever process drove it). Kwargs: task_id,
    # board, assignee, run_id, profile_name; completed adds summary, blocked adds reason.
    "kanban_task_claimed", "kanban_task_completed", "kanban_task_blocked",
    # Kanban worker/mutation/tick observers; returns ignored; fire sites short-circuit on
    # has_hook(). Kwargs: task_id, profile_name, board, assignee, run_id plus, per hook:
    # worker_spawned (DISPATCHER, after PID persisted, inside the dispatch lock — stay fast):
    #   worker_pid, workspace_path (privacy: project layout/usernames).
    # worker_exited (tick-derived on dead-PID reclaim): worker_pid, exit_kind ("clean_exit" |
    #   "rate_limited" | "nonzero_exit" | "signaled" | "unknown"), exit_code, outcome, retry_status.
    # worker_stale_claim (TTL-expired claim reclaimed; live-PID extensions do NOT fire):
    #   worker_pid, heartbeat_stale, retry_status.
    # task_updated (committed task-row write outside claim/complete/block, in whichever process
    #   committed it): changed_fields — field NAMES only, never values.
    # dispatch_tick (once per dispatch_once, strictly AFTER the dispatch lock is released): board,
    #   profile_name, dry_run, outcome ("ok"|"skipped_locked"|"idle"), result: DispatchResult
    #   (privacy: task ids, assignees, workspace paths).
    "on_kanban_worker_spawned", "on_kanban_worker_exited", "on_kanban_worker_stale_claim",
    "on_kanban_task_updated", "on_kanban_dispatch_tick",
    # gateway_platform_event: normalized envelopes only, never raw SDK objects or adapter handles.
    # Kwargs: platform, event_type, payload (event_type-local; see hooks.md). New event types land
    # only together with real fire-sites.
    # on_kanban_dispatch_tick fires once per dispatcher tick in dispatch_once, strictly AFTER the board's
    # single-writer dispatch lock has been released (the #56066 original fired inside the lock — the #64231
    # disposition mandates the post-lock re-port), so a slow subscriber can never extend the writer critical
    # section. Kwargs: board: str | None, profile_name: str, dry_run: bool, outcome: "ok" | "skipped_locked"
    # | "idle", result: hermes_cli.kanban_db.DispatchResult (spawned, reclaimed, promoted,
    # reconciled_orphans, crashed, stale, timed_out, auto_blocked, rate_limited, auto_assigned_default,
    # respawn_guarded, skipped_per_profile_capped, skipped_unassigned, skipped_nonspawnable,
    # skipped_locked). Privacy: result carries task ids, assignees, and workspace paths.
    # Gateway platform-boundary observer hooks (#64176). Observer-only; each callback isolated by
    # invoke_hook. This surface grants no adapter handles or platform actions. Fired today: Telegram
    # "reaction" + "message_edited"; Discord "message_edited", "message_deleted", "thread_created",
    # "thread_renamed". Each event type carries its own event-local additive payload contract (see
    # hooks.md). Other event types and hook names land here only together with real fire-sites and payload
    # contracts; no inert VALID_HOOKS surface is registered ahead of implementation.
    "gateway_platform_event",
    # pre_command: BEFORE a recognized slash command's handler on CLI and gateway canonical dispatch;
    # returns IGNORED in v1. Deliberately NOT fired for the gateway's running-agent intercept path
    # (/stop, /approve, busy_policy) — a slow/hostile plugin must not touch the operator's escape
    # hatches. Kwargs: surface, command (canonical), alias_used, args_raw, session_key, platform.
    # Slash-command dispatch observer (#64204, observer-first per #64182 ground rule 3). Return values are
    # IGNORED in v1 — a plugin returning a directive-shaped dict gets a debug log so future block/rewrite
    # adopters are discoverable once the middleware variant ships against the #64231 taxonomy.
    "pre_command",
}

# Hooks whose directive the shell-hook response parser has no channel for. VALID_HOOKS doubles as
# the shell-hook allow-list, so these are refused loudly instead of having output silently ignored.
SHELL_UNSUPPORTED_HOOKS: Set[str] = {"transform_api_error_classification"}

_env_enabled = env_var_enabled  # imported by plugins/memory
_UNSET = object()


@dataclass
class LoadedPlugin:
    """Runtime state for a single loaded plugin."""

    manifest: PluginManifest
    module: Optional[types.ModuleType] = None
    tools_registered: List[str] = field(default_factory=list)
    hooks_registered: List[str] = field(default_factory=list)
    middleware_registered: List[str] = field(default_factory=list)
    commands_registered: List[str] = field(default_factory=list)
    enabled: bool = False
    error: Optional[str] = None
    # Bundled platform recorded as a not-yet-imported loader (see _register_deferred_platform).
    deferred: bool = False


@dataclass
class PluginRegistration:
    """One host-owned registration made while loading a plugin.

    Plugins only receive the context registration APIs; the manager owns the
    matching cleanup operation.  Keeping that inverse operation beside the
    registration lets a force reload unwind global registries in reverse
    order, including an override that needs to restore the entry it replaced.
    """

    kind: str
    key: str
    release: Callable[[], None]
    plugin_key: str = ""
    # Process-global host infrastructure (e.g. dashboard-auth providers) whose
    # lifetime is the server, not this per-home manager: kept out of
    # ``_registration_order`` so a routine unload-all cannot dispose it
    # (#91701), but still disposed by a *targeted* unload (plugin disable/
    # uninstall) and evicted on force re-discovery when the plugin no longer
    # re-registers it.
    persistent: bool = False
    _disposed: bool = field(default=False, init=False, repr=False)
    _on_dispose: Optional[Callable[["PluginRegistration"], None]] = field(
        default=None, init=False, repr=False
    )

    @property
    def active(self) -> bool:
        """Whether this handle still owns an active registration."""
        return not self._disposed

    def dispose(self) -> None:
        """Release this registration once; repeated disposal is harmless."""
        if self._disposed:
            return
        self._disposed = True
        try:
            self.release()
        finally:
            if self._on_dispose is not None:
                self._on_dispose(self)


# ---------------------------------------------------------------------------
# PluginContext  – handed to each plugin's ``register()`` function
# ---------------------------------------------------------------------------

_PLUGIN_SETTING_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_PLUGIN_SETTING_RESERVED_ROOTS = frozenset({"model", "plugins", "security", "settings"})
_PLUGIN_STATE_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_PLUGIN_STATE_QUOTA_BYTES = 10 * 1024 * 1024
_PLUGIN_STATE_LOCKS: Dict[str, threading.RLock] = {}
_PLUGIN_STATE_LOCKS_GUARD = threading.Lock()


def _plugin_relative_segments(key: str) -> tuple[str, ...]:
    """Validate and split a plugin-relative settings key.

    The public API accepts only relative keys (``endpoint`` or
    ``retry.policy``).  Full Hermes paths, traversal syntax, and the security-
    sensitive core roots called out in #64227 are rejected before any config
    read occurs.
    """
    if not isinstance(key, str):
        raise ValueError("Expected a plugin-relative config key string")
    segments = tuple(key.split("."))
    if (
        not key
        or "/" in key
        or "\\" in key
        or any(
            not _PLUGIN_SETTING_SEGMENT_RE.fullmatch(segment) for segment in segments
        )
        or segments[0].lower() in _PLUGIN_SETTING_RESERVED_ROOTS
    ):
        raise ValueError(
            "Expected a plugin-relative config key such as 'endpoint' or "
            "'retry.policy'; global, cross-plugin, and traversal paths are forbidden"
        )
    return segments


def _nested_plugin_value(root: object, segments: tuple[str, ...], default: Any) -> Any:
    current = root
    for segment in segments:
        if not isinstance(current, Mapping) or segment not in current:
            return default
        current = current[segment]
    return current


def _nested_plugin_mapping(segments: tuple[str, ...], value: Any) -> dict[str, Any]:
    nested: Any = value
    for segment in reversed(segments):
        nested = {segment: nested}
    return nested


def _plugin_data_namespace(plugin_id: str, skill_namespace: str) -> str:
    """Return one Windows-safe directory component for plugin-owned data."""
    candidate = skill_namespace or plugin_id
    if (
        skill_namespace
        and candidate.startswith("agent-plugin-")
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,191}", candidate)
    ):
        # Portable Agent Plugins already receive this exact PLUGIN_DATA path.
        return candidate
    # Reuse the portable namespace algorithm for native/nested ids too. Its
    # fixed prefix avoids Windows reserved device names (CON, NUL, COM1...),
    # while the digest prevents collisions after unsafe characters are folded.
    return _portable_skill_namespace(candidate)


def _state_thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve(strict=False))
    with _PLUGIN_STATE_LOCKS_GUARD:
        return _PLUGIN_STATE_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _locked_plugin_state(path: Path):
    """Serialize state read-modify-write across threads and processes.

    ``fcntl`` is used on POSIX and ``msvcrt`` on native Windows.  The lock is
    kept in a sibling file because atomic replacement changes the inode/file
    handle of the target itself.
    """
    lock_path = path.with_name(f".{path.name}.lock")
    thread_lock = _state_thread_lock(lock_path)
    with thread_lock:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+b") as handle:
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                import msvcrt

                if handle.seek(0, os.SEEK_END) == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class PluginState:
    """Atomic, quota-bounded JSON key/value state owned by one plugin."""

    def __init__(self, plugin_id: str, skill_namespace: str = "") -> None:
        self._data_namespace = _plugin_data_namespace(plugin_id, skill_namespace)

    @property
    def data_dir(self) -> Path:
        """Profile-scoped directory matching portable plugins' PLUGIN_DATA."""
        return get_hermes_home() / "plugin-data" / self._data_namespace

    @property
    def path(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def quota_bytes(self) -> int:
        return _PLUGIN_STATE_QUOTA_BYTES

    @staticmethod
    def _validate_key(key: str) -> None:
        if (
            not isinstance(key, str)
            or not _PLUGIN_STATE_KEY_RE.fullmatch(key)
            or ".." in key
        ):
            raise ValueError(
                "Plugin state keys must be 1-128 characters using letters, "
                "numbers, '_', '-', '.', or ':' (without '..')"
            )

    def _read_unlocked(self) -> dict[str, Any]:
        try:
            with open(self.path, encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot parse plugin state {self.path}: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(
                f"Cannot parse plugin state {self.path}: root must be an object"
            )
        return data

    def get(self, key: str, default: Any = None) -> Any:
        """Read a JSON value, returning *default* when the key is absent."""
        self._validate_key(key)
        with _locked_plugin_state(self.path):
            return self._read_unlocked().get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Atomically set one JSON value without dropping concurrent updates."""
        self._validate_key(key)
        with _locked_plugin_state(self.path):
            data = self._read_unlocked()
            data[key] = value
            try:
                encoded = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Plugin state value for {key!r} is not JSON-serializable"
                ) from exc
            if len(encoded) > self.quota_bytes:
                raise ValueError(
                    f"Plugin state quota exceeded: {len(encoded)} bytes is greater "
                    f"than the {self.quota_bytes}-byte per-plugin quota"
                )
            from utils import atomic_json_write

            atomic_json_write(self.path, data, mode=0o600)


class PluginContext:
    """Facade given to plugins so they can register tools and hooks."""

    def __init__(self, manifest: PluginManifest, manager: "PluginManager"):
        self.manifest = manifest
        self._manager = manager
        self._llm: Any = None  # lazy; tests preseed it (see ``llm``)

    @property
    def plugin_id(self) -> str:
        """Return the effective registry id used for this plugin's namespaces."""
        return manifest_key(self.manifest)

    def has_plugin(self, plugin_id: str) -> bool:
        """Return True when another plugin is loaded and enabled (runtime probe for advisory
        ``requires_plugins``). Matches on registry key or manifest name.

        See #64165.
        """
        return any(
            loaded.enabled and (key == plugin_id or loaded.manifest.name == plugin_id)
            for key, loaded in self._manager._plugins.items()
        )

    def _segments(self, key: str) -> tuple[str, ...]:
        """Validated plugin-relative settings path (warn + re-raise on rejection)."""
        try:
            return _plugin_relative_segments(key)
        except ValueError:
            logger.warning("Rejected config path %r from plugin %s", key, self.plugin_id)
            raise

    def get_config(self, key: str, default: Any = None) -> Any:
        """Read plugin-relative ``plugins.entries.<plugin_id>.settings.<key>`` (falls back to the
        legacy ``config`` subtree for migration compatibility)."""
        segments = self._segments(key)
        entry = _plugin_settings_entry(load_config_readonly() or {}, self.plugin_id)
        if entry is None:
            return default
        value = _nested_plugin_value(entry.get("settings"), segments, _UNSET)
        if value is not _UNSET:
            return value
        return _nested_plugin_value(entry.get("config"), segments, default)

    def set_config(self, key: str, value: Any) -> None:
        """Atomically write one value in this plugin's ``settings`` subtree."""
        segments = self._segments(key)
        from hermes_cli import config as config_mod
        if config_mod.is_managed():
            raise PermissionError("Plugin settings cannot be changed in a managed install")
        from hermes_cli import managed_scope
        full_path = ("plugins", "entries", self.plugin_id, "settings", *segments)
        dotted_path = ".".join(full_path)
        if managed_scope.is_key_managed(dotted_path):
            raise PermissionError(f"Plugin setting {dotted_path!r} is administrator-managed")
        partial = _nested_plugin_mapping(full_path[:4], _nested_plugin_mapping(segments, value))
        # The lock covers merge-read plus atomic save so sibling plugin writes (threads or
        # processes) cannot race between the two steps.
        with _locked_plugin_state(config_mod.get_config_path()), config_mod._CONFIG_LOCK:
            # Fail closed on malformed YAML: save_config degrades parse failures to {} — safe
            # for reads, destructive for read-modify-write.
            config_mod.read_user_config_raw()
            config_mod.save_config(partial, preserve_keys={full_path}, merge_existing=True)

    @cached_property
    def state(self) -> PluginState:
        """This plugin's profile-scoped durable JSON state facade."""
        return PluginState(self.plugin_id, self.manifest.skill_namespace)

    @cached_property
    def platform_actions(self):
        """Capability-gated platform action facade (``add_reaction``, ``set_thread_title``). Every call
        re-checks ``gateway.platform_actions`` (legacy ``plugins.entries.<id>.allow_platform_actions``,
        default OFF) and returns ``{"ok": bool, ...}`` — verbs never raise into hook dispatch; no adapter
        handles or raw SDK objects."""
        from hermes_cli.platform_actions import PlatformActions
        return PlatformActions(self.plugin_id)

    def _wrong_type(self, obj: Any, base_class: type, label: str, article: str = "a") -> bool:
        """Warn-and-ignore gate shared by every registrar that requires a base class."""
        if isinstance(obj, base_class):
            return False
        logger.warning("Plugin '%s' tried to register %s %s that does not inherit from %s. Ignoring.",
                       self.manifest.name, article, label, base_class.__name__)
        return True

    def _refuse(self, what: str) -> ValueError:
        """``ValueError`` for a malformed registration (``what`` completes "tried to register ...")."""
        return ValueError(f"Plugin '{self.manifest.name}' tried to register {what}.")

    def _track(
        self,
        kind: str,
        key: str,
        release: Callable[[], None],
        *,
        persistent: bool = False,
    ) -> PluginRegistration:
        """Record host-owned cleanup for a successful registration.

        ``persistent`` registrations are returned as live handles but kept
        out of the manager's reverse-order teardown, so a routine per-home
        manager unload does not dispose them (see
        :meth:`PluginManager._track_registration`).
        """
        return self._manager._track_registration(
            self.manifest, kind, key, release, persistent=persistent
        )

    def _track_replacement(
        self, kind: str, key: str, *, slot: tuple, current: Any, previous: Any,
        restore: Callable[[Any], bool],
    ) -> PluginRegistration:
        """Track one generation in a replaceable manager-local registration slot."""
        lease = replacement_coordinator.acquire(slot, current=current, previous=previous, restore=restore)
        return self._track(kind, key, lease.dispose)

    def _track_mapping_entry(
        self, kind: str, key: str, mapping: Dict[str, Any], entry: Any, previous: Any = _UNSET,
    ) -> PluginRegistration:
        """Store ``entry`` under ``key`` in a manager-local mapping and lease the slot; unload restores
        ``previous`` (default: the displaced entry, or removes the key) only while ``entry`` is still
        current."""
        if previous is _UNSET:
            previous = mapping.get(key)
        mapping[key] = entry
        return self._track_replacement(
            kind, key, slot=("manager_mapping", id(mapping), key), current=entry, previous=previous,
            restore=lambda replacement: self._manager._restore_mapping(mapping, key, entry, replacement),
        )

    def _register_entry(
        self, kind: str, key: str, mapping: Dict[str, Any], entry: Any, log_fmt: str, *log_args: Any,
        previous: Any = _UNSET,
    ) -> PluginRegistration:
        """Shared tail of the manager-mapping registrars: store + lease the entry, then log
        ``log_fmt % (plugin name, *log_args)`` at debug."""
        handle = self._track_mapping_entry(kind, key, mapping, entry, previous)
        logger.debug(log_fmt, self.manifest.name, *log_args)
        return handle

    def _register_scoped_provider(
        self, provider: Any, *, kind: str, base_class: type, registry: Any, label: str,
        article: str = "a", normalize: Optional[Callable[[str], str]] = lambda n: n.strip(),
        register: Optional[Callable[..., Any]] = None, reject_message: Optional[str] = None,
    ) -> Optional[PluginRegistration]:
        """Shared body of the ``register_<category>_provider`` methods: type-check (warn + ignore),
        register in the scope-keyed ``registry``, lease the slot so unload restores the displaced entry.
        ``None`` when the registry refused/replaced the provider (``ValueError`` with ``reject_message``
        set, or a falsy ``register``)."""
        if self._wrong_type(provider, base_class, label, article):
            return None
        registry_name = provider.name if normalize is None else normalize(provider.name)
        scope = self._manager.scope_key
        previous = registry.snapshot_registration(registry_name, scope=scope)
        try:
            accepted = (register or registry.register_provider)(provider, scope=scope)
        except ValueError as exc:
            if reject_message is None:
                raise
            logger.warning(reject_message, self.manifest.name, exc)
            return None
        if (register is not None and not accepted) or registry.snapshot_registration(
            registry_name, scope=scope
        ) is not provider:
            return None
        handle = self._manager._track_scoped_registration(
            self.manifest, kind, registry_name, registry, provider, previous
        )
        logger.info("Plugin '%s' registered %s: %s", self.manifest.name, label, registry_name)
        return handle

    @property
    def llm(self) -> Any:
        """Host-owned :class:`agent.plugin_llm.PluginLlm` facade: completions on the user's active
        model/auth. Overrides (model, agent id, auth profile) are fail-closed, gated by
        ``plugins.entries.<plugin_id>.llm.*``."""
        if self._llm is None:
            from agent.plugin_llm import PluginLlm
            self._llm = PluginLlm(plugin_id=self.plugin_id)
        return self._llm

    @cached_property
    def subagent_lifecycle(self) -> Any:
        """Plugin-safe subagent lifecycle service: serializable handles and immutable snapshots,
        never a live agent or private registry."""
        from agent.subagent_lifecycle import SubagentLifecycleService, get_active_subagent_parent
        return SubagentLifecycleService(get_active_subagent_parent)

    @property
    def profile_name(self) -> str:
        """Active profile name (``"default"``, the ``~/.hermes/profiles/<name>`` id, or ``"custom"``),
        derived from ``HERMES_HOME`` — not ``_cli_ref``, which is None outside the interactive CLI —
        so gateway and kanban workers get it too."""
        try:
            from hermes_cli.profiles import get_active_profile_name
            return get_active_profile_name()
        except Exception:
            return "default"

    def on_unload(self, callback: Callable[[], None]) -> PluginRegistration:
        """Register a cleanup callback for unload: runs in reverse acquisition order interleaved
        with registration teardown; exceptions are logged, never propagated."""
        if not callable(callback):
            raise TypeError("on_unload callback must be callable")
        handle = self._track("on_unload", getattr(callback, "__name__", "callback"), callback)
        logger.debug("Plugin %s registered on_unload callback", self.manifest.name)
        return handle

    def spawn_task(self, coro, *, name: Optional[str] = None) -> "asyncio.Task":
        """Spawn a supervised asyncio task; unload/force reload cancels it. Needs a running loop."""
        if not asyncio.iscoroutine(coro):
            raise TypeError("spawn_task expects a coroutine")
        loop = asyncio.get_running_loop()
        task_name = name or f"plugin:{self.plugin_id}:task"
        task = loop.create_task(coro, name=task_name)
        handle = self._track("background_task", task_name, lambda: task.done() or task.cancel())
        task.add_done_callback(lambda _t: handle.dispose())
        logger.debug("Plugin %s spawned supervised task: %s", self.manifest.name, task_name)
        return task

    def register_approval_transport(self, name: str, present_fn: Callable) -> None:
        """Register a human approval transport, inactive until ``security.approval.transport:
        <name>`` selects it. It receives a redacted ``ApprovalRequest`` and returns only a
        correlated decision; policy and persistence stay host-owned. ``present_fn`` may be async."""
        from hermes_cli.approval_transport import RegisteredApprovalTransport
        transports = self._manager._approval_transports
        clean = str(name).strip().lower()
        if clean == "builtin":
            raise ValueError("approval transport name 'builtin' is reserved")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", clean):
            raise ValueError("approval transport name must match [a-z0-9][a-z0-9_-]{0,63}")
        if not callable(present_fn):
            raise TypeError("approval transport present_fn must be callable")
        if clean in transports:
            owner = transports[clean].plugin_id
            raise ValueError(f"approval transport {clean!r} is already registered by {owner!r}")
        entry = RegisteredApprovalTransport(
            name=clean, present=present_fn, plugin_id=self.plugin_id,
            profile_home=str(get_hermes_home().resolve()),
        )
        logger.info("Plugin %s registered approval transport: %s", self.plugin_id, clean)
        # Duplicate names are rejected above, so there is never a displaced previous entry to restore;
        # tracking makes unload/force-reload remove this transport.
        self._track_mapping_entry("approval_transport", clean, transports, entry, None)

    @_serialized_replacement
    def register_tool(
        self, name: str, toolset: str, schema: dict, handler: Callable,
        check_fn: Callable | None = None, requires_env: list | None = None, is_async: bool = False,
        description: str = "", emoji: str = "", override: bool = False,
    ) -> Optional[PluginRegistration]:
        """Register a tool in the global registry and track it as plugin-provided. ``override=True``
        replaces a same-named built-in (without it a name claimed by another toolset is rejected) and
        needs operator opt-in via ``plugins.entries.<plugin_id>.allow_tool_override: true`` — otherwise
        any enabled plugin could silently replace a privileged built-in like ``write_file``.

        ``override=True`` against a built-in tool requires the operator to opt in via
        ``plugins.entries.<plugin_id>.allow_tool_override: true`` in config.yaml — mirrors the trust gate
        pattern used for ``ctx.llm`` provider/model overrides (#23194).
        """
        if override and not self._tool_override_allowed(name):
            raise PluginToolOverrideError(
                f"Plugin {self.manifest.name!r} cannot override built-in tool {name!r}. Set "
                f"plugins.entries.{self.plugin_id}.allow_tool_override: true "
                f"in config.yaml to allow this plugin to replace built-in tools."
            )
        from tools.registry import registry
        scope = self._manager.scope_key
        previous = registry.snapshot_registration(name, scope=scope)
        if previous is None and not override and registry.get_entry(name, scope=scope) is not None:
            logger.warning("Plugin %s tried to shadow global tool %s without override=True",
                           self.manifest.name, name)
            return None
        registry.register(
            name=name, toolset=toolset, schema=schema, handler=handler, check_fn=check_fn,
            requires_env=requires_env, is_async=is_async, description=description, emoji=emoji,
            override=override, scope=scope,
        )
        registered = registry.snapshot_registration(name, scope=scope)
        handle = None
        if registered is not None and registered is not previous and registered.handler is handler:
            self._manager._plugin_tool_names.add(name)
            handle = self._manager._track_scoped_registration(
                self.manifest, "tool", name, registry, registered, previous,
                finalize=lambda: self._manager._remove_tool_name_if_unowned(name),
            )
        logger.debug("Plugin %s registered tool: %s%s", self.manifest.name, name,
                     " (override)" if override else "")
        return handle

    # -- capability probing (#64228) -----------------------------------------
    def has_capability(self, capability: str) -> bool:
        """True when *capability* is live for this plugin (probe, then degrade gracefully). Bundled
        plugins are trusted for ``tools.override``; otherwise granted_capabilities or the legacy
        ``allow_*`` key decides. Unknown ids / unreadable consent -> False (fail closed)."""
        if self.manifest.source == "bundled" and capability == "tools.override":
            return True
        return plugin_capability_granted(self.plugin_id, capability)

    def call_mcp(
        self, server: str, tool: str, arguments: Optional[Dict[str, Any]] = None,
        timeout: float = 30,
    ) -> Dict[str, Any]:
        """Call ``tool`` on MCP ``server`` synchronously through :mod:`tools.mcp_tool`'s native client
        (same trust gates, breaker, reconnect — never a parallel connection). Servers not in
        ``plugins.entries.<plugin_id>.mcp_allowlist`` raise ``PermissionError`` (default-deny). ``timeout``
        clamps to 1–600s; results over ~64KB are truncated with a marker.

        This is a per-server grant, deliberately not ambient authority over every configured server.
        TODO(#64228): swap the per-server allowlist for the declared capability model once it lands
        (per-tool grants, expiry, ro/rw).
        """
        if server not in self._mcp_allowlist(self.plugin_id):
            raise PermissionError(
                f"Plugin {self.manifest.name!r} is not allowed to call MCP "
                f"server {server!r}. Add it to "
                f"plugins.entries.{self.plugin_id}.mcp_allowlist in config.yaml "
                f"to grant access (default is no MCP access)."
            )
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            timeout = 30.0
        timeout = max(1.0, min(timeout, 600.0))
        from tools.mcp_tool_handlers import _make_tool_handler
        raw = _make_tool_handler(server, tool, timeout)(dict(arguments or {}))
        logger.debug("Plugin %s called MCP %s/%s (timeout=%ss, %d chars returned)",
                     self.manifest.name, server, tool, timeout, len(raw or ""))
        return self._mcp_envelope(raw)

    _MCP_RESULT_CHAR_CAP = 65536

    @classmethod
    def _mcp_envelope(cls, raw: Any) -> Dict[str, Any]:
        """Normalize an MCP handler result string into a stable envelope."""
        if not isinstance(raw, str):
            raw = "" if raw is None else str(raw)
        truncated = len(raw) > cls._MCP_RESULT_CHAR_CAP
        if truncated:
            raw = raw[: cls._MCP_RESULT_CHAR_CAP] + "… [truncated]"
        try:
            parsed: Any = json.loads(raw)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict) and "error" in parsed:
            envelope: Dict[str, Any] = {"ok": False, "error": parsed["error"]}
        elif isinstance(parsed, dict) and "result" in parsed:
            envelope = {"ok": True, "result": parsed["result"]}
            if "structuredContent" in parsed:
                envelope["structuredContent"] = parsed["structuredContent"]
        else:
            envelope = {"ok": True, "result": parsed if parsed is not None else raw}
        return {**envelope, "truncated": True} if truncated else envelope

    @staticmethod
    def _mcp_allowlist(plugin_id: str) -> List[str]:
        """Operator-granted MCP server allowlist; missing/unreadable -> [] (default-deny)."""
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
        except Exception:
            return []
        allowlist = (_plugin_settings_entry(cfg, plugin_id) or {}).get("mcp_allowlist")
        return [str(item) for item in allowlist] if isinstance(allowlist, list) else []

    def _tool_override_allowed(self, tool_name: str) -> bool:
        """Whether this plugin may override built-in tools: bundled plugins are trusted (a maintainer
        choice, not privilege escalation); others need ``tools.override`` via
        :func:`plugin_capability_granted` (granted_capabilities OR legacy ``allow_tool_override: true``).

        Bundled plugins (shipped with Hermes core) are trusted by default — an override there is a
        deliberate maintainer choice, not a third-party plugin trying to elevate privilege. For every other
        source, the canonical check is :func:`plugin_capability_granted` with the ``tools.override``
        capability — satisfied by EITHER the consent-flow grant
        (``plugins.entries.<plugin_id>.granted_capabilities``) OR the deprecated legacy key
        ``allow_tool_override: true`` (still honored for backward compatibility; #64228 reference
        migration).
        """
        if self.manifest.source == "bundled":
            return True
        try:
            from hermes_cli.config import load_config
            with _plugin_home_scope(self._manager.home_path):
                cfg = load_config() or {}
        except Exception:
            return False  # fail closed: better to break the override than silently grant it
        # Pass THIS manager's profile-scoped config so a multi-profile process never consults the
        # active profile's consent state instead.
        return plugin_capability_granted(self.plugin_id, "tools.override", config=cfg)

    # Fail-closed by construction: any failure to read consent state inside plugin_capability_granted
    # returns False. The profile-scoped config is passed through so a multi-profile process consults THIS
    # manager's home, never the active profile's (#65593 constraint).
    def inject_message(
        self, content: str, role: str = "user", *, session_key: str | None = None,
    ) -> bool:
        """Inject a message into a CLI or gateway conversation (new turn if idle, interrupt if running).
        Gateway injection needs an existing ``session_key`` plus
        ``plugins.entries.<plugin_id>.allow_gateway_injection``; ``True`` means the gateway accepted the
        request for async dispatch, not that delivery completed."""
        cli = self._manager._cli_ref
        msg = content if role == "user" else f"[{role}] {content}"
        if cli is not None:
            queue_ = cli._interrupt_queue if getattr(cli, "_agent_running", False) else cli._pending_input
            queue_.put(msg)
            return True
        if not session_key:
            logger.warning("inject_message: gateway mode requires an existing session_key")
            return False
        if not self._gateway_injection_allowed():
            logger.warning("inject_message: gateway injection denied for plugin %s; set "
                           "plugins.entries.%s.allow_gateway_injection: true to allow it",
                           self.plugin_id, self.plugin_id)
            return False
        if not self._manager.has_gateway_message_injector:
            logger.warning("inject_message: no live gateway is available")
            return False
        try:
            return bool(self._manager.inject_gateway_message(
                session_key=session_key, content=msg, plugin_id=self.plugin_id,
            ))
        except Exception:
            logger.warning("inject_message: gateway scheduling failed for plugin %s", self.plugin_id,
                           exc_info=True)
            return False

    def _gateway_injection_allowed(self) -> bool:
        """Return whether this plugin may trigger gateway session turns."""
        try:
            cfg = load_config_readonly() or {}
        except Exception:
            return False
        return (_plugin_settings_entry(cfg, self.plugin_id) or {}).get("allow_gateway_injection") is True

    @_serialized_replacement
    def register_cli_command(
        self, name: str, help: str, setup_fn: Callable, handler_fn: Callable | None = None,
        description: str = "",
    ) -> PluginRegistration:
        """Register a CLI subcommand (``hermes <name> ...``). *setup_fn* receives the argparse
        subparser; *handler_fn* becomes ``set_defaults(func=...)``."""
        entry = {
            "name": name, "help": help, "description": description, "setup_fn": setup_fn,
            "handler_fn": handler_fn, "plugin": self.manifest.name, "plugin_key": self.plugin_id,
        }
        return self._register_entry("cli_command", name, self._manager._cli_commands, entry,
                                    "Plugin %s registered CLI command: %s", name)

    @_serialized_replacement
    def register_command(
        self, name: str, handler: Callable, description: str = "", args_hint: str = "",
        argument_mode: str | None = None,
    ) -> Optional[PluginRegistration]:
        """Register an in-session slash command (``/name``); handler ``fn(raw_args: str) -> str | None``
        (sync or async). ``args_hint`` (e.g. ``"<file>"``) lets adapters like Discord surface an argument
        field; without it the command registers parameterless there but still accepts trailing text."""
        clean = name.lower().strip().lstrip("/").replace(" ", "-")
        if not clean:
            logger.warning("Plugin '%s' tried to register a command with an empty name.", self.manifest.name)
            return
        with suppress(Exception):  # reject if it conflicts with a built-in command
            from hermes_cli.commands import resolve_command
            if resolve_command(clean) is not None:
                logger.warning("Plugin '%s' tried to register command '/%s' which conflicts "
                               "with a built-in command. Skipping.", self.manifest.name, clean)
                return
        hint = (args_hint or "").strip()
        entry = {
            "handler": handler, "description": description or "Plugin command",
            "plugin": self.manifest.name, "plugin_key": self.plugin_id, "args_hint": hint,
            "argument_mode": argument_mode if argument_mode in {"options", "text", "mixed"}
            else ("text" if hint else None),
        }
        return self._register_entry("command", clean, self._manager._plugin_commands, entry,
                                    "Plugin %s registered command: /%s", clean)

    def dispatch_tool(self, tool_name: str, args: dict, **kwargs) -> str:
        """Dispatch a tool call through the registry with the parent agent (when available)
        resolved automatically; returns the handler's JSON string. ``kwargs`` forward to dispatch."""
        from tools.registry import registry
        # In gateway mode _cli_ref is None — tools degrade gracefully (no spinner, TERMINAL_CWD).
        if "parent_agent" not in kwargs:
            agent = getattr(self._manager._cli_ref, "agent", None)
            if agent is not None:
                kwargs["parent_agent"] = agent
        return registry.dispatch(tool_name, args, scope=self._manager.scope_key, **kwargs)

    @_serialized_replacement
    def register_context_engine(self, engine) -> Optional[PluginRegistration]:
        """Register the (single) ``agent.context_engine.ContextEngine`` replacing the built-in
        ContextCompressor; a second registration is rejected with a warning."""
        if self._manager._context_engine is not None:
            logger.warning("Plugin '%s' tried to register a context engine, but one is "
                           "already registered. Only one context engine plugin is allowed.",
                           self.manifest.name)
            return
        from agent.context_engine import ContextEngine
        if self._wrong_type(engine, ContextEngine, "context engine"):
            return
        previous = self._manager._context_engine  # always None here; kept for the restore contract
        self._manager._context_engine = engine
        handle = self._track_replacement(
            "context_engine", engine.name, slot=("manager_value", id(self._manager), "_context_engine"),
            current=engine, previous=previous,
            restore=lambda replacement: self._manager._restore_value("_context_engine", engine, replacement),
        )
        logger.info("Plugin '%s' registered context engine: %s", self.manifest.name, engine.name)
        return handle

    def register_context_reference(self, provider) -> None:
        """Register a :class:`agent.context_references.ContextReferenceProvider`; ``provider.prefix``
        defines ``@<prefix>:``. Built-in prefixes (diff, staged, file, folder, git, url) are
        rejected."""
        from agent.context_references import (
            ContextReferenceProvider as _CRP, register_context_reference_provider as _register,
        )
        if self._wrong_type(provider, _CRP, "context reference provider"):
            return
        try:
            _register(provider)
        except ValueError as exc:
            logger.warning("Plugin '%s' context reference registration failed: %s", self.manifest.name, exc)
            return
        logger.info("Plugin '%s' registered context reference: @%s:", self.manifest.name, provider.prefix)

    def register_memory_provider(self, provider) -> None:
        """Record a memory provider (inert). Activation is owned by ``plugins/memory`` via
        ``memory.provider``; a provider reaching here was loaded by the general manager, and
        without this method its ``register()`` would fail on a missing attribute."""
        from agent.memory_provider import MemoryProvider
        if self._wrong_type(provider, MemoryProvider, "memory provider"):
            return
        self._memory_provider = provider
        logger.debug("Plugin '%s' registered memory provider: %s", self.manifest.name,
                     getattr(provider, "name", "?"))

    @_serialized_replacement
    def register_dashboard_auth_provider(self, provider) -> Optional[PluginRegistration]:
        """Register a dashboard authentication provider.

        ``provider`` must be an instance of
        :class:`hermes_cli.dashboard_auth.DashboardAuthProvider`. Used by
        the dashboard OAuth auth gate, which engages when the dashboard
        binds to a non-loopback host without ``--insecure``.

        Misbehaving providers (wrong type, duplicate name) are logged at
        WARNING and silently ignored — never raised — so a broken plugin
        cannot crash the host. Same convention as
        ``register_image_gen_provider``.
        """
        from hermes_cli.dashboard_auth import DashboardAuthProvider
        from hermes_cli.dashboard_auth.registry import (
            register_global_provider,
            unregister_global_provider,
        )

        if not isinstance(provider, DashboardAuthProvider):
            logger.warning(
                "Plugin '%s' tried to register a dashboard-auth provider "
                "that does not inherit from DashboardAuthProvider. Ignoring.",
                self.manifest.name,
            )
            return
        registry_name = provider.name
        # The dashboard auth registry is process-global — its lifetime is the
        # web server, not this per-home plugin manager. A per-home manager is
        # torn down routinely (profile-scoped dashboard activity, force
        # re-discovery), and disposing this registration on that teardown
        # emptied the auth registry for the WHOLE process, permanently
        # disabling sign-in until restart (#91701). So register it in the
        # global slot (upsert) and, crucially, keep it OUT of the manager's
        # reverse-order teardown (``persistent=True``): a per-home unload can
        # no longer clear it. The handle still disposes explicitly (identity-
        # conditional), and a forced re-discovery rotates the provider in
        # place via the upsert.
        try:
            register_global_provider(provider)
        except (TypeError, ValueError) as e:
            logger.warning("Plugin '%s' failed to register dashboard-auth provider %r: %s",
                           self.manifest.name, getattr(provider, "name", "?"), e)
            return
        handle = self._track(
            "dashboard_auth_provider",
            registry_name,
            lambda: unregister_global_provider(registry_name, provider),
            persistent=True,
        )
        logger.info(
            "Plugin '%s' registered dashboard-auth provider: %s (%s)",
            self.manifest.name, registry_name, provider.display_name,
        )
        return handle

    # -- video gen provider registration -------------------------------------

    @_serialized_replacement
    def register_video_gen_provider(self, provider) -> Optional[PluginRegistration]:
        """Register a video generation backend.

        ``provider`` must be an instance of
        :class:`agent.video_gen_provider.VideoGenProvider`. The
        ``provider.name`` attribute is what ``video_gen.provider`` in
        ``config.yaml`` matches against when routing ``video_generate``
        tool calls.
        """
        from agent.video_gen_provider import VideoGenProvider
        from agent.video_gen_registry import (
            register_provider as _register_video_provider,
            restore_registration,
            snapshot_registration,
        )

        if not isinstance(provider, VideoGenProvider):
            logger.warning(
                "Plugin '%s' tried to register a video_gen provider that does "
                "not inherit from VideoGenProvider. Ignoring.",
                self.manifest.name,
            )
            return
        registry_name = provider.name.strip()
        scope = self._manager.scope_key
        previous = snapshot_registration(registry_name, scope=scope)
        _register_video_provider(provider, scope=scope)
        registered = snapshot_registration(registry_name, scope=scope)
        if registered is not provider:
            return None
        handle = self._track_replacement(
            "video_gen_provider",
            registry_name,
            slot=("video_gen_provider", scope, registry_name),
            current=provider,
            previous=previous,
            restore=lambda replacement: restore_registration(
                registry_name, provider, replacement, scope=scope
            ),
        )
        logger.info(
            "Plugin '%s' registered video_gen provider: %s",
            self.manifest.name, registry_name,
        )
        return handle

    # -- web search/extract provider registration ----------------------------

    @_serialized_replacement
    def register_web_search_provider(self, provider) -> Optional[PluginRegistration]:
        """Register a web search/extract backend.

        ``provider`` must be an instance of
        :class:`agent.web_search_provider.WebSearchProvider`. The
        ``provider.name`` attribute is what ``web.search_backend`` /
        ``web.extract_backend`` / ``web.backend`` in ``config.yaml``
        matches against when routing ``web_search`` / ``web_extract``
        tool calls.
        """
        from agent.web_search_provider import WebSearchProvider
        from agent.web_search_registry import (
            register_provider as _register_web_provider,
            restore_registration,
            snapshot_registration,
        )

        if not isinstance(provider, WebSearchProvider):
            logger.warning(
                "Plugin '%s' tried to register a web provider that does "
                "not inherit from WebSearchProvider. Ignoring.",
                self.manifest.name,
            )
            return
        registry_name = provider.name.strip()
        scope = self._manager.scope_key
        previous = snapshot_registration(registry_name, scope=scope)
        _register_web_provider(provider, scope=scope)
        registered = snapshot_registration(registry_name, scope=scope)
        if registered is not provider:
            return None
        handle = self._track_replacement(
            "web_search_provider",
            registry_name,
            slot=("web_search_provider", scope, registry_name),
            current=provider,
            previous=previous,
            restore=lambda replacement: restore_registration(
                registry_name, provider, replacement, scope=scope
            ),
        )
        logger.info(
            "Plugin '%s' registered web provider: %s",
            self.manifest.name, registry_name,
        )
        return handle

    # -- browser provider registration ---------------------------------------

    @_serialized_replacement
    def register_browser_provider(self, provider) -> Optional[PluginRegistration]:
        """Register a cloud browser backend.

        ``provider`` must be an instance of
        :class:`agent.browser_provider.BrowserProvider`. The
        ``provider.name`` attribute is what ``browser.cloud_provider`` in
        ``config.yaml`` matches against when routing cloud-mode
        ``browser_*`` tool calls.

        Mirrors :meth:`register_web_search_provider` exactly — same
        registration shape, same gating, same logging. The browser
        subsystem's dispatcher (:func:`tools.browser_tool._get_cloud_provider`)
        consults the registry built up by these calls.
        """
        from agent.browser_provider import BrowserProvider
        from agent.browser_registry import (
            register_provider as _register_browser_provider,
            restore_registration,
            snapshot_registration,
        )

        if not isinstance(provider, BrowserProvider):
            logger.warning(
                "Plugin '%s' tried to register a browser provider that does "
                "not inherit from BrowserProvider. Ignoring.",
                self.manifest.name,
            )
            return
        registry_name = provider.name.strip()
        scope = self._manager.scope_key
        previous = snapshot_registration(registry_name, scope=scope)
        _register_browser_provider(provider, scope=scope)
        registered = snapshot_registration(registry_name, scope=scope)
        if registered is not provider:
            return None
        handle = self._track_replacement(
            "browser_provider",
            registry_name,
            slot=("browser_provider", scope, registry_name),
            current=provider,
            previous=previous,
            restore=lambda replacement: restore_registration(
                registry_name, provider, replacement, scope=scope
            ),
        )
        logger.info(
            "Plugin '%s' registered browser provider: %s",
            self.manifest.name, registry_name,
        )
        return handle

    # -- terminal environment provider registration ----------------------------

    @_serialized_replacement
    def register_terminal_environment_provider(self, provider) -> Optional[PluginRegistration]:
        """Register a pluggable terminal execution backend.

        ``provider`` must be an instance of
        :class:`agent.terminal_env_provider.TerminalEnvironmentProvider`.
        The ``provider.name`` attribute is what ``terminal.backend`` in
        ``config.yaml`` (bridged to ``TERMINAL_ENV``) matches against when
        the dispatch ladder in :func:`tools.terminal_tool._create_environment`
        finds no built-in backend of that name.

        Names colliding with built-in backends (local, docker, singularity,
        modal, daytona, vercel_sandbox, ssh) are rejected by the registry —
        plugins extend the backend set, they never shadow in-tree backends.

        Mirrors :meth:`register_browser_provider` — same registration shape,
        same gating, same logging.
        """
        from agent.terminal_env_provider import TerminalEnvironmentProvider
        from agent.terminal_env_registry import (
            register_provider as _register_terminal_env_provider,
            restore_registration,
            snapshot_registration,
        )

        if not isinstance(provider, TerminalEnvironmentProvider):
            logger.warning(
                "Plugin '%s' tried to register a terminal environment "
                "provider that does not inherit from "
                "TerminalEnvironmentProvider. Ignoring.",
                self.manifest.name,
            )
            return
        registry_name = provider.name.strip().lower()
        scope = self._manager.scope_key
        previous = snapshot_registration(registry_name, scope=scope)
        try:
            _register_terminal_env_provider(provider, scope=scope)
        except ValueError as exc:
            logger.warning(
                "Plugin '%s' terminal environment provider rejected: %s",
                self.manifest.name, exc,
            )
            return
        registered = snapshot_registration(registry_name, scope=scope)
        if registered is not provider:
            return None
        handle = self._track_replacement(
            "terminal_environment_provider",
            registry_name,
            slot=("terminal_environment_provider", scope, registry_name),
            current=provider,
            previous=previous,
            restore=lambda replacement: restore_registration(
                registry_name, provider, replacement, scope=scope
            ),
        )
        logger.info(
            "Plugin '%s' registered terminal environment provider: %s",
            self.manifest.name, registry_name,
        )
        return handle

    # -- secret source registration -------------------------------------------

    @_serialized_replacement
    def register_secret_source(self, source) -> Optional[PluginRegistration]:
        """Register an external secret-manager backend.

        ``source`` must be an instance of
        :class:`agent.secret_sources.base.SecretSource`.  Registered
        sources run during ``load_hermes_dotenv()`` startup — after
        ``~/.hermes/.env`` loads, before Hermes reads credentials — when
        their ``secrets.<source.name>`` config section is enabled.  The
        orchestrator (``agent.secret_sources.registry.apply_all``) owns
        ordering, mapped-vs-bulk precedence, conflict warnings, and
        provenance; the source only fetches.

        NOTE ON TIMING: ``load_hermes_dotenv()`` usually runs at import
        *before* plugin discovery.  After discovery completes, the plugin
        manager re-pulls enabled plugin secret sources (``reset_secret_source_cache``
        + ``load_hermes_dotenv``) so the first process sees them (#64177).
        Child processes that load env after plugins still work without that
        re-pull.  Failed re-pulls never block startup.

        Contract requirements (rejected with a warning otherwise):
        inherit from ``SecretSource``, ``api_version`` matching
        ``SECRET_SOURCE_API_VERSION``, lowercase unique ``name``,
        ``shape`` of ``"mapped"`` or ``"bulk"``, unique ``scheme`` (when
        set), and a ``fetch()`` that never raises and never prompts.
        See the base-module docstring for the full contract.
        """
        from agent.secret_sources.base import SecretSource
        from agent.secret_sources.registry import (
            register_source,
            restore_registration,
            snapshot_registration,
        )

        if not isinstance(source, SecretSource):
            logger.warning(
                "Plugin '%s' tried to register a secret source that does "
                "not inherit from SecretSource. Ignoring.",
                self.manifest.name,
            )
            return
        registry_name = source.name
        scope = self._manager.scope_key
        previous = snapshot_registration(registry_name, scope=scope)
        if register_source(source, scope=scope):
            registered = snapshot_registration(registry_name, scope=scope)
            if registered is not source:
                return None
            handle = self._track_replacement(
                "secret_source",
                registry_name,
                slot=("secret_source", scope, registry_name),
                current=source,
                previous=previous,
                restore=lambda replacement: restore_registration(
                    registry_name, source, replacement, scope=scope
                ),
            )
            logger.info(
                "Plugin '%s' registered secret source: %s",
                self.manifest.name, registry_name,
            )
            return handle
        return None

    # -- TTS provider registration -------------------------------------------

    @_serialized_replacement
    def register_tts_provider(self, provider) -> Optional[PluginRegistration]:
        """Register a text-to-speech backend.

        ``provider`` must be an instance of
        :class:`agent.tts_provider.TTSProvider`. The ``provider.name``
        attribute is what ``tts.provider`` in ``config.yaml`` matches
        against when routing ``text_to_speech`` tool calls — **but
        only when**:

        1. ``provider.name`` is NOT a built-in TTS provider name
           (``edge``, ``openai``, ``elevenlabs``, …). Built-ins always
           win — the registry rejects shadowing names with a warning.
        2. There is NO ``tts.providers.<name>: type: command`` entry
           with the same name. Command-providers (PR #17843) win on
           name collision because config is more local than plugin
           install.

        Coexists with the command-provider registry rather than
        replacing it — see issue #30398 for the full design rationale.
        """
        from agent.tts_provider import TTSProvider
        from agent.tts_registry import (
            register_provider as _register_tts_provider,
            restore_registration,
            snapshot_registration,
        )

        if not isinstance(provider, TTSProvider):
            logger.warning(
                "Plugin '%s' tried to register a TTS provider that does "
                "not inherit from TTSProvider. Ignoring.",
                self.manifest.name,
            )
            return
        registry_name = provider.name.strip().lower()
        scope = self._manager.scope_key
        previous = snapshot_registration(registry_name, scope=scope)
        _register_tts_provider(provider, scope=scope)
        registered = snapshot_registration(registry_name, scope=scope)
        if registered is not provider:
            return None
        handle = self._track_replacement(
            "tts_provider",
            registry_name,
            slot=("tts_provider", scope, registry_name),
            current=provider,
            previous=previous,
            restore=lambda replacement: restore_registration(
                registry_name, provider, replacement, scope=scope
            ),
        )
        logger.info(
            "Plugin '%s' registered TTS provider: %s",
            self.manifest.name, registry_name,
        )
        return handle

    # -- transcription (STT) provider registration ---------------------------

    @_serialized_replacement
    def register_transcription_provider(self, provider) -> Optional[PluginRegistration]:
        """Register a speech-to-text backend.

        ``provider`` must be an instance of
        :class:`agent.transcription_provider.TranscriptionProvider`.
        The ``provider.name`` attribute is what ``stt.provider`` in
        ``config.yaml`` matches against when routing
        :func:`tools.transcription_tools.transcribe_audio` calls —
        **but only when**:

        1. ``provider.name`` is NOT a built-in STT provider name
           (``local``, ``local_command``, ``groq``, ``openai``,
           ``mistral``, ``xai``). Built-ins always win — the registry
           rejects shadowing names with a warning.
        2. There is NO ``stt.providers.<name>: type: command`` entry
           with the same name. Command-providers win on name
           collision because config is more local than plugin install
           — same precedence rule as TTS.

        Coexists with the in-tree dispatcher and the STT
        command-provider registry rather than replacing them. The 6
        built-in STT backends keep their native implementations in
        ``tools/transcription_tools.py``; this hook is for *new* Python
        engines (OpenRouter, SenseAudio, Gemini-STT, custom proprietary
        backends).
        """
        from agent.transcription_provider import TranscriptionProvider
        from agent.transcription_registry import (
            register_provider as _register_stt_provider,
            restore_registration,
            snapshot_registration,
        )

        if not isinstance(provider, TranscriptionProvider):
            logger.warning(
                "Plugin '%s' tried to register a transcription provider that "
                "does not inherit from TranscriptionProvider. Ignoring.",
                self.manifest.name,
            )
            return
        registry_name = provider.name.strip().lower()
        scope = self._manager.scope_key
        previous = snapshot_registration(registry_name, scope=scope)
        _register_stt_provider(provider, scope=scope)
        registered = snapshot_registration(registry_name, scope=scope)
        if registered is not provider:
            return None
        handle = self._track_replacement(
            "transcription_provider",
            registry_name,
            slot=("transcription_provider", scope, registry_name),
            current=provider,
            previous=previous,
            restore=lambda replacement: restore_registration(
                registry_name, provider, replacement, scope=scope
            ),
        )
        logger.info(
            "Plugin '%s' registered transcription provider: %s",
            self.manifest.name, registry_name,
        )
        return handle

    # -- platform adapter registration ---------------------------------------

    @_serialized_replacement
    def register_platform(
        self, name: str, label: str, adapter_factory: Callable, check_fn: Callable,
        validate_config: Callable | None = None, required_env: list | None = None,
        install_hint: str = "", **entry_kwargs: Any,
    ) -> Optional[PluginRegistration]:
        """Register a gateway platform adapter (``adapter_factory(PlatformConfig) -> BasePlatformAdapter``).
        ``check_fn`` is a PASSIVE "deps importable?" probe that must never install (status displays call
        it freely); an ACTIVE installer goes in ``ensure_deps_fn`` (called from ``create_adapter()`` when
        ``check_fn`` is False). Extra kwargs (``setup_fn``, ``emoji``, ``allowed_users_env``,
        ``platform_hint``, ``ensure_deps_fn``) forward to ``PlatformEntry``; unknown keys raise TypeError."""
        from gateway.platform_registry import platform_registry, PlatformEntry
        entry_kwargs.setdefault("plugin_name", self.manifest.name)
        entry = PlatformEntry(
            name=name, label=label, adapter_factory=adapter_factory, check_fn=check_fn,
            validate_config=validate_config, required_env=required_env or [],
            install_hint=install_hint, source="plugin", **entry_kwargs,
        )
        scope = self._manager.scope_key
        previous = platform_registry.snapshot_registration(name, scope=scope)
        platform_registry.register(entry, scope=scope)
        current = platform_registry.snapshot_registration(name, scope=scope)
        if current[0] is not entry or current[1] is not None:
            return None
        self._manager._plugin_platform_names.add(name)
        handle = self._manager._track_scoped_registration(
            self.manifest, "platform", name, platform_registry, current, previous,
            finalize=lambda: self._manager._remove_platform_name_if_unowned(name),
        )
        logger.debug("Plugin %s registered platform: %s", self.manifest.name, name)
        return handle

    def register_slack_action_handler(
        self, action_id: Any, callback: Callable,
    ) -> PluginRegistration:
        """Register a Slack Block Kit action handler, wired into ``slack_bolt.AsyncApp`` at connect.
        ``action_id`` is anything ``slack_bolt.App.action()`` accepts; ``callback`` is
        ``async def handler(ack, body, action)`` (``await ack()`` within 3s). Raises ``ValueError`` for
        a non-callable callback or empty ``action_id``."""
        if not callable(callback):
            raise self._refuse("a Slack action handler with a non-callable callback")
        if action_id is None or (isinstance(action_id, str) and not action_id.strip()):
            raise self._refuse("a Slack action handler with an empty action_id")
        entry = (action_id, callback, self.manifest.name)
        handlers = self._manager._slack_action_handlers
        handlers.append(entry)
        handle = self._track("slack_action_handler", repr(action_id),
                             lambda: self._manager._remove_identity(handlers, entry))
        logger.debug("Plugin %s registered Slack action handler: %s", self.manifest.name, action_id)
        return handle

    def register_platform_handler(self, platform: str, factory: Callable) -> None:
        """Register ``factory(native, adapter)``, invoked at ``connect()`` before/as the core handlers
        register (``adapter`` read-only). ``native``: telegram PTB ``Application``, discord
        ``commands.Bot``, slack ``AsyncApp``, matrix client, teams ``App``, dingtalk
        ``DingTalkStreamClient``, line aiohttp ``web.Application``, others ``None``. Keep SDK imports
        inside the factory; exceptions are logged and the platform still connects. Scope handlers in
        first-match dispatch tables so core flows keep working. Raises ``ValueError`` when not callable
        or platform is empty."""
        if not callable(factory):
            raise self._refuse("a platform handler factory with a non-callable factory")
        key = (platform or "").strip().lower()
        if not key:
            raise self._refuse("a platform handler factory with an empty platform name")
        self._manager._platform_handler_factories.setdefault(key, []).append((factory, self.manifest.name))
        logger.debug("Plugin %s registered %s handler factory: %s", self.manifest.name, key,
                     getattr(factory, "__name__", repr(factory)))

    def register_telegram_handler(self, factory: Callable) -> None:
        """``register_platform_handler("telegram", factory)``. PTB dispatches only the FIRST matching
        handler per group and core registers a catch-all ``CallbackQueryHandler`` — always scope with
        ``pattern=`` or you swallow the core button flows."""
        self.register_platform_handler("telegram", factory)

    @_serialized_replacement
    def register_auxiliary_task(
        self, key: str, *, display_name: str, description: str,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> PluginRegistration:
        """Register a plugin-defined auxiliary LLM task.

        Auxiliary tasks are LLM-backed side jobs (vision analysis, web extraction,
        compression, smart-approval, etc.) that route through ``auxiliary_client.py``.
        Each task has its own ``auxiliary.<key>`` config block where users can
        pin a provider/model independent of the main chat model.

        Plugins use this to declare their own auxiliary tasks without touching
        core files. After registration, the task:

          - Appears in the ``hermes model → Configure auxiliary models`` picker
          - Has its provider/model/base_url/api_key bridged from config.yaml to
            ``AUXILIARY_<KEY_UPPER>_*`` env vars at gateway startup
          - Gets default routing fields (provider="auto", model="", etc.) merged
            into loaded configs so ``cfg.get("auxiliary", {}).get(key)`` works

        Args:
            key: stable task key (snake_case). Used in config ``auxiliary.<key>``
                and env vars ``AUXILIARY_<KEY_UPPER>_*``. Must not shadow a
                built-in task key (vision, compression, approval,
                mcp, title_generation, skills_hub, curator).
            display_name: human-readable name shown in the picker.
            description: short one-line description shown next to the name.
            defaults: optional dict of default routing fields. Recognized keys:
                ``provider`` (default "auto"), ``model`` (default ""),
                ``base_url`` (default ""), ``api_key`` (default ""),
                ``timeout`` (default 60), ``extra_body`` (default {}),
                plus any task-specific extras (e.g. ``download_timeout``).
                Unknown keys are preserved verbatim — the plugin owns the
                schema for its own task.

        Raises:
            ValueError: if *key* is empty, contains invalid characters, or
                shadows a built-in auxiliary task key.

        Example:
            ctx.register_auxiliary_task(
                key="memory_retain_filter",
                display_name="Memory retain filter",
                description="hindsight pre-retain dedup/extract",
                defaults={"provider": "auto", "timeout": 30},
            )
        """
        # Validate key shape
        if not key or not isinstance(key, str):
            raise ValueError(f"Plugin '{me}' tried to register auxiliary task with invalid key {key!r}")
        if not all(c.isalnum() or c == "_" for c in key):
            raise ValueError(f"Plugin '{me}' auxiliary task key {key!r} "
                             f"must contain only alphanumeric characters and underscores")
        from hermes_cli.main_provider_setup import _AUX_TASKS as _BUILTIN_AUX_TASKS
        if key in {k for k, _name, _desc in _BUILTIN_AUX_TASKS}:
            raise ValueError(f"Plugin '{me}' cannot register auxiliary task {key!r} — that key is reserved "
                             f"for a built-in task. Pick a plugin-namespaced key (e.g. '{me}_{key}').")
        # Owner is the canonical id ``ctx.llm`` is bound to, so agent/plugin_llm.py can match it.
        owner_id = self.plugin_id
        existing = self._manager._aux_tasks.get(key)
        if existing is not None and existing.get("plugin") != owner_id:
            raise ValueError(f"Plugin '{me}' cannot register auxiliary task {key!r} — already registered "
                             f"by plugin '{existing.get('plugin')}'")
        # Plugin owns the schema; routing fields are guaranteed present so consumers don't crash.
        entry = {
            "key": key, "display_name": display_name, "description": description,
            "defaults": {"provider": "auto", "model": "", "base_url": "", "api_key": "", "timeout": 60,
                         "extra_body": {}, **(defaults or {})},
            "plugin": owner_id, "plugin_key": owner_id,
        }
        return self._register_entry("auxiliary_task", key, self._manager._aux_tasks, entry,
                                    "Plugin %s registered auxiliary task: %s (%s)", key, display_name,
                                    previous=existing)

    def register_redaction_patterns(self, patterns) -> int:
        """Additively register secret-token regexes with :mod:`agent.redact`; returns the count accepted.
        Plugins can over-redact, never weaken built-ins; ``security.redact_secrets: false`` applies
        equally. Each pattern must compile and start with >= 2 literal characters; invalid entries warn
        and are skipped."""
        from agent.redact import register_redaction_patterns as _register
        try:
            count = _register(patterns, source=f"plugin:{self.manifest.name}")
        except Exception as exc:
            logger.warning("Plugin '%s' redaction pattern registration failed: %s", self.manifest.name, exc)
            return 0
        logger.debug("Plugin %s registered %d redaction pattern(s)", self.manifest.name, count)
        return count

    def register_hook(self, hook_name: str, callback: Callable) -> PluginRegistration:
        """Register a lifecycle hook callback (unknown names warn but are still stored)."""
        return self._track_callback("hook", hook_name, callback, self._manager._hooks, VALID_HOOKS)

    def register_middleware(self, kind: str, callback: Callable) -> PluginRegistration:
        """Register behavior-changing middleware (request kinds rewrite the payload, execution kinds
        wrap the callback). Unknown kinds warn but are stored."""
        return self._track_callback(
            "middleware", kind, callback, self._manager._middleware, VALID_MIDDLEWARE
        )

    def _track_callback(
        self, kind: str, key: str, callback: Callable, mapping: Dict[str, List[Callable]],
        valid: Set[str],
    ) -> PluginRegistration:
        """Append ``callback`` under ``key`` (warning on unknown ``key``) and lease its removal."""
        if key not in valid:
            logger.warning("Plugin '%s' registered unknown %s '%s' (valid: %s)", self.manifest.name, kind,
                           key, ", ".join(sorted(valid)))
        mapping.setdefault(key, []).append(callback)
        handle = self._track(kind, key, lambda: self._manager._remove_callback(mapping, key, callback))
        logger.debug("Plugin %s registered %s: %s", self.manifest.name, kind, key)
        return handle

    def register_system_prompt_section(
        self, id: str, content: Union[str, Callable[[Mapping[str, Any]], str]], *,
        position: str = "after_memory", max_chars: int = DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS,
    ) -> PluginRegistration:
        """Register bounded context frozen into each new session prompt. Callables receive a
        read-only session-info mapping; the rendered prompt is persisted by core verbatim."""
        if not is_valid_system_prompt_section_id(id):
            raise ValueError("system prompt section id must be 1-128 lowercase characters "
                             "using letters, numbers, '.', '_', or '-'")
        if not isinstance(content, str) and not callable(content):
            raise TypeError("system prompt section content must be a string or callable")
        if position not in SYSTEM_PROMPT_SECTION_POSITIONS:
            raise ValueError("system prompt section position must be one of: "
                             + ", ".join(sorted(SYSTEM_PROMPT_SECTION_POSITIONS)))
        if (isinstance(max_chars, bool) or not isinstance(max_chars, int)
                or not 0 < max_chars <= MAX_SYSTEM_PROMPT_SECTION_CHARS):
            raise ValueError(f"system prompt section max_chars must be between 1 and {MAX_SYSTEM_PROMPT_SECTION_CHARS}")
        existing = self._manager._system_prompt_sections.get(id)
        if existing is not None:
            raise ValueError(f"system prompt section {id!r} is already registered by plugin {existing.plugin!r}")
        section = PluginSystemPromptSection(
            id=id, content=content, position=position, max_chars=max_chars, plugin=self.plugin_id,
        )
        return self._register_entry("system_prompt_section", id, self._manager._system_prompt_sections,
                                    section, "Plugin %s registered system prompt section: %s", id,
                                    previous=existing)

    def emit(self, event: str, payload: Optional[dict] = None) -> int:
        """Publish bare *event* as ``<plugin_key>:<event>`` (namespace FORCED to this plugin); return
        the subscriber count scheduled. Any ``':'`` in the name (``hermes:x`` is reserved for core,
        foreign namespaces forbidden) raises ``ValueError``. Delivery is fire-and-forget via a
        single-worker queue: order preserved, a blocking subscriber cannot stall the emitter."""
        plugin_key = self.plugin_id
        if not event or not isinstance(event, str):
            logger.warning("Plugin '%s' tried to emit an invalid event name %r", plugin_key, event)
            raise ValueError(f"Plugin '{plugin_key}' emit() requires a non-empty event name")
        if ":" in event:
            logger.warning("Plugin '%s' tried to emit namespaced/reserved event '%s' — a plugin may only emit "
                           "bare event names under its own '%s:' namespace (the '%s:' prefix is reserved "
                           "for core, and foreign namespaces are forbidden)",
                           plugin_key, event, plugin_key, HERMES_EVENT_NAMESPACE)
            raise ValueError(f"Plugin '{plugin_key}' may not emit '{event}': emit only the bare event name; "
                             f"the namespace is forced to '{plugin_key}:' and the '{HERMES_EVENT_NAMESPACE}:' "
                             f"prefix is reserved for core")
        if payload is not None and not isinstance(payload, dict):
            raise TypeError(f"Plugin '{plugin_key}' emit() payload must be a dict or None")
        return self._manager._dispatch_event(f"{plugin_key}:{event}", payload or {})

    def subscribe(self, event: str, callback: Callable) -> None:
        """Subscribe to a fully-qualified ``<plugin_key>:<event>`` name (unrestricted — only
        emitting is namespace-gated). Owner-tagged so unload removes zombie callbacks."""
        if not event or not isinstance(event, str):
            raise ValueError(f"Plugin '{self.manifest.name}' subscribe() requires a non-empty event name")
        self._manager._subscribe_event(self.plugin_id, event, callback)
        logger.debug("Plugin %s subscribed to event: %s", self.manifest.name, event)

    @_serialized_replacement
    def register_skill(
        self, name: str, path: Path, description: str = "",
        frontmatter: Optional[Mapping[str, Any]] = None,
    ) -> PluginRegistration:
        """Register a read-only skill resolvable as ``'<plugin_name>:<name>'`` via ``skill_view()``.
        Not in ``~/.hermes/skills/`` nor ``<available_skills>`` — explicit loads only. Raises
        ``ValueError`` (``':'``/invalid chars) or ``FileNotFoundError``."""
        from agent.skill_utils import _NAMESPACE_RE
        if ":" in name:
            raise ValueError(f"Skill name '{name}' must not contain ':' (the namespace is derived from the "
                             f"plugin name '{self.manifest.name}' automatically).")
        if not name or not _NAMESPACE_RE.match(name):
            raise ValueError(f"Invalid skill name '{name}'. Must match [a-zA-Z0-9_-]+.")
        if not path.exists():
            raise FileNotFoundError(f"SKILL.md not found at {path}")
        namespace = self.manifest.skill_namespace or self.manifest.name
        qualified = f"{namespace}:{name}"
        if self.manifest.portable and qualified in self._manager._plugin_skills:
            raise ValueError(f"Plugin skill '{qualified}' is already registered")
        entry = {
            "path": path, "plugin": namespace, "plugin_key": self.plugin_id, "bare_name": name,
            "description": description, "frontmatter": dict(frontmatter or {}),
        }
        return self._register_entry("skill", qualified, self._manager._plugin_skills, entry,
                                    "Plugin %s registered skill: %s", qualified)


# -- scoped provider registrars ------------------------------------------------------------------
# Every ``register_<category>_provider`` shares one body (:meth:`PluginContext._register_scoped_provider`):
# type-check, register in the scope-keyed process-global registry, lease the slot so unload restores
# the displaced entry. Rows: (method, kind, registry module, base-class module:attr, label, docstring,
# options). ``normalize``: ``strip`` (default), ``lower`` (strip+lowercase) or ``None`` (raw name).
_SCOPED_PROVIDER_REGISTRARS: Tuple[Tuple[str, str, str, str, str, str, Dict[str, Any]], ...] = (
    ("register_image_gen_provider", "image_gen_provider", "agent.image_gen_registry",
     "agent.image_gen_provider:ImageGenProvider", "image_gen provider",
     "Register an :class:`agent.image_gen_provider.ImageGenProvider`; "
     "``provider.name`` is matched by ``image_gen.provider``.", {"article": "an"}),
    ("register_video_gen_provider", "video_gen_provider", "agent.video_gen_registry",
     "agent.video_gen_provider:VideoGenProvider", "video_gen provider",
     "Register an :class:`agent.video_gen_provider.VideoGenProvider`; "
     "``provider.name`` is matched by ``video_gen.provider``.", {}),
    ("register_web_search_provider", "web_search_provider", "agent.web_search_registry",
     "agent.web_search_provider:WebSearchProvider", "web provider",
     "Register an :class:`agent.web_search_provider.WebSearchProvider`; "
     "``provider.name`` is matched by ``web.search_backend`` / ``web.extract_backend`` / ``web.backend``.",
     {}),
    ("register_browser_provider", "browser_provider", "agent.browser_registry",
     "agent.browser_provider:BrowserProvider", "browser provider",
     "Register an :class:`agent.browser_provider.BrowserProvider`; "
     "``provider.name`` is matched by ``browser.cloud_provider`` (consulted by "
     "``tools.browser_tool_cloud._get_cloud_provider``).", {}),
    ("register_terminal_environment_provider", "terminal_environment_provider",
     "agent.terminal_env_registry", "agent.terminal_env_provider:TerminalEnvironmentProvider",
     "terminal environment provider",
     "Register a :class:`agent.terminal_env_provider.TerminalEnvironmentProvider`; ``provider.name`` "
     "is matched by ``terminal.backend`` when no built-in backend has that name. Built-in names (local, "
     "docker, singularity, modal, daytona, vercel_sandbox, ssh) are rejected — plugins never shadow "
     "in-tree backends.",
     {"normalize": "lower", "reject_message": "Plugin '%s' terminal environment provider rejected: %s"}),
    ("register_secret_source", "secret_source", "agent.secret_sources.registry",
     "agent.secret_sources.base:SecretSource", "secret source",
     "Register a :class:`agent.secret_sources.base.SecretSource`, run by ``load_hermes_dotenv()`` "
     "(after ``~/.hermes/.env``, before credentials are read) when ``secrets.<name>`` is enabled. The "
     "orchestrator owns ordering/precedence/provenance; the source only fetches. Since dotenv usually "
     "loads before discovery, the manager re-pulls enabled plugin sources afterwards.",
     {"normalize": None, "register": "register_source", "param": "source"}),
    ("register_tts_provider", "tts_provider", "agent.tts_registry",
     "agent.tts_provider:TTSProvider", "TTS provider",
     "Register an :class:`agent.tts_provider.TTSProvider`; ``provider.name`` is matched by "
     "``tts.provider`` unless it is a built-in name (rejected with a warning) or a "
     "``tts.providers.<name>: type: command`` entry shares it (command-providers win).",
     {"normalize": "lower"}),
    ("register_transcription_provider", "transcription_provider", "agent.transcription_registry",
     "agent.transcription_provider:TranscriptionProvider", "transcription provider",
     "Register an :class:`agent.transcription_provider.TranscriptionProvider`; ``provider.name`` is "
     "matched by ``stt.provider`` unless it is a built-in name (rejected) or a ``stt.providers.<name>: "
     "type: command`` entry shares it (command-providers win).", {"normalize": "lower"}),
)

_NAME_NORMALIZERS: Dict[Optional[str], Optional[Callable[[str], str]]] = {
    "strip": lambda n: n.strip(), "lower": lambda n: n.strip().lower(), None: None,
}


def _make_scoped_provider_registrar(method_name, kind, registry_mod, base_ref, label, doc, options):
    """Build one ``register_<category>_provider`` method from a ``_SCOPED_PROVIDER_REGISTRARS`` row."""
    base_mod, base_attr = base_ref.split(":")
    normalize_fn = _NAME_NORMALIZERS[options.get("normalize", "strip")]
    register_name = options.get("register")

    def register(self, provider) -> Optional[PluginRegistration]:
        registry = importlib.import_module(registry_mod)
        return self._register_scoped_provider(
            provider, kind=kind, base_class=getattr(importlib.import_module(base_mod), base_attr),
            registry=registry, label=label, article=options.get("article", "a"), normalize=normalize_fn,
            register=getattr(registry, register_name) if register_name else None,
            reject_message=options.get("reject_message"),
        )

    def register_source(self, source) -> Optional[PluginRegistration]:  # secret sources: ``source``
        return register(self, source)

    method = register_source if options.get("param") == "source" else register
    method.__name__, method.__qualname__, method.__doc__ = method_name, f"PluginContext.{method_name}", doc
    return _serialized_replacement(method)


for _row in _SCOPED_PROVIDER_REGISTRARS:
    setattr(PluginContext, _row[0], _make_scoped_provider_registrar(*_row))
del _row


def _resolve_hook_callback_timeout() -> float:
    """Effective hook-callback timeout from ``plugins.hook_callback_timeout`` (default 30s; ``<= 0``
    disables the threaded path; clamped to ``_MAX_HOOK_CALLBACK_TIMEOUT_SECS``)."""
    default = _HOOK_CALLBACK_TIMEOUT_SECS
    try:
        from hermes_cli.config import load_config_readonly
        plugins_cfg = (load_config_readonly() or {}).get("plugins")
        if not isinstance(plugins_cfg, dict) or plugins_cfg.get("hook_callback_timeout") is None:
            return default
        timeout = float(plugins_cfg["hook_callback_timeout"])
    except (TypeError, ValueError):
        logger.warning("plugins.hook_callback_timeout is not a number; using default %gs", default)
        return default
    except Exception:
        return default
    if timeout < 0:
        logger.warning("plugins.hook_callback_timeout=%g is negative; using default %gs", timeout, default)
        return default
    if timeout > _MAX_HOOK_CALLBACK_TIMEOUT_SECS:
        logger.warning("plugins.hook_callback_timeout=%g exceeds max %gs; clamping", timeout,
                       _MAX_HOOK_CALLBACK_TIMEOUT_SECS)
        return _MAX_HOOK_CALLBACK_TIMEOUT_SECS
    return timeout


class PluginManager(PluginLoaderMixin, PluginDispatchMixin, PluginLedgerMixin):
    """Central manager that discovers, loads, and invokes plugins."""

    def __init__(self, scope_key: Optional[str] = None) -> None:
        # Home is captured immutably: unload may run from another profile context, but every
        # inverse must target the registration's original scope.
        self.scope_key = scope_key or hermes_home_key()
        self.home_path = Path(self.scope_key)
        self._discovery_lock = threading.RLock()
        self._discovered: bool = False
        self._cli_ref = None  # Set by CLI after plugin discovery
        self._gateway_message_injector: tuple[object, Callable] | None = None
        self._context_engine = None  # Set by a plugin via register_context_engine()
        # Manager-local registries keyed by name (see the matching ``PluginContext.register_*``):
        # plugins, hooks, middleware, CLI + slash commands, prompt sections, skills (qualified name ->
        # metadata), portable MCP servers, auxiliary tasks, approval transports, Slack action handlers
        # (matcher, callback, plugin_name), platform handler factories (lowercase platform -> list).
        self._plugins: Dict[str, LoadedPlugin] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        # Fallback hooks registered by a memory provider before general discovery.
        self._memory_hook_registrations: Dict[Tuple[str, str], List[PluginRegistration]] = {}
        self._middleware: Dict[str, List[Callable]] = {}
        self._plugin_tool_names: Set[str] = set()
        self._plugin_platform_names: Set[str] = set()
        self._cli_commands: Dict[str, dict] = {}
        self._plugin_commands: Dict[str, dict] = {}
        self._system_prompt_sections: Dict[str, PluginSystemPromptSection] = {}
        self._plugin_skills: Dict[str, Dict[str, Any]] = {}
        self._portable_mcp_servers: Dict[str, Dict[str, Any]] = {}
        self._aux_tasks: Dict[str, Dict[str, Any]] = {}
        self._approval_transports: Dict[str, Any] = {}
        self._slack_action_handlers: List[tuple] = []
        self._platform_handler_factories: Dict[str, List[tuple]] = {}
        # Event bus: owner-tagged subscriptions (unload removes zombies); one daemon worker keeps
        # registration order while emitters never block; per-worker chain depth caps mutual emitters.
        self._subscriptions: Dict[str, List[_EventSubscription]] = {}
        self._event_lock = threading.RLock()
        self._event_idle = threading.Condition(self._event_lock)
        self._event_generation = 0
        self._event_pending_by_generation: Dict[int, int] = {0: 0}
        self._event_queue: queue.Queue[Any] = queue.Queue(maxsize=_EVENT_PENDING_CAP)
        self._event_worker: Optional[threading.Thread] = None
        self._emit_depth = threading.local()
        # In-flight / recently-timed-out hook callbacks keyed by (hook_name, id(cb), call_identity)
        # so a stuck policy hook cannot spawn a new abandoned thread on every fire.
        self._hook_running_callbacks: Dict[tuple, object] = {}
        self._hook_abandoned: Dict[tuple, set] = {}
        self._hook_timeout_suppressed_until: Dict[tuple, float] = {}
        self._hook_timeout_lock = threading.Lock()
        self._hook_timeout_suppression_seconds = _HOOK_TIMEOUT_SUPPRESSION_SECONDS
        # Ledger per plugin (ownership) plus global order (reverse teardown across plugins). Process-
        # global registries are shared across profiles while several managers coexist, so the ledger
        # is keyed per (hermes_home, plugin_id) and every inverse is identity-conditional — one
        # profile's unload can never clear another's. Persistent registrations that survived an
        # unload-all park in ``_persistent_carryover`` until force re-discovery evicts the stale ones.
        # Registration handles are kept both per plugin (ownership lookup) and globally (reverse-order
        # teardown for overrides spanning plugins). Registry overlays keyed by scope_key (see
        # tools/registry.py and gateway/platform_registry.py) carry the profile dimension; anything still
        # process-global is guarded by the identity checks. TODO(#64178): extend explicit profile keying to
        # any remaining process-global slots when the symmetric force-reload lands.
        self._ownership_ledger: Dict[str, List[PluginRegistration]] = {}
        self._registration_order: List[PluginRegistration] = []
        # Persistent (process-global) registrations that survived an
        # unload-all. Force re-discovery drains this via
        # _evict_stale_persistent_registrations(): entries whose plugin
        # re-registered the same (kind, key) are kept (the upsert rotated
        # them in place), the rest are disposed so a disabled/removed auth
        # plugin's provider does not outlive its plugin (#91701 follow-up).
        self._persistent_carryover: List[PluginRegistration] = []
        # Deferred platform plugins whose client tools were registered at
        # discovery time (see _register_deferred_platform_tools). Keyed by
        # plugin id: the already-imported package module, so materializing the
        # adapter later doesn't re-execute it, and the tool names it
        # contributed, so `hermes plugins list` still attributes them once the
        # full plugin loads.
        self._predeclared_modules: Dict[str, types.ModuleType] = {}
        self._predeclared_tools: Dict[str, List[str]] = {}

    # -----------------------------------------------------------------------
    # Registration ledger internals
    # -----------------------------------------------------------------------

    def _track_registration(
        self,
        manifest: PluginManifest,
        kind: str,
        key: str,
        release: Callable[[], None],
        *,
        persistent: bool = False,
    ) -> PluginRegistration:
        """Record one successful registration under its canonical plugin key.

        ``persistent`` registrations (process-global host infrastructure such
        as dashboard-auth providers, whose lifetime is the server rather than
        a per-home plugin manager) are still tracked in the ownership ledger
        for attribution, but are NOT enrolled in ``_registration_order`` — so
        a routine per-home manager unload cannot dispose them (#91701). The
        returned handle still releases on explicit ``dispose()``.
        """
        plugin_key = manifest.key or manifest.name
        registration = PluginRegistration(
            kind=kind,
            key=key,
            release=release,
            plugin_key=plugin_key,
            persistent=persistent,
        )
        registration._on_dispose = lambda disposed: self._forget_registrations(
            [disposed]
        )
        self._ownership_ledger.setdefault(plugin_key, []).append(registration)
        if not persistent:
            self._registration_order.append(registration)
        return registration

    def _evict_stale_persistent_registrations(self) -> None:
        """Dispose carried-over persistent registrations not re-registered.

        Persistent registrations (process-global host infrastructure such as
        dashboard-auth providers) survive an unload-all by design (#91701);
        ``_unload_scoped`` parks their handles in ``_persistent_carryover``.
        After a re-discovery pass, three cases exist for each parked handle:

        - the plugin re-registered the same ``(kind, key)`` → the upsert
          rotated the entry in place. The old handle is superseded; drop it
          WITHOUT disposing (a plugin that re-registered the *same object*
          would otherwise pass the identity check and evict the live entry).
        - the plugin did not come back (disabled, uninstalled, omitted) →
          dispose, releasing the process-global registration.
        - the handle was already disposed elsewhere (targeted unload) → drop.
        """
        if not self._persistent_carryover:
            return
        parked = self._persistent_carryover
        self._persistent_carryover = []
        current = {
            (registration.kind, registration.key)
            for owned in self._ownership_ledger.values()
            for registration in owned
            if registration.persistent and registration.active
        }
        stale = [
            registration
            for registration in parked
            if registration.active
            and (registration.kind, registration.key) not in current
        ]
        for registration in stale:
            logger.info(
                "Evicting persistent registration %s/%s: plugin '%s' no "
                "longer supplies it after re-discovery",
                registration.kind,
                registration.key,
                registration.plugin_key,
            )
        self._dispose_registrations(stale)

    @staticmethod
    def _remove_identity(values: list, target: Any) -> bool:
        """Remove the last exact object match from a registration list."""
        for index in range(len(values) - 1, -1, -1):
            if values[index] is target:
                del values[index]
                return True
        return False

    def _remove_callback(
        self,
        mapping: Dict[str, List[Callable]],
        key: str,
        callback: Callable,
    ) -> None:
        callbacks = mapping.get(key)
        if callbacks is None:
            return
        self._remove_identity(callbacks, callback)
        if not callbacks:
            mapping.pop(key, None)

    def _restore_mapping(
        self,
        mapping: Dict[str, Any],
        key: str,
        current: Any,
        previous: Optional[Any],
    ) -> bool:
        """Restore a manager-local mapping only when *current* is still present."""
        if mapping.get(key) is not current:
            return False
        if previous is None:
            mapping.pop(key, None)
        else:
            mapping[key] = previous
        return True

    def _restore_value(
        self,
        attribute: str,
        current: Any,
        previous: Any,
    ) -> bool:
        """Restore a manager-local value only when *current* is still active."""
        if getattr(self, attribute) is not current:
            return False
        setattr(self, attribute, previous)
        return True

    def _remove_tool_name_if_unowned(self, name: str) -> None:
        if not any(
            registration.active
            and registration.kind == "tool"
            and registration.key == name
            for registration in self._registration_order
        ):
            self._plugin_tool_names.discard(name)

    def _remove_platform_name_if_unowned(self, name: str) -> None:
        if not any(
            registration.active
            and registration.kind == "platform"
            and registration.key == name
            for registration in self._registration_order
        ):
            self._plugin_platform_names.discard(name)

    def _forget_registrations(
        self,
        registrations: List[PluginRegistration],
    ) -> None:
        if not registrations:
            return
        registration_ids = {id(registration) for registration in registrations}
        self._registration_order = [
            registration
            for registration in self._registration_order
            if id(registration) not in registration_ids
        ]
        for plugin_key, owned in list(self._ownership_ledger.items()):
            remaining = [
                registration
                for registration in owned
                if id(registration) not in registration_ids
            ]
            if remaining:
                self._ownership_ledger[plugin_key] = remaining
            else:
                self._ownership_ledger.pop(plugin_key, None)

    def _dispose_registrations(
        self,
        registrations: List[PluginRegistration],
    ) -> None:
        """Dispose registrations in reverse acquisition order, best effort."""
        for registration in reversed(registrations):
            try:
                registration.dispose()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.warning(
                    "Failed to unload plugin registration %s/%s: %s",
                    registration.plugin_key,
                    registration.key,
                    exc,
                    exc_info=_PLUGINS_DEBUG,
                )

    @staticmethod
    def _resolve_plugin_key(
        plugin: Union[str, PluginManifest, LoadedPlugin],
    ) -> str:
        if isinstance(plugin, LoadedPlugin):
            return plugin.manifest.key or plugin.manifest.name
        if isinstance(plugin, PluginManifest):
            return plugin.key or plugin.name
        return str(plugin)

    def unload(
        self,
        plugin: Union[str, PluginManifest, LoadedPlugin, None] = None,
    ) -> bool:
        """Unload registrations while excluding discovery/deferred loading."""
        with self._discovery_lock, _plugin_home_scope(self.home_path):
            return self._unload_scoped(plugin)

    def _unload_scoped(
        self,
        plugin: Union[str, PluginManifest, LoadedPlugin, None] = None,
    ) -> bool:
        """Unload one plugin or all plugins owned by this manager.

        Every registration made through :class:`PluginContext` is disposed in
        reverse acquisition order.  Registry inverses are conditional on the
        exact object still being current, so a later registration is never
        removed accidentally.  ``plugin=None`` is the lifecycle operation
        used by force rediscovery.  ``on_unload`` callbacks and supervised
        background tasks registered through :class:`PluginContext` are
        disposed through the same reverse-order ledger walk.

        Returns ``True`` when at least one plugin or registration was found.
        """
        unload_all = plugin is None
        if unload_all:
            target_keys = set(self._ownership_ledger) | set(self._plugins)
            registrations = list(self._registration_order)
        else:
            requested = self._resolve_plugin_key(plugin)
            exact = {
                requested,
            } if requested in self._ownership_ledger or requested in self._plugins else set()
            if exact:
                target_keys = exact
            else:
                target_keys = {
                    key
                    for key, loaded in self._plugins.items()
                    if loaded.manifest.name == requested
                }
                target_keys.update(
                    key
                    for key in self._ownership_ledger
                    if key == requested
                )
            registrations = [
                registration
                for registration in self._registration_order
                if registration.plugin_key in target_keys
            ]
            # Persistent registrations (process-global host infrastructure,
            # e.g. dashboard-auth providers) are deliberately absent from
            # _registration_order so an unload-all cannot dispose them
            # (#91701). A *targeted* unload is different: it is the plugin
            # disable/uninstall path, and a disabled auth plugin's provider
            # must NOT stay live process-wide. Gather them from the ledger.
            registrations.extend(
                registration
                for key in target_keys
                for registration in self._ownership_ledger.get(key, [])
                if registration.persistent and registration.active
            )

        found = bool(target_keys or registrations)
        self._dispose_registrations(registrations)
        self._forget_registrations(registrations)

        if unload_all:
            # The handles are authoritative for global registries, while the
            # manager-local containers are also reset to clear legacy/manual
            # state that predates the ledger.
            #
            # Platform names may exist in _plugin_platform_names without a
            # ledger entry (state predating the ledger, or set manually in
            # long-lived processes). Main's force path always unregistered
            # them from the global registry — keep that sweep so disabled
            # plugins can't leak parsers/send handlers into the next
            # discovery pass.
            from gateway.platform_registry import platform_registry

            for platform_name in tuple(self._plugin_platform_names):
                platform_registry.unregister(platform_name)
            # Symmetric sweep for tools: names in _plugin_tool_names that no
            # ledger registration covers (pre-ledger state, or set manually)
            # would survive in the process-global tools.registry as zombie
            # entries after a force reload (#60050; tracking #64178 —
            # extracted from PR #64188). Ledger-owned names are excluded:
            # their handles were already disposed above with precise
            # previous-entry restoration, and blanket deregistration here
            # would remove entries the ledger just restored.
            ledger_tool_names = {
                registration.key
                for registration in registrations
                if registration.kind == "tool"
            }
            preledger_tools = tuple(
                name
                for name in self._plugin_tool_names
                if name not in ledger_tool_names
            )
            if preledger_tools:
                try:
                    from tools.registry import registry as tool_registry
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("unload: tools.registry unavailable: %s", exc)
                else:
                    for tool_name in preledger_tools:
                        try:
                            tool_registry.deregister(tool_name)
                        except Exception as exc:
                            logger.debug(
                                "unload: tool deregister %s failed: %s",
                                tool_name,
                                exc,
                            )
            # Persistent registrations survive this unload-all by design
            # (#91701) but must not be orphaned by the ledger clear below:
            # carry them over so a force re-discovery can evict the ones
            # whose plugin does not come back (disabled/removed/omitted).
            carryover_ids = {
                id(registration) for registration in self._persistent_carryover
            }
            self._persistent_carryover.extend(
                registration
                for owned in self._ownership_ledger.values()
                for registration in owned
                if registration.persistent
                and registration.active
                and id(registration) not in carryover_ids
            )
            self._ownership_ledger.clear()
            self._plugins.clear()
            self._hooks.clear()
            self._middleware.clear()
            self._plugin_tool_names.clear()
            self._plugin_platform_names.clear()
            self._cli_commands.clear()
            self._plugin_commands.clear()
            self._plugin_skills.clear()
            self._portable_mcp_servers.clear()
            self._aux_tasks.clear()
            self._system_prompt_sections.clear()
            self._approval_transports.clear()
            self._slack_action_handlers.clear()
            self._predeclared_modules.clear()
            self._predeclared_tools.clear()
            self._context_engine = None
            self._discovered = False
        else:
            for key in target_keys:
                self._plugins.pop(key, None)

        return found

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    @property
    def has_gateway_message_injector(self) -> bool:
        """Return whether a live gateway can accept plugin-triggered turns."""
        return self._gateway_message_injector is not None

    def set_gateway_message_injector(self, owner: object, injector: Callable[..., bool]) -> None:
        """Publish a live gateway injector and its lifecycle owner."""
        self._gateway_message_injector = (owner, injector)

    def clear_gateway_message_injector(self, owner: object) -> None:
        """Clear the injector only when it still belongs to ``owner``."""
        if self._gateway_message_injector is not None and self._gateway_message_injector[0] is owner:
            self._gateway_message_injector = None

    def inject_gateway_message(self, **kwargs: Any) -> bool:
        """Submit a plugin-triggered turn to the live gateway."""
        registered = self._gateway_message_injector
        return registered is not None and bool(registered[1](**kwargs))

    def discover_and_load(self, force: bool = False) -> None:
        """Scan all plugin sources and load each plugin found; ``force`` unloads first so config
        changes / new bundled backends become visible in long-lived sessions."""
        with self._discovery_lock, _plugin_home_scope(self.home_path):
            if self._discovered and not force:
                return
            if force:
                self.unload()  # the ledger owns teardown of process-global registries
            if env_var_enabled("HERMES_SAFE_MODE"):
                logger.info("HERMES_SAFE_MODE=1 — plugin discovery skipped")
                self._discovered = True
                return
            # Flag set up front as a re-entrancy guard (register() can trigger discovery again) but
            # reset on failure so a failed scan is NOT cached as "discovered with an empty registry"
            # — callers swallow the exception and would be stranded on the early return above.
            self._discovered = True
            try:
                self._discover_and_load_inner()
                # Persistent registrations deliberately survived the
                # unload-all above (#91701). Now that plugins have had their
                # chance to re-register, dispose the ones whose plugin did
                # not come back (disabled, removed, or omitted from this
                # discovery pass) so e.g. a disabled auth plugin's provider
                # does not stay live process-wide until restart.
                self._evict_stale_persistent_registrations()
                # Plugin secret sources register during discover; the initial
                # load_hermes_dotenv() already ran at import time. Re-pull so the
                # first process sees plugin backends (tracking #64177).
                self._refresh_secret_sources_after_discovery()
                if force:
                    # config.yaml shell hooks / outbound webhooks live in ``_hooks`` but are
                    # config-owned; unload() wiped them and cannot restore them.
                    # Re-register so force-reload is symmetric (#60036; tracking #64178 — salvaged from PR
                    # #64188; outbound webhooks added per #92682 review).
                    self._re_register_config_hooks_after_force()
            except BaseException:
                self._discovered = False
                raise

    def _re_register_config_hooks_after_force(self) -> None:
        """Restore config-owned shell hooks/outbound webhooks after a force clear; each guarded
        independently so one failing does not skip the other."""
        for label, module_name in (("shell-hook", "agent.shell_hooks"),
                                   ("outbound-webhook", "agent.outbound_webhooks")):
            try:
                importlib.import_module(module_name).re_register_config_hooks()
            except Exception as exc:
                logger.debug("force-reload %s re-register skipped: %s", label, exc)

    def _refresh_secret_sources_after_discovery(self) -> None:
        """If any plugin secret source is enabled (per its own ``is_enabled(cfg)``, honoring custom
        activation), reset the cache and re-apply. Fail-open: never raises into discover_and_load."""
        try:
            from agent.secret_sources.registry import list_plugin_sources
            from hermes_cli.env_loader import load_hermes_dotenv, reset_secret_source_cache
            plugin_sources = list_plugin_sources()
        except Exception:
            return
        if not plugin_sources:
            return
        try:
            from hermes_cli.config import load_config
            secrets = (load_config() or {}).get("secrets") or {}
        except Exception:
            secrets = {}

        def _enabled(source) -> bool:
            section = secrets.get(getattr(source, "name", ""))
            try:
                return bool(source.is_enabled(section if isinstance(section, dict) else {}))
            except Exception:
                return False  # mirrors the orchestrator: a raising is_enabled() is skipped

        enabled_names = [getattr(s, "name", "") for s in plugin_sources if _enabled(s)]
        if not enabled_names:
            return
        try:
            # Reset and reload the SAME home the process (or routed turn) resolves to: under multiplex this
            # runs at gateway boot after sibling profiles may already have hydrated, and a global clear
            # wiped their snapshots; a routed discovery must rebuild the profile it just dropped.
            from hermes_constants import get_hermes_home
            home = get_hermes_home()
            reset_secret_source_cache(home)
            load_hermes_dotenv(hermes_home=home)
            logger.debug("Re-applied secret sources after plugin discovery for: %s",
                         ", ".join(sorted(enabled_names)))
        except Exception as exc:
            logger.debug("secret source re-apply after discovery failed: %s", exc)

    def _discover_and_load_inner(self) -> None:
        """The actual discovery sweep — see :meth:`discover_and_load`."""
        manifests: List[PluginManifest] = self._collect_directory_manifests()
        # Entry points are separate from the directory scan: the startup MCP probe must not import
        # or register them.
        ep_manifests = self._scan_entry_points()
        logger.debug("  entrypoints: %d manifest(s)", len(ep_manifests))
        manifests.extend(ep_manifests)
        disabled = _get_disabled_plugins()
        enabled = _get_enabled_plugins()  # None = opt-in default (nothing enabled)
        stale_relay_keys = legacy_relay_plugin_keys(enabled)
        if stale_relay_keys:
            logger.warning(
                "Removed Hermes plugin %s is still listed in plugins.enabled; "
                "remove it and configure native Relay plugins with %s",
                ", ".join(stale_relay_keys),
                RELAY_PLUGINS_CONFIG_ENV,
            )
        winners: Dict[str, PluginManifest] = {}
        for manifest in manifests:
            winners[manifest.key or manifest.name] = manifest
        # Standalone/user plugins that pass the gates below are collected
        # here and loaded AFTER the sweep in dependency-respecting order
        # (requires_plugins topological sort, #64165).
        to_load: Dict[str, PluginManifest] = {}
        for manifest in winners.values():
            lookup_key = manifest.key or manifest.name

            # Relay lifecycle ownership now lives in the Hermes core. Loading
            # an old user or entry-point copy would let plugin.initialize()
            # compete for the same process-global Relay registries.
            if (
                lookup_key in LEGACY_RELAY_PLUGIN_KEYS
                or manifest.name in LEGACY_RELAY_PLUGIN_KEYS
            ):
                loaded = LoadedPlugin(manifest=manifest, enabled=False)
                loaded.error = (
                    "removed — Relay lifecycle is owned by Hermes core; configure "
                    f"{RELAY_PLUGINS_CONFIG_ENV} instead"
                )
                self._plugins[lookup_key] = loaded
                logger.warning(
                    "Refusing to load removed Hermes Relay plugin '%s'; %s",
                    lookup_key,
                    loaded.error,
                )
                continue

            # Explicit disable always wins (matches on key or on legacy
            # bare name for back-compat with existing user configs).
            if lookup_key in disabled or manifest.name in disabled:
                loaded = LoadedPlugin(manifest=manifest, enabled=False)
                loaded.error = "disabled via config"
                self._plugins[lookup_key] = loaded
                logger.debug("Skipping disabled plugin '%s'", lookup_key)
                continue

            # Exclusive plugins (memory providers) have their own
            # discovery/activation path. The general loader records the
            # manifest for introspection but does not load the module.
            if manifest.kind == "exclusive":
                loaded = LoadedPlugin(manifest=manifest, enabled=False)
                loaded.error = (
                    "exclusive plugin — activate via <category>.provider config"
                )
                self._plugins[lookup_key] = loaded
                logger.debug(
                    "Skipping '%s' (exclusive, handled by category discovery)",
                    lookup_key,
                )
                continue

            # Model provider plugins are loaded by providers/__init__.py
            # (its own lazy discovery keyed off first get_provider_profile()
            # call). We record the manifest here for introspection but do
            # not import the module — a second import would create two
            # ProviderProfile instances and break the "last writer wins"
            # override semantics between bundled and user plugins.
            if manifest.kind == "model-provider":
                loaded = LoadedPlugin(manifest=manifest, enabled=True)
                self._plugins[lookup_key] = loaded
                logger.debug(
                    "Skipping '%s' (model-provider, handled by providers/ discovery)",
                    lookup_key,
                )
                continue

            # Built-in backends auto-load — they ship with hermes and must
            # just work. Selection among them (e.g. which image_gen backend
            # services calls) is driven by ``<category>.provider`` config,
            # enforced by the tool wrapper.
            if manifest.source == "bundled" and manifest.kind == "backend":
                self._load_plugin(manifest)
                continue

            # Bundled platform plugins (gateway adapters: telegram, discord,
            # feishu, teams, ...) are registered LAZILY. Their modules import
            # heavy, platform-specific SDKs at module level (lark_oapi,
            # microsoft_teams, discord.py, slack_bolt, ...), so eagerly loading
            # all ~20 of them added several seconds to every `hermes`
            # invocation — including plain `hermes chat`, which never touches a
            # gateway platform. Instead we register a cheap deferred loader in
            # the platform_registry keyed on the platform name; the real module
            # is imported only when the gateway / cron / setup / send_message
            # path actually asks for that platform. Every platform Hermes ships
            # remains available out of the box — it just loads on first use.
            if manifest.source == "bundled" and manifest.kind == "platform":
                self._register_deferred_platform(manifest)
                continue

            # Everything else (standalone, user-installed backends,
            # entry-point plugins) is opt-in via plugins.enabled.
            # Accept both the path-derived key and the legacy bare name
            # so existing configs keep working.
            is_enabled = (
                enabled is not None
                and (lookup_key in enabled or manifest.name in enabled)
            )
            if not is_enabled:
                loaded = LoadedPlugin(manifest=manifest, enabled=False)
                loaded.error = (
                    "not enabled in config (run `hermes plugins enable {}` to activate)"
                    .format(lookup_key)
                )
                self._plugins[lookup_key] = loaded
                logger.debug(
                    "Skipping '%s' (not in plugins.enabled)", lookup_key
                )
                continue
            to_load[lookup_key] = manifest

        # Load the surviving standalone plugins in dependency order:
        # when A requires B, B's register() runs before A's (topological
        # sort, stable alphabetical tiebreak; cycles warn and fall back to
        # alphabetical order). Missing deps warn but never block the load.
        for lookup_key in resolve_plugin_load_order(to_load):
            manifest = to_load[lookup_key]
            self._warn_python_dependencies(manifest)
            self._validate_plugin_config_schema(manifest)
            self._load_plugin(manifest)
        if manifests:
            logger.info("Plugin discovery complete: %d found, %d enabled", len(self._plugins),
                        sum(1 for p in self._plugins.values() if p.enabled))
        self._refresh_plugin_compat_report(list(to_load.values()))

    def _refresh_plugin_compat_report(self, manifests: List[PluginManifest]) -> None:
        """Refresh HERMES_HOME/.plugin-compat-report.json from this discovery pass (hermes_cli.plugin_compat).

        The Desktop boot modal has no Python runtime of its own and reads that file after the ``serve``
        backend is up, so the scan must run wherever plugins are discovered — not only under the CLI
        banner / doctor / update, which never run inside the Desktop's backend. Fail-open: never raises.
        """
        try:
            from hermes_cli.plugin_compat import compat_report
            compat_report(manifests, force=True)
        except Exception as exc:
            logger.debug("plugin compat report refresh skipped: %s", exc)

    def _gate_manifest(
        self, manifest: PluginManifest, disabled: Set[str], enabled: Optional[Set[str]],
    ) -> bool:
        """Route one winning manifest per :func:`gate_manifest`: load now, defer, or record as
        skipped (introspection-only placeholder). Returns True only for plugins that go through the
        dependency-ordered load pass."""
        verdict = gate_manifest(manifest, disabled, enabled)
        if verdict.action == "load":
            return True
        if verdict.action == "load_now":
            self._load_plugin(manifest)
        elif verdict.action == "defer":
            self._register_deferred_platform(manifest)
        else:
            self._plugins[manifest_key(manifest)] = LoadedPlugin(
                manifest=manifest, enabled=verdict.enabled, error=verdict.error)
        if verdict.log:
            logger.log(*verdict.log)
        return False

    def register_approval_transport(self, name: str, present_fn: Callable, *, plugin_id: str) -> None:
        """Manager-level registration (public API kept for out-of-tree plugins); the PluginContext
        method is the tracked path plugins normally use. Same validation, no unload tracking."""
        from hermes_cli.approval_transport import RegisteredApprovalTransport
        clean = str(name).strip().lower()
        if clean == "builtin":
            raise ValueError("approval transport name 'builtin' is reserved")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", clean):
            raise ValueError("approval transport name must match [a-z0-9][a-z0-9_-]{0,63}")
        if not callable(present_fn):
            raise TypeError("approval transport present_fn must be callable")
        if clean in self._approval_transports:
            owner = self._approval_transports[clean].plugin_id
            raise ValueError(f"approval transport {clean!r} is already registered by {owner!r}")
        self._approval_transports[clean] = RegisteredApprovalTransport(
            name=clean, present=present_fn, plugin_id=plugin_id, profile_home=str(get_hermes_home().resolve()),
        )
        logger.info("Plugin %s registered approval transport: %s", plugin_id, clean)

    def get_approval_transport(self, name: str):
        """Return a transport only inside the profile that registered it."""
        registered = self._approval_transports.get(str(name).strip().lower())
        if registered is None or registered.profile_home != str(get_hermes_home().resolve()):
            return None
        return registered

    def _collect_directory_manifests(self) -> List[PluginManifest]:
        """Directory manifests in full-discovery order (see :func:`collect_directory_manifests`)."""
        return collect_directory_manifests()

    def has_enabled_portable_mcp(self, raw_config: Mapping[str, Any]) -> bool:
        """Probe enabled portable MCP packages without loading plugins (shares the full-discovery
        manifest collection so precedence/gating cannot diverge)."""
        if _env_enabled("HERMES_SAFE_MODE"):
            return False
        plugins_config = raw_config.get("plugins")
        if not isinstance(plugins_config, dict):
            return False

        def _names(value: Any) -> Set[str]:
            return {v for v in value if isinstance(v, str)} if isinstance(value, list) else set()

        enabled = _names(plugins_config.get("enabled"))
        disabled = _names(plugins_config.get("disabled", []))
        if not enabled:
            return False
        for lookup_key, manifest in {manifest_key(m): m for m in self._collect_directory_manifests()}.items():
            names = {lookup_key, manifest.name}
            if not manifest.portable or names & disabled or not names & enabled:
                continue
            try:  # lazy: this is a startup probe, keep agent_plugins unimported unless needed
                from hermes_cli.agent_plugins import _discover_mcp
                if _discover_mcp(Path(manifest.path), get_hermes_home() / "plugin-data"
                                 / (manifest.skill_namespace or lookup_key), [], create_data=False):
                    return True
            except (OSError, RuntimeError, ValueError):
                continue  # fail closed on an unreadable package; full discovery reports it
        return False

    def _scan_directory(
        self, path: Path, source: str, skip_names: Optional[Set[str]] = None,
    ) -> List[PluginManifest]:
        """Read manifests under *path* (see :func:`scan_directory`)."""
        return scan_directory(path, source, skip_names=skip_names)

    def _scan_entry_points(self) -> List[PluginManifest]:
        """Read installed plugin entry points (see :func:`discover_entrypoint_manifests`)."""
        return discover_entrypoint_manifests()

    def get_slack_action_handlers(self) -> List[tuple]:
        """``(action_id, callback, plugin_name)`` tuples for the Slack adapter to wire at connect."""
        return list(self._slack_action_handlers)

    def get_platform_handler_factories(self, platform: str) -> List[tuple]:
        """``(factory, plugin_name)`` tuples for one platform; adapters call ``factory(native,
        adapter)`` at connect (see :meth:`PluginContext.register_platform_handler`)."""
        return list(self._platform_handler_factories.get((platform or "").strip().lower(), []))

    def get_telegram_handler_factories(self) -> List[tuple]:
        """Back-compat alias for ``get_platform_handler_factories("telegram")``."""
        return self.get_platform_handler_factories("telegram")

    def list_plugins(self) -> List[Dict[str, Any]]:
        """Return a list of info dicts for all discovered plugins."""
        return [
            {
                "name": p.manifest.name, "key": manifest_key(p.manifest), "kind": p.manifest.kind,
                "version": p.manifest.version, "description": p.manifest.description,
                "source": p.manifest.source, "enabled": p.enabled, "tools": len(p.tools_registered),
                "hooks": len(p.hooks_registered), "middleware": len(p.middleware_registered),
                "commands": len(p.commands_registered), "error": p.error,
            } for _key, p in sorted(self._plugins.items())
        ]

    def find_plugin_skill(self, qualified_name: str) -> Optional[Path]:
        """Return the ``Path`` to a plugin skill's SKILL.md, or ``None``."""
        entry = self._plugin_skills.get(qualified_name)
        return entry["path"] if entry else None

    def list_plugin_skills(self, plugin_name: str) -> List[str]:
        """Return sorted bare names of all skills registered by *plugin_name*."""
        prefix = f"{plugin_name}:"
        return sorted(e["bare_name"] for qn, e in self._plugin_skills.items() if qn.startswith(prefix))

    def list_plugin_skill_metadata(self) -> List[Dict[str, Any]]:
        """Return progressive-disclosure metadata for registered plugin skills."""
        return [
            {
                "name": qualified, "description": str(entry.get("description", "")),
                "category": "plugin", "frontmatter": dict(entry.get("frontmatter", {})),
            } for qualified, entry in sorted(self._plugin_skills.items())
        ]

    def has_portable_mcp_servers(self) -> bool:
        return bool(self._portable_mcp_servers)

    def get_portable_mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """Return a defensive copy of enabled portable MCP server configs."""
        return {name: dict(config) for name, config in self._portable_mcp_servers.items()}

    def remove_plugin_skill(self, qualified_name: str) -> None:
        """Remove a stale registry entry (silently ignores missing keys)."""
        self._plugin_skills.pop(qualified_name, None)


# Module-level singleton & convenience functions.

# Legacy single-slot "current" manager, kept so tests that monkeypatch ``_plugin_manager`` keep
# working — ``get_plugin_manager()`` still reads/writes this name.
_plugin_manager: Optional[PluginManager] = None

# Resolved Hermes home -> PluginManager. A process can switch profiles via
# ``set_hermes_home_override()``; a single slot would leak one profile's plugin/context-engine state
# into another, and keying by resolved home lets a re-entered profile reuse its imported modules.
_plugin_managers_by_home: Dict[Path, PluginManager] = {}
_plugin_managers_lock = threading.RLock()


def _plugin_home_key() -> Path:
    """Resolved active Hermes home — the key for per-profile plugin managers (plugins capture the
    home at registration, so a process serving several profiles cannot share one manager)."""
    try:
        return get_hermes_home().expanduser().resolve()
    except Exception:
        return get_hermes_home().expanduser()


def _clear_plugin_submodules(manager: Optional[PluginManager]) -> None:
    """Purge ``sys.modules`` entries for this manager's directory plugins (package AND submodules —
    otherwise a same-slug plugin in another profile reuses the previous profile's submodule state).
    """
    if manager is None:
        return
    for loaded in getattr(manager, "_plugins", {}).values():  # tolerates test-double managers
        module_name = getattr(getattr(loaded, "module", None), "__name__", None)
        if not module_name or not module_name.startswith(f"{_NS_PARENT}."):
            continue
        _evict_modules(module_name)
        with _MODULE_NAMESPACE_LOCK:
            if _BARE_MODULE_SCOPE.get(module_name) == manager.scope_key:
                _BARE_MODULE_SCOPE.pop(module_name, None)


def get_plugin_manager() -> PluginManager:
    """Return the plugin manager for the active Hermes profile/home (cached per resolved home; a
    profile switch gets its own manager and plugin submodules)."""
    global _plugin_manager
    current_home = _plugin_home_key()
    with _plugin_managers_lock:
        # Tests/embedders monkeypatch ``_plugin_manager`` directly: adopt a single-slot manager the
        # keyed cache doesn't know about at all.
        if _plugin_manager is not None and _plugin_manager not in _plugin_managers_by_home.values():
            _plugin_managers_by_home[current_home] = _plugin_manager
            return _plugin_manager
        manager = _plugin_managers_by_home.get(current_home)
        if manager is None:
            manager = PluginManager(scope_key=hermes_home_key(current_home))
            _plugin_managers_by_home[current_home] = manager
        _plugin_manager = manager
        return manager


def _reset_plugin_managers_for_tests() -> None:
    """Test-only: drop every cached manager and its submodules for a fully clean slate."""
    global _plugin_manager
    with _plugin_managers_lock:
        managers = list(dict.fromkeys(_plugin_managers_by_home.values()))
        if _plugin_manager is not None and _plugin_manager not in managers:
            managers.append(_plugin_manager)
        for manager in managers:
            _clear_plugin_submodules(manager)
            try:
                manager.unload()
            except Exception:
                logger.debug("test plugin-manager unload failed", exc_info=True)
        _plugin_managers_by_home.clear()
        _plugin_manager = None
    # Dashboard-auth providers are persistent host-owned registrations that
    # deliberately survive a routine manager unload (#91701), so the "clean
    # slate" reset must drop the process-global auth registry explicitly —
    # otherwise a provider auto-registered during one test leaks into the next.
    try:
        from hermes_cli.dashboard_auth.registry import (
            clear_providers as _clear_dashboard_auth_providers,
        )

        _clear_dashboard_auth_providers()
    except Exception:
        logger.debug("dashboard-auth registry clear failed", exc_info=True)


def has_enabled_agent_plugin_mcp(raw_config: Mapping[str, Any]) -> bool:
    """Whether config enables a portable package with MCP servers (manifest-only scan on a fresh
    manager; imports nothing, mutates no registry)."""
    return PluginManager().has_enabled_portable_mcp(raw_config)


def discover_plugins(force: bool = False) -> None:
    """Discover and load all plugins (idempotent; ``force=True`` rescans). Joins an in-flight
    background discovery instead of racing a second scan."""
    _join_background_discovery()
    get_plugin_manager().discover_and_load(force=force)


_background_discovery_thread: Optional[threading.Thread] = None
_background_discovery_lock = threading.Lock()


def start_background_plugin_discovery() -> None:
    """Run discovery in a daemon thread to overlap the rest of CLI startup (~150ms). Every
    synchronous consumer joins it via :func:`discover_plugins`, so no one sees a half-loaded
    registry. No-op when already done or in flight."""
    global _background_discovery_thread
    manager = get_plugin_manager()
    if manager._discovered:
        return
    with _background_discovery_lock:
        if _background_discovery_thread is not None and _background_discovery_thread.is_alive():
            return

        def _run() -> None:
            try:
                manager.discover_and_load()
                _persist_plugin_toolset_keys()
            except Exception:
                logger.warning("background plugin discovery failed", exc_info=True)

        _background_discovery_thread = threading.Thread(target=_run, name="plugin-discovery", daemon=True)
        _background_discovery_thread.start()


def _join_background_discovery(timeout: float = 30.0) -> None:
    """Wait for an in-flight background discovery (no-op from its own thread)."""
    t = _background_discovery_thread
    if t is None or not t.is_alive() or t is threading.current_thread():
        return
    t.join(timeout=timeout)


def _plugin_toolset_keys_cache_path() -> Path:
    return get_hermes_home() / "cache" / "plugin_toolset_keys.json"


def _persist_plugin_toolset_keys() -> None:
    """Persist discovered plugin toolset keys + portable MCP names (best-effort)."""
    try:
        from utils import atomic_json_write
        keys = sorted({ts_key for ts_key, _, _ in get_plugin_toolsets()})
        try:
            portable = sorted(get_plugin_manager().get_portable_mcp_servers())
        except Exception:
            portable = []
        atomic_json_write(_plugin_toolset_keys_cache_path(), {"toolset_keys": keys, "portable_mcp": portable}, indent=None, mode=0o600)
    except Exception:
        logger.debug("plugin toolset key persist failed", exc_info=True)


def _nowait_plugin_set(cache_field: str, live: Callable[[PluginManager], "set[str]"]) -> "set[str]":
    """Shared body of the ``*_nowait`` probes: live registry, else last launch's cache, else block."""
    manager = get_plugin_manager()
    t = _background_discovery_thread
    in_flight = t is not None and t.is_alive()
    if manager._discovered and not in_flight:
        return live(manager)
    if in_flight:
        with suppress(Exception):
            blob = json.loads(_plugin_toolset_keys_cache_path().read_text(encoding="utf-8"))
            values = blob.get(cache_field) if isinstance(blob, dict) else None
            if isinstance(values, list) and all(isinstance(v, str) for v in values):
                return set(values)
    discover_plugins()
    return live(manager)


def get_plugin_toolset_keys_nowait() -> "set[str]":
    """Plugin toolset keys without blocking on in-flight discovery: live registry when done, last
    launch's persisted set while a background scan runs (callers only EXCLUDE these keys, so a stale
    set is harmless and self-heals), else block via discover_plugins()."""
    return _nowait_plugin_set("toolset_keys", lambda _m: {ts_key for ts_key, _, _ in get_plugin_toolsets()})


def get_portable_mcp_server_names_nowait() -> "set[str]":
    """Portable MCP server names; same contract as :func:`get_plugin_toolset_keys_nowait`."""
    return _nowait_plugin_set("portable_mcp", lambda m: set(m.get_portable_mcp_servers()))


def _delivery_manager() -> PluginManager:
    """Active manager, lazily discovering if it never ran — delivery must not depend on WHICH
    surface imported us (dashboards/TUI/cron never import model_tools). ``getattr`` default
    ``True`` leaves test doubles untouched.

    Hook/middleware delivery must not depend on WHICH surface imported us: dashboards, TUI slash workers,
    query mode, and cron delivery paths never import ``model_tools`` (whose import side-effect is the
    discovery trigger on the interactive CLI path), so hooks registered by user plugins were silently dead
    on those surfaces (#50776, #67597, #67890, #50937; tracking #64178 — salvaged from PR #64188).
    """
    manager = get_plugin_manager()
    if not getattr(manager, "_discovered", True):
        _join_background_discovery()
        manager.discover_and_load()
    return manager


def invoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """Invoke a lifecycle hook (lazy-discovers first); return non-``None`` callback results.

    Hot-path / observer hooks in ``_HOOK_TIMEOUT_BOUNDED_HOOKS`` and the policy hook ``pre_tool_call`` are
    bounded by ``plugins.hook_callback_timeout`` (default 30s). On timeout the worker is abandoned (not
    joined) so we do not reintroduce the #6622 hang. Timed-out or still-running ``pre_tool_call`` callbacks
    fail closed with a block directive; other bounded hooks fail open (skip).
    Ensures plugins are discovered on first invocation so callers in processes that never explicitly call
    ``discover_plugins()`` (gateway platform events, TUI slash workers, query mode, cron) still fire
    callbacks registered by user plugins (tracking #64178).
    """
    return _delivery_manager().invoke_hook(hook_name, **kwargs)


def render_system_prompt_sections(session_info: Mapping[str, Any]) -> List[RenderedPluginSystemPromptSection]:
    """Render plugin prompt sections after idempotent plugin discovery."""
    return _ensure_plugins_discovered().render_system_prompt_sections(session_info)


def invoke_middleware(kind: str, **kwargs: Any) -> List[Any]:
    """Invoke registered middleware callbacks (lazy-discovers like :func:`invoke_hook`).

    Lazy-discovers plugins on first use — same delivery-parity guarantee as :func:`invoke_hook` (tracking
    #64178).
    """
    return _delivery_manager().invoke_middleware(kind, **kwargs)


def has_middleware(kind: str) -> bool:
    """True when middleware is registered for ``kind``; lazy-discovers first since callers gate
    :func:`invoke_middleware` on it.

    Lazy-discovers first: callers use this as a gate before :func:`invoke_middleware`, so a pre-discovery
    ``False`` here would silently skip delivery on surfaces that never ran discovery (#64178).
    """
    manager = _delivery_manager()
    method = getattr(manager, "has_middleware", None)
    if callable(method):
        return bool(method(kind))
    return bool(getattr(manager, "_middleware", {}).get(kind))


def has_hook(hook_name: str) -> bool:
    """True when a loaded plugin handles a hook (lazy-discovers first, like :func:`has_middleware`).

    Lazy-discovers first — same gate-before-invoke rationale as :func:`has_middleware` (tracking #64178).
    """
    return _delivery_manager().has_hook(hook_name)


def iter_hook_callbacks(hook_name: str) -> tuple[Callable, ...]:
    """Return a stable snapshot of callbacks registered for a hook."""
    return get_plugin_manager().iter_hook_callbacks(hook_name)


def fire_pre_command_hook(
    *, surface: str, command: str, alias_used: str, args_raw: str,
    session_key: Optional[str] = None, platform: Optional[str] = None,
) -> None:
    """Fire the observer-only ``pre_command`` hook; never raises. Directive-shaped returns are
    logged at debug so future block/rewrite adopters are discoverable."""
    try:
        manager = get_plugin_manager()
        if not manager.has_hook("pre_command"):
            return
        results = manager.invoke_hook(
            "pre_command", surface=surface, command=command, alias_used=alias_used,
            args_raw=args_raw, session_key=session_key, platform=platform,
        )
        for result in results:
            if isinstance(result, dict) and ("action" in result or "decision" in result):
                logger.debug("pre_command is observer-only in v1: ignoring directive %r for /%s (surface=%s). "
                             "Block/rewrite will arrive with the command middleware variant (#64204/#64231).",
                             result, command, surface)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("pre_command hook dispatch failed (non-fatal): %s", exc)


_thread_tool_whitelist = threading.local()


@dataclass(frozen=True)
class _PreToolCallDirective:
    action: Optional[str] = None
    message: Optional[str] = None
    rule_key: Optional[str] = None
    modified_args: Optional[Dict[str, Any]] = None


def set_thread_tool_whitelist(
    allowed: Optional[Set[str]],
    deny_msg_fmt: str = "Tool '{tool_name}' denied: not in this thread's tool whitelist",
) -> None:
    _thread_tool_whitelist.allowed = allowed
    _thread_tool_whitelist.fmt = deny_msg_fmt


def clear_thread_tool_whitelist() -> None:
    _thread_tool_whitelist.allowed = None


def _get_pre_tool_call_directive_details(
    tool_name: str, args: Optional[Dict[str, Any]], task_id: str = "", session_id: str = "",
    tool_call_id: str = "", turn_id: str = "", api_request_id: str = "",
    middleware_trace: Optional[List[Dict[str, Any]]] = None,
) -> _PreToolCallDirective:
    """Check ``pre_tool_call`` hooks for ``{"action": "block", "message"}`` (veto; message becomes
    the tool result) or ``{"action": "approve", "message", "rule_key"?}`` (escalate ANY tool to the
    human-approval gate; ``rule_key`` picks the ``[a]lways`` allowlist grain). First valid directive
    wins; irrelevant returns are ignored."""
    allowed = getattr(_thread_tool_whitelist, "allowed", None)
    if allowed is not None and tool_name not in allowed:
        fmt = getattr(_thread_tool_whitelist, "fmt", "Tool '{tool_name}' denied")
        return _PreToolCallDirective(action="block", message=fmt.format(tool_name=tool_name))
    from hermes_cli.lifecycle import invoke_hook as invoke_lifecycle_hook
    hook_results = invoke_lifecycle_hook(
        "pre_tool_call", tool_name=tool_name, args=args if isinstance(args, dict) else {},
        task_id=task_id, session_id=session_id, tool_call_id=tool_call_id, turn_id=turn_id,
        api_request_id=api_request_id, middleware_trace=list(middleware_trace or []),
    )
    modified_args: Optional[Dict[str, Any]] = None
    for result in hook_results:
        if not isinstance(result, dict):
            continue
        action = result.get("action")
        # "modify" — transform tool_input before dispatch. Processed before the block/approve gate
        # so modify directives are visible even when a later hook blocks. Each modify directive
        # shallow-merges its keys into one accumulated dict built from the original args.
        if action == "modify":
            partial = result.get("args")
            if isinstance(partial, dict) and partial:
                modified_args = {**(modified_args if modified_args is not None else
                                    (args if isinstance(args, dict) else {})), **partial}
            continue
        if action not in ("block", "approve"):
            continue
        message = result.get("message")
        message = message if isinstance(message, str) and message else None
        # A block directive requires a message (it becomes the tool result); approve's is optional.
        if action == "block" and not message:
            continue
        rule_key = result.get("rule_key") if action == "approve" else None
        rule_key = (rule_key.strip() or None) if isinstance(rule_key, str) else None
        return _PreToolCallDirective(action=action, message=message, rule_key=rule_key, modified_args=modified_args)
    return _PreToolCallDirective(modified_args=modified_args)


def get_pre_tool_call_directive(
    tool_name: str, args: Optional[Dict[str, Any]], **hook_kwargs: Any
) -> tuple[Optional[str], Optional[str]]:
    """Back-compat: ``(directive, message)`` with directive ``"block"`` / ``"approve"`` / ``None``.
    ``hook_kwargs`` are the observability ids of :func:`_get_pre_tool_call_directive_details`."""
    details = _get_pre_tool_call_directive_details(tool_name, args, **hook_kwargs)
    return (details.action, details.message)


def get_pre_tool_call_block_message(
    tool_name: str, args: Optional[Dict[str, Any]], **hook_kwargs: Any
) -> Optional[str]:
    """Deprecated shim: only the ``block`` message (or ``None``); ``approve`` is invisible here."""
    directive, message = get_pre_tool_call_directive(tool_name, args, **hook_kwargs)
    return message if directive == "block" else None


def resolve_pre_tool_block(
    tool_name: str, args: Optional[Dict[str, Any]], **hook_kwargs: Any
) -> Optional[str]:
    """Resolve the pre_tool_call directive to a final block message (or ``None`` to proceed),
    running the human-approval gate for ``approve``. See :func:`_resolve_block_from_details`."""
    return _dispatch_pre_tool_call_hooks(tool_name, args, **hook_kwargs)[0]


def _resolve_block_from_details(
    details: "_PreToolCallDirective", tool_name: str, *, turn_id: str = "", tool_call_id: str = "",
    session_id: str = "",
) -> Optional[str]:
    """The ONE place for the fail-closed approval logic: ``block`` blocks with its message; an
    ``approve`` whose gate errors, denies, or times out is blocked; anything else proceeds."""
    if details.action == "block":
        return details.message
    if details.action != "approve":
        return None
    try:
        from tools.approval import request_tool_approval
        from tools.approval_context import reset_current_observability_context, set_current_observability_context
        approval_tokens = None
        with suppress(Exception):
            approval_tokens = set_current_observability_context(
                turn_id=turn_id, tool_call_id=tool_call_id, session_id=session_id)
        try:
            result = request_tool_approval(tool_name, details.message or "", rule_key=details.rule_key or tool_name)
        finally:
            if approval_tokens is not None:
                with suppress(Exception):
                    reset_current_observability_context(approval_tokens)
    except Exception:
        # Fail-closed: if the gate itself errors, block rather than silently execute an action a
        # plugin flagged for approval.
        return f"BLOCKED: plugin approval gate failed for {tool_name}"
    if not result.get("approved"):
        return str(result.get("message") or f"BLOCKED: plugin approval required for {tool_name}")
    return None


def _dispatch_pre_tool_call_hooks(
    tool_name: str, args: Optional[Dict[str, Any]], **hook_kwargs: Any
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Invoke ``pre_tool_call`` hooks once; return ``(block_message, modified_args)`` — the resolved
    block/approve message (``None`` to proceed) and merged ``modify`` args (``None`` if none)."""
    details = _get_pre_tool_call_directive_details(tool_name, args, **hook_kwargs)
    block_msg = _resolve_block_from_details(
        details, tool_name, **{k: hook_kwargs.get(k, "") for k in ("turn_id", "tool_call_id", "session_id")})
    return (block_msg, details.modified_args)


def get_pre_verify_continue_message(
    *, session_id: str = "", platform: str = "", model: str = "", coding: bool = False,
    attempt: int = 0, final_response: str = "", changed_paths: Optional[List[str]] = None,
) -> Optional[str]:
    """Check ``pre_verify`` hooks for ``{"action": "continue", "message"}`` (or Claude-Code Stop
    ``{"decision": "block", "reason"}``) to keep the turn going; first non-empty message wins, any
    other return lets the turn finish. ``coding``/``attempt`` let hooks scope and self-throttle."""
    hook_results = invoke_hook(
        "pre_verify", session_id=session_id, platform=platform, model=model, coding=coding,
        attempt=attempt, final_response=final_response, changed_paths=list(changed_paths or []),
    )
    for result in hook_results:
        if not isinstance(result, dict):
            continue
        action = str(result.get("action") or result.get("decision") or "").strip().lower()
        message = result.get("message") or result.get("reason")
        if action in ("continue", "block") and isinstance(message, str) and message.strip():
            return message.strip()
    return None


def get_plugin_error_classification(
    *, provider: str = "", model: str = "", status_code: Optional[int] = None, error_type: str = "",
    error_code: str = "", error_message: str = "", error_body: Optional[Dict[str, Any]] = None,
    error: Optional[BaseException] = None, approx_tokens: int = 0, context_length: int = 0,
    num_messages: int = 0,
) -> Optional[Dict[str, Any]]:
    """Consult ``transform_api_error_classification`` hooks BEFORE the built-in classifier.
    Run-all-then-pick-first: the first valid result in registration order wins, losing valid results
    warn (conflicts visible, not shadowed). Returns a sanitized dict (``reason`` -> ``FailoverReason``,
    hint flags -> bool, ``message`` capped at 500) or ``None``. Privacy: inputs may be unredacted.

    A callback returns ``None`` to decline, or a dict with a required ``"reason"`` (a
    :class:`agent.error_classifier.FailoverReason` member or its string name) plus optional recovery-hint
    overrides. Dispatch is run-all-then-pick-first: ``invoke_hook`` runs every registered callback with
    failures isolated, then the first result carrying a valid reason wins in registration order — mirroring
    :func:`get_pre_tool_call_block_message`, invalid or irrelevant returns are silently ignored so a
    misbehaving plugin degrades to a no-op. When more than one callback returns a valid classification, the
    losing results are skipped with a runtime warning (the #64714 skipped-transform rule) so conflicting
    provider plugins are visible in logs instead of silently shadowed.
    """
    from agent.error_classifier import FailoverReason
    hook_results = invoke_hook(
        "transform_api_error_classification", provider=provider, model=model,
        status_code=status_code, error_type=error_type, error_code=error_code,
        error_message=error_message, error_body=error_body if isinstance(error_body, dict) else {},
        error=error, approx_tokens=approx_tokens, context_length=context_length,
        num_messages=num_messages,
    )

    def _reason(result: Any) -> Any:
        reason = result.get("reason") if isinstance(result, dict) else None
        if isinstance(reason, str):
            with suppress(ValueError):
                return FailoverReason(reason.strip().lower())
            return None
        return reason if isinstance(reason, FailoverReason) else None

    valid = [(result, reason) for result in hook_results if (reason := _reason(result)) is not None]
    if not valid:
        return None
    result, reason = valid[0]
    winner: Dict[str, Any] = {"reason": reason}
    for key in ("retryable", "should_compress", "should_rotate_credential", "should_fallback"):
        if key in result:
            winner[key] = bool(result[key])
    message = result.get("message")
    if isinstance(message, str) and message.strip():
        winner["message"] = message.strip()[:500]
    if isinstance(result.get("error_context"), dict):
        winner["error_context"] = result["error_context"]
    if len(valid) > 1:
        logger.warning("transform_api_error_classification: skipped %d valid classification(s) after the "
                       "first result in registration order won (run-all-then-pick-first)", len(valid) - 1)
    return winner


def _ensure_plugins_discovered(force: bool = False) -> PluginManager:
    """Return the global manager after idempotent (or ``force``d) discovery."""
    manager = get_plugin_manager()
    manager.discover_and_load(force=force)
    return manager


def get_plugin_context_engine():
    """Return the plugin-registered context engine, or None."""
    return _ensure_plugins_discovered()._context_engine


def get_plugin_command_handler(name: str) -> Optional[Callable]:
    """Return the handler for a plugin-registered slash command, or ``None``."""
    entry = _ensure_plugins_discovered()._plugin_commands.get(name)
    return entry["handler"] if entry else None


_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS = 30.0


def resolve_plugin_command_result(result: Any) -> Any:
    """Resolve a plugin command result, awaiting async handlers: ``asyncio.run`` when no loop is
    running, else a helper thread with its own loop (30s bound so a hung handler cannot wedge the
    terminal)."""
    if not inspect.isawaitable(result):
        return result
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(result)
    outcome: Dict[str, Any] = {}
    failure: Dict[str, BaseException] = {}
    done = threading.Event()

    def _runner() -> None:
        try:
            outcome["value"] = asyncio.run(result)
        except BaseException as exc:  # pragma: no cover - re-raised below
            failure["exc"] = exc
        finally:
            done.set()

    # copy_context: the helper thread must see the caller's profile/secret scope, else an
    # async hook under a running loop reads the default HERMES_HOME and get_secret raises.
    threading.Thread(target=contextvars.copy_context().run, args=(_runner,),
                     name="hermes-plugin-command-await", daemon=True).start()
    if not done.wait(timeout=_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS):
        raise TimeoutError("Plugin command async handler did not complete within "
                           f"{_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS:.0f}s")
    if "exc" in failure:
        raise failure["exc"]
    return outcome.get("value")


def get_plugin_commands() -> Dict[str, dict]:
    """Plugin commands dict (name -> {handler, description, plugin}) after idempotent discovery."""
    return _ensure_plugins_discovered()._plugin_commands


def get_plugin_auxiliary_tasks() -> List[Dict[str, Any]]:
    """Plugin auxiliary-task registration dicts sorted by ``key`` (after idempotent discovery)."""
    manager = _ensure_plugins_discovered()
    return [manager._aux_tasks[k] for k in sorted(manager._aux_tasks)]


def get_plugin_toolsets() -> List[tuple]:
    """Plugin toolsets as ``(key, label, description)`` tuples for the ``hermes tools`` TUI."""
    manager = get_plugin_manager()
    if not manager._plugin_tool_names:
        return []
    try:
        from tools.registry import registry
    except Exception:
        return []
    # Group plugin tool names by their toolset, then map each toolset back to the plugin that
    # registered it (first owner wins) for the description.
    toolset_tools: Dict[str, List[str]] = {}
    for tool_name in manager._plugin_tool_names:
        entry = registry.get_entry(tool_name)
        if entry:
            toolset_tools.setdefault(entry.toolset, []).append(entry.name)
    toolset_plugin: Dict[str, LoadedPlugin] = {}
    for loaded in manager._plugins.values():
        for tool_name in loaded.tools_registered:
            entry = registry.get_entry(tool_name)
            if entry and entry.toolset in toolset_tools:
                toolset_plugin.setdefault(entry.toolset, loaded)
    result = []
    for ts_key in sorted(toolset_tools):
        plugin = toolset_plugin.get(ts_key)
        desc = (plugin.manifest.description if plugin else "") or ", ".join(sorted(toolset_tools[ts_key]))
        result.append((ts_key, f"🔌 {ts_key.replace('_', ' ').title()}", desc))
    return result


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Iterable  # noqa: F401,E402
from typing import Type  # noqa: F401,E402
from contextlib import contextmanager  # noqa: F401,E402
import contextvars  # noqa: F401,E402
import copy  # noqa: F401,E402
import hashlib  # noqa: F401,E402
import time  # noqa: F401,E402
from functools import wraps  # noqa: F401,E402

def get_plugin_subscriptions() -> Dict[str, List[Callable]]:
    """Return the inter-plugin event bus subscription registry.

    Returns a snapshot mapping each fully-qualified event name
    (``<plugin_key>:<event>`` or ``hermes:<event>``) to subscriber callbacks in
    registration order. Owner ledger metadata stays private to the manager.
    Triggers idempotent plugin discovery before reading the snapshot.
    """
    manager = _ensure_plugins_discovered()
    with manager._event_lock:
        return {
            event: [entry.callback for entry in entries]
            for event, entries in manager._subscriptions.items()
        }

def unload_plugins(
    plugin: Union[str, PluginManifest, LoadedPlugin, None] = None,
) -> bool:
    """Unload one plugin or all plugins from the process-global manager.

    Wait for background discovery first so teardown cannot race an in-flight
    registration sweep introduced by the warm-start discovery path.
    """
    _join_background_discovery()
    return get_plugin_manager().unload(plugin)


_PLUGIN_COMPAT_LAZY = {
    'CAPABILITY_REGISTRY': ('hermes_cli.plugin_capabilities', 'CAPABILITY_REGISTRY'),
    'ENTRY_POINT_CAPABILITIES_GROUP': ('hermes_cli.plugins_discovery', 'ENTRY_POINT_CAPABILITIES_GROUP'),
    'LEGACY_RELAY_PLUGIN_KEYS': ('hermes_cli.relay_plugin_cutover', 'LEGACY_RELAY_PLUGIN_KEYS'),
    'MAX_SYSTEM_PROMPT_SECTIONS': ('hermes_cli.plugins_dispatch', 'MAX_SYSTEM_PROMPT_SECTIONS'),
    'OBSERVER_SCHEMA_VERSION': ('hermes_cli.middleware', 'OBSERVER_SCHEMA_VERSION'),
    'VALID_CAPABILITY_IDS': ('hermes_cli.plugin_capabilities', 'VALID_CAPABILITY_IDS'),
    'cfg_get': ('hermes_cli.config', 'cfg_get'),
    'fast_safe_load': ('utils', 'fast_safe_load'),
    'format_system_prompt_section': ('hermes_cli.plugins_dispatch', 'format_system_prompt_section'),
    'reset_hermes_home_override': ('hermes_constants', 'reset_hermes_home_override'),
    'set_hermes_home_override': ('hermes_constants', 'set_hermes_home_override'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
