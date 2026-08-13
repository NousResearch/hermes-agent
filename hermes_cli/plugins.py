"""
Hermes Plugin System
====================

Discovers, loads, and manages plugins from four sources:

1. **Bundled plugins** – ``<repo>/plugins/<name>/`` (shipped with hermes-agent;
   ``memory/`` and ``context_engine/`` subdirs are excluded — they have their
   own discovery paths)
2. **User plugins**   – ``~/.hermes/plugins/<name>/``
3. **Project plugins** – ``./.hermes/plugins/<name>/`` (opt-in via
   ``HERMES_ENABLE_PROJECT_PLUGINS``)
4. **Pip plugins**     – packages that expose the ``hermes_agent.plugins``
   entry-point group.

Later sources override earlier ones on name collision, so a user or project
plugin with the same name as a bundled plugin replaces it.

Each directory plugin must contain a ``plugin.yaml`` manifest **and** an
``__init__.py`` with a ``register(ctx)`` function.

Lifecycle hooks
---------------
Plugins may register callbacks for any of the hooks in ``VALID_HOOKS``.
The agent core calls ``invoke_hook(name, **kwargs)`` at the appropriate
points.

Tool registration
-----------------
``PluginContext.register_tool()`` delegates to ``tools.registry.register()``
so plugin-defined tools appear alongside the built-in tools.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import logging
import os
import queue
import re
import sys
import threading
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Union

from hermes_constants import (
    get_hermes_home,
    hermes_home_key,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from registration_lifecycle import replacement_coordinator
from utils import env_var_enabled, fast_safe_load
from hermes_cli.config import cfg_get, load_config_readonly
from hermes_cli.middleware import OBSERVER_SCHEMA_VERSION, VALID_MIDDLEWARE
from hermes_cli.plugin_capabilities import (  # noqa: F401 — re-exported
    CAPABILITY_REGISTRY,
    plugin_capability_granted,
)
from hermes_cli.plugin_capabilities import (
    parse_declared_capabilities as _parse_declared_capabilities,
)


def get_bundled_plugins_dir() -> Path:
    """Locate the bundled ``plugins/`` directory.

    Honours ``HERMES_BUNDLED_PLUGINS`` (set by the Nix wrapper / packaged
    installs) so read-only store paths are consulted first.  Falls back to
    the in-repo path used during development.
    """
    env_override = os.getenv("HERMES_BUNDLED_PLUGINS")
    if env_override:
        return Path(env_override)
    return Path(__file__).resolve().parent.parent / "plugins"

try:
    import yaml
except ImportError:  # pragma: no cover – yaml is optional at import time
    yaml = None  # type: ignore[assignment]


class PluginToolOverrideError(PermissionError):
    """Raised when a plugin attempts to override a built-in tool without
    operator opt-in via ``plugins.entries.<plugin_id>.allow_tool_override``.
    """


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plugin developer debug logging
# ---------------------------------------------------------------------------
#
# Set ``HERMES_PLUGINS_DEBUG=1`` to surface verbose plugin-discovery logs to
# stderr in addition to ~/.hermes/logs/agent.log. Aimed at plugin authors
# trying to figure out why their plugin isn't showing up: which directories
# were scanned, which manifests parsed, which plugins were skipped (and why),
# what each ``register(ctx)`` call registered, and full tracebacks on load
# failure.
#
# The env var is read once at import time; tests that need to flip it
# mid-process can call ``_install_plugin_debug_handler(force=True)``.

_PLUGINS_DEBUG = os.getenv("HERMES_PLUGINS_DEBUG", "").strip().lower() in {
    "1", "true", "yes", "on",
}
_DEBUG_HANDLER_INSTALLED = False


def _install_plugin_debug_handler(force: bool = False) -> None:
    """When HERMES_PLUGINS_DEBUG is on, tee plugin logs to stderr at DEBUG.

    Idempotent: only attaches the handler once per process unless ``force``
    is passed. Does not touch the root logger or other Hermes loggers.
    """
    global _DEBUG_HANDLER_INSTALLED, _PLUGINS_DEBUG
    if force:
        _PLUGINS_DEBUG = os.getenv("HERMES_PLUGINS_DEBUG", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
    if not _PLUGINS_DEBUG or _DEBUG_HANDLER_INSTALLED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("[plugins] %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    # Don't double-emit through the root logger when the central logging
    # config also writes to stderr. agent.log still captures everything.
    logger.propagate = True
    _DEBUG_HANDLER_INSTALLED = True
    logger.debug(
        "HERMES_PLUGINS_DEBUG=1 — verbose plugin discovery logging enabled"
    )


_install_plugin_debug_handler()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_HOOKS: Set[str] = {
    "pre_tool_call",
    "post_tool_call",
    "transform_terminal_output",
    "transform_tool_result",
    # Transform LLM output before it's returned to the user.
    # Plugins return a string to replace the response text, or None/empty to leave unchanged.
    # First non-None string wins. Useful for vocabulary/personality transformation.
    "transform_llm_output",
    "pre_llm_call",
    "post_llm_call",
    # Streaming LLM output observer hooks. Fired asynchronously off the token
    # path by agent.plugin_stream_hooks; callbacks observe immutable normalized
    # text/lifecycle payloads and cannot transform the stream.
    "on_stream_start",
    "on_stream_delta",
    "on_stream_end",
    "on_interim_message",
    # Verification-loop gate. Fired once per turn when the agent has edited code
    # and is about to verify/finish (after the verify-on-stop guard). A callback
    # may keep the agent going — run a check, defer it, tidy the diff — instead
    # of stopping by returning:
    #   {"action": "continue", "message": "<follow-up instruction>"}
    # The Claude-Code Stop shape {"decision": "block", "reason": "..."} (block
    # the stop == keep going) is accepted too. Anything else lets the turn
    # finish. Hermes' shipped guidance lives in the evidence-based
    # verification-stop nudge; this hook is for user/plugin policy and is
    # bounded by agent.max_verify_nudges.
    "pre_verify",
    "pre_api_request",
    "post_api_request",
    "api_request_error",
    "on_session_start",
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
    # Successful skill lifecycle facts. The local skill name is available to
    # plugins, while built-in shared metrics emit only bounded classifications.
    "on_skill_lifecycle",
    "subagent_start",
    "subagent_stop",
    # Gateway pre-dispatch hook. Fired once per incoming MessageEvent
    # after the internal-event guard but BEFORE auth/pairing and agent
    # dispatch. Plugins may return a dict to influence flow:
    #   {"action": "skip",    "reason": "..."}  -> drop message (no reply)
    #   {"action": "rewrite", "text": "..."}    -> replace event.text, continue
    #   {"action": "allow"}  /  None             -> normal dispatch
    # Kwargs: event: MessageEvent, gateway: GatewayRunner, session_store.
    "pre_gateway_dispatch",
    # Approval lifecycle hooks. Fired by tools/approval.py when a dangerous
    # command needs an approval decision -- fires for CLI-interactive prompts,
    # gateway/ACP approvals, and smart-mode auxiliary-LLM decisions.
    # Observers only: return values are ignored. Plugins cannot veto or
    # pre-answer an approval from these hooks (use pre_tool_call to block
    # a tool before it reaches approval).
    #
    # Kwargs for pre_approval_request:
    #   command: str, description: str, pattern_key: str, pattern_keys: list[str],
    #   session_key: str, surface: "cli" | "gateway" | "smart"
    # Kwargs for post_approval_response: same as above plus
    #   choice: "once" | "session" | "always" | "deny" | "timeout"
    #           | "smart_approve" | "smart_deny"
    #   decided_by: "aux_llm"  -- only on surface="smart"
    "pre_approval_request",
    "post_approval_response",
    # Pre-transcription transform hook. Fired by the STT dispatcher
    # (tools.transcription_tools.transcribe_audio) after provider resolution
    # and BEFORE any backend — built-in, command-type, or plugin-registered —
    # is invoked. Callbacks receive keyword args:
    #   file_path, provider, model, language, prompt, source
    # and may return None (unchanged) or a dict mutating any of
    # ``prompt`` / ``language`` / ``model``. Results are applied in
    # registration order, last-writer-wins per field. ``file_path`` is
    # read-only — attempts to change it are logged and dropped. The static
    # ``stt.prompt`` config value is the base; hook results mutate on top.
    "pre_transcription",
    # Kanban task lifecycle hooks. Fired by hermes_cli.kanban_db when a task
    # transitions state, AFTER the change is committed to the board DB (so the
    # hook always sees durable state and a slow plugin can never hold the
    # SQLite write lock). Observers only: return values are ignored.
    #
    # WHICH PROCESS each fires in matters, because kanban workers run as
    # separate `hermes -p <profile> chat -q` subprocesses:
    #   - kanban_task_claimed   -> the DISPATCHER process (gateway-embedded
    #                              dispatcher or `hermes kanban dispatch`),
    #                              right before the worker subprocess spawns.
    #   - kanban_task_completed -> the WORKER process, when it calls
    #                              kanban_complete (or a CLI/manual complete).
    #   - kanban_task_blocked   -> the WORKER process (worker-initiated block)
    #                              or whichever process drove the block.
    # A plugin that needs to observe every transition centrally should hook in
    # the dispatcher; one that needs per-task in-session context should hook in
    # the worker.
    #
    # Common kwargs: task_id: str, board: str | None, assignee: str | None,
    #   run_id: int | None, profile_name: str.
    # kanban_task_completed adds: summary: str | None.
    # kanban_task_blocked adds:   reason: str | None.
    "kanban_task_claimed",
    "kanban_task_completed",
    "kanban_task_blocked",
    # Gateway platform-boundary observer hooks (#64176). Observer-only; each
    # callback isolated by invoke_hook. Payloads are normalized envelopes only,
    # never raw platform SDK objects (per #64176 / #64182 ground rule). This
    # surface grants no adapter handles or platform actions.
    #
    #   gateway_platform_event: inbound platform event as a normalized envelope.
    #       Kwargs: platform, event_type, payload (event_type-specific dict).
    #       Telegram reactions fire today. Other event types and their hook
    #       names land here only together with real fire-sites and payload
    #       contracts; no inert VALID_HOOKS surface is registered ahead of
    #       implementation.
    "gateway_platform_event",
}

ENTRY_POINTS_GROUP = "hermes_agent.plugins"

# System-prompt sections are deliberately more constrained than lifecycle
# hooks. They become high-trust prompt bytes and are charged on every turn.
SYSTEM_PROMPT_SECTION_POSITIONS = frozenset({"after_memory"})
DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS = 4_000
MAX_SYSTEM_PROMPT_SECTION_CHARS = 4_000
MAX_SYSTEM_PROMPT_SECTIONS = 32
MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS = 8_000
_SYSTEM_PROMPT_SECTION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SYSTEM_PROMPT_SECTION_HEADING_PREFIX = "## Plugin Context: "
PLUGIN_SECTIONS_START = "<!-- hermes-plugin-sections:start -->"
PLUGIN_SECTIONS_END = "<!-- hermes-plugin-sections:end -->"


def is_valid_system_prompt_section_id(value: Any) -> bool:
    """Return whether *value* is a stable, heading-safe section identifier."""
    return isinstance(value, str) and bool(_SYSTEM_PROMPT_SECTION_ID_RE.fullmatch(value))


def format_system_prompt_section(section_id: str, content: str) -> str:
    """Render an auditable, length-framed block recoverable from the full prompt."""
    return (
        f"{_SYSTEM_PROMPT_SECTION_HEADING_PREFIX}{section_id}\n"
        f"<!-- hermes-plugin-section-chars:{len(content)} -->\n\n"
        f"{content}"
    )


def format_system_prompt_sections(sections: list) -> str:
    """Render the canonical container used for persistence recovery."""
    if not sections:
        return ""
    blocks = [format_system_prompt_section(item.id, item.content) for item in sections]
    return f"{PLUGIN_SECTIONS_START}\n" + "\n\n".join(blocks) + f"\n{PLUGIN_SECTIONS_END}"
# Reserved event namespace prefix — only core may publish ``hermes:<event>``.
HERMES_EVENT_NAMESPACE = "hermes"

# Max inter-plugin event dispatch recursion depth. A subscriber may itself
# call ``ctx.emit``; this bound stops mutually-emitting plugins from looping
# forever. When exceeded the over-deep emit is dropped (with a warning), not
# raised, so delivery always terminates cleanly.
_EVENT_EMIT_DEPTH_CAP = 8
# Maximum number of queued + currently-running events per manager generation.
# ``emit`` never waits for capacity: a full budget drops the new event with a
# warning so a blocked subscriber cannot back-pressure the emitter forever.
_EVENT_PENDING_CAP = 64
_EVENT_WORKER_STOP = object()

_NS_PARENT = "hermes_plugins"
_MODULE_NAMESPACE_LOCK = threading.RLock()
_BARE_MODULE_SCOPE: Dict[str, str] = {}


def _serialized_replacement(method):
    """Make snapshot → write → lease attachment one atomic transaction."""
    @wraps(method)
    def wrapped(*args, **kwargs):
        with replacement_coordinator.transaction():
            return method(*args, **kwargs)

    return wrapped


@contextmanager
def _plugin_home_scope(home: Path):
    """Bind discovery and loading to the manager's immutable Hermes home."""
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _env_enabled(name: str) -> bool:
    """Return True when an env var is set to a truthy opt-in value."""
    return env_var_enabled(name)


def _get_disabled_plugins() -> set:
    """Read the disabled plugins list from config.yaml.

    Kept for backward compat and explicit deny-list semantics. A plugin
    name in this set will never load, even if it appears in
    ``plugins.enabled``.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
        disabled = cfg_get(config, "plugins", "disabled", default=[])
        return set(disabled) if isinstance(disabled, list) else set()
    except Exception:
        return set()


def _get_enabled_plugins() -> Optional[set]:
    """Read the enabled-plugins allow-list from config.yaml.

    Plugins are opt-in by default — only plugins whose name appears in
    this set are loaded. Returns:

    * ``None`` — the key is missing or malformed. Callers should treat
      this as "nothing enabled yet" (the opt-in default); the first
      ``migrate_config`` run populates the key with a grandfathered set
      of currently-installed user plugins so existing setups don't
      break on upgrade.
    * ``set()`` — an empty list was explicitly set; nothing loads.
    * ``set(...)`` — the concrete allow-list.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
        plugins_cfg = config.get("plugins")
        if not isinstance(plugins_cfg, dict):
            return None
        if "enabled" not in plugins_cfg:
            return None
        enabled = plugins_cfg.get("enabled")
        if not isinstance(enabled, list):
            return None
        return set(enabled)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

_VALID_PLUGIN_KINDS: Set[str] = {"standalone", "backend", "exclusive", "platform", "model-provider"}


def _portable_skill_namespace(key: str) -> str:
    """Return a readable, collision-resistant namespace for a portable plugin."""

    slug = "".join(
        ch if ch.isascii() and (ch.isalnum() or ch in "_-") else "-"
        for ch in key.lower()
    )
    slug = slug.strip("-_") or "plugin"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return f"agent-plugin-{slug}-{digest}"


def _display_author(value: object) -> str:
    """Normalize a manifest author value for the string PluginManifest field."""
    if isinstance(value, Mapping):
        return ", ".join(
            str(value[field])
            for field in ("name", "email", "url")
            if value.get(field)
        )
    return "" if value is None else str(value)


# ── Manifest v2 (#64165) parsing helpers ──────────────────────────────────

# Fields the current parser understands. Anything else in plugin.yaml is
# forward-compat surface: warn (once per manifest, at debug for v1 files to
# avoid churning existing plugins, at warning for v2+) and continue loading.
_KNOWN_MANIFEST_FIELDS: Set[str] = {
    # v1
    "name", "version", "description", "author", "requires_env",
    "provides_tools", "provides_hooks", "kind", "hooks", "label",
    "optional_env", "platforms", "external_dependencies", "pip_dependencies",
    "provides_browser_providers", "provides_web_providers",
    # v2 (#64165)
    "manifest_version", "api_version", "requires_plugins",
    "python_dependencies", "config_schema", "license", "homepage", "tags",
    # owned by sibling sub-issues but reserved so their manifests don't warn
    "capabilities", "emits", "listens", "hermes", "depends",
}

# Highest manifest schema version this Hermes understands.
SUPPORTED_MANIFEST_VERSION = 2

_CONFIG_SCHEMA_TYPES: Dict[str, tuple] = {
    "str": (str,),
    "string": (str,),
    "int": (int,),
    "integer": (int,),
    "float": (int, float),
    "number": (int, float),
    "bool": (bool,),
    "boolean": (bool,),
    "list": (list,),
    "array": (list,),
    "dict": (dict,),
    "object": (dict,),
}


def _parse_manifest_v2_fields(data: Mapping, key: str) -> Dict[str, Any]:
    """Validate and normalize the manifest v2 fields (#64165).

    Returns kwargs for :class:`PluginManifest`. Every problem is a warning,
    never a load failure — v2 metadata is advisory and additive.
    """
    out: Dict[str, Any] = {}

    # manifest_version — absent means v1 (supported forever).
    raw_mv = data.get("manifest_version", 1)
    try:
        mv = int(raw_mv)
    except (TypeError, ValueError):
        logger.warning(
            "Plugin %s: manifest_version %r is not an integer; treating as 1",
            key, raw_mv,
        )
        mv = 1
    if mv > SUPPORTED_MANIFEST_VERSION:
        logger.warning(
            "Plugin %s: manifest_version %d is newer than this Hermes "
            "supports (%d); loading anyway and ignoring unknown fields",
            key, mv, SUPPORTED_MANIFEST_VERSION,
        )
    out["manifest_version"] = mv

    # api_version — plugin API generation (independent of manifest_version).
    raw_api = data.get("api_version")
    if raw_api is None:
        out["api_version"] = None
    else:
        try:
            out["api_version"] = int(raw_api)
        except (TypeError, ValueError):
            logger.warning(
                "Plugin %s: api_version %r is not an integer; ignoring", key, raw_api,
            )
            out["api_version"] = None

    # requires_plugins — list of {id, version_range?} (str shorthand ok).
    deps: List[Dict[str, Any]] = []
    raw_deps = data.get("requires_plugins")
    if raw_deps is not None and not isinstance(raw_deps, list):
        logger.warning(
            "Plugin %s: requires_plugins must be a list; ignoring", key,
        )
        raw_deps = None
    for item in raw_deps or []:
        if isinstance(item, str):
            deps.append({"id": item, "version_range": None})
        elif isinstance(item, Mapping) and isinstance(item.get("id"), str) and item["id"]:
            vr = item.get("version_range")
            deps.append({
                "id": item["id"],
                "version_range": str(vr) if vr is not None else None,
            })
        else:
            logger.warning(
                "Plugin %s: requires_plugins entry %r must be a plugin id "
                "string or a {id, version_range} mapping; skipping", key, item,
            )
    out["requires_plugins"] = deps

    # python_dependencies — declared pip requirement strings. Validated and
    # surfaced ONLY; never auto-installed (isolation design deferred).
    pydeps: List[str] = []
    raw_pydeps = data.get("python_dependencies")
    if raw_pydeps is not None and not isinstance(raw_pydeps, list):
        logger.warning(
            "Plugin %s: python_dependencies must be a list of requirement "
            "strings; ignoring", key,
        )
        raw_pydeps = None
    for item in raw_pydeps or []:
        if isinstance(item, str) and item.strip():
            pydeps.append(item.strip())
        else:
            logger.warning(
                "Plugin %s: python_dependencies entry %r must be a non-empty "
                "requirement string; skipping", key, item,
            )
    out["python_dependencies"] = pydeps

    # config_schema — mapping of key -> {type?, default?, description?, required?}.
    raw_schema = data.get("config_schema")
    schema: Dict[str, Any] = {}
    if raw_schema is not None and not isinstance(raw_schema, Mapping):
        logger.warning(
            "Plugin %s: config_schema must be a mapping; ignoring", key,
        )
        raw_schema = None
    for skey, spec in (raw_schema or {}).items():
        if not isinstance(spec, Mapping):
            logger.warning(
                "Plugin %s: config_schema entry %r must be a mapping "
                "(e.g. {type: str}); skipping", key, skey,
            )
            continue
        stype = spec.get("type")
        if stype is not None and str(stype).lower() not in _CONFIG_SCHEMA_TYPES:
            logger.warning(
                "Plugin %s: config_schema key %r declares unknown type %r "
                "(known: %s); type check will be skipped for it",
                key, skey, stype, ", ".join(sorted(_CONFIG_SCHEMA_TYPES)),
            )
        schema[str(skey)] = dict(spec)
    out["config_schema"] = schema

    # Standard metadata.
    out["license"] = str(data.get("license") or "")
    out["homepage"] = str(data.get("homepage") or "")
    raw_tags = data.get("tags")
    if raw_tags is not None and not isinstance(raw_tags, list):
        logger.warning("Plugin %s: tags must be a list; ignoring", key)
        raw_tags = None
    out["tags"] = [str(t) for t in (raw_tags or [])]

    # Forward compat: unknown fields warn (never fail). Keep v1 manifests
    # quiet at warning level — they predate the known-field census.
    unknown = sorted(set(data.keys()) - _KNOWN_MANIFEST_FIELDS)
    if unknown:
        log = logger.warning if mv >= 2 else logger.debug
        log(
            "Plugin %s: unknown manifest field(s) ignored: %s "
            "(newer manifest schema or typo; plugin still loads)",
            key, ", ".join(unknown),
        )

    return out


def validate_config_schema(
    plugin_id: str,
    schema: Mapping,
    settings: Mapping,
) -> List[str]:
    """Validate a plugin's config entry against its declared config_schema.

    Returns a list of human-actionable warning strings. Never raises;
    schema mismatches must not block plugin load (#64165).
    """
    warnings: List[str] = []
    if not isinstance(schema, Mapping) or not isinstance(settings, Mapping):
        return warnings
    for skey, spec in schema.items():
        if not isinstance(spec, Mapping):
            continue
        present = skey in settings
        if not present:
            if spec.get("required") and "default" not in spec:
                warnings.append(
                    f"plugins.entries.{plugin_id}.settings.{skey} is required "
                    "by the plugin's config_schema but is not set"
                )
            continue
        stype = spec.get("type")
        expected = _CONFIG_SCHEMA_TYPES.get(str(stype).lower()) if stype else None
        if expected is not None:
            value = settings[skey]
            # bool is an int subclass — don't let True satisfy int/float.
            ok = isinstance(value, expected) and not (
                isinstance(value, bool) and bool not in expected
            )
            if not ok:
                warnings.append(
                    f"plugins.entries.{plugin_id}.settings.{skey} should be "
                    f"{stype} (got {type(value).__name__})"
                )
    return warnings


def resolve_plugin_load_order(
    manifests: Mapping[str, "PluginManifest"],
) -> List[str]:
    """Return plugin keys in dependency-respecting load order (#64165).

    When A requires B, B sorts before A (so B's ``register()`` runs first).
    Ties break alphabetically for determinism. Dependency cycles are
    detected, warned about, and the members of the cycle fall back to
    alphabetical order after every non-cycle plugin they depend on.
    Missing dependencies are warned about here (once, at discovery) but do
    not remove the dependent plugin from the order — loads never hard-fail
    on a missing advisory dependency.
    """
    import graphlib

    keys = sorted(manifests.keys())
    by_name: Dict[str, str] = {}
    for k in keys:
        name = manifests[k].name
        if name and name not in by_name:
            by_name[name] = k

    def _resolve_dep(dep_id: str) -> Optional[str]:
        if dep_id in manifests:
            return dep_id
        return by_name.get(dep_id)

    edges: Dict[str, Set[str]] = {k: set() for k in keys}
    for k in keys:
        for dep in manifests[k].requires_plugins:
            dep_id = dep.get("id") if isinstance(dep, Mapping) else None
            if not dep_id:
                continue
            resolved = _resolve_dep(dep_id)
            if resolved is None:
                logger.warning(
                    "Plugin %s requires plugin '%s' which is not enabled/"
                    "installed; loading anyway (probe availability at runtime "
                    "via ctx.has_plugin). Run `hermes plugins enable %s` if "
                    "it is installed.",
                    k, dep_id, dep_id,
                )
                continue
            if resolved == k:
                logger.warning("Plugin %s declares a dependency on itself; ignoring", k)
                continue
            edges[k].add(resolved)

    sorter = graphlib.TopologicalSorter(edges)
    try:
        sorter.prepare()
    except graphlib.CycleError as exc:
        cycle = exc.args[1] if len(exc.args) > 1 else []
        logger.warning(
            "Plugin dependency cycle detected (%s); falling back to "
            "alphabetical load order for all plugins",
            " -> ".join(str(c) for c in cycle),
        )
        return keys

    ordered: List[str] = []
    while sorter.is_active():
        ready = sorted(sorter.get_ready())
        ordered.extend(ready)
        sorter.done(*ready)
    return ordered


@dataclass
class PluginManifest:
    """Parsed representation of a plugin.yaml manifest."""

    name: str
    version: str = ""
    description: str = ""
    author: str = ""
    requires_env: List[Union[str, Dict[str, Any]]] = field(default_factory=list)
    provides_tools: List[str] = field(default_factory=list)
    provides_hooks: List[str] = field(default_factory=list)
    source: str = ""        # "user", "project", or "entrypoint"
    path: Optional[str] = None
    # Plugin kind — see plugins.py module docstring for semantics.
    # ``standalone`` (default): hooks/tools of its own; opt-in via
    #                           ``plugins.enabled``.
    # ``backend``: pluggable backend for an existing core tool (e.g.
    #              image_gen). Built-in (bundled) backends auto-load;
    #              user-installed still gated by ``plugins.enabled``.
    # ``exclusive``: category with exactly one active provider (memory).
    #              Selection via ``<category>.provider`` config key; the
    #              category's own discovery system handles loading and the
    #              general scanner skips these.
    # ``platform``: gateway messaging platform adapter (e.g. IRC). Bundled
    #              platform plugins auto-load so every shipped platform is
    #              available out of the box; user-installed platform plugins
    #              in ~/.hermes/plugins/ still gated by ``plugins.enabled``
    #              (untrusted code).
    kind: str = "standalone"
    # Registry key — path-derived, used by ``plugins.enabled``/``disabled``
    # lookups and by ``hermes plugins list``. For a flat plugin at
    # ``plugins/disk-cleanup/`` the key is ``disk-cleanup``; for a nested
    # category plugin at ``plugins/image_gen/openai/`` the key is
    # ``image_gen/openai``. When empty, falls back to ``name``.
    key: str = ""
    portable: bool = False
    skill_namespace: str = ""
    # Declared capability ids from the manifest ``capabilities:`` list
    # (#64228). Normalized to KNOWN ids only — see
    # ``hermes_cli.plugin_capabilities.CAPABILITY_REGISTRY``. Declaration is
    # consent metadata, not a grant: a capability is live only when the user
    # granted it (``plugins.entries.<id>.granted_capabilities``) or the
    # deprecated legacy ``allow_*`` key is set.
    capabilities: List[str] = field(default_factory=list)
    # ── Manifest v2 fields (#64165) — all optional and additive ──────────
    # Manifest SCHEMA version. Absent (v1) manifests are fully supported
    # forever. This versions the *file format* only; it is deliberately
    # independent from ``api_version`` (the runtime plugin API generation).
    manifest_version: int = 1
    # Runtime plugin API generation the plugin targets (ctx surface /
    # hook signatures). ``None`` = unspecified (treated as current-compatible).
    api_version: Optional[int] = None
    # Inter-plugin dependencies: list of {"id": str, "version_range": str|None}.
    # Advisory: a missing dependency logs a warning but the plugin still
    # loads (plugins can probe availability via ``ctx.has_plugin``). Load
    # ORDER honors these edges: if A requires B, B registers first.
    requires_plugins: List[Dict[str, Any]] = field(default_factory=list)
    # Declared pip dependencies. VALIDATED AND SURFACED ONLY — Hermes never
    # auto-installs these (isolation design for the install seam is a
    # deferred follow-up; see #64165 round-2 review and #15220).
    python_dependencies: List[str] = field(default_factory=list)
    # JSON-schema-ish mapping describing keys under
    # ``plugins.entries.<id>.settings``. Validated at load; mismatches are
    # warnings, never load failures.
    config_schema: Dict[str, Any] = field(default_factory=dict)
    # Formalized standard metadata.
    license: str = ""
    homepage: str = ""
    tags: List[str] = field(default_factory=list)
    # Inter-plugin event bus declarations (advisory in v1 — NOT enforced).
    # ``emits`` lists the bare event names this plugin publishes under its own
    # ``<key>:`` namespace (e.g. ``["ping"]`` → publishes ``<key>:ping``).
    # ``listens`` lists the fully-qualified ``<plugin>:<event>`` names this
    # plugin subscribes to. Both are purely for discoverability
    # (``hermes plugins show``); a plugin may emit/subscribe without declaring.
    emits: List[str] = field(default_factory=list)
    listens: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PluginSystemPromptSection:
    """A plugin-owned section rendered once for each new session."""

    id: str
    content: Union[str, Callable[[Mapping[str, Any]], str]]
    position: str
    max_chars: int
    plugin: str


@dataclass(frozen=True)
class RenderedPluginSystemPromptSection:
    """Validated prompt bytes frozen on the owning AIAgent."""

    id: str
    content: str
    position: str
    plugin: str


@dataclass(frozen=True)
class _EventSubscription:
    """Host-owned subscription ledger entry."""

    owner: str
    callback: Callable


@dataclass(frozen=True)
class _QueuedPluginEvent:
    """Immutable dispatch envelope consumed by the event worker."""

    event: str
    payload: Dict[str, Any]
    subscriptions: tuple[_EventSubscription, ...]
    depth: int
    generation: int


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
    # True for a bundled platform plugin recorded as a deferred (not-yet-
    # imported) loader. The module loads on first real use via the
    # platform_registry; see PluginManager._register_deferred_platform.
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
        # Lazy-built host-owned LLM facade — see ctx.llm property below.
        self._llm: Any = None
        self._subagent_lifecycle: Any = None
        self._state: PluginState | None = None

    @property
    def plugin_id(self) -> str:
        """Return the effective registry id used for this plugin's namespaces."""
        return self.manifest.key or self.manifest.name

    def has_plugin(self, plugin_id: str) -> bool:
        """Return True when another plugin is loaded and enabled (#64165).

        Companion to the advisory ``requires_plugins`` manifest field: a
        missing dependency never blocks load, so plugins probe availability
        at runtime with this. Matches on registry key or manifest name.
        """
        for key, loaded in self._manager._plugins.items():
            if not loaded.enabled:
                continue
            if key == plugin_id or loaded.manifest.name == plugin_id:
                return True
        return False

    # -- namespaced config and durable state --------------------------------

    def get_config(self, key: str, default: Any = None) -> Any:
        """Read ``plugins.entries.<plugin_id>.settings.<key>``.

        ``key`` is always plugin-relative.  For migration compatibility, a
        missing canonical value falls back to the former ``config`` subtree;
        no global config paths are exposed.
        """
        try:
            segments = _plugin_relative_segments(key)
        except ValueError:
            logger.warning(
                "Rejected config path %r from plugin %s", key, self.plugin_id
            )
            raise
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
        plugins = config.get("plugins") if isinstance(config, Mapping) else None
        entries = plugins.get("entries") if isinstance(plugins, Mapping) else None
        entry = entries.get(self.plugin_id) if isinstance(entries, Mapping) else None
        if not isinstance(entry, Mapping):
            return default
        missing = object()
        value = _nested_plugin_value(entry.get("settings"), segments, missing)
        if value is not missing:
            return value
        return _nested_plugin_value(entry.get("config"), segments, default)

    def set_config(self, key: str, value: Any) -> None:
        """Atomically write one value in this plugin's ``settings`` subtree."""
        try:
            segments = _plugin_relative_segments(key)
        except ValueError:
            logger.warning(
                "Rejected config path %r from plugin %s", key, self.plugin_id
            )
            raise
        from hermes_cli import config as config_mod

        if config_mod.is_managed():
            raise PermissionError(
                "Plugin settings cannot be changed in a managed install"
            )
        from hermes_cli import managed_scope

        dotted_path = ".".join((
            "plugins",
            "entries",
            self.plugin_id,
            "settings",
            *segments,
        ))
        if managed_scope.is_key_managed(dotted_path):
            raise PermissionError(
                f"Plugin setting {dotted_path!r} is administrator-managed"
            )
        partial = {
            "plugins": {
                "entries": {
                    self.plugin_id: {
                        "settings": _nested_plugin_mapping(segments, value),
                    }
                }
            }
        }
        full_path = ("plugins", "entries", self.plugin_id, "settings", *segments)
        # The lock covers the merge read plus atomic save, preventing sibling
        # plugin writes from racing between those two steps.
        # Serialize bridge-to-bridge writes across processes as well as
        # threads. Other Hermes config writers still retain their existing
        # atomic-replace semantics; this lock specifically prevents two
        # plugin read/merge/write transactions from dropping siblings.
        with _locked_plugin_state(config_mod.get_config_path()):
            with config_mod._CONFIG_LOCK:
                # Fail closed on malformed YAML. save_config's raw-cache reader
                # intentionally degrades parse failures to {}, which is safe for
                # reads but destructive for read-modify-write.
                config_mod.read_user_config_raw()
                config_mod.save_config(
                    partial,
                    preserve_keys={full_path},
                    merge_existing=True,
                )

    @property
    def state(self) -> PluginState:
        """Return this plugin's profile-scoped durable JSON state facade."""
        if self._state is None:
            self._state = PluginState(self.plugin_id, self.manifest.skill_namespace)
        return self._state

    def _track(
        self,
        kind: str,
        key: str,
        release: Callable[[], None],
    ) -> PluginRegistration:
        """Record host-owned cleanup for a successful registration."""
        return self._manager._track_registration(
            self.manifest, kind, key, release
        )

    def _track_replacement(
        self,
        kind: str,
        key: str,
        *,
        slot: tuple,
        current: Any,
        previous: Any,
        restore: Callable[[Any], bool],
        finalize: Optional[Callable[[], None]] = None,
    ) -> PluginRegistration:
        """Track one generation in a replaceable registration slot."""
        lease = replacement_coordinator.acquire(
            slot,
            current=current,
            previous=previous,
            restore=restore,
            finalize=finalize,
        )
        return self._track(kind, key, lease.dispose)

    # -- host-owned LLM access ----------------------------------------------

    @property
    def llm(self) -> Any:
        """Return the plugin's :class:`agent.plugin_llm.PluginLlm` facade.

        Lets trusted plugins run host-owned chat or structured completions
        against the user's active model and auth without bringing their
        own provider keys. Override capability (model, agent id, auth
        profile) is fail-closed by default and gated through
        ``plugins.entries.<plugin_id>.llm.*`` config keys.

        See :mod:`agent.plugin_llm` for the full surface."""
        if self._llm is None:
            from agent.plugin_llm import PluginLlm
            plugin_id = self.manifest.key or self.manifest.name
            self._llm = PluginLlm(plugin_id=plugin_id)
        return self._llm

    @property
    def subagent_lifecycle(self) -> Any:
        """Return the public, plugin-safe subagent lifecycle service.

        The service only resolves the active host-owned parent agent when a
        child is launched. Plugins receive serializable handles and immutable
        snapshots; they never receive a live agent or a private registry.
        """
        if self._subagent_lifecycle is None:
            from agent.subagent_lifecycle import (
                SubagentLifecycleService,
                get_active_subagent_parent,
            )
            self._subagent_lifecycle = SubagentLifecycleService(
                get_active_subagent_parent
            )
        return self._subagent_lifecycle

    # -- profile awareness --------------------------------------------------

    @property
    def profile_name(self) -> str:
        """Return the active Hermes profile name (e.g. ``"default"``).

        Derived from ``HERMES_HOME`` via
        :func:`hermes_cli.profiles.get_active_profile_name`, so it works in
        every execution context — interactive CLI, gateway, and
        kanban-spawned worker sessions alike — without depending on
        ``_cli_ref`` (which is ``None`` outside an interactive CLI run).

        Returns ``"default"`` for the default profile, the profile id when
        running under ``~/.hermes/profiles/<name>``, or ``"custom"`` when
        ``HERMES_HOME`` points somewhere unrecognized.
        """
        try:
            from hermes_cli.profiles import get_active_profile_name
            return get_active_profile_name()
        except Exception:
            return "default"

    # -- lifecycle: unload callbacks and supervised tasks --------------------

    def on_unload(self, callback: Callable[[], None]) -> PluginRegistration:
        """Register a cleanup callback that runs when this plugin unloads.

        Callbacks are recorded in the ownership ledger, so they run in
        reverse acquisition order interleaved with registration teardown,
        and each is isolated — an exception is logged, never propagated
        (see :meth:`PluginManager._dispose_registrations`).
        """
        if not callable(callback):
            raise TypeError("on_unload callback must be callable")
        handle = self._track("on_unload", getattr(callback, "__name__", "callback"), callback)
        logger.debug("Plugin %s registered on_unload callback", self.manifest.name)
        return handle

    def spawn_task(self, coro, *, name: Optional[str] = None) -> "asyncio.Task":
        """Spawn a supervised background asyncio task owned by this plugin.

        The task is recorded in the ownership ledger; unloading the plugin
        (or a force reload) cancels it. Requires a running event loop.
        """
        if not asyncio.iscoroutine(coro):
            raise TypeError("spawn_task expects a coroutine")
        loop = asyncio.get_running_loop()
        task_name = name or f"plugin:{self.plugin_id}:task"
        task = loop.create_task(coro, name=task_name)

        def _cancel_task() -> None:
            if not task.done():
                task.cancel()

        handle = self._track("background_task", task_name, _cancel_task)
        task.add_done_callback(lambda _t: handle.dispose())
        logger.debug(
            "Plugin %s spawned supervised task: %s", self.manifest.name, task_name
        )
        return task

    # -- approval transport registration ------------------------------------

    def register_approval_transport(self, name: str, present_fn: Callable) -> None:
        """Register a human approval presentation transport.

        The transport is inactive until the operator explicitly selects
        ``security.approval.transport: <name>``. It receives a host-created,
        redacted ``ApprovalRequest`` and may only return a correlated human
        decision; command policy and approval persistence remain host-owned.
        ``present_fn`` may be synchronous or async.
        """
        self._manager.register_approval_transport(
            name,
            present_fn,
            plugin_id=self.manifest.key or self.manifest.name,
        )
        # Record ownership so unload/force-reload removes this transport.
        # Duplicate names are rejected above (raise), so there is never a
        # displaced previous entry to restore.
        clean = str(name).strip().lower()
        entry = self._manager._approval_transports.get(clean)
        if entry is not None:
            self._track_replacement(
                "approval_transport",
                clean,
                slot=(
                    "manager_mapping",
                    id(self._manager._approval_transports),
                    clean,
                ),
                current=entry,
                previous=None,
                restore=lambda replacement: self._manager._restore_mapping(
                    self._manager._approval_transports, clean, entry, replacement
                ),
            )

    # -- tool registration --------------------------------------------------

    @_serialized_replacement
    def register_tool(
        self,
        name: str,
        toolset: str,
        schema: dict,
        handler: Callable,
        check_fn: Callable | None = None,
        requires_env: list | None = None,
        is_async: bool = False,
        description: str = "",
        emoji: str = "",
        override: bool = False,
    ) -> Optional[PluginRegistration]:
        """Register a tool in the global registry **and** track it as plugin-provided.

        Pass ``override=True`` to replace an existing built-in tool with the
        same name (e.g. swap the default ``browser_navigate`` for a custom
        CDP-backed implementation). Without it, attempting to register a name
        already claimed by a different toolset is rejected.

        ``override=True`` against a built-in tool requires the operator to
        opt in via ``plugins.entries.<plugin_id>.allow_tool_override: true``
        in config.yaml — mirrors the trust gate pattern used for
        ``ctx.llm`` provider/model overrides (#23194). Without that gate,
        any enabled plugin could silently replace a privileged built-in
        like ``shell_exec`` or ``write_file`` and exfiltrate everything
        the model invokes through it.
        """
        if override and not self._tool_override_allowed(name):
            plugin_id = self.manifest.key or self.manifest.name
            raise PluginToolOverrideError(
                f"Plugin {self.manifest.name!r} cannot override built-in tool "
                f"{name!r}. Set "
                f"plugins.entries.{plugin_id}.allow_tool_override: true "
                f"in config.yaml to allow this plugin to replace built-in tools."
            )

        from tools.registry import registry

        scope = self._manager.scope_key
        previous = registry.snapshot_registration(name, scope=scope)
        effective = registry.get_entry(name, scope=scope)
        if previous is None and effective is not None and not override:
            logger.warning(
                "Plugin %s tried to shadow global tool %s without override=True",
                self.manifest.name,
                name,
            )
            return None
        registry.register(
            name=name,
            toolset=toolset,
            schema=schema,
            handler=handler,
            check_fn=check_fn,
            requires_env=requires_env,
            is_async=is_async,
            description=description,
            emoji=emoji,
            override=override,
            scope=scope,
        )
        registered = registry.snapshot_registration(name, scope=scope)
        if (
            registered is not None
            and registered is not previous
            and registered.handler is handler
        ):
            self._manager._plugin_tool_names.add(name)
            def _restore_tool(replacement: Any) -> bool:
                return registry.restore_registration(
                    name, registered, replacement, scope=scope
                )

            handle = self._track_replacement(
                "tool",
                name,
                slot=("tool", scope, name),
                current=registered,
                previous=previous,
                restore=_restore_tool,
                finalize=lambda: self._manager._remove_tool_name_if_unowned(name),
            )
        else:
            handle = None
        logger.debug(
            "Plugin %s registered tool: %s%s",
            self.manifest.name, name, " (override)" if override else "",
        )
        return handle

    # -- capability probing (#64228) -----------------------------------------

    def has_capability(self, capability: str) -> bool:
        """Return True when *capability* is live for this plugin.

        Plugins should probe with this and degrade gracefully instead of
        crashing when a gated host surface refuses them. Bundled plugins are
        trusted for ``tools.override`` (mirrors the registration gate); for
        everything else the answer comes from the granted-capability set or
        the deprecated legacy ``allow_*`` config key. Unknown capability ids
        and unreadable consent state return False (fail closed).
        """
        source = getattr(self.manifest, "source", "") or ""
        if source == "bundled" and capability == "tools.override":
            return True
        plugin_id = self.manifest.key or self.manifest.name
        return plugin_capability_granted(plugin_id, capability)

    # -- override trust gate ------------------------------------------------

    def _tool_override_allowed(self, tool_name: str) -> bool:
        """Return True if this plugin is configured to override built-in tools.

        Bundled plugins (shipped with Hermes core) are trusted by default —
        an override there is a deliberate maintainer choice, not a third-party
        plugin trying to elevate privilege. For every other source, the
        canonical check is :func:`plugin_capability_granted` with the
        ``tools.override`` capability — satisfied by EITHER the consent-flow
        grant (``plugins.entries.<plugin_id>.granted_capabilities``) OR the
        deprecated legacy key ``allow_tool_override: true`` (still honored
        for backward compatibility; #64228 reference migration).
        """
        source = getattr(self.manifest, "source", "") or ""
        if source == "bundled":
            return True
        try:
            from hermes_cli.config import load_config

            with _plugin_home_scope(self._manager.home_path):
                cfg = load_config() or {}
        except Exception:
            # If we can't load config, fail closed — better to break the
            # override than silently grant it.
            return False
        plugin_id = self.manifest.key or self.manifest.name
        # Fail-closed by construction: any failure to read consent state
        # inside plugin_capability_granted returns False. The profile-scoped
        # config is passed through so a multi-profile process consults THIS
        # manager's home, never the active profile's (#65593 constraint).
        return plugin_capability_granted(plugin_id, "tools.override", config=cfg)

    # -- message injection --------------------------------------------------

    def inject_message(
        self,
        content: str,
        role: str = "user",
        *,
        session_key: str | None = None,
    ) -> bool:
        """Inject a message into a CLI or gateway conversation.

        If the agent is idle (waiting for user input), this starts a new turn.
        If the agent is running, this interrupts and injects the message.

        This enables plugins (e.g. remote control viewers, messaging bridges)
        to send messages into the conversation from external sources.

        Gateway injection requires an existing ``session_key`` and an explicit
        ``plugins.entries.<plugin_id>.allow_gateway_injection`` config grant.
        A ``True`` return means the live gateway accepted the request for
        asynchronous dispatch, not that platform delivery has completed.

        Returns True if the message was queued successfully.
        """
        cli = self._manager._cli_ref
        msg = content if role == "user" else f"[{role}] {content}"

        if cli is not None:
            if getattr(cli, "_agent_running", False):
                # Agent is mid-turn - interrupt with the message
                cli._interrupt_queue.put(msg)
            else:
                # Agent is idle - queue as next input
                cli._pending_input.put(msg)
            return True

        if not session_key:
            logger.warning(
                "inject_message: gateway mode requires an existing session_key"
            )
            return False
        if not self._gateway_injection_allowed():
            plugin_id = self.manifest.key or self.manifest.name
            logger.warning(
                "inject_message: gateway injection denied for plugin %s; set "
                "plugins.entries.%s.allow_gateway_injection: true to allow it",
                plugin_id,
                plugin_id,
            )
            return False

        if not self._manager.has_gateway_message_injector:
            logger.warning("inject_message: no live gateway is available")
            return False

        plugin_id = self.manifest.key or self.manifest.name
        try:
            return bool(
                self._manager.inject_gateway_message(
                    session_key=session_key,
                    content=msg,
                    plugin_id=plugin_id,
                )
            )
        except Exception:
            logger.warning(
                "inject_message: gateway scheduling failed for plugin %s",
                plugin_id,
                exc_info=True,
            )
            return False

    def _gateway_injection_allowed(self) -> bool:
        """Return whether this plugin may trigger gateway session turns."""
        try:
            cfg = load_config_readonly() or {}
        except Exception:
            return False

        plugin_id = self.manifest.key or self.manifest.name
        return (
            cfg_get(
                cfg,
                "plugins",
                "entries",
                plugin_id,
                "allow_gateway_injection",
                default=False,
            )
            is True
        )

    # -- CLI command registration --------------------------------------------

    @_serialized_replacement
    def register_cli_command(
        self,
        name: str,
        help: str,
        setup_fn: Callable,
        handler_fn: Callable | None = None,
        description: str = "",
    ) -> PluginRegistration:
        """Register a CLI subcommand (e.g. ``hermes honcho ...``).

        The *setup_fn* receives an argparse subparser and should add any
        arguments/sub-subparsers.  If *handler_fn* is provided it is set
        as the default dispatch function via ``set_defaults(func=...)``."""
        previous = self._manager._cli_commands.get(name)
        entry = {
            "name": name,
            "help": help,
            "description": description,
            "setup_fn": setup_fn,
            "handler_fn": handler_fn,
            "plugin": self.manifest.name,
            "plugin_key": self.manifest.key or self.manifest.name,
        }
        self._manager._cli_commands[name] = entry
        handle = self._track_replacement(
            "cli_command",
            name,
            slot=("manager_mapping", id(self._manager._cli_commands), name),
            current=entry,
            previous=previous,
            restore=lambda replacement: self._manager._restore_mapping(
                self._manager._cli_commands, name, entry, replacement
            ),
        )
        logger.debug("Plugin %s registered CLI command: %s", self.manifest.name, name)
        return handle

    # -- slash command registration -------------------------------------------

    @_serialized_replacement
    def register_command(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        args_hint: str = "",
    ) -> Optional[PluginRegistration]:
        """Register a slash command (e.g. ``/lcm``) available in CLI and gateway sessions.

        The handler signature is ``fn(raw_args: str) -> str | None``.
        It may also be an async callable — the gateway dispatch handles both.

        Unlike ``register_cli_command()`` (which creates ``hermes <subcommand>``
        terminal commands), this registers in-session slash commands that users
        invoke during a conversation.

        ``args_hint`` is an optional short string (e.g. ``"<file>"`` or
        ``"dias:7 formato:json"``) used by gateway adapters to surface the
        command with an argument field — for example Discord's native slash
        command picker. Plugin commands without ``args_hint`` register as
        parameterless in Discord and still accept trailing text when invoked
        as free-form chat.

        Names conflicting with built-in commands are rejected with a warning.
        """
        clean = name.lower().strip().lstrip("/").replace(" ", "-")
        if not clean:
            logger.warning(
                "Plugin '%s' tried to register a command with an empty name.",
                self.manifest.name,
            )
            return

        # Reject if it conflicts with a built-in command
        try:
            from hermes_cli.commands import resolve_command
            if resolve_command(clean) is not None:
                logger.warning(
                    "Plugin '%s' tried to register command '/%s' which conflicts "
                    "with a built-in command. Skipping.",
                    self.manifest.name, clean,
                )
                return
        except Exception:
            pass  # If commands module isn't available, skip the check

        previous = self._manager._plugin_commands.get(clean)
        entry = {
            "handler": handler,
            "description": description or "Plugin command",
            "plugin": self.manifest.name,
            "plugin_key": self.manifest.key or self.manifest.name,
            "args_hint": (args_hint or "").strip(),
        }
        self._manager._plugin_commands[clean] = entry
        handle = self._track_replacement(
            "command",
            clean,
            slot=("manager_mapping", id(self._manager._plugin_commands), clean),
            current=entry,
            previous=previous,
            restore=lambda replacement: self._manager._restore_mapping(
                self._manager._plugin_commands, clean, entry, replacement
            ),
        )
        logger.debug("Plugin %s registered command: /%s", self.manifest.name, clean)
        return handle

    # -- tool dispatch -------------------------------------------------------

    def dispatch_tool(self, tool_name: str, args: dict, **kwargs) -> str:
        """Dispatch a tool call through the registry, with parent agent context.

        This is the public interface for plugin slash commands that need to call
        tools like ``delegate_task`` without reaching into framework internals.
        The parent agent (if available) is resolved automatically — plugins never
        need to access the agent directly.

        Args:
            tool_name: Registry name of the tool (e.g. ``"delegate_task"``).
            args: Tool arguments dict (same as what the model would pass).
            **kwargs: Extra keyword args forwarded to the registry dispatch.

        Returns:
            JSON string from the tool handler (same format as model tool calls).
        """
        from tools.registry import registry

        # Wire up parent agent context when available (CLI mode).
        # In gateway mode _cli_ref is None — tools degrade gracefully
        # (workspace hints fall back to TERMINAL_CWD, no spinner).
        if "parent_agent" not in kwargs:
            cli = self._manager._cli_ref
            agent = getattr(cli, "agent", None) if cli else None
            if agent is not None:
                kwargs["parent_agent"] = agent

        return registry.dispatch(
            tool_name, args, scope=self._manager.scope_key, **kwargs
        )

    # -- context engine registration -----------------------------------------

    @_serialized_replacement
    def register_context_engine(self, engine) -> Optional[PluginRegistration]:
        """Register a context engine to replace the built-in ContextCompressor.

        Only one context engine plugin is allowed. If a second plugin tries
        to register one, it is rejected with a warning.

        The engine must be an instance of ``agent.context_engine.ContextEngine``.
        """
        if self._manager._context_engine is not None:
            logger.warning(
                "Plugin '%s' tried to register a context engine, but one is "
                "already registered. Only one context engine plugin is allowed.",
                self.manifest.name,
            )
            return
        # Defer the import to avoid circular deps at module level
        from agent.context_engine import ContextEngine
        if not isinstance(engine, ContextEngine):
            logger.warning(
                "Plugin '%s' tried to register a context engine that does not "
                "inherit from ContextEngine. Ignoring.",
                self.manifest.name,
            )
            return
        previous = self._manager._context_engine
        self._manager._context_engine = engine
        handle = self._track_replacement(
            "context_engine",
            engine.name,
            slot=("manager_value", id(self._manager), "_context_engine"),
            current=engine,
            previous=previous,
            restore=lambda replacement: self._manager._restore_value(
                "_context_engine", engine, replacement
            ),
        )
        logger.info(
            "Plugin '%s' registered context engine: %s",
            self.manifest.name, engine.name,
        )
        return handle

    # -- context reference registration -------------------------------------

    def register_context_reference(self, provider) -> None:
        """Register a custom @-prefix context reference provider.

        ``provider`` must be an instance of
        :class:`agent.context_references.ContextReferenceProvider`.  The
        ``provider.prefix`` attribute defines the @-prefix (e.g. ``"issue"``
        creates ``@issue:...``).  Built-in prefixes (diff, staged, file,
        folder, git, url) are reserved and will be rejected.
        """
        from agent.context_references import (
            ContextReferenceProvider as _CRP,
            register_context_reference_provider as _register,
        )

        if not isinstance(provider, _CRP):
            logger.warning(
                "Plugin '%s' tried to register a context reference provider "
                "that does not inherit from ContextReferenceProvider. Ignoring.",
                self.manifest.name,
            )
            return
        try:
            _register(provider)
        except ValueError as exc:
            logger.warning(
                "Plugin '%s' context reference registration failed: %s",
                self.manifest.name, exc,
            )
            return
        logger.info(
            "Plugin '%s' registered context reference: @%s:",
            self.manifest.name, provider.prefix,
        )

    # -- image gen provider registration ------------------------------------

    @_serialized_replacement
    def register_image_gen_provider(self, provider) -> Optional[PluginRegistration]:
        """Register an image generation backend.

        ``provider`` must be an instance of
        :class:`agent.image_gen_provider.ImageGenProvider`. The
        ``provider.name`` attribute is what ``image_gen.provider`` in
        ``config.yaml`` matches against when routing ``image_generate``
        tool calls.
        """
        from agent.image_gen_provider import ImageGenProvider
        from agent.image_gen_registry import (
            register_provider,
            restore_registration,
            snapshot_registration,
        )

        if not isinstance(provider, ImageGenProvider):
            logger.warning(
                "Plugin '%s' tried to register an image_gen provider that does "
                "not inherit from ImageGenProvider. Ignoring.",
                self.manifest.name,
            )
            return
        registry_name = provider.name.strip()
        scope = self._manager.scope_key
        previous = snapshot_registration(registry_name, scope=scope)
        register_provider(provider, scope=scope)
        registered = snapshot_registration(registry_name, scope=scope)
        if registered is not provider:
            return None
        handle = self._track_replacement(
            "image_gen_provider",
            registry_name,
            slot=("image_gen_provider", scope, registry_name),
            current=provider,
            previous=previous,
            restore=lambda replacement: restore_registration(
                registry_name, provider, replacement, scope=scope
            ),
        )
        logger.info(
            "Plugin '%s' registered image_gen provider: %s",
            self.manifest.name, registry_name,
        )
        return handle

    # -- dashboard auth provider registration --------------------------------

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
        from hermes_cli.dashboard_auth import (
            DashboardAuthProvider,
            register_provider,
        )
        from hermes_cli.dashboard_auth.registry import restore_registration
        from hermes_cli.dashboard_auth.registry import snapshot_registration

        if not isinstance(provider, DashboardAuthProvider):
            logger.warning(
                "Plugin '%s' tried to register a dashboard-auth provider "
                "that does not inherit from DashboardAuthProvider. Ignoring.",
                self.manifest.name,
            )
            return
        registry_name = provider.name
        scope = self._manager.scope_key
        previous = snapshot_registration(registry_name, scope=scope)
        try:
            register_provider(provider, scope=scope)
        except (TypeError, ValueError) as e:
            logger.warning(
                "Plugin '%s' failed to register dashboard-auth provider "
                "%r: %s",
                self.manifest.name, getattr(provider, "name", "?"), e,
            )
            return
        registered = snapshot_registration(registry_name, scope=scope)
        if registered is not provider:
            return None
        handle = self._track_replacement(
            "dashboard_auth_provider",
            registry_name,
            slot=("dashboard_auth_provider", scope, registry_name),
            current=provider,
            previous=previous,
            restore=lambda replacement: restore_registration(
                registry_name, provider, replacement, scope=scope
            ),
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
        self,
        name: str,
        label: str,
        adapter_factory: Callable,
        check_fn: Callable,
        validate_config: Callable | None = None,
        required_env: list | None = None,
        install_hint: str = "",
        **entry_kwargs: Any,
    ) -> Optional[PluginRegistration]:
        """Register a gateway platform adapter.

        The adapter_factory receives a ``PlatformConfig`` and returns a
        ``BasePlatformAdapter`` subclass instance.

        ``check_fn`` is a PASSIVE dependency probe — "are deps importable
        right now?".  It must never install anything: status displays and
        config loading call it freely.  If your platform's SDK is
        lazy-installable, pass the ACTIVE installer separately as
        ``ensure_deps_fn`` (forwarded via ``entry_kwargs``); the gateway
        calls it from ``create_adapter()`` when ``check_fn`` is False,
        right before connecting the platform.

        Extra keyword arguments are forwarded to ``PlatformEntry`` (e.g.
        ``setup_fn``, ``emoji``, ``allowed_users_env``, ``platform_hint``,
        ``ensure_deps_fn``).  Unknown keys raise TypeError from the
        dataclass constructor.

        Example::

            ctx.register_platform(
                name="irc",
                label="IRC",
                adapter_factory=lambda cfg: IRCAdapter(cfg),
                check_fn=lambda: True,
                emoji="💬",
                setup_fn=irc_interactive_setup,
            )
        """
        from gateway.platform_registry import platform_registry, PlatformEntry

        entry_kwargs.setdefault("plugin_name", self.manifest.name)
        entry = PlatformEntry(
            name=name,
            label=label,
            adapter_factory=adapter_factory,
            check_fn=check_fn,
            validate_config=validate_config,
            required_env=required_env or [],
            install_hint=install_hint,
            source="plugin",
            **entry_kwargs,
        )
        scope = self._manager.scope_key
        previous = platform_registry.snapshot_registration(name, scope=scope)
        platform_registry.register(entry, scope=scope)
        current = platform_registry.snapshot_registration(name, scope=scope)
        if current[0] is not entry or current[1] is not None:
            return None
        self._manager._plugin_platform_names.add(name)
        handle = self._track_replacement(
            "platform",
            name,
            slot=("platform", scope, name),
            current=current,
            previous=previous,
            restore=lambda replacement: self._restore_platform_registration(
                platform_registry, name, current, replacement, scope
            ),
            finalize=lambda: self._manager._remove_platform_name_if_unowned(name),
        )
        logger.debug(
            "Plugin %s registered platform: %s",
            self.manifest.name,
            name,
        )
        return handle

    def _restore_platform_registration(
        self,
        platform_registry,
        name: str,
        current,
        replacement,
        scope: str,
    ) -> bool:
        return platform_registry.restore_registration(
            name, current, replacement, scope=scope
        )

    # -- slack action handler registration ----------------------------------

    def register_slack_action_handler(
        self,
        action_id: Any,
        callback: Callable,
    ) -> PluginRegistration:
        """Register a Slack Block Kit action handler from a plugin.

        Hermes' Slack adapter wires registered handlers into its
        ``slack_bolt.AsyncApp`` at connect time. The callback is invoked
        when a user clicks a button (or interacts with another Block Kit
        action element) whose ``action_id`` matches.

        Callback signature follows the slack_bolt convention::

            async def handler(ack, body, action) -> None:
                await ack()  # required, within 3 seconds
                ...

        Args:
            action_id: Whatever ``slack_bolt.App.action()`` accepts —
                a literal ``action_id`` string, a compiled ``re.Pattern``
                for matching multiple ids, or a constraint dict
                (e.g. ``{"action_id": "...", "block_id": "..."}``).
            callback: Async callable receiving ``(ack, body, action)``.

        Raises:
            ValueError: if ``callback`` is not callable, or ``action_id``
                is empty/None.

        Example::

            async def _on_approve(ack, body, action):
                await ack()
                # apply some workflow keyed on action["value"]

            ctx.register_slack_action_handler("inbox_sweep_approve", _on_approve)
        """
        if not callable(callback):
            raise ValueError(
                f"Plugin '{self.manifest.name}' tried to register a Slack "
                f"action handler with a non-callable callback."
            )
        if action_id is None or (isinstance(action_id, str) and not action_id.strip()):
            raise ValueError(
                f"Plugin '{self.manifest.name}' tried to register a Slack "
                f"action handler with an empty action_id."
            )
        entry = (action_id, callback, self.manifest.name)
        self._manager._slack_action_handlers.append(entry)
        handle = self._track(
            "slack_action_handler",
            repr(action_id),
            lambda: self._manager._remove_identity(
                self._manager._slack_action_handlers, entry
            ),
        )
        logger.debug(
            "Plugin %s registered Slack action handler: %s",
            self.manifest.name,
            action_id,
        )
        return handle

    # -- hook registration --------------------------------------------------

    # -- auxiliary task registration ---------------------------------------

    @_serialized_replacement
    def register_auxiliary_task(
        self,
        key: str,
        *,
        display_name: str,
        description: str,
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
                built-in task key (vision, compression, web_extract, approval,
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
            raise ValueError(
                f"Plugin '{self.manifest.name}' tried to register auxiliary task "
                f"with invalid key {key!r}"
            )
        if not all(c.isalnum() or c == "_" for c in key):
            raise ValueError(
                f"Plugin '{self.manifest.name}' auxiliary task key {key!r} "
                f"must contain only alphanumeric characters and underscores"
            )

        # Lazy import to avoid circular: hermes_cli.main imports plugins indirectly
        from hermes_cli.main import _AUX_TASKS as _BUILTIN_AUX_TASKS

        builtin_keys = {k for k, _name, _desc in _BUILTIN_AUX_TASKS}
        if key in builtin_keys:
            raise ValueError(
                f"Plugin '{self.manifest.name}' cannot register auxiliary task "
                f"{key!r} — that key is reserved for a built-in task. "
                f"Pick a plugin-namespaced key (e.g. '{self.manifest.name}_{key}')."
            )

        # Owner is the plugin's canonical id (``key or name``) — the same id
        # ``ctx.llm`` is bound to (PluginLlm._plugin_id) — so the task trust
        # gate in agent/plugin_llm.py can match ownership. For the common
        # case where a manifest sets no explicit ``key`` this equals the name.
        owner_id = self.manifest.key or self.manifest.name

        # Reject duplicate registrations across plugins
        existing = self._manager._aux_tasks.get(key)
        if existing is not None and existing.get("plugin") != owner_id:
            raise ValueError(
                f"Plugin '{self.manifest.name}' cannot register auxiliary task "
                f"{key!r} — already registered by plugin "
                f"'{existing.get('plugin')}'"
            )

        # Normalize defaults — plugin owns the schema, but we ensure routing
        # fields exist with sensible types so consumers don't crash.
        merged_defaults: Dict[str, Any] = {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 60,
            "extra_body": {},
        }
        if defaults:
            for k, v in defaults.items():
                merged_defaults[k] = v

        self._manager._aux_tasks[key] = {
            "key": key,
            "display_name": display_name,
            "description": description,
            "defaults": merged_defaults,
            "plugin": owner_id,
            "plugin_key": owner_id,
        }
        entry = self._manager._aux_tasks[key]
        handle = self._track_replacement(
            "auxiliary_task",
            key,
            slot=("manager_mapping", id(self._manager._aux_tasks), key),
            current=entry,
            previous=existing,
            restore=lambda replacement: self._manager._restore_mapping(
                self._manager._aux_tasks, key, entry, replacement
            ),
        )
        logger.debug(
            "Plugin %s registered auxiliary task: %s (%s)",
            self.manifest.name,
            key,
            display_name,
        )
        return handle

    # -- redaction pattern registration --------------------------------------

    def register_redaction_patterns(self, patterns) -> int:
        """Additively register secret-token regexes with the redaction engine.

        Accepted patterns join the vendor-prefix alternation in
        :mod:`agent.redact` and are masked everywhere built-in patterns
        apply — logs, terminal output, transport errors, transcripts —
        with the same head/tail masking and the same non-reusable
        sentinel on ``file_read`` content. Historically every new vendor
        token format required a core PR appending to
        ``_PREFIX_PATTERNS``; provider plugins should own their own
        format instead.

        The registry is **additive-only**: plugins can extend what gets
        masked but cannot remove or weaken built-in patterns, so a
        plugin can only ever over-redact, never expose. The operator's
        global opt-out (``security.redact_secrets: false``) applies to
        plugin patterns exactly as it does to built-ins.

        Each pattern must compile as a regex and start with at least 2
        literal characters (e.g. ``r"nvapi-[A-Za-z0-9_-]{20,}"``).
        Invalid entries are warned and skipped — never raised.

        Returns the number of patterns accepted.
        """
        from agent.redact import register_redaction_patterns as _register

        try:
            count = _register(
                patterns, source=f"plugin:{self.manifest.name}",
            )
        except Exception as exc:
            logger.warning(
                "Plugin '%s' redaction pattern registration failed: %s",
                self.manifest.name, exc,
            )
            return 0
        logger.debug(
            "Plugin %s registered %d redaction pattern(s)",
            self.manifest.name, count,
        )
        return count

    def register_hook(self, hook_name: str, callback: Callable) -> PluginRegistration:
        """Register a lifecycle hook callback.

        Unknown hook names produce a warning but are still stored so
        forward-compatible plugins don't break.
        """
        if hook_name not in VALID_HOOKS:
            logger.warning(
                "Plugin '%s' registered unknown hook '%s' "
                "(valid: %s)",
                self.manifest.name,
                hook_name,
                ", ".join(sorted(VALID_HOOKS)),
            )
        callbacks = self._manager._hooks.setdefault(hook_name, [])
        callbacks.append(callback)
        handle = self._track(
            "hook", hook_name,
            lambda: self._manager._remove_callback(
                self._manager._hooks, hook_name, callback
            ),
        )
        logger.debug("Plugin %s registered hook: %s", self.manifest.name, hook_name)
        return handle

    def register_system_prompt_section(
        self,
        id: str,
        content: Union[str, Callable[[Mapping[str, Any]], str]],
        *,
        position: str = "after_memory",
        max_chars: int = DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS,
    ) -> PluginRegistration:
        """Register bounded context that is frozen into each new session prompt.

        Callables receive a read-only session-info mapping. The rendered full
        system prompt is already persisted by core and restored verbatim, so no
        parallel plugin-section state is needed for process restarts.
        """
        if not is_valid_system_prompt_section_id(id):
            raise ValueError(
                "system prompt section id must be 1-128 lowercase characters "
                "using letters, numbers, '.', '_', or '-'"
            )
        if not isinstance(content, str) and not callable(content):
            raise TypeError("system prompt section content must be a string or callable")
        if position not in SYSTEM_PROMPT_SECTION_POSITIONS:
            raise ValueError(
                "system prompt section position must be one of: "
                + ", ".join(sorted(SYSTEM_PROMPT_SECTION_POSITIONS))
            )
        if (
            isinstance(max_chars, bool)
            or not isinstance(max_chars, int)
            or not 0 < max_chars <= MAX_SYSTEM_PROMPT_SECTION_CHARS
        ):
            raise ValueError(
                "system prompt section max_chars must be between 1 and "
                f"{MAX_SYSTEM_PROMPT_SECTION_CHARS}"
            )
        existing = self._manager._system_prompt_sections.get(id)
        if existing is not None:
            raise ValueError(
                f"system prompt section {id!r} is already registered by "
                f"plugin {existing.plugin!r}"
            )
        plugin_id = self.manifest.key or self.manifest.name
        section = PluginSystemPromptSection(
            id=id,
            content=content,
            position=position,
            max_chars=max_chars,
            plugin=plugin_id,
        )
        self._manager._system_prompt_sections[id] = section
        # Record ownership so unload/force-reload removes this section.
        # Duplicate ids are rejected above (raise), so there is never a
        # displaced previous entry to restore. The parameter ``id`` shadows
        # the builtin, so capture the mapping identity via ``builtins.id``.
        import builtins

        handle = self._track_replacement(
            "system_prompt_section",
            id,
            slot=(
                "manager_mapping",
                builtins.id(self._manager._system_prompt_sections),
                id,
            ),
            current=section,
            previous=existing,
            restore=lambda replacement: self._manager._restore_mapping(
                self._manager._system_prompt_sections, id, section, replacement
            ),
        )
        logger.debug(
            "Plugin %s registered system prompt section: %s",
            self.manifest.name,
            id,
        )
        return handle

    # -- inter-plugin event bus --------------------------------------------

    def emit(self, event: str, payload: Optional[dict] = None) -> int:
        """Publish *event* to all subscribers; return the number invoked.

        The event is delivered as ``<plugin_key>:<event>`` where
        ``plugin_key`` is FORCED to this plugin's own registry key
        (``manifest.key or manifest.name``). Pass only the bare event name —
        a plugin may only publish under its own namespace.

        Passing an already-namespaced name (anything containing ``':'``,
        including ``hermes:x`` or a foreign ``other:x``) is rejected with a
        ``ValueError`` and a logged warning — fail-closed. The ``hermes:``
        prefix is reserved for core.

        Delivery is fire-and-forget through a host-owned, single-worker queue:
        registration order is preserved, while a blocking subscriber cannot
        stall the emitter. The queue has a bounded pending budget; a full
        budget drops the new event with a warning. Each subscriber receives a
        deep-copied payload and is isolated in its own ``try/except``. Awaitable
        results are resolved through the existing loop-safe plugin path.

        Returns the count of subscriber callbacks scheduled (0 when there are
        no subscribers, or when the pending/recursion budget drops the emit).
        """
        plugin_key = self.manifest.key or self.manifest.name
        if not event or not isinstance(event, str):
            logger.warning(
                "Plugin '%s' tried to emit an invalid event name %r",
                plugin_key, event,
            )
            raise ValueError(
                f"Plugin '{plugin_key}' emit() requires a non-empty event name"
            )
        if ":" in event:
            logger.warning(
                "Plugin '%s' tried to emit namespaced/reserved event '%s' — "
                "a plugin may only emit bare event names under its own '%s:' "
                "namespace (the '%s:' prefix is reserved for core, and foreign "
                "namespaces are forbidden)",
                plugin_key, event, plugin_key, HERMES_EVENT_NAMESPACE,
            )
            raise ValueError(
                f"Plugin '{plugin_key}' may not emit '{event}': emit only the "
                f"bare event name; the namespace is forced to '{plugin_key}:' "
                f"and the '{HERMES_EVENT_NAMESPACE}:' prefix is reserved for core"
            )
        if payload is not None and not isinstance(payload, dict):
            raise TypeError(
                f"Plugin '{plugin_key}' emit() payload must be a dict or None"
            )
        full_event = f"{plugin_key}:{event}"
        return self._manager._dispatch_event(full_event, payload or {})

    def subscribe(self, event: str, callback: Callable) -> None:
        """Subscribe *callback* to a fully-qualified event name.

        *event* is the full ``<plugin_key>:<event>`` name (or ``hermes:<event>``
        if core ever emits). Subscribing is unrestricted — any plugin may
        listen to any published event; only *emitting* is namespace-gated.

        Callbacks are stored in registration order as host-owned ledger
        entries. The owner key lets plugin unload/reload remove subscriptions
        before any later event can invoke a zombie callback.
        """
        if not event or not isinstance(event, str):
            raise ValueError(
                f"Plugin '{self.manifest.name}' subscribe() requires a "
                f"non-empty event name"
            )
        plugin_key = self.manifest.key or self.manifest.name
        self._manager._subscribe_event(plugin_key, event, callback)
        logger.debug(
            "Plugin %s subscribed to event: %s", self.manifest.name, event,
        )

    # -- middleware registration -------------------------------------------

    def register_middleware(self, kind: str, callback: Callable) -> PluginRegistration:
        """Register a behavior-changing middleware callback.

        Middleware is separate from observer hooks: request middleware may
        rewrite the effective payload, and execution middleware may wrap the
        real callback. Unknown kinds are stored for forward compatibility but
        warned so plugin authors can catch typos.
        """
        if kind not in VALID_MIDDLEWARE:
            logger.warning(
                "Plugin '%s' registered unknown middleware '%s' "
                "(valid: %s)",
                self.manifest.name,
                kind,
                ", ".join(sorted(VALID_MIDDLEWARE)),
            )
        callbacks = self._manager._middleware.setdefault(kind, [])
        callbacks.append(callback)
        handle = self._track(
            "middleware", kind,
            lambda: self._manager._remove_callback(
                self._manager._middleware, kind, callback
            ),
        )
        logger.debug("Plugin %s registered middleware: %s", self.manifest.name, kind)
        return handle

    # -- skill registration -------------------------------------------------

    @_serialized_replacement
    def register_skill(
        self,
        name: str,
        path: Path,
        description: str = "",
        frontmatter: Optional[Mapping[str, Any]] = None,
    ) -> PluginRegistration:
        """Register a read-only skill provided by this plugin.

        The skill becomes resolvable as ``'<plugin_name>:<name>'`` via
        ``skill_view()``.  It does **not** enter the flat
        ``~/.hermes/skills/`` tree and is **not** listed in the system
        prompt's ``<available_skills>`` index — plugin skills are
        opt-in explicit loads only.

        Raises:
            ValueError: if *name* contains ``':'`` or invalid characters.
            FileNotFoundError: if *path* does not exist.
        """
        from agent.skill_utils import _NAMESPACE_RE

        if ":" in name:
            raise ValueError(
                f"Skill name '{name}' must not contain ':' "
                f"(the namespace is derived from the plugin name "
                f"'{self.manifest.name}' automatically)."
            )
        if not name or not _NAMESPACE_RE.match(name):
            raise ValueError(
                f"Invalid skill name '{name}'. Must match [a-zA-Z0-9_-]+."
            )
        if not path.exists():
            raise FileNotFoundError(f"SKILL.md not found at {path}")

        namespace = self.manifest.skill_namespace or self.manifest.name
        qualified = f"{namespace}:{name}"
        if self.manifest.portable and qualified in self._manager._plugin_skills:
            raise ValueError(f"Plugin skill '{qualified}' is already registered")
        previous = self._manager._plugin_skills.get(qualified)
        entry = {
            "path": path,
            "plugin": namespace,
            "plugin_key": self.manifest.key or self.manifest.name,
            "bare_name": name,
            "description": description,
            "frontmatter": dict(frontmatter or {}),
        }
        self._manager._plugin_skills[qualified] = entry
        handle = self._track_replacement(
            "skill",
            qualified,
            slot=("manager_mapping", id(self._manager._plugin_skills), qualified),
            current=entry,
            previous=previous,
            restore=lambda replacement: self._manager._restore_mapping(
                self._manager._plugin_skills, qualified, entry, replacement
            ),
        )
        logger.debug(
            "Plugin %s registered skill: %s",
            self.manifest.name, qualified,
        )
        return handle


# ---------------------------------------------------------------------------
# PluginManager
# ---------------------------------------------------------------------------

class PluginManager:
    """Central manager that discovers, loads, and invokes plugins."""

    def __init__(self, scope_key: Optional[str] = None) -> None:
        # Capture the home immutably. Unload can run from a different ambient
        # profile context, but every inverse must target the registration's
        # original scope.
        self.scope_key = scope_key or hermes_home_key()
        self.home_path = Path(self.scope_key)
        self._discovery_lock = threading.RLock()
        self._plugins: Dict[str, LoadedPlugin] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        self._middleware: Dict[str, List[Callable]] = {}
        self._plugin_tool_names: Set[str] = set()
        self._plugin_platform_names: Set[str] = set()
        self._cli_commands: Dict[str, dict] = {}
        self._context_engine = None  # Set by a plugin via register_context_engine()
        self._plugin_commands: Dict[str, dict] = {}  # Slash commands registered by plugins
        self._system_prompt_sections: Dict[str, PluginSystemPromptSection] = {}
        self._discovered: bool = False
        self._cli_ref = None  # Set by CLI after plugin discovery
        self._gateway_message_injector: tuple[object, Callable] | None = None
        # Plugin skill registry: qualified name → metadata dict.
        self._plugin_skills: Dict[str, Dict[str, Any]] = {}
        self._portable_mcp_servers: Dict[str, Dict[str, Any]] = {}
        # Plugin-registered auxiliary tasks: key → {key, display_name,
        # description, defaults, plugin}. See PluginContext.register_auxiliary_task.
        self._aux_tasks: Dict[str, Dict[str, Any]] = {}
        # Explicitly-selected, profile-scoped human approval transports.
        self._approval_transports: Dict[str, Any] = {}
        # Inter-plugin event bus. Subscriptions are owner-tagged ledger entries
        # so unload/reload can remove zombie callbacks. A single daemon worker
        # preserves registration order while keeping emitters non-blocking.
        self._subscriptions: Dict[str, List[_EventSubscription]] = {}
        self._event_lock = threading.RLock()
        self._event_idle = threading.Condition(self._event_lock)
        self._event_generation = 0
        self._event_pending_by_generation: Dict[int, int] = {0: 0}
        self._event_queue: queue.Queue[Any] = queue.Queue(
            maxsize=_EVENT_PENDING_CAP
        )
        self._event_worker: Optional[threading.Thread] = None
        # Per-worker chain depth caps mutually-emitting plugins even though each
        # re-entrant emit is queued rather than invoked recursively.
        self._emit_depth = threading.local()
        # Slack Block Kit action handlers registered by plugins. Each entry
        # is (matcher, callback, plugin_name); the Slack adapter wires them
        # into its slack_bolt App at connect() time. ``matcher`` is whatever
        # ``app.action()`` accepts (a literal action_id string, a compiled
        # ``re.Pattern``, or a constraint dict); ``callback`` is an async
        # function with the slack_bolt signature ``(ack, body, action)``.
        self._slack_action_handlers: List[tuple] = []
        # Registration handles are kept both per plugin (ownership lookup) and
        # globally (reverse-order teardown for overrides spanning plugins).
        #
        # Multi-profile constraint (#65593): several process-global registries
        # (tools, platforms, providers) are shared across profiles while
        # multiple PluginManager instances may coexist in one process (keyed
        # by resolved hermes home). The ledger is therefore keyed per manager
        # — i.e. per (hermes_home, plugin_id) — and every release/restore
        # closure is identity-conditional, so one profile's unload can never
        # clear another profile's registrations. Registry overlays keyed by
        # scope_key (see tools/registry.py and gateway/platform_registry.py)
        # carry the profile dimension; anything still process-global is
        # guarded by the identity checks. TODO(#64178): extend explicit
        # profile keying to any remaining process-global slots when the
        # symmetric force-reload lands.
        self._ownership_ledger: Dict[str, List[PluginRegistration]] = {}
        self._registration_order: List[PluginRegistration] = []

    # -----------------------------------------------------------------------
    # Registration ledger internals
    # -----------------------------------------------------------------------

    def _track_registration(
        self,
        manifest: PluginManifest,
        kind: str,
        key: str,
        release: Callable[[], None],
    ) -> PluginRegistration:
        """Record one successful registration under its canonical plugin key."""
        plugin_key = manifest.key or manifest.name
        registration = PluginRegistration(
            kind=kind,
            key=key,
            release=release,
            plugin_key=plugin_key,
        )
        registration._on_dispose = lambda disposed: self._forget_registrations(
            [disposed]
        )
        self._ownership_ledger.setdefault(plugin_key, []).append(registration)
        self._registration_order.append(registration)
        return registration

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

    def set_gateway_message_injector(
        self,
        owner: object,
        injector: Callable[..., bool],
    ) -> None:
        """Publish a live gateway injector and its lifecycle owner."""
        self._gateway_message_injector = (owner, injector)

    def clear_gateway_message_injector(self, owner: object) -> None:
        """Clear the injector only when it still belongs to ``owner``."""
        registered = self._gateway_message_injector
        if registered is not None and registered[0] is owner:
            self._gateway_message_injector = None

    def inject_gateway_message(self, **kwargs: Any) -> bool:
        """Submit a plugin-triggered turn to the live gateway."""
        registered = self._gateway_message_injector
        if registered is None:
            return False
        return bool(registered[1](**kwargs))

    def discover_and_load(self, force: bool = False) -> None:
        """Scan all plugin sources and load each plugin found.

        When ``force`` is true, clear cached discovery state first so config
        changes or newly-added bundled backends become visible in long-lived
        sessions without requiring a full agent restart.
        """
        with self._discovery_lock, _plugin_home_scope(self.home_path):
            if self._discovered and not force:
                return
            if force:
                # The ledger owns teardown.  Clearing manager-local containers by
                # itself leaves process-global tools/platforms/providers installed.
                self.unload()
            if env_var_enabled("HERMES_SAFE_MODE"):
                logger.info("HERMES_SAFE_MODE=1 — plugin discovery skipped")
                self._discovered = True
                return
            # Set the flag up front as a re-entrancy guard (a plugin's register()
            # can transitively trigger discovery again), but reset it if the sweep
            # raises so a failed scan is NOT cached as "discovered with an empty
            # registry" — callers swallow the exception and would otherwise be
            # permanently stranded on the early-return above (the "No web provider
            # configured" class of failures).
            self._discovered = True
            try:
                self._discover_and_load_inner()
                # Plugin secret sources register during discover; the initial
                # load_hermes_dotenv() already ran at import time. Re-pull so the
                # first process sees plugin backends (tracking #64177).
                self._refresh_secret_sources_after_discovery()
            except BaseException:
                self._discovered = False
                raise

    def _refresh_secret_sources_after_discovery(self) -> None:
        """If any plugin secret source is enabled, reset cache and re-apply.

        Enablement is delegated to each source's ``is_enabled(cfg)`` — the
        same contract the orchestrator uses (``registry._ordered_enabled_sources``)
        — so a source with custom activation logic is honored, not just
        ``secrets.<name>.enabled``.

        No-op when only bundled sources exist or none are enabled.
        Fail-open: never raise into discover_and_load.
        """
        try:
            from agent.secret_sources.registry import list_plugin_sources
            from hermes_cli.env_loader import load_hermes_dotenv, reset_secret_source_cache
        except Exception:
            return
        try:
            plugin_sources = list_plugin_sources()
        except Exception:
            return
        if not plugin_sources:
            return
        # Load the secrets config once; hand each source its own section and
        # let its is_enabled() decide (honours custom activation extensions).
        try:
            from hermes_cli.config import load_config

            cfg = load_config() or {}
            secrets = cfg.get("secrets") or {}
        except Exception:
            secrets = {}
        enabled_names = []
        for source in plugin_sources:
            name = getattr(source, "name", "")
            section = secrets.get(name)
            section = section if isinstance(section, dict) else {}
            try:
                if source.is_enabled(section):
                    enabled_names.append(name)
            except Exception:
                # A source whose is_enabled() raises is skipped, mirroring
                # the orchestrator's defensive posture.
                continue
        if not enabled_names:
            return
        try:
            reset_secret_source_cache()
            load_hermes_dotenv()
            logger.debug(
                "Re-applied secret sources after plugin discovery for: %s",
                ", ".join(sorted(enabled_names)),
            )
        except Exception as exc:
            logger.debug("secret source re-apply after discovery failed: %s", exc)

    def _discover_and_load_inner(self) -> None:
        """The actual discovery sweep — see :meth:`discover_and_load`."""
        manifests: List[PluginManifest] = self._collect_directory_manifests()

        # Directory plugins are collected above. Pip / entry-point plugins
        # are intentionally separate: portable packages are directory-only
        # and the startup MCP probe must not import or register entry points.
        ep_manifests = self._scan_entry_points()
        logger.debug("  entrypoints: %d manifest(s)", len(ep_manifests))
        manifests.extend(ep_manifests)

        # Load each manifest (skip user-disabled plugins).
        # Later sources override earlier ones on key collision — user
        # plugins take precedence over bundled, project plugins take
        # precedence over user. Dedup here so we only load the final
        # winner. Keys are path-derived (``image_gen/openai``,
        # ``disk-cleanup``) so ``tts/openai`` and ``image_gen/openai``
        # don't collide even when both manifests say ``name: openai``.
        disabled = _get_disabled_plugins()
        enabled = _get_enabled_plugins()  # None = opt-in default (nothing enabled)
        winners: Dict[str, PluginManifest] = {}
        for manifest in manifests:
            winners[manifest.key or manifest.name] = manifest
        # Standalone/user plugins that pass the gates below are collected
        # here and loaded AFTER the sweep in dependency-respecting order
        # (requires_plugins topological sort, #64165).
        to_load: Dict[str, PluginManifest] = {}
        for manifest in winners.values():
            lookup_key = manifest.key or manifest.name

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
            logger.info(
                "Plugin discovery complete: %d found, %d enabled",
                len(self._plugins),
                sum(1 for p in self._plugins.values() if p.enabled),
            )

    def register_approval_transport(
        self,
        name: str,
        present_fn: Callable,
        *,
        plugin_id: str,
    ) -> None:
        """Register one plugin-owned approval transport for this profile."""
        import re

        from hermes_cli.approval_transport import RegisteredApprovalTransport

        clean = str(name).strip().lower()
        if clean == "builtin":
            raise ValueError("approval transport name 'builtin' is reserved")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", clean):
            raise ValueError(
                "approval transport name must match [a-z0-9][a-z0-9_-]{0,63}"
            )
        if not callable(present_fn):
            raise TypeError("approval transport present_fn must be callable")
        if clean in self._approval_transports:
            owner = self._approval_transports[clean].plugin_id
            raise ValueError(
                f"approval transport {clean!r} is already registered by {owner!r}"
            )
        self._approval_transports[clean] = RegisteredApprovalTransport(
            name=clean,
            present=present_fn,
            plugin_id=plugin_id,
            profile_home=str(get_hermes_home().resolve()),
        )
        logger.info("Plugin %s registered approval transport: %s", plugin_id, clean)

    def get_approval_transport(self, name: str):
        """Return a transport only inside the profile that registered it."""
        registered = self._approval_transports.get(str(name).strip().lower())
        if registered is None:
            return None
        if registered.profile_home != str(get_hermes_home().resolve()):
            return None
        return registered

    def _collect_directory_manifests(self) -> List[PluginManifest]:
        """Collect directory manifests in the same order as full discovery.

        This method only reads manifests. It does not load native plugin
        modules, register deferred platforms, or otherwise mutate manager
        registries. Keeping the source ordering and scanner calls here lets
        startup probes share the exact precedence and containment rules used
        by :meth:`_discover_and_load_inner`.
        """
        manifests: List[PluginManifest] = []

        # 1. Bundled plugins (<repo>/plugins/<name>/). The excluded top-level
        # categories have their own discovery systems; bundled platforms are
        # scanned explicitly one level below.
        repo_plugins = get_bundled_plugins_dir()
        logger.debug("Scanning bundled plugins: %s", repo_plugins)
        bundled = self._scan_directory(
            repo_plugins,
            source="bundled",
            skip_names={"memory", "context_engine", "platforms", "model-providers"},
        )
        logger.debug("  bundled (top-level): %d manifest(s)", len(bundled))
        manifests.extend(bundled)
        bundled_platforms = self._scan_directory(
            repo_plugins / "platforms", source="bundled"
        )
        logger.debug("  bundled/platforms: %d manifest(s)", len(bundled_platforms))
        manifests.extend(bundled_platforms)

        # 2. User plugins (~/.hermes/plugins/)
        user_dir = get_hermes_home() / "plugins"
        logger.debug("Scanning user plugins: %s", user_dir)
        user_manifests = self._scan_directory(user_dir, source="user")
        logger.debug("  user: %d manifest(s)", len(user_manifests))
        manifests.extend(user_manifests)

        # 3. Project plugins (./.hermes/plugins/), only when explicitly opted
        # in. This must match the full discovery gate exactly.
        if _env_enabled("HERMES_ENABLE_PROJECT_PLUGINS"):
            project_dir = Path.cwd() / ".hermes" / "plugins"
            logger.debug("Scanning project plugins: %s", project_dir)
            project_manifests = self._scan_directory(project_dir, source="project")
            logger.debug("  project: %d manifest(s)", len(project_manifests))
            manifests.extend(project_manifests)
        else:
            logger.debug(
                "Project plugins disabled (set HERMES_ENABLE_PROJECT_PLUGINS=1 to enable)"
            )

        return manifests

    def has_enabled_portable_mcp(self, raw_config: Mapping[str, Any]) -> bool:
        """Probe enabled portable MCP packages without loading plugins.

        The directory manifest collection is shared with full discovery, so
        native ``plugin.yaml`` precedence, source ordering, depth limits, and
        project-plugin gating cannot diverge between startup and runtime.
        """
        if _env_enabled("HERMES_SAFE_MODE"):
            return False

        plugins_config = raw_config.get("plugins")
        if not isinstance(plugins_config, dict):
            return False
        enabled_value = plugins_config.get("enabled")
        if not isinstance(enabled_value, list):
            return False
        enabled = {value for value in enabled_value if isinstance(value, str)}
        disabled_value = plugins_config.get("disabled", [])
        disabled = (
            {value for value in disabled_value if isinstance(value, str)}
            if isinstance(disabled_value, list)
            else set()
        )
        if not enabled:
            return False

        winners: Dict[str, PluginManifest] = {}
        for manifest in self._collect_directory_manifests():
            winners[manifest.key or manifest.name] = manifest

        for manifest in winners.values():
            if not manifest.portable:
                continue
            lookup_key = manifest.key or manifest.name
            if lookup_key in disabled or manifest.name in disabled:
                continue
            if lookup_key not in enabled and manifest.name not in enabled:
                continue
            try:
                from hermes_cli.agent_plugins import _discover_mcp

                if _discover_mcp(
                    Path(manifest.path),
                    get_hermes_home()
                    / "plugin-data"
                    / (manifest.skill_namespace or lookup_key),
                    [],
                    create_data=False,
                ):
                    return True
            except (OSError, RuntimeError, ValueError):
                # Full discovery will report component diagnostics. Startup
                # probing should fail closed for an unreadable package.
                continue
        return False

    # -----------------------------------------------------------------------
    # Directory scanning
    # -----------------------------------------------------------------------

    def _scan_directory(
        self,
        path: Path,
        source: str,
        skip_names: Optional[Set[str]] = None,
    ) -> List[PluginManifest]:
        """Read ``plugin.yaml`` manifests from subdirectories of *path*.

        Supports two layouts, mixed freely:

        * **Flat** — ``<root>/<plugin-name>/plugin.yaml``. Key is
          ``<plugin-name>`` (e.g. ``disk-cleanup``).
        * **Category** — ``<root>/<category>/<plugin-name>/plugin.yaml``,
          where the ``<category>`` directory itself has no ``plugin.yaml``.
          Key is ``<category>/<plugin-name>`` (e.g. ``image_gen/openai``).
          Depth is capped at two segments.

        *skip_names* is an optional allow-list of names to ignore at the
        top level (kept for back-compat; the current call sites no longer
        pass it now that categories are first-class).
        """
        return self._scan_directory_level(
            path, source, skip_names=skip_names, prefix="", depth=0
        )

    def _scan_directory_level(
        self,
        path: Path,
        source: str,
        *,
        skip_names: Optional[Set[str]],
        prefix: str,
        depth: int,
    ) -> List[PluginManifest]:
        """Recursive implementation of :meth:`_scan_directory`.

        ``prefix`` is the category path already accumulated ("" at root,
        "image_gen" one level in). ``depth`` is the recursion depth; we
        cap at 2 so ``<root>/a/b/c/`` is ignored.
        """
        manifests: List[PluginManifest] = []
        if not path.is_dir():
            return manifests

        for child in sorted(path.iterdir()):
            if not child.is_dir():
                continue
            if depth == 0 and skip_names and child.name in skip_names:
                continue
            manifest_file = child / "plugin.yaml"
            if not manifest_file.exists():
                manifest_file = child / "plugin.yml"

            if manifest_file.exists():
                manifest = self._parse_manifest(
                    manifest_file, child, source, prefix
                )
                if manifest is not None:
                    manifests.append(manifest)
                continue

            portable_file = child / "plugin.json"
            if portable_file.exists() or portable_file.is_symlink():
                try:
                    from hermes_cli.agent_plugins import read_agent_plugin_manifest

                    data, diagnostics = read_agent_plugin_manifest(child)
                    for diagnostic in diagnostics:
                        logger.warning(
                            "Agent Plugin '%s': %s",
                            child,
                            diagnostic.message,
                        )
                    key = f"{prefix}/{child.name}" if prefix else data["name"]
                    manifests.append(
                        PluginManifest(
                            name=data["name"],
                            version=data.get("version", ""),
                            description=data.get("description", ""),
                            author=_display_author(data.get("author", "")),
                            source=source,
                            path=str(child),
                            key=key,
                            portable=True,
                            skill_namespace=_portable_skill_namespace(key),
                        )
                    )
                except Exception as exc:
                    logger.warning("Failed to parse %s: %s", portable_file, exc)
                continue

            # No manifest at this level. If we're still within the depth
            # cap, treat this directory as a category namespace and recurse
            # one level in looking for children with manifests.
            if depth >= 1:
                logger.debug("Skipping %s (no plugin.yaml, depth cap reached)", child)
                continue

            sub_prefix = f"{prefix}/{child.name}" if prefix else child.name
            manifests.extend(
                self._scan_directory_level(
                    child,
                    source,
                    skip_names=None,
                    prefix=sub_prefix,
                    depth=depth + 1,
                )
            )

        return manifests

    def _parse_manifest(
        self,
        manifest_file: Path,
        plugin_dir: Path,
        source: str,
        prefix: str,
    ) -> Optional[PluginManifest]:
        """Parse a single ``plugin.yaml`` into a :class:`PluginManifest`.

        Returns ``None`` on parse failure (logs a warning).
        """
        try:
            if yaml is None:
                logger.warning("PyYAML not installed – cannot load %s", manifest_file)
                return None
            data = fast_safe_load(manifest_file.read_text(encoding="utf-8")) or {}

            name = data.get("name", plugin_dir.name)
            key = f"{prefix}/{plugin_dir.name}" if prefix else name

            raw_kind = data.get("kind", "standalone")
            if not isinstance(raw_kind, str):
                raw_kind = "standalone"
            kind = raw_kind.strip().lower()
            if kind not in _VALID_PLUGIN_KINDS:
                logger.warning(
                    "Plugin %s: unknown kind '%s' (valid: %s); treating as 'standalone'",
                    key, raw_kind, ", ".join(sorted(_VALID_PLUGIN_KINDS)),
                )
                kind = "standalone"

            # Auto-coerce user-installed memory providers to kind="exclusive"
            # so they're routed to plugins/memory discovery instead of being
            # loaded by the general PluginManager (which has no
            # register_memory_provider on PluginContext). Mirrors the
            # heuristic in plugins/memory/__init__.py:_is_memory_provider_dir.
            # Bundled memory providers are already skipped via skip_names.
            if kind == "standalone" and "kind" not in data:
                init_file = plugin_dir / "__init__.py"
                if init_file.exists():
                    try:
                        source_text = init_file.read_text(errors="replace", encoding="utf-8")[:8192]
                        if (
                            "register_memory_provider" in source_text
                            or "MemoryProvider" in source_text
                        ):
                            kind = "exclusive"
                            logger.debug(
                                "Plugin %s: detected memory provider, "
                                "treating as kind='exclusive'",
                                key,
                            )
                        elif (
                            "register_provider" in source_text
                            and "ProviderProfile" in source_text
                        ):
                            # Model provider plugin (calls register_provider()
                            # from ``providers`` with a ProviderProfile). Route
                            # to providers/__init__.py discovery.
                            kind = "model-provider"
                            logger.debug(
                                "Plugin %s: detected model provider, "
                                "treating as kind='model-provider'",
                                key,
                            )
                    except Exception:
                        pass

            logger.debug(
                "Parsed manifest: key=%s name=%s kind=%s source=%s path=%s",
                key, name, kind, source, plugin_dir,
            )
            v2_fields = _parse_manifest_v2_fields(data, key)
            return PluginManifest(
                name=name,
                version=str(data.get("version", "")),
                description=data.get("description", ""),
                author=_display_author(data.get("author", "")),
                requires_env=data.get("requires_env", []),
                provides_tools=data.get("provides_tools", []),
                provides_hooks=data.get("provides_hooks", []),
                source=source,
                path=str(plugin_dir),
                kind=kind,
                key=key,
                capabilities=_parse_declared_capabilities(
                    data.get("capabilities"), name
                ),
                **v2_fields,
                emits=data.get("emits") or [],
                listens=data.get("listens") or [],
            )
        except Exception as exc:
            logger.warning(
                "Failed to parse %s: %s", manifest_file, exc, exc_info=_PLUGINS_DEBUG,
            )
            return None

    # -----------------------------------------------------------------------
    # Entry-point scanning
    # -----------------------------------------------------------------------

    def _scan_entry_points(self) -> List[PluginManifest]:
        """Check ``importlib.metadata`` for pip-installed plugins."""
        manifests: List[PluginManifest] = []
        try:
            eps = importlib.metadata.entry_points()
            # Python 3.12+ returns a SelectableGroups; earlier returns dict
            if hasattr(eps, "select"):
                group_eps = eps.select(group=ENTRY_POINTS_GROUP)
            elif isinstance(eps, dict):
                group_eps = eps.get(ENTRY_POINTS_GROUP, [])
            else:
                group_eps = [ep for ep in eps if ep.group == ENTRY_POINTS_GROUP]

            for ep in group_eps:
                manifest = PluginManifest(
                    name=ep.name,
                    source="entrypoint",
                    path=ep.value,
                    key=ep.name,
                )
                manifests.append(manifest)
        except Exception as exc:
            logger.debug("Entry-point scan failed: %s", exc)

        return manifests

    # -----------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------

    def _platform_name_from_manifest(self, manifest: PluginManifest) -> str:
        """Derive the gateway platform name (e.g. ``feishu``) for a platform plugin.

        The platform name registered via ``register_platform(name=...)`` lives
        inside the adapter module (which we are explicitly trying NOT to import
        early). It is not carried in ``plugin.yaml``. Across every bundled
        platform plugin the manifest name is ``<platform>-platform`` and the
        plugin directory basename is ``<platform>``, so we derive the name
        without importing: strip a trailing ``-platform`` from the manifest
        name, falling back to the directory basename. This is also a sensible
        convention for third-party platform plugins.
        """
        name = manifest.name or ""
        if name.endswith("-platform"):
            return name[: -len("-platform")]
        if manifest.path:
            return Path(manifest.path).name
        return name

    @_serialized_replacement
    def _register_deferred_platform(self, manifest: PluginManifest) -> None:
        """Register a lazy loader for a bundled platform plugin.

        The platform adapter module is imported only when the gateway / cron /
        setup / send_message path first asks the ``platform_registry`` for this
        platform. Until then we record a lightweight ``LoadedPlugin`` so
        ``hermes plugins list`` still shows the platform as available, and we
        hand the registry a loader that runs the normal eager-load path.
        """
        lookup_key = manifest.key or manifest.name
        platform_name = self._platform_name_from_manifest(manifest)

        # Record an enabled placeholder for introspection (`hermes plugins
        # list`). The real module load swaps in a fully-populated LoadedPlugin
        # (tools/hooks/commands attribution) when the loader fires.
        loaded = LoadedPlugin(manifest=manifest, enabled=True)
        loaded.deferred = True
        self._plugins[lookup_key] = loaded

        try:
            from gateway.platform_registry import platform_registry

            scope = self.scope_key

            def _loader(_manifest: PluginManifest = manifest) -> None:
                # Acquire the manager lock before checking cancellation. If an
                # unload won the race after the registry marked this loader
                # in-flight, it restores the predecessor and this loader exits
                # without publishing any plugin registrations. If loading won,
                # unload waits and then disposes the completed registration set.
                with self._discovery_lock, _plugin_home_scope(self.home_path):
                    if platform_registry.is_deferred_load_cancelled(
                        platform_name, scope=scope
                    ):
                        return
                    self._load_plugin_scoped(_manifest)

            previous = platform_registry.snapshot_registration(
                platform_name, scope=scope
            )
            platform_registry.register_deferred(
                platform_name, _loader, scope=scope
            )
            current = platform_registry.snapshot_registration(
                platform_name, scope=scope
            )
            if current[0] is None and current[1] is _loader:
                self._plugin_platform_names.add(platform_name)
                lease = replacement_coordinator.acquire(
                    ("platform", scope, platform_name),
                    current=current,
                    previous=previous,
                    restore=lambda replacement: self._restore_deferred_platform(
                        platform_registry,
                        platform_name,
                        current,
                        replacement,
                        scope,
                    ),
                    finalize=lambda: self._remove_platform_name_if_unowned(
                        platform_name
                    ),
                )
                self._track_registration(
                    manifest,
                    "platform",
                    platform_name,
                    lease.dispose,
                )
            logger.debug(
                "Registered deferred platform loader: %s (plugin=%s)",
                platform_name,
                lookup_key,
            )
        except Exception:
            # If the registry import fails for any reason, fall back to eager
            # loading so the platform is never silently lost.
            logger.debug(
                "Deferred platform registration failed for '%s'; eager-loading",
                lookup_key,
                exc_info=True,
            )
            self._load_plugin(manifest)

    def _warn_python_dependencies(self, manifest: PluginManifest) -> None:
        """Surface declared pip dependencies (#64165).

        python_dependencies is a declaration seam ONLY: Hermes validates and
        prints the requirements with an install hint but NEVER auto-installs
        them. The isolation design (constraints installs vs. vendored dirs
        vs. conflict-detection-and-refusal) is an explicitly deferred
        follow-up — see the round-2 review on #64165 and #15220.
        """
        deps = manifest.python_dependencies
        if not deps:
            return
        key = manifest.key or manifest.name
        missing: List[str] = []
        for req in deps:
            # Best-effort presence probe on the distribution name.
            dist = re.split(r"[<>=!~\[;\s]", req, maxsplit=1)[0].strip()
            if not dist:
                continue
            try:
                importlib.metadata.version(dist)
            except importlib.metadata.PackageNotFoundError:
                missing.append(req)
            except Exception:
                continue
        if missing:
            logger.warning(
                "Plugin %s declares Python dependencies that are not "
                "installed: %s. Hermes does not install plugin dependencies "
                "automatically; install them yourself, e.g.: pip install %s",
                key, ", ".join(missing),
                " ".join(f"'{m}'" for m in missing),
            )
        else:
            logger.debug(
                "Plugin %s python_dependencies satisfied: %s",
                key, ", ".join(deps),
            )

    def _validate_plugin_config_schema(self, manifest: PluginManifest) -> None:
        """Check plugins.entries.<id> settings against config_schema (#64165).

        Mismatches log actionable warnings naming the key and expected type;
        they never block the plugin from loading.
        """
        if not manifest.config_schema:
            return
        plugin_id = manifest.key or manifest.name
        settings: Mapping[str, Any] = {}
        try:
            from hermes_cli.config import load_config

            cfg = load_config() or {}
            entries = (cfg.get("plugins") or {}).get("entries") or {}
            entry = entries.get(plugin_id) if isinstance(entries, Mapping) else None
            raw = entry.get("settings") if isinstance(entry, Mapping) else None
            if not isinstance(raw, Mapping):
                # Migration fallback mirroring ctx.get_config.
                raw = entry.get("config") if isinstance(entry, Mapping) else None
            settings = raw if isinstance(raw, Mapping) else {}
        except Exception:
            settings = {}
        for warning in validate_config_schema(
            plugin_id, manifest.config_schema, settings
        ):
            logger.warning("Plugin %s config: %s", plugin_id, warning)

    def _restore_deferred_platform(
        self,
        platform_registry,
        name: str,
        current,
        replacement,
        scope: str,
    ) -> bool:
        return platform_registry.restore_registration(
            name, current, replacement, scope=scope
        )

    def _load_plugin(self, manifest: PluginManifest) -> None:
        """Import a plugin module and call its ``register(ctx)`` function."""
        with self._discovery_lock, _plugin_home_scope(self.home_path):
            self._load_plugin_scoped(manifest)

    def _load_plugin_scoped(self, manifest: PluginManifest) -> None:
        """Load one plugin with the manager's home bound as current."""
        loaded = LoadedPlugin(manifest=manifest)
        logger.debug(
            "Loading plugin '%s' (source=%s, kind=%s, path=%s)",
            manifest.key or manifest.name, manifest.source, manifest.kind, manifest.path,
        )

        if manifest.portable:
            self._load_portable_plugin(manifest, loaded)
            return

        from tools.registry import registry as _registry
        registration_start = len(self._registration_order)
        plugin_key = manifest.key or manifest.name
        _module_name = self._policy_module_name(manifest)
        with replacement_coordinator.transaction():
            previous_policy = _registry.snapshot_plugin_override_policy(
                _module_name, scope=self.scope_key
            )
            current_policy = _registry.register_plugin_override_policy(
                _module_name,
                PluginContext(manifest, self)._tool_override_allowed(""),
                scope=self.scope_key,
            )
            policy_lease = replacement_coordinator.acquire(
                ("tool_override_policy", self.scope_key, _module_name),
                current=current_policy,
                previous=previous_policy,
                restore=lambda replacement: _registry.restore_plugin_override_policy(
                    _module_name,
                    current_policy,
                    replacement,
                    scope=self.scope_key,
                ),
            )
            self._track_registration(
                manifest,
                "tool_override_policy",
                _module_name,
                policy_lease.dispose,
            )
        try:
            if manifest.source in {"user", "project", "bundled"}:
                module = self._load_directory_module(
                    manifest, module_name=_module_name
                )
            else:
                module = self._load_entrypoint_module(manifest)

            loaded.module = module

            # Call register()
            register_fn = getattr(module, "register", None)
            if register_fn is None:
                loaded.error = "no register() function"
                logger.warning("Plugin '%s' has no register() function", manifest.name)
            else:
                ctx = PluginContext(manifest, self)
                register_fn(ctx)
                registrations = [
                    registration
                    for registration in self._registration_order[registration_start:]
                    if registration.plugin_key == plugin_key and registration.active
                ]
                loaded.tools_registered = [
                    registration.key
                    for registration in registrations
                    if registration.kind == "tool"
                ]
                loaded.hooks_registered = [
                    registration.key
                    for registration in registrations
                    if registration.kind == "hook"
                ]
                loaded.middleware_registered = [
                    registration.key
                    for registration in registrations
                    if registration.kind == "middleware"
                ]
                loaded.commands_registered = [
                    registration.key
                    for registration in registrations
                    if registration.kind == "command"
                ]
                loaded.enabled = True
                logger.debug(
                    "  registered: %d tool(s), %d hook(s), %d middleware, %d slash command(s), %d CLI command(s)",
                    len(loaded.tools_registered),
                    len(loaded.hooks_registered),
                    len(loaded.middleware_registered),
                    len(loaded.commands_registered),
                    sum(
                        1 for c in self._cli_commands
                        if any(
                            registration.active
                            and registration.plugin_key == plugin_key
                            and registration.kind == "cli_command"
                            and registration.key == c
                            for registration in registrations
                        )
                    ),
                )

        except Exception as exc:
            owned = [
                registration
                for registration in self._registration_order
                if registration.plugin_key == plugin_key
            ]
            self._dispose_registrations(owned)
            self._forget_registrations(owned)
            loaded.error = str(exc)
            # register() may have subscribed before raising. Remove those
            # owner-tagged entries so a failed/unloaded plugin cannot leave a
            # callable reachable from later event dispatch.
            self._remove_plugin_subscriptions(plugin_key)
            logger.warning(
                "Failed to load plugin '%s': %s",
                manifest.name, exc, exc_info=_PLUGINS_DEBUG,
            )
        self._plugins[manifest.key or manifest.name] = loaded

    def _load_portable_plugin(
        self, manifest: PluginManifest, loaded: LoadedPlugin
    ) -> None:
        """Load validated portable components without importing Python code."""

        lookup_key = manifest.key or manifest.name
        try:
            from hermes_cli.agent_plugins import load_agent_plugin

            package = load_agent_plugin(
                Path(manifest.path),
                get_hermes_home() / "plugin-data" / manifest.skill_namespace,
            )
            ctx = PluginContext(manifest, self)
            for diagnostic in package.diagnostics:
                logger.warning(
                    "Agent Plugin '%s' [%s]: %s",
                    lookup_key,
                    diagnostic.scope,
                    diagnostic.message,
                )
            for skill in package.skills:
                try:
                    ctx.register_skill(
                        skill.name,
                        skill.skill_md,
                        skill.description,
                        skill.frontmatter,
                    )
                except Exception as exc:
                    logger.warning(
                        "Agent Plugin '%s' skill '%s' skipped: %s",
                        lookup_key,
                        skill.name,
                        exc,
                    )
            for server_name, config in package.mcp_servers.items():
                internal_name = f"{manifest.skill_namespace}__{server_name}"
                if internal_name in self._portable_mcp_servers:
                    logger.warning(
                        "Agent Plugin '%s' MCP server collision: %s",
                        lookup_key,
                        internal_name,
                    )
                    continue
                self._portable_mcp_servers[internal_name] = dict(config)
            loaded.enabled = True
        except Exception as exc:
            loaded.error = str(exc)
            logger.warning("Failed to load Agent Plugin '%s': %s", lookup_key, exc)
        self._plugins[lookup_key] = loaded

    def _directory_module_name(self, manifest: PluginManifest) -> str:
        """Return a profile-safe import namespace for a directory plugin."""
        key = manifest.key or manifest.name
        slug = key.replace("/", "__").replace("-", "_")
        bare_name = f"{_NS_PARENT}.{slug}"
        with _MODULE_NAMESPACE_LOCK:
            owner = _BARE_MODULE_SCOPE.get(bare_name)
            if owner is None:
                _BARE_MODULE_SCOPE[bare_name] = self.scope_key
                return bare_name
            if owner == self.scope_key:
                return bare_name
            digest = hashlib.sha256(self.scope_key.encode("utf-8")).hexdigest()[:12]
            return f"{bare_name}__home_{digest}"

    def _policy_module_name(self, manifest: PluginManifest) -> str:
        """Return the module prefix whose callbacks inherit plugin policy."""
        if manifest.source == "entrypoint" and manifest.path:
            module_name = str(manifest.path).partition(":")[0].strip()
            if module_name:
                return module_name
        return self._directory_module_name(manifest)

    def _load_directory_module(
        self,
        manifest: PluginManifest,
        *,
        module_name: Optional[str] = None,
    ) -> types.ModuleType:
        """Import a directory-based plugin as ``hermes_plugins.<slug>``.

        The module slug is derived from ``manifest.key`` so category-namespaced
        plugins (``image_gen/openai``) import as
        ``hermes_plugins.image_gen__openai`` without colliding with any
        future ``tts/openai``.
        """
        plugin_dir = Path(manifest.path)  # type: ignore[arg-type]
        init_file = plugin_dir / "__init__.py"
        if not init_file.exists():
            raise FileNotFoundError(f"No __init__.py in {plugin_dir}")

        # Ensure the namespace parent package exists
        if _NS_PARENT not in sys.modules:
            ns_pkg = types.ModuleType(_NS_PARENT)
            ns_pkg.__path__ = []  # type: ignore[attr-defined]
            ns_pkg.__package__ = _NS_PARENT
            sys.modules[_NS_PARENT] = ns_pkg

        module_name = module_name or self._directory_module_name(manifest)

        # Evict any stale sys.modules entries for this slug before
        # (re-)importing. A same-slug module may already be cached here
        # from a different Hermes home (profile switch reusing a slug
        # like "hermes-lcm") or from an earlier force=True reload in the
        # same home. Replacing only sys.modules[module_name] below is not
        # enough: the plugin's own relative imports (`from . import foo`)
        # are cached separately under "module_name + '.' + submodule",
        # and Python's import system resolves those from sys.modules
        # first — so a stale submodule would silently keep serving the
        # previous load's code/state instead of the fresh one we're
        # about to exec. Evict the package and everything nested under
        # it so this import starts clean.
        stale_prefix = f"{module_name}."
        for name in [n for n in sys.modules if n == module_name or n.startswith(stale_prefix)]:
            del sys.modules[name]

        spec = importlib.util.spec_from_file_location(
            module_name,
            init_file,
            submodule_search_locations=[str(plugin_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {init_file}")

        module = importlib.util.module_from_spec(spec)
        module.__package__ = module_name
        module.__path__ = [str(plugin_dir)]  # type: ignore[attr-defined]
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            # Don't leave a half-initialized module (or the partially
            # imported relative submodules it pulled in before failing)
            # cached in sys.modules — a retry or a same-slug plugin in a
            # different profile would otherwise inherit broken state.
            for name in [n for n in sys.modules if n == module_name or n.startswith(stale_prefix)]:
                del sys.modules[name]
            raise
        return module

    def _load_entrypoint_module(self, manifest: PluginManifest) -> types.ModuleType:
        """Load a pip-installed plugin via its entry-point reference."""
        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            group_eps = eps.select(group=ENTRY_POINTS_GROUP)
        elif isinstance(eps, dict):
            group_eps = eps.get(ENTRY_POINTS_GROUP, [])
        else:
            group_eps = [ep for ep in eps if ep.group == ENTRY_POINTS_GROUP]

        for ep in group_eps:
            if ep.name == manifest.name:
                return ep.load()

        raise ImportError(
            f"Entry point '{manifest.name}' not found in group '{ENTRY_POINTS_GROUP}'"
        )

    # -----------------------------------------------------------------------
    # Hook invocation
    # -----------------------------------------------------------------------

    @staticmethod
    def _invoke_hook_callback(callback: Callable, payload: Dict[str, Any]) -> Any:
        """Invoke a hook while withholding additive fields from old callbacks."""
        try:
            parameters = inspect.signature(callback).parameters
        except (TypeError, ValueError):
            # Some extension/builtin callables do not expose a signature. Keep
            # the historical behavior for those callables rather than guessing.
            return callback(**payload)

        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            return callback(**payload)

        accepted_payload = {
            name: value
            for name, value in payload.items()
            if name in parameters
            and parameters[name].kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }
        return callback(**accepted_payload)

    def invoke_hook(self, hook_name: str, **kwargs: Any) -> List[Any]:
        """Call all registered callbacks for *hook_name*.

        Hook payloads evolve additively. Callbacks that accept ``**kwargs``
        receive the complete payload; older callbacks with a narrow signature
        receive only the keyword arguments they declare. Each callback is
        wrapped in its own try/except so a misbehaving plugin cannot break the
        core agent loop.

        Returns a list of non-``None`` return values from callbacks.

        For ``pre_llm_call``, callbacks may return a dict describing
        context to inject into the current turn's user message::

            {"context": "recalled text..."}
            "recalled text..."          # plain string, equivalent

        Context is ALWAYS injected into the user message, never the
        system prompt.  This preserves the prompt cache prefix — the
        system prompt stays identical across turns so cached tokens
        are reused.  All injected context is ephemeral — never
        persisted to session DB.
        """
        # Most legacy observer hooks carry the shared telemetry marker. Gateway
        # platform events define event-local additive envelopes instead: injecting
        # a bus-wide version here would turn unrelated adapter payloads into one
        # monolithic compatibility contract (#64176).
        if hook_name != "gateway_platform_event":
            kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
        callbacks = self._hooks.get(hook_name, [])
        results: List[Any] = []
        for cb in callbacks:
            try:
                ret = self._invoke_hook_callback(cb, kwargs)
                if ret is not None:
                    results.append(ret)
            except Exception as exc:
                logger.warning(
                    "Hook '%s' callback %s raised: %s",
                    hook_name,
                    getattr(cb, "__name__", repr(cb)),
                    exc,
                )
        return results

    def _subscribe_event(
        self,
        owner: str,
        event: str,
        callback: Callable,
    ) -> None:
        """Add an owner-tagged event subscription in registration order."""
        if not callable(callback):
            raise TypeError("Event subscriber callback must be callable")
        entry = _EventSubscription(owner=owner, callback=callback)
        with self._event_lock:
            self._subscriptions.setdefault(event, []).append(entry)

    def _remove_plugin_subscriptions(self, owner: str) -> int:
        """Remove every subscription owned by *owner* and return the count.

        Queued dispatch envelopes re-check ledger membership before each
        callback, so removing an owner also cancels callbacks already snapshotted
        by an event that has not reached that subscriber yet.

        TODO(#64229): when the central plugin ownership ledger / registration
        handles land, route this owner-tagged bookkeeping through that ledger
        so per-plugin unload cancels event subscriptions alongside every other
        registration surface. This method is the integration seam.
        """
        removed = 0
        with self._event_lock:
            for event in list(self._subscriptions):
                entries = self._subscriptions[event]
                retained = [entry for entry in entries if entry.owner != owner]
                removed += len(entries) - len(retained)
                if retained:
                    self._subscriptions[event] = retained
                else:
                    del self._subscriptions[event]
        return removed

    def _reset_event_bus(self) -> None:
        """Cancel the current event generation and clear its subscriptions."""
        with self._event_lock:
            old_queue = self._event_queue
            had_worker = self._event_worker is not None
            self._event_generation += 1
            self._subscriptions.clear()
            self._event_queue = queue.Queue(maxsize=_EVENT_PENDING_CAP)
            self._event_worker = None
            self._event_pending_by_generation.setdefault(
                self._event_generation, 0
            )

            # Drop work that has not started. A currently-running callback
            # cannot be force-killed safely, but generation + ledger checks stop
            # it before the next subscriber and prevent all queued callbacks.
            while True:
                try:
                    item = old_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    if item is not _EVENT_WORKER_STOP:
                        self._mark_event_done(item.generation)
                finally:
                    old_queue.task_done()
            if had_worker:
                old_queue.put_nowait(_EVENT_WORKER_STOP)
            self._event_idle.notify_all()

    def _ensure_event_worker_locked(self) -> None:
        worker = self._event_worker
        if worker is not None and worker.is_alive():
            return
        dispatch_queue = self._event_queue
        worker = threading.Thread(
            target=self._event_worker_loop,
            args=(dispatch_queue,),
            name="hermes-plugin-events",
            daemon=True,
        )
        self._event_worker = worker
        worker.start()

    def _event_worker_loop(self, dispatch_queue: queue.Queue[Any]) -> None:
        while True:
            item = dispatch_queue.get()
            try:
                if item is _EVENT_WORKER_STOP:
                    return
                self._deliver_event(item)
            finally:
                if item is not _EVENT_WORKER_STOP:
                    self._mark_event_done(item.generation)
                dispatch_queue.task_done()

    def _mark_event_done(self, generation: int) -> None:
        with self._event_idle:
            pending = self._event_pending_by_generation.get(generation, 0)
            if pending > 0:
                self._event_pending_by_generation[generation] = pending - 1
            self._event_idle.notify_all()

    def _deliver_event(self, item: _QueuedPluginEvent) -> None:
        """Deliver one queued event on the host-owned worker thread."""
        with self._event_lock:
            if item.generation != self._event_generation:
                return
        previous_depth = getattr(self._emit_depth, "value", 0)
        self._emit_depth.value = item.depth
        try:
            for subscription in item.subscriptions:
                with self._event_lock:
                    if item.generation != self._event_generation:
                        break
                    # Owner unload may remove this exact ledger entry after the
                    # event was queued but before its callback starts.
                    if not any(
                        current is subscription
                        for current in self._subscriptions.get(item.event, [])
                    ):
                        continue
                callback = subscription.callback
                try:
                    # A fresh deep copy per subscriber prevents one callback
                    # from mutating the emitter's nested values or the payload
                    # observed by the next subscriber.
                    owned_payload = copy.deepcopy(item.payload)
                    result = callback(**owned_payload)
                    resolve_plugin_command_result(result)
                except Exception as exc:
                    logger.warning(
                        "Event '%s' subscriber %s raised: %s",
                        item.event,
                        getattr(callback, "__name__", repr(callback)),
                        exc,
                    )
        finally:
            self._emit_depth.value = previous_depth

    def _wait_for_event_dispatch(self, timeout: float = 2.0) -> bool:
        """Wait for the current event generation to become idle (test helper)."""
        with self._event_idle:
            generation = self._event_generation
            return self._event_idle.wait_for(
                lambda: self._event_pending_by_generation.get(generation, 0) == 0,
                timeout=timeout,
            )

    def _dispatch_event(self, event: str, payload: Dict[str, Any]) -> int:
        """Queue *event* without blocking; return subscriber count scheduled.

        A single daemon worker preserves registration order. Pending work is
        bounded per manager generation so a blocking subscriber can consume at
        most one worker while later emits are dropped once the budget is full.
        """
        depth = getattr(self._emit_depth, "value", 0)
        if depth >= _EVENT_EMIT_DEPTH_CAP:
            logger.warning(
                "Event bus recursion cap (%d) exceeded while dispatching '%s' "
                "— dropping this emit to prevent an infinite loop",
                _EVENT_EMIT_DEPTH_CAP, event,
            )
            return 0

        with self._event_lock:
            subscriptions = tuple(self._subscriptions.get(event, []))
            if not subscriptions:
                return 0
            generation = self._event_generation
            pending = self._event_pending_by_generation.get(generation, 0)
            if pending >= _EVENT_PENDING_CAP:
                logger.warning(
                    "Event bus pending budget (%d) exhausted while dispatching "
                    "'%s' — dropping this emit",
                    _EVENT_PENDING_CAP,
                    event,
                )
                return 0
            item = _QueuedPluginEvent(
                event=event,
                payload=dict(payload),
                subscriptions=subscriptions,
                depth=depth + 1,
                generation=generation,
            )
            try:
                self._event_queue.put_nowait(item)
            except queue.Full:
                logger.warning(
                    "Event bus pending budget (%d) exhausted while dispatching "
                    "'%s' — dropping this emit",
                    _EVENT_PENDING_CAP,
                    event,
                )
                return 0
            self._event_pending_by_generation[generation] = pending + 1
            self._ensure_event_worker_locked()
            return len(subscriptions)

    def has_hook(self, hook_name: str) -> bool:
        """Return True when at least one callback is registered for a hook."""
        return bool(self._hooks.get(hook_name))

    def iter_hook_callbacks(self, hook_name: str) -> tuple[Callable, ...]:
        """Return a stable snapshot of callbacks registered for a hook."""
        return tuple(self._hooks.get(hook_name, ()))

    def render_system_prompt_sections(
        self, session_info: Mapping[str, Any]
    ) -> List[RenderedPluginSystemPromptSection]:
        """Render all registered sections deterministically and fail open."""
        frozen_info = types.MappingProxyType(dict(session_info))
        rendered: List[RenderedPluginSystemPromptSection] = []
        total_chars = len(PLUGIN_SECTIONS_START) + len(PLUGIN_SECTIONS_END) + 2
        for section_id in sorted(self._system_prompt_sections):
            section = self._system_prompt_sections[section_id]
            if len(rendered) >= MAX_SYSTEM_PROMPT_SECTIONS:
                logger.warning(
                    "Plugin system prompt section %s exceeded the section-count "
                    "budget (%d) and was skipped",
                    section.id,
                    MAX_SYSTEM_PROMPT_SECTIONS,
                )
                continue
            try:
                value = (
                    section.content(frozen_info)
                    if callable(section.content)
                    else section.content
                )
            except Exception as exc:
                logger.warning(
                    "Plugin system prompt section %s (%s) raised and was skipped: %s",
                    section.id,
                    section.plugin,
                    exc,
                )
                continue
            if not isinstance(value, str):
                logger.warning(
                    "Plugin system prompt section %s (%s) returned %s, not str; skipped",
                    section.id,
                    section.plugin,
                    type(value).__name__,
                )
                continue
            text = value.strip()
            if not text:
                continue
            if PLUGIN_SECTIONS_START in text or PLUGIN_SECTIONS_END in text:
                logger.warning(
                    "Plugin system prompt section %s (%s) contained a reserved "
                    "persistence marker and was skipped",
                    section.id,
                    section.plugin,
                )
                continue
            if len(text) > section.max_chars:
                logger.warning(
                    "Plugin system prompt section %s (%s) exceeded max_chars "
                    "(%d > %d) and was skipped",
                    section.id,
                    section.plugin,
                    len(text),
                    section.max_chars,
                )
                continue
            rendered_chars = len(format_system_prompt_section(section.id, text))
            if rendered:
                rendered_chars += 2  # canonical ``\n\n`` separator
            if total_chars + rendered_chars > MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS:
                logger.warning(
                    "Plugin system prompt section %s (%s) exceeded the aggregate "
                    "session budget (%d chars) and was skipped",
                    section.id,
                    section.plugin,
                    MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS,
                )
                continue
            rendered.append(
                RenderedPluginSystemPromptSection(
                    id=section.id,
                    content=text,
                    position=section.position,
                    plugin=section.plugin,
                )
            )
            total_chars += rendered_chars
            logger.info(
                "Session plugin prompt section: id=%s plugin=%s position=%s chars=%d",
                section.id,
                section.plugin,
                section.position,
                len(text),
            )
        return rendered

    def has_middleware(self, kind: str) -> bool:
        """Return True when at least one callback is registered for middleware."""
        return bool(self._middleware.get(kind))

    def invoke_middleware(self, kind: str, **kwargs: Any) -> List[Any]:
        """Call registered middleware callbacks for *kind*.

        Each callback is isolated so one plugin cannot break the base runtime
        path. Middleware that wants to change behavior must return the shape
        documented by the caller-specific contract.
        """
        callbacks = self._middleware.get(kind, [])
        results: List[Any] = []
        for cb in callbacks:
            try:
                ret = cb(**kwargs)
                if ret is not None:
                    results.append(ret)
            except Exception as exc:
                logger.warning(
                    "Middleware '%s' callback %s raised: %s",
                    kind,
                    getattr(cb, "__name__", repr(cb)),
                    exc,
                )
        return results

    # -----------------------------------------------------------------------
    # Slack action handler accessor
    # -----------------------------------------------------------------------

    def get_slack_action_handlers(self) -> List[tuple]:
        """Return the list of plugin-registered Slack action handlers.

        Each entry is a ``(action_id, callback, plugin_name)`` tuple.
        Consumed by the Slack adapter at connect time to wire callbacks
        into its ``slack_bolt.AsyncApp``.

        Plugins register handlers via
        :meth:`PluginContext.register_slack_action_handler`.
        """
        return list(self._slack_action_handlers)

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    def list_plugins(self) -> List[Dict[str, Any]]:
        """Return a list of info dicts for all discovered plugins."""
        result: List[Dict[str, Any]] = []
        for key, loaded in sorted(self._plugins.items()):
            result.append(
                {
                    "name": loaded.manifest.name,
                    "key": loaded.manifest.key or loaded.manifest.name,
                    "kind": loaded.manifest.kind,
                    "version": loaded.manifest.version,
                    "description": loaded.manifest.description,
                    "source": loaded.manifest.source,
                    "enabled": loaded.enabled,
                    "tools": len(loaded.tools_registered),
                    "hooks": len(loaded.hooks_registered),
                    "middleware": len(loaded.middleware_registered),
                    "commands": len(loaded.commands_registered),
                    "error": loaded.error,
                }
            )
        return result

    # -----------------------------------------------------------------------
    # Plugin skill lookups
    # -----------------------------------------------------------------------

    def find_plugin_skill(self, qualified_name: str) -> Optional[Path]:
        """Return the ``Path`` to a plugin skill's SKILL.md, or ``None``."""
        entry = self._plugin_skills.get(qualified_name)
        return entry["path"] if entry else None

    def list_plugin_skills(self, plugin_name: str) -> List[str]:
        """Return sorted bare names of all skills registered by *plugin_name*."""
        prefix = f"{plugin_name}:"
        return sorted(
            e["bare_name"]
            for qn, e in self._plugin_skills.items()
            if qn.startswith(prefix)
        )

    def list_plugin_skill_metadata(self) -> List[Dict[str, Any]]:
        """Return progressive-disclosure metadata for registered plugin skills."""

        return [
            {
                "name": qualified,
                "description": str(entry.get("description", "")),
                "category": "plugin",
                "frontmatter": dict(entry.get("frontmatter", {})),
            }
            for qualified, entry in sorted(self._plugin_skills.items())
        ]

    def get_portable_mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """Return a defensive copy of enabled portable MCP server configs."""

        return {
            name: dict(config)
            for name, config in self._portable_mcp_servers.items()
        }

    def has_portable_mcp_servers(self) -> bool:
        return bool(self._portable_mcp_servers)

    def remove_plugin_skill(self, qualified_name: str) -> None:
        """Remove a stale registry entry (silently ignores missing keys)."""
        self._plugin_skills.pop(qualified_name, None)


# ---------------------------------------------------------------------------
# Module-level singleton & convenience functions
# ---------------------------------------------------------------------------

# Legacy single-slot singleton. Kept as the storage for the "current"
# manager so existing test code that does
# ``monkeypatch.setattr(plugins_mod, "_plugin_manager", some_manager)``
# keeps working — ``get_plugin_manager()`` still reads/writes this name.
_plugin_manager: Optional[PluginManager] = None

# Keyed cache: resolved Hermes home -> PluginManager. Hermes supports
# multiple profiles via different HERMES_HOME directories, and a single
# long-lived process (gateway multiplexer, test session, embedder) can
# switch between them via ``set_hermes_home_override()`` — which is a
# ContextVar and deliberately does NOT touch os.environ (see
# hermes_constants.set_hermes_home_override). A process-wide single-slot
# cache leaks one profile's plugin/context-engine state into another. We
# key the cache by the *resolved* home path so re-entering a previously
# seen profile reuses its manager (and picks up any modules it already
# imported) instead of rebuilding from scratch every switch.
_plugin_managers_by_home: Dict[Path, PluginManager] = {}
_plugin_managers_lock = threading.RLock()


def _plugin_home_key() -> Path:
    """Return the profile/home key for process-global plugin state.

    Plugins are discovered from ``get_hermes_home() / "plugins"`` and some
    plugins (notably context engines such as hermes-lcm) capture that home
    at registration time for profile-scoped storage. A long-lived process
    can temporarily switch Hermes home (env var *or* the context-local
    ``set_hermes_home_override()``) while serving another profile, so the
    plugin manager must be scoped to the active Hermes home instead of
    being one process-wide singleton.
    """
    try:
        return get_hermes_home().expanduser().resolve()
    except Exception:
        return get_hermes_home().expanduser()


def _clear_plugin_submodules(manager: Optional[PluginManager]) -> None:
    """Purge ``sys.modules`` entries for directory-loaded plugins.

    ``PluginManager._load_directory_module`` imports each plugin as
    ``hermes_plugins.<slug>`` and registers that top-level module in
    ``sys.modules``. Anything the plugin's ``__init__.py`` imports with a
    *relative* import (``from . import foo``, ``from .sub import bar``)
    ends up cached in ``sys.modules`` too, under
    ``hermes_plugins.<slug>.<submodule>``. When we swap in a fresh manager
    for a new home, replacing only the parent module leaves those
    submodules behind: if a same-named plugin in the new profile does a
    relative import, Python resolves it from ``sys.modules`` first and
    silently reuses the *previous* profile's already-imported submodule
    (and any module-level state it captured), instead of re-executing the
    new profile's code. We must evict the package itself and every module
    whose name is prefixed with ``"<module_name>."`` before (or when)
    discarding a manager, not just drop our reference to it.
    """
    if manager is None:
        return
    for loaded in getattr(manager, "_plugins", {}).values():
        module = getattr(loaded, "module", None)
        module_name = getattr(module, "__name__", None)
        if not module_name or not module_name.startswith(f"{_NS_PARENT}."):
            continue
        prefix = f"{module_name}."
        for name in [n for n in sys.modules if n == module_name or n.startswith(prefix)]:
            del sys.modules[name]
        with _MODULE_NAMESPACE_LOCK:
            if _BARE_MODULE_SCOPE.get(module_name) == manager.scope_key:
                _BARE_MODULE_SCOPE.pop(module_name, None)


def get_plugin_manager() -> PluginManager:
    """Return the plugin manager for the active Hermes profile/home.

    Managers are cached per resolved home so repeated calls within the
    same profile reuse discovery state (normal performance), while a
    profile switch — via ``HERMES_HOME`` or the context-local
    ``set_hermes_home_override()`` — gets its own manager with its own
    plugin submodules, instead of silently inheriting another profile's
    context engine or stale relative-import state.
    """
    global _plugin_manager
    current_home = _plugin_home_key()

    with _plugin_managers_lock:
        # Tests and embedders historically monkeypatch ``_plugin_manager``
        # directly (``monkeypatch.setattr(plugins_mod, "_plugin_manager", mgr)``).
        # Detect that specifically by checking whether the single-slot pointer
        # references a manager our keyed cache doesn't know about *at all*.
        if (
            _plugin_manager is not None
            and _plugin_manager not in _plugin_managers_by_home.values()
        ):
            _plugin_managers_by_home[current_home] = _plugin_manager
            return _plugin_manager

        manager = _plugin_managers_by_home.get(current_home)
        if manager is None:
            manager = PluginManager(scope_key=hermes_home_key(current_home))
            _plugin_managers_by_home[current_home] = manager

        _plugin_manager = manager
        return manager


def _reset_plugin_managers_for_tests() -> None:
    """Test-only helper: drop every cached manager and its submodules.

    Not used by production code paths — tests that want a fully clean
    slate (rather than adopting/injecting a specific manager) can call
    this instead of reaching into the module's private dict directly.
    """
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


def has_enabled_agent_plugin_mcp(raw_config: Mapping[str, Any]) -> bool:
    """Return whether config enables a portable package with MCP servers.

    A fresh manager performs manifest-only scanning, so this startup gate does
    not mutate the process-wide plugin registry or import native plugin code.
    """
    return PluginManager().has_enabled_portable_mcp(raw_config)


def discover_plugins(force: bool = False) -> None:
    """Discover and load all plugins.

    Default behavior is idempotent. Pass ``force=True`` to rescan plugin
    manifests and reload state in the current process.

    If a background discovery started via
    :func:`start_background_plugin_discovery` is still running, this waits
    for it instead of racing a second scan.
    """
    _join_background_discovery()
    get_plugin_manager().discover_and_load(force=force)


_background_discovery_thread: Optional[threading.Thread] = None
_background_discovery_lock = threading.Lock()


def start_background_plugin_discovery() -> None:
    """Run plugin discovery in a daemon thread (startup-latency overlap).

    Discovery costs ~150ms of manifest scanning + module imports on the CLI
    startup path. Interactive chat doesn't need plugins until the first
    agent turn, so callers on that path can start discovery here and let it
    overlap the CPU/subprocess-heavy rest of startup. Every synchronous
    consumer goes through :func:`discover_plugins`, which joins this thread
    first — so no caller can observe a half-loaded registry. Idempotent;
    no-op when discovery already ran or is already in flight.
    """
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

        _background_discovery_thread = threading.Thread(
            target=_run, name="plugin-discovery", daemon=True
        )
        _background_discovery_thread.start()


def _join_background_discovery(timeout: float = 30.0) -> None:
    """Wait for an in-flight background discovery (no-op from its own thread)."""
    t = _background_discovery_thread
    if t is None or not t.is_alive() or t is threading.current_thread():
        return
    t.join(timeout=timeout)


def _plugin_toolset_keys_cache_path():
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / "plugin_toolset_keys.json"


def _persist_plugin_toolset_keys() -> None:
    """Persist discovered plugin toolset keys + portable MCP names (best-effort)."""
    try:
        import json as _json
        import os as _os
        import tempfile as _tempfile
        keys = sorted({ts_key for ts_key, _, _ in get_plugin_toolsets()})
        try:
            portable = sorted(get_plugin_manager().get_portable_mcp_servers())
        except Exception:
            portable = []
        path = _plugin_toolset_keys_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = _tempfile.mkstemp(dir=str(path.parent), prefix=".pt_keys.")
        with _os.fdopen(fd, "w", encoding="utf-8") as fh:
            _json.dump({"toolset_keys": keys, "portable_mcp": portable}, fh)
        _os.replace(tmp, path)
    except Exception:
        logger.debug("plugin toolset key persist failed", exc_info=True)


def _read_plugin_keys_cache() -> Optional[dict]:
    try:
        import json as _json
        blob = _json.loads(
            _plugin_toolset_keys_cache_path().read_text(encoding="utf-8")
        )
        if isinstance(blob, dict):
            return blob
    except Exception:
        pass
    return None


def get_plugin_toolset_keys_nowait() -> "set[str]":
    """Plugin toolset keys without blocking on in-flight discovery.

    When discovery already completed in this process, reads the live
    registry. While a background discovery is still running, falls back to
    the key set persisted by the previous run — callers on the startup path
    (platform toolset resolution) only use these keys to EXCLUDE plugin
    toolsets from composite expansion, so a stale set from the last launch
    is harmless and self-heals as soon as discovery lands. When neither is
    available, blocks via discover_plugins() (correctness first).
    """
    manager = get_plugin_manager()
    t = _background_discovery_thread
    if manager._discovered and (t is None or not t.is_alive()):
        return {ts_key for ts_key, _, _ in get_plugin_toolsets()}
    if t is not None and t.is_alive():
        blob = _read_plugin_keys_cache()
        if blob is not None:
            keys = blob.get("toolset_keys")
            if isinstance(keys, list) and all(isinstance(k, str) for k in keys):
                return set(keys)
    discover_plugins()
    return {ts_key for ts_key, _, _ in get_plugin_toolsets()}


def get_portable_mcp_server_names_nowait() -> "set[str]":
    """Portable MCP server names without blocking on in-flight discovery.

    Same contract as :func:`get_plugin_toolset_keys_nowait`: live registry
    when discovery finished, last launch's persisted set while a background
    discovery is running, blocking discovery otherwise.
    """
    manager = get_plugin_manager()
    t = _background_discovery_thread
    if manager._discovered and (t is None or not t.is_alive()):
        return set(manager.get_portable_mcp_servers())
    if t is not None and t.is_alive():
        blob = _read_plugin_keys_cache()
        if blob is not None:
            names = blob.get("portable_mcp")
            if isinstance(names, list) and all(isinstance(n, str) for n in names):
                return set(names)
    discover_plugins()
    return set(manager.get_portable_mcp_servers())


def unload_plugins(
    plugin: Union[str, PluginManifest, LoadedPlugin, None] = None,
) -> bool:
    """Unload one plugin or all plugins from the process-global manager.

    Wait for background discovery first so teardown cannot race an in-flight
    registration sweep introduced by the warm-start discovery path.
    """
    _join_background_discovery()
    return get_plugin_manager().unload(plugin)


def invoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """Invoke a lifecycle hook on loaded plugins.

    Returns a list of non-``None`` return values from plugin callbacks.
    """
    return get_plugin_manager().invoke_hook(hook_name, **kwargs)


def render_system_prompt_sections(
    session_info: Mapping[str, Any],
) -> List[RenderedPluginSystemPromptSection]:
    """Render plugin prompt sections after idempotent plugin discovery."""
    return _ensure_plugins_discovered().render_system_prompt_sections(session_info)


def invoke_middleware(kind: str, **kwargs: Any) -> List[Any]:
    """Invoke registered middleware callbacks.

    Returns a list of non-``None`` return values from middleware callbacks.
    """
    return get_plugin_manager().invoke_middleware(kind, **kwargs)


def has_middleware(kind: str) -> bool:
    """Return True when middleware callbacks are registered for ``kind``."""
    manager = get_plugin_manager()
    method = getattr(manager, "has_middleware", None)
    if callable(method):
        return bool(method(kind))
    return bool(getattr(manager, "_middleware", {}).get(kind))


def has_hook(hook_name: str) -> bool:
    """Return True when a loaded plugin handles a hook."""
    return get_plugin_manager().has_hook(hook_name)


def iter_hook_callbacks(hook_name: str) -> tuple[Callable, ...]:
    """Return a stable snapshot of callbacks registered for a hook."""
    return get_plugin_manager().iter_hook_callbacks(hook_name)


_thread_tool_whitelist = threading.local()


@dataclass(frozen=True)
class _PreToolCallDirective:
    action: Optional[str] = None
    message: Optional[str] = None
    rule_key: Optional[str] = None


def set_thread_tool_whitelist(
    allowed: Optional[Set[str]],
    deny_msg_fmt: str = "Tool '{tool_name}' denied: not in this thread's tool whitelist",
) -> None:
    _thread_tool_whitelist.allowed = allowed
    _thread_tool_whitelist.fmt = deny_msg_fmt


def clear_thread_tool_whitelist() -> None:
    _thread_tool_whitelist.allowed = None


def _get_pre_tool_call_directive_details(
    tool_name: str,
    args: Optional[Dict[str, Any]],
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    middleware_trace: Optional[List[Dict[str, Any]]] = None,
) -> _PreToolCallDirective:
    """Check ``pre_tool_call`` hooks for a blocking or approval directive.

    Plugins that need to enforce policy (rate limiting, security
    restrictions, approval workflows) can return one of::

        {"action": "block",   "message": "Reason the tool was blocked"}
        {"action": "approve", "message": "Why this needs human confirmation"}
        {"action": "approve", "message": "...", "rule_key": "write_file:ssh"}

    from their ``pre_tool_call`` callback.

    - ``block`` vetoes the tool call outright (the message becomes the tool
      result the model sees).
    - ``approve`` ESCALATES to the existing human-approval gate
      (``prompt_dangerous_approval`` on CLI, the approval callback on the
      gateway) — the same mechanism Tier-2 dangerous shell patterns use.
      This lets a plugin require a human ``[o]nce/[s]ession/[a]lways/[d]eny``
      decision on ANY tool, not just terminal command strings. The caller is
      responsible for invoking the gate (see
      :func:`tools.approval.request_tool_approval`).
    - ``rule_key`` is optional and only honored for ``approve`` directives. It
      lets plugins choose the allowlist grain for `[a]lways` approvals.

    The first valid directive wins. Invalid or irrelevant hook return values
    are silently ignored so existing observer-only hooks are unaffected.
    """
    allowed = getattr(_thread_tool_whitelist, "allowed", None)
    if allowed is not None and tool_name not in allowed:
        fmt = getattr(_thread_tool_whitelist, "fmt", "Tool '{tool_name}' denied")
        return _PreToolCallDirective(
            action="block",
            message=fmt.format(tool_name=tool_name),
        )

    from hermes_cli.lifecycle import invoke_hook as invoke_lifecycle_hook

    hook_results = invoke_lifecycle_hook(
        "pre_tool_call",
        tool_name=tool_name,
        args=args if isinstance(args, dict) else {},
        task_id=task_id,
        session_id=session_id,
        tool_call_id=tool_call_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
        middleware_trace=list(middleware_trace or []),
    )

    for result in hook_results:
        if not isinstance(result, dict):
            continue
        action = result.get("action")
        if action not in ("block", "approve"):
            continue
        message = result.get("message")
        message = message if isinstance(message, str) and message else None
        # A block directive requires a message (it becomes the tool result);
        # an approve directive can carry an optional reason.
        if action == "block" and not message:
            continue
        rule_key = result.get("rule_key") if action == "approve" else None
        rule_key = rule_key.strip() if isinstance(rule_key, str) else None
        if not rule_key:
            rule_key = None
        return _PreToolCallDirective(action=action, message=message, rule_key=rule_key)

    return _PreToolCallDirective()


def get_pre_tool_call_directive(
    tool_name: str,
    args: Optional[Dict[str, Any]],
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    middleware_trace: Optional[List[Dict[str, Any]]] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Check ``pre_tool_call`` hooks for a blocking or approval directive.

    Backward-compatible public helper: returns ``(directive, message)`` where
    ``directive`` is ``"block"``, ``"approve"``, or ``None``. Internal callers
    that need approve-specific metadata use
    :func:`_get_pre_tool_call_directive_details`.
    """
    details = _get_pre_tool_call_directive_details(
        tool_name, args, task_id=task_id, session_id=session_id,
        tool_call_id=tool_call_id, turn_id=turn_id,
        api_request_id=api_request_id, middleware_trace=middleware_trace,
    )
    return (details.action, details.message)


def get_pre_tool_call_block_message(
    tool_name: str,
    args: Optional[Dict[str, Any]],
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    middleware_trace: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Back-compat shim: return only a ``block`` message (or ``None``).

    Deprecated in favor of :func:`get_pre_tool_call_directive`, which also
    surfaces the ``approve`` escalation directive. Kept so any external caller
    importing the old name keeps working; ``approve`` directives are invisible
    to this shim (it only reports blocks).
    """
    directive, message = get_pre_tool_call_directive(
        tool_name, args, task_id=task_id, session_id=session_id,
        tool_call_id=tool_call_id, turn_id=turn_id,
        api_request_id=api_request_id, middleware_trace=middleware_trace,
    )
    return message if directive == "block" else None


def resolve_pre_tool_block(
    tool_name: str,
    args: Optional[Dict[str, Any]],
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    middleware_trace: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Resolve the pre_tool_call directive to a final block message (or None).

    Single entry point for every tool-dispatch site: fetches the plugin
    directive and, for an ``approve`` escalation, invokes the human-approval
    gate (:func:`tools.approval.request_tool_approval`). Returns the message
    the tool result should carry when the call is blocked, or ``None`` when
    the call may proceed.

    Centralizing this keeps the security-critical fail-closed logic in ONE
    place instead of copy-pasted across the concurrent/sequential/helper
    dispatch paths: an ``approve`` directive whose gate errors, denies, or
    times out is fail-closed to a block; ``block`` blocks with its message;
    anything else proceeds.
    """
    details = _get_pre_tool_call_directive_details(
        tool_name, args, task_id=task_id, session_id=session_id,
        tool_call_id=tool_call_id, turn_id=turn_id,
        api_request_id=api_request_id, middleware_trace=middleware_trace,
    )
    if details.action == "block":
        return details.message
    if details.action == "approve":
        try:
            from tools.approval import (
                request_tool_approval,
                reset_current_observability_context,
                set_current_observability_context,
            )

            approval_tokens = None
            try:
                approval_tokens = set_current_observability_context(
                    turn_id=turn_id,
                    tool_call_id=tool_call_id,
                )
            except Exception:
                pass
            try:
                result = request_tool_approval(
                    tool_name,
                    details.message or "",
                    rule_key=details.rule_key or tool_name,
                )
            finally:
                if approval_tokens is not None:
                    try:
                        reset_current_observability_context(approval_tokens)
                    except Exception:
                        pass
        except Exception:
            # Fail-closed: if the gate itself errors, block rather than
            # silently execute an action a plugin flagged for approval.
            return f"BLOCKED: plugin approval gate failed for {tool_name}"
        if not result.get("approved"):
            return str(
                result.get("message")
                or f"BLOCKED: plugin approval required for {tool_name}"
            )
    return None


def get_pre_verify_continue_message(
    *,
    session_id: str = "",
    platform: str = "",
    model: str = "",
    coding: bool = False,
    attempt: int = 0,
    final_response: str = "",
    changed_paths: Optional[List[str]] = None,
) -> Optional[str]:
    """Check user ``pre_verify`` hooks for a directive to keep the agent going.

    Fired once per turn when the agent edited code and is about to verify/finish.
    A hook keeps the turn going (run a check, defer it, tidy the diff) by
    returning::

        {"action": "continue", "message": "<follow-up for the model>"}

    The Claude-Code Stop shape ``{"decision": "block", "reason": "..."}`` (block
    the stop == keep going) is accepted too. The first directive carrying a
    non-empty message wins; any other return lets the turn finish. Mirrors
    :func:`get_pre_tool_call_block_message` — the call site stays a one-liner.

    ``coding`` / ``attempt`` let a hook scope itself (``if not coding`` …) and
    self-throttle (``if attempt`` …), the same way a ``pre_tool_call`` hook
    scopes on ``tool_name``.
    """
    hook_results = invoke_hook(
        "pre_verify",
        session_id=session_id,
        platform=platform,
        model=model,
        coding=coding,
        attempt=attempt,
        final_response=final_response,
        changed_paths=list(changed_paths or []),
    )

    for result in hook_results:
        if not isinstance(result, dict):
            continue
        action = str(result.get("action") or result.get("decision") or "").strip().lower()
        if action not in ("continue", "block"):
            continue
        message = result.get("message") or result.get("reason")
        if isinstance(message, str) and message.strip():
            return message.strip()

    return None


def _ensure_plugins_discovered(force: bool = False) -> PluginManager:
    """Return the global manager after ensuring plugin discovery has run.

    Pass ``force=True`` to rescan in the current process.
    """
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
    """Resolve a plugin command return value, awaiting async handlers when needed.

    Sync CLI/TUI dispatch sites call plugin handlers from plain functions.
    If a handler is async, await it directly when no loop is running; if
    we're already inside an active loop, run it in a helper thread with its
    own loop so the caller still gets a concrete result synchronously. The
    threaded path is bounded by a 30s timeout so a hung async handler cannot
    wedge the terminal indefinitely.
    """
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

    thread = threading.Thread(
        target=_runner,
        name="hermes-plugin-command-await",
        daemon=True,
    )
    thread.start()
    if not done.wait(timeout=_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS):
        raise TimeoutError(
            "Plugin command async handler did not complete within "
            f"{_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS:.0f}s"
        )
    if "exc" in failure:
        raise failure["exc"]
    return outcome.get("value")


def get_plugin_commands() -> Dict[str, dict]:
    """Return the full plugin commands dict (name → {handler, description, plugin}).

    Triggers idempotent plugin discovery so callers can use plugin commands
    before any explicit discover_plugins() call.
    """
    return _ensure_plugins_discovered()._plugin_commands


def get_plugin_auxiliary_tasks() -> List[Dict[str, Any]]:
    """Return all plugin-registered auxiliary tasks as a stable-ordered list.

    Each entry is the registration dict from
    :meth:`PluginContext.register_auxiliary_task`:
    ``{key, display_name, description, defaults, plugin}``.

    Triggers idempotent plugin discovery so callers can read the registry
    before any explicit ``discover_plugins()`` call. Sorted by ``key`` for
    deterministic ordering in pickers and tests.
    """
    manager = _ensure_plugins_discovered()
    return [manager._aux_tasks[k] for k in sorted(manager._aux_tasks)]


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


def get_plugin_toolsets() -> List[tuple]:
    """Return plugin toolsets as ``(key, label, description)`` tuples.

    Used by the ``hermes tools`` TUI so plugin-provided toolsets appear
    alongside the built-in ones and can be toggled on/off per platform.
    """
    manager = get_plugin_manager()
    if not manager._plugin_tool_names:
        return []

    try:
        from tools.registry import registry
    except Exception:
        return []

    # Group plugin tool names by their toolset
    toolset_tools: Dict[str, List[str]] = {}
    toolset_plugin: Dict[str, LoadedPlugin] = {}
    for tool_name in manager._plugin_tool_names:
        entry = registry.get_entry(tool_name)
        if not entry:
            continue
        ts = entry.toolset
        toolset_tools.setdefault(ts, []).append(entry.name)

    # Map toolsets back to the plugin that registered them
    for _name, loaded in manager._plugins.items():
        for tool_name in loaded.tools_registered:
            entry = registry.get_entry(tool_name)
            if entry and entry.toolset in toolset_tools:
                toolset_plugin.setdefault(entry.toolset, loaded)

    result = []
    for ts_key in sorted(toolset_tools):
        plugin = toolset_plugin.get(ts_key)
        label = f"🔌 {ts_key.replace('_', ' ').title()}"
        if plugin and plugin.manifest.description:
            desc = plugin.manifest.description
        else:
            desc = ", ".join(sorted(toolset_tools[ts_key]))
        result.append((ts_key, label, desc))

    return result
