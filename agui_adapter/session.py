"""Hermes agent construction and run-scoped tool dispatch for AG-UI.

The server retains one agent and canonical history per live conversation.
A frontend-tool continuation stays in its original worker and does not
reconstruct history or inject another user turn.

Frontend (client-executed) tools are implemented WITHOUT any change to Hermes
core. The HTTP server supplies a run-scoped relay that blocks the frontend
handler until the next HTTP request returns its matching result. Call identity
comes from Hermes' existing tool-start callback.

Dispatch without a live relay fails closed. Tool schemas are merged into
``agent.tools`` and ``valid_tool_names`` so the model sees dispatchable tools.

A batch that contains a frontend tool is never all-parallel-safe, so Hermes
runs it sequentially. A server action before a frontend wait completes before
handoff; an action after it waits for the frontend result. No action is replayed
by reconstructing an interrupted turn.
"""

from __future__ import annotations

import contextvars
import copy
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

# The shared-state store for the currently-executing run, so the (module-level,
# shared) state-writer tool handler can mutate the right run's state. Set per
# run; read on the run's execution thread.
_CURRENT_STATE: contextvars.ContextVar[Optional[RunState]] = contextvars.ContextVar(
    "agui_current_state", default=None
)


@dataclass(frozen=True)
class StateWriterSpec:
    """How one state-writer tool call maps into shared state (immutable per run).

    * ``state_key`` — the top-level shared-state key this tool writes.
    * ``arg`` — which tool argument carries the value. ``None`` means "merge the
      whole args dict into ``state[state_key]``" (or, if ``state_key`` is empty,
      merge into the top-level state).
    * ``mode`` — ``"replace"`` (default): ``state[state_key] = value``;
      ``"append"``: treat ``state[state_key]`` as a list and append ``value``.
      ``RunState.apply`` treats any non-``"append"`` value as ``"replace"``.
    """

    state_key: str = ""
    arg: Optional[str] = None
    mode: Literal["replace", "append"] = "replace"


@dataclass
class RunState:
    """Run-scoped shared-state store.

    Seeded from inbound ``RunAgentInput.state`` so snapshots carry UI-set keys
    (e.g. ``preferences``) alongside agent-written keys (e.g. ``notes``). Each
    state-writer tool call mutates ``state`` and records a deep-copied snapshot
    so the server can emit one ``StateSnapshotEvent`` per call, in call order.
    """

    state: dict[str, Any] = field(default_factory=dict)
    specs: dict[str, StateWriterSpec] = field(default_factory=dict)
    # (tool_call args snapshot) recorded per state-writer invocation, in order.
    # Each entry is the FULL state as it stood right after that tool applied.
    snapshots: list[dict[str, Any]] = field(default_factory=list)

    def apply(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Merge a state-writer tool's args into state; return the new snapshot."""
        spec = self.specs.get(tool_name)
        if spec is not None:
            value = args if spec.arg is None else args.get(spec.arg)
            if spec.state_key:
                if spec.mode == "append":
                    current = self.state.get(spec.state_key)
                    lst = list(current) if isinstance(current, list) else []
                    lst.append(value)
                    self.state[spec.state_key] = lst
                else:
                    self.state[spec.state_key] = value
            elif isinstance(value, dict):
                # No state_key: merge the args dict into the top-level state.
                self.state.update(value)
        snap = copy.deepcopy(self.state)
        self.snapshots.append(snap)
        return snap


_registered_frontend_names: set[str] = set()
_reg_lock = threading.RLock()

# Toolsets this adapter itself registers into the process-global registry, one
# per kind of client declaration. An entry in one of these is the adapter's OWN
# handler left over from an earlier run (registration is idempotent and
# process-global, declarations are per-run) rather than a server tool.
_FRONTEND_TOOLSET = "agui-frontend"
_STATE_WRITER_TOOLSET = "agui-state-writer"
_ADAPTER_TOOLSETS = frozenset({_FRONTEND_TOOLSET, _STATE_WRITER_TOOLSET})


class ToolNameCollisionError(ValueError):
    """A client-declared tool name is already owned by a server tool.

    ``registry.dispatch`` resolves purely by name (``tools/registry.py``), so a
    frontend or state-writer declaration that reuses a server tool name such as
    ``terminal`` or ``write_file`` would make the model's call execute that
    server tool instead of handing off to the client. The run is rejected
    outright rather than silently degraded.
    """

    def __init__(self, names) -> None:
        self.names = sorted(names)
        joined = ", ".join(self.names)
        super().__init__(
            f"Client-declared tool name(s) conflict with existing tools: {joined}. "
            "Rename the client tool(s); server tool names are reserved, and a "
            "name cannot switch between frontend and state-writer."
        )


def _reject_name_collisions(
    frontend_names: set[str], state_writer_names: set[str]
) -> None:
    """Raise :class:`ToolNameCollisionError` if a declaration shadows a server tool.

    Call this BEFORE any schema merge or registration. It must also run AFTER
    ``run_agent`` has been imported: ``model_tools`` executes
    ``discover_builtin_tools()`` at import time, and until that happens the
    registry is empty and every check below passes vacuously.
    """
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS
    from model_tools import _LEGACY_TOOL_ALIASES
    from tools.connectors.gateway.names import is_connector_name
    from tools.registry import registry
    from tools.tool_search import BRIDGE_TOOL_NAMES

    both = set(frontend_names) & set(state_writer_names)
    if both:
        raise ToolNameCollisionError(both)

    # The adapter's own leftover registration is exempt only when the name is
    # being re-declared as the SAME kind of tool. A blanket "is it one of our
    # toolsets?" exemption would let a state-writer name be re-declared as a
    # frontend tool on a later run: registration is skipped (the name is
    # already in the registry), so the name stays bound to the state-writer
    # handler while being advertised as client-executed — the same shadowing
    # bug this guard exists to prevent, one layer in.
    reserved = set(_LEGACY_TOOL_ALIASES) | set(BRIDGE_TOOL_NAMES)
    reserved |= set(INLINE_TOOL_EXECUTORS)
    collisions = (set(frontend_names) | set(state_writer_names)) & reserved
    for names, own_toolset in (
        (frontend_names, _FRONTEND_TOOLSET),
        (state_writer_names, _STATE_WRITER_TOOLSET),
    ):
        for name in names:
            entry = registry.get_entry(name)
            if is_connector_name(name) or (
                entry is not None and entry.toolset != own_toolset
            ):
                collisions.add(name)
    if collisions:
        raise ToolNameCollisionError(collisions)


def _load_agui_config() -> dict:
    """Return the ``agui`` section of ``config.yaml`` (empty dict if unavailable).

    Behavioral adapter settings live here, not in environment variables (Hermes
    reserves ``.env`` for secrets). A missing/invalid config is non-fatal: the
    caller falls back to the ``DEFAULT_CONFIG`` defaults baked into each field.
    """
    try:
        from hermes_cli.config import load_config

        section = load_config().get("agui")
        return section if isinstance(section, dict) else {}
    except Exception:
        logger.debug(
            "load_config() failed for the agui section; using defaults", exc_info=True
        )
        return {}


class AgentConfig:
    """Deployment-level settings shared across runs.

    Behavioral settings come from the ``agui`` section of ``config.yaml`` (see
    ``DEFAULT_CONFIG`` in ``hermes_cli.config``); credentials come through the
    owning profile's secret scope. By default the adapter plugs into the
    local Hermes setup: provider, credentials, and ``api_mode`` are resolved per
    conversation from the ``hermes model`` config + auth pools via
    ``hermes_cli.runtime_provider.resolve_runtime_provider`` (see
    ``build_run_agent`` / ``_resolve_agent_settings``).

    The ``agui`` config keys ``provider`` / ``model`` / ``api_mode`` /
    ``base_url`` are optional per-adapter overrides; left blank they defer to
    that resolver. Setting ``base_url`` pins an explicit OpenAI-compatible
    endpoint and bypasses the resolver entirely — the self-hosted / aimock-e2e
    path (tests set these fields directly on the instance).
    """

    def __init__(self) -> None:
        from agent.secret_scope import get_secret

        cfg = _load_agui_config()
        # Explicit endpoint override (config.yaml `agui.base_url`). When set, the
        # hermes provider resolver is skipped and the agent talks directly to it.
        self.base_url = (cfg.get("base_url") or "").strip() or None
        # Credentials are secrets, never a behavioral config key: resolved from
        # hermes auth on the resolver path, or OPENAI_API_KEY for an explicit
        # base_url (self-hosted / aimock).
        self.api_key = get_secret("OPENAI_API_KEY")
        # Model override; empty means "use the hermes config default model".
        self.model = (cfg.get("model") or "").strip()
        # Provider override; None means "let the hermes config / resolver decide".
        self.provider = (cfg.get("provider") or "").strip() or None
        # API-mode override; None means resolver-derived (or Hermes auto-detect
        # on the explicit-endpoint path).
        self.api_mode = (cfg.get("api_mode") or "").strip() or None
        toolsets = cfg.get("toolsets")
        if isinstance(toolsets, str):
            toolsets = toolsets.split(",")
        if not isinstance(toolsets, list):
            toolsets = ["hermes-acp"]
        self.enabled_toolsets = [str(t).strip() for t in toolsets if str(t).strip()]


def set_current_state(state: Optional[RunState]) -> contextvars.Token:
    return _CURRENT_STATE.set(state)


def reset_current_state(token: contextvars.Token) -> None:
    _CURRENT_STATE.reset(token)


def _frontend_tool_handler(args, **kwargs) -> str:
    """Wait for the matching client result on the original execution thread."""
    from agui_adapter.continuation import CURRENT_FRONTEND, ContinuationError

    relay = CURRENT_FRONTEND.get()
    if relay is None:
        raise ContinuationError("Frontend dispatch requires a live frontend relay.")
    return relay.execute(args)


_STATE_WRITER_CONFIRMATION = "State updated."

_registered_state_writer_names: set[str] = set()


def _make_state_writer_handler(tool_name: str):
    """Build the dispatch handler for one state-writer tool.

    The name is bound in the closure (the registry does not pass the tool name
    to handlers). The handler merges the call's args into the run-scoped shared
    state and returns a confirmation. Unlike the frontend-tool handler it does
    NOT interrupt — the run continues so the model reads the tool result and
    produces its follow-up. The server emits a ``StateSnapshotEvent`` (built
    from the recorded snapshot) after the run."""

    def _handler(args, **kwargs) -> str:
        run_state = _CURRENT_STATE.get()
        if run_state is not None:
            try:
                run_state.apply(tool_name, args if isinstance(args, dict) else {})
            except Exception:
                logger.debug(
                    "state-writer apply failed for %s", tool_name, exc_info=True
                )
        return _STATE_WRITER_CONFIRMATION

    return _handler


def _ensure_state_writer_tools_registered(specs: dict[str, StateWriterSpec]) -> None:
    """Register a server-side dispatch handler for each state-writer tool name.

    The handler merges the call into the run-scoped state. Registration is
    idempotent and never clobbers a real backend tool of the same name. Which
    state key each tool writes is resolved per run from ``_CURRENT_STATE`` (the
    ``RunState`` carries the specs), so the same registered handler serves every
    run — only the run-scoped store differs.
    """
    from tools.registry import registry

    with _reg_lock:
        for name, spec in specs.items():
            if name in _registered_state_writer_names:
                continue
            if registry.get_entry(name) is not None:
                continue
            arg = spec.arg
            properties = {arg: {}} if arg else {}
            registry.register(
                name=name,
                toolset="agui-state-writer",
                schema={
                    "name": name,
                    "description": "Update shared UI state.",
                    "parameters": {"type": "object", "properties": properties},
                },
                handler=_make_state_writer_handler(name),
                check_fn=lambda: True,
                emoji="🗂️",
            )
            _registered_state_writer_names.add(name)


def _ensure_frontend_tools_registered(names: set[str]) -> None:
    """Register a dispatch handler for each frontend tool name (idempotent).

    The advertised schema (with real parameters) comes from ``agent.tools``;
    this registration only supplies a dispatchable handler so a call resolves
    to the interrupt path instead of an "unknown tool" error. A name already
    owned by a real backend tool is left untouched.
    """
    from tools.registry import registry

    with _reg_lock:
        for name in names:
            if name in _registered_frontend_names:
                continue
            if registry.get_entry(name) is not None:
                continue
            registry.register(
                name=name,
                toolset="agui-frontend",
                schema={
                    "name": name,
                    "description": "client-executed tool",
                    "parameters": {"type": "object", "properties": {}},
                },
                handler=_frontend_tool_handler,
                check_fn=lambda: True,
                emoji="🖥️",
            )
            _registered_frontend_names.add(name)


def _resolve_agent_settings(config: AgentConfig) -> dict:
    """Resolve the effective provider/model/credentials/api_mode for a run.

    Two paths, splitting the hermes-runtime deployment (which resolves from the
    hermes setup) from the aimock tests (which pin an explicit endpoint):

    * **Explicit endpoint** (``config.base_url`` set): use the config as-is and
      skip the resolver. This is the aimock e2e path and any self-hosted
      OpenAI-compatible deployment. Sensible defaults fill provider/model/key.
    * **Resolver path** (no ``base_url``): resolve provider, credentials, and
      ``api_mode`` from the ``hermes model`` config + auth pools via
      ``resolve_runtime_provider``. Any explicit ``config.provider`` /
      ``config.model`` / ``config.api_mode`` overrides the resolved values.
    """
    if config.base_url:
        return {
            "provider": config.provider or "custom",
            "model": config.model or "gpt-4o",
            "base_url": config.base_url,
            "api_key": config.api_key or "sk-aimock",
            "api_mode": config.api_mode,
            "command": None,
            "args": None,
        }

    try:
        from hermes_cli.config import load_config
        from hermes_cli.runtime_provider import resolve_runtime_provider
    except Exception:
        logger.debug(
            "hermes runtime resolver unavailable; using env config", exc_info=True
        )
        return {
            "provider": config.provider or "custom",
            "model": config.model or "gpt-4o",
            "base_url": None,
            "api_key": config.api_key,
            "api_mode": config.api_mode,
            "command": None,
            "args": None,
        }

    # Parse the `hermes model` config for a default model + provider.
    default_model = ""
    config_provider = None
    try:
        model_cfg = load_config().get("model")
        if isinstance(model_cfg, dict):
            default_model = str(model_cfg.get("default") or "")
            config_provider = model_cfg.get("provider")
        elif isinstance(model_cfg, str):
            default_model = model_cfg.strip()
    except Exception:
        logger.debug(
            "load_config() failed; proceeding with resolver defaults", exc_info=True
        )

    # Model comes from the hermes setup via `model or default_model` — no
    # hardcoded fallback that could mask a missing/mis-set model with a guess
    # that doesn't exist on the resolved provider.
    settings = {
        "provider": config.provider or "",
        "model": config.model or default_model,
        "base_url": None,
        "api_key": None,
        "api_mode": config.api_mode,
        "command": None,
        "args": None,
    }
    try:
        runtime = resolve_runtime_provider(
            requested=config.provider or config_provider,
            target_model=settings["model"] or None,
        )
        settings.update(
            provider=runtime.get("provider") or settings["provider"],
            base_url=runtime.get("base_url"),
            api_key=runtime.get("api_key"),
            api_mode=config.api_mode or runtime.get("api_mode"),
            command=runtime.get("command"),
            args=list(runtime.get("args") or []),
            credential_pool=runtime.get("credential_pool"),
        )
    except Exception:
        logger.warning(
            "hermes provider resolution failed; agent may lack credentials",
            exc_info=True,
        )

    # No adapter-level guard on an empty model: `model or default_model` is
    # passed straight through to AIAgent, letting a missing model surface
    # downstream rather than masking it with a hardcoded guess.
    return settings


def build_run_agent(
    config: AgentConfig,
    *,
    frontend_tool_schemas: Optional[list[dict]] = None,
    frontend_tool_names: Optional[set[str]] = None,
    state_writer_specs: Optional[dict[str, StateWriterSpec]] = None,
    state_writer_schemas: Optional[list[dict]] = None,
    cwd: Optional[str] = None,
    default_headers: Optional[dict] = None,
):
    """Construct and configure an ``AIAgent`` for one AG-UI run.

    ``state_writer_specs`` declares server-executed tools that write shared UI
    state (name -> :class:`StateWriterSpec`); ``state_writer_schemas`` are the
    OpenAI function schemas advertised to the model for those tools (so the
    model knows how to call them). See ``server.py`` for how the resulting
    ``RunState`` is seeded and its snapshots are emitted.
    """
    from run_agent import AIAgent
    from agent.skill_utils import parse_config_string_list
    from hermes_cli.config import load_config
    from hermes_constants import resolve_reasoning_config

    # Reject shadowing declarations before anything is merged or registered.
    # Keep this BELOW the run_agent import: that import is what populates the
    # tool registry (model_tools runs discover_builtin_tools() at import), so
    # hoisting the check above it would silently make it a no-op.
    _reject_name_collisions(
        set(frontend_tool_names or set()),
        set(state_writer_specs or {}),
    )

    settings = _resolve_agent_settings(config)
    profile = load_config()

    names = frontend_tool_names or set()
    specs = state_writer_specs or {}
    # Resolution can yield to another build. Claim both kinds together before
    # constructing an agent; the helpers re-enter this shared registry lock.
    with _reg_lock:
        _reject_name_collisions(set(names), set(specs))
        _ensure_frontend_tools_registered(names)
        _ensure_state_writer_tools_registered(specs)

    kwargs: dict[str, Any] = {
        "platform": "agui",
        "enabled_toolsets": list(config.enabled_toolsets),
        "disabled_toolsets": parse_config_string_list(
            (profile.get("agent") or {}).get("disabled_toolsets")
        )
        or None,
        "reasoning_config": resolve_reasoning_config(profile, settings["model"]),
        "credential_pool": settings.get("credential_pool"),
        "quiet_mode": True,
        "model": settings["model"],
        "provider": settings["provider"],
    }
    if settings.get("base_url"):
        kwargs["base_url"] = settings["base_url"]
    if settings.get("api_key"):
        kwargs["api_key"] = settings["api_key"]
    if settings.get("api_mode"):
        kwargs["api_mode"] = settings["api_mode"]
    if settings.get("command"):
        kwargs["command"] = settings["command"]
        kwargs["args"] = list(settings.get("args") or [])

    agent = AIAgent(**kwargs)
    manager = getattr(agent, "_memory_manager", None)
    collisions = {
        name
        for name in set(names) | set(specs)
        if name in getattr(agent, "_context_engine_tool_names", set())
        or (manager is not None and manager.has_tool(name))
    }
    if collisions:
        agent.close()
        raise ToolNameCollisionError(collisions)
    _expose_tools_eagerly(agent)
    if cwd:
        agent.session_cwd = cwd
    if default_headers:
        _apply_default_headers(agent, default_headers)

    if names:
        _merge_frontend_tools(agent, frontend_tool_schemas or [], names)

    if specs:
        # State-writer tools are server-executed; advertise them exactly like
        # frontend tools (name + parameters) so the model can call them.
        _merge_frontend_tools(agent, state_writer_schemas or [], set(specs))

    from agui_adapter.tool_snapshot import pin_tool_snapshot

    pin_tool_snapshot(agent)
    return agent


def _expose_tools_eagerly(agent) -> None:
    """Re-assemble the agent's model-facing tools WITHOUT the tool_search bridge.

    Newer Hermes core defers non-core (MCP/plugin) tools behind a
    ``tool_search`` / ``tool_describe`` / ``tool_call`` bridge (config
    ``tools.tool_search``, default ``"auto"``) to save context on large tool
    catalogs. That indirection is wrong for the AG-UI surface: the client
    renders one card per tool call and the adapter streams ``TOOL_CALL_*`` keyed
    on the *real* tool name (see :class:`events.AGUIEventBridge`) and detects
    frontend-tool handoffs by name. A bridged ``tool_call`` dispatch would
    surface ``tool_call`` instead of the real tool and break both. It also
    matches the adapter's pre-existing contract — it always advertised its
    (curated) toolset directly.

    So we force the eager, pre-assembly tool list. Scoped to this adapter's
    agent instance via the public ``skip_tool_search_assembly`` flag — no core
    change and no global-config mutation. Runs BEFORE the frontend/state-writer
    merge so those schemas still append on top.
    """
    try:
        import model_tools
        from tools.registry import registry

        eager = (
            model_tools.get_tool_definitions(
                enabled_toolsets=getattr(agent, "enabled_toolsets", None),
                disabled_toolsets=getattr(agent, "disabled_toolsets", None),
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
            or []
        )
        # Registry entries supply dispatch only; their placeholder schemas are
        # process-global. Advertise only this agent's declarations below.
        eager = [
            schema
            for schema in eager
            if not (
                (
                    entry := registry.get_entry(
                        (schema.get("function") or {}).get("name")
                    )
                )
                and entry.toolset in _ADAPTER_TOOLSETS
            )
        ]
    except Exception:
        logger.debug(
            "eager tool re-assembly failed; leaving agent.tools as built", exc_info=True
        )
        return
    # Initialization already gated instance-owned provider tools. They are not
    # registry entries, so retain their schemas without repeating lifecycle hooks.
    manager = getattr(agent, "_memory_manager", None)
    context_names = getattr(agent, "_context_engine_tool_names", set())
    eager_names = {(schema.get("function") or {}).get("name") for schema in eager}
    for schema in agent.tools or []:
        name = (schema.get("function") or {}).get("name")
        if name not in eager_names and (
            name in context_names or (manager is not None and manager.has_tool(name))
        ):
            eager.append(schema)
            eager_names.add(name)
    agent.tools = eager
    agent.valid_tool_names = {
        t["function"]["name"]
        for t in eager
        if isinstance(t, dict)
        and isinstance(t.get("function"), dict)
        and t["function"].get("name")
    }
    invalidate = getattr(agent, "_invalidate_system_prompt", None)
    if callable(invalidate):
        try:
            invalidate()
        except Exception:
            logger.debug("system-prompt invalidation failed", exc_info=True)


def _merge_frontend_tools(agent, schemas: list[dict], names: set[str]) -> None:
    """Advertise frontend tool schemas to the model (name + real parameters)."""
    existing = list(agent.tools or [])
    existing_names = {
        (t.get("function") or {}).get("name") for t in existing if isinstance(t, dict)
    }
    for schema in schemas:
        fn_name = (schema.get("function") or {}).get("name")
        if fn_name and fn_name not in existing_names:
            existing.append(schema)
    agent.tools = existing
    agent.valid_tool_names = set(agent.valid_tool_names or set()) | set(names)

    invalidate = getattr(agent, "_invalidate_system_prompt", None)
    if callable(invalidate):
        try:
            invalidate()
        except Exception:
            logger.debug("system-prompt invalidation failed", exc_info=True)


def _apply_default_headers(agent, headers: dict) -> None:
    """Attach default headers to the agent's OpenAI-compatible client (aimock).

    Hermes' ``AIAgent`` builds a FRESH per-request OpenAI client from
    ``agent._client_kwargs`` (see ``_create_request_openai_client``) for the
    actual chat-completions call — the ``self.client`` instance the adapter can
    reach here is only the shared/primary client and is bypassed on the request
    path. So a ``with_options(default_headers=...)`` override on ``agent.client``
    alone is silently dropped when the request client is rebuilt. We therefore
    merge the forwarded headers into ``_client_kwargs["default_headers"]`` too,
    so every rebuilt client (streaming and non-streaming) carries them. Without
    this the forwarded ``x-aimock-context`` never reaches aimock and every run
    404s on a fixture miss."""
    # 1) Merge into the kwargs every (re)built client is constructed from — the
    #    load-bearing path for the actual request client.
    try:
        client_kwargs = getattr(agent, "_client_kwargs", None)
        if isinstance(client_kwargs, dict):
            merged = dict(client_kwargs.get("default_headers") or {})
            merged.update(headers)
            client_kwargs["default_headers"] = merged
    except Exception:
        logger.debug("merging headers into _client_kwargs failed", exc_info=True)

    # 2) Also patch the already-constructed shared client for any code path that
    #    reuses it directly.
    client = getattr(agent, "client", None)
    if client is None:
        return
    try:
        with_options = getattr(client, "with_options", None)
        if callable(with_options):
            agent.client = with_options(default_headers=headers)
            return
    except Exception:
        logger.debug("client.with_options(default_headers) failed", exc_info=True)
    try:
        store = getattr(client, "_custom_headers", None)
        if isinstance(store, dict):
            store.update(headers)
    except Exception:
        logger.debug("client header mutation failed", exc_info=True)
