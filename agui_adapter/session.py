"""Per-run Hermes agent construction for the AG-UI adapter.

AG-UI is stateless per run: the client sends the full message list every time,
so the adapter rebuilds the conversation from ``RunAgentInput.messages`` each
run rather than holding server-side history.

Frontend (client-executed) tools are implemented WITHOUT any change to Hermes
core, using the interrupt mechanism Hermes already exposes:

* each frontend tool name is registered with a handler that calls
  ``agent.interrupt()`` and returns a placeholder — so when the model calls it,
  the tool loop unwinds at its next top-of-loop interrupt check
  (``conversation_loop.py``) instead of making another model call;
* the schema is merged into ``agent.tools`` / ``valid_tool_names`` so the model
  sees the tool and the loop does not treat the call as a hallucination.

A batch that contains a frontend tool is never all-parallel-safe, so Hermes
runs it sequentially (``_should_parallelize_tool_batch``). That means any
server-side tools in the same turn finish and append their results BEFORE the
frontend tool's handler fires the interrupt — which is what makes mixed
server+frontend batches deterministic. The adapter reads the resulting
messages to emit events and to resume.
"""

from __future__ import annotations

import contextvars
import copy
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

# The agent for the currently-executing run, so the (module-level, shared)
# frontend-tool handler can interrupt the right agent. Set per run before
# ``run_conversation``; read on the run's execution thread.
_CURRENT_AGENT: contextvars.ContextVar[Any] = contextvars.ContextVar("agui_current_agent", default=None)

# The shared-state store for the currently-executing run, so the (module-level,
# shared) state-writer tool handler can mutate the right run's state. Set per
# run alongside ``_CURRENT_AGENT``; read on the run's execution thread.
_CURRENT_STATE: contextvars.ContextVar[Optional["RunState"]] = contextvars.ContextVar(
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

    state: Dict[str, Any] = field(default_factory=dict)
    specs: Dict[str, StateWriterSpec] = field(default_factory=dict)
    # (tool_call args snapshot) recorded per state-writer invocation, in order.
    # Each entry is the FULL state as it stood right after that tool applied.
    snapshots: List[Dict[str, Any]] = field(default_factory=list)

    def apply(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
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

# Placeholder tool result returned by a client-side tool. Never surfaced to the
# client (the adapter emits the tool call without a result and the real result
# arrives on the next run); only present in history so the interrupted turn
# stays API-valid.
CLIENT_TOOL_PLACEHOLDER = json.dumps({"status": "pending_client_execution"})

_registered_frontend_names: set[str] = set()
_reg_lock = threading.Lock()


class AgentConfig:
    """Deployment-level settings shared across runs.

    By default the adapter plugs into the local Hermes setup: provider,
    credentials, and ``api_mode`` are resolved per run from the ``hermes model``
    config + auth pools via
    ``hermes_cli.runtime_provider.resolve_runtime_provider`` (see
    ``build_run_agent`` / ``_resolve_agent_settings``).

    Env vars override that resolution. In particular an explicit endpoint
    (``HERMES_AGUI_BASE_URL`` / ``OPENAI_BASE_URL``, or a ``base_url`` set on the
    instance) bypasses the resolver entirely and talks directly to that URL —
    which is what the aimock e2e tests and self-hosted OpenAI-compatible
    deployments rely on.
    """

    def __init__(self) -> None:
        # Explicit endpoint override. When set, the hermes provider resolver is
        # skipped and the agent talks directly to this base_url.
        self.base_url = os.environ.get("HERMES_AGUI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        # Explicit credential override. Only consulted on the explicit-endpoint
        # path; on the resolver path the resolved provider supplies the key.
        self.api_key = os.environ.get("HERMES_AGUI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        # Model override; empty means "use the hermes config default model".
        self.model = os.environ.get("HERMES_AGUI_MODEL") or ""
        # Provider override; None means "let the hermes config / resolver decide".
        self.provider = os.environ.get("HERMES_AGUI_PROVIDER") or None
        # API-mode override; None means resolver-derived (or Hermes auto-detect
        # on the explicit-endpoint path).
        self.api_mode = os.environ.get("HERMES_AGUI_API_MODE") or None
        raw_toolsets = os.environ.get("HERMES_AGUI_TOOLSETS", "hermes-acp")
        self.enabled_toolsets = [t for t in (raw_toolsets.split(",") if raw_toolsets else []) if t.strip()]


def set_current_agent(agent) -> contextvars.Token:
    return _CURRENT_AGENT.set(agent)


def reset_current_agent(token: contextvars.Token) -> None:
    _CURRENT_AGENT.reset(token)


def set_current_state(state: Optional[RunState]) -> contextvars.Token:
    return _CURRENT_STATE.set(state)


def reset_current_state(token: contextvars.Token) -> None:
    _CURRENT_STATE.reset(token)


def _frontend_tool_handler(args, **kwargs) -> str:
    """Handler shared by every client-side tool: interrupt the run and return a
    placeholder. The run unwinds without a further model call; the adapter then
    hands the tool call to the client."""
    agent = _CURRENT_AGENT.get()
    if agent is not None:
        try:
            agent.interrupt("client-side tool handoff")
        except Exception:
            logger.debug("agent.interrupt() failed in frontend tool handler", exc_info=True)
    return CLIENT_TOOL_PLACEHOLDER


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
                logger.debug("state-writer apply failed for %s", tool_name, exc_info=True)
        return _STATE_WRITER_CONFIRMATION

    return _handler


def _ensure_state_writer_tools_registered(specs: Dict[str, StateWriterSpec]) -> None:
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
                schema={"name": name, "description": "client-executed tool", "parameters": {"type": "object", "properties": {}}},
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
    except Exception:  # noqa: BLE001 - hermes CLI unavailable; fall back to env
        logger.debug("hermes runtime resolver unavailable; using env config", exc_info=True)
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
    except Exception:  # noqa: BLE001 - missing/invalid config is non-fatal
        logger.debug("load_config() failed; proceeding with resolver defaults", exc_info=True)

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
        runtime = resolve_runtime_provider(requested=config.provider or config_provider)
        settings.update(
            provider=runtime.get("provider") or settings["provider"],
            base_url=runtime.get("base_url"),
            api_key=runtime.get("api_key"),
            api_mode=config.api_mode or runtime.get("api_mode"),
            command=runtime.get("command"),
            args=list(runtime.get("args") or []),
        )
    except Exception:  # noqa: BLE001 - surface as a run without creds rather than crash
        logger.warning("hermes provider resolution failed; agent may lack credentials", exc_info=True)

    # No adapter-level guard on an empty model: `model or default_model` is
    # passed straight through to AIAgent, letting a missing model surface
    # downstream rather than masking it with a hardcoded guess.
    return settings


def build_run_agent(
    config: AgentConfig,
    *,
    frontend_tool_schemas: Optional[List[dict]] = None,
    frontend_tool_names: Optional[set[str]] = None,
    state_writer_specs: Optional[Dict[str, StateWriterSpec]] = None,
    state_writer_schemas: Optional[List[dict]] = None,
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

    settings = _resolve_agent_settings(config)

    kwargs: dict[str, Any] = {
        "platform": "agui",
        "enabled_toolsets": list(config.enabled_toolsets),
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
    if cwd:
        agent.session_cwd = cwd
    if default_headers:
        _apply_default_headers(agent, default_headers)

    names = frontend_tool_names or set()
    if names:
        _ensure_frontend_tools_registered(names)
        _merge_frontend_tools(agent, frontend_tool_schemas or [], names)

    specs = state_writer_specs or {}
    if specs:
        _ensure_state_writer_tools_registered(specs)
        # State-writer tools are server-executed; advertise them exactly like
        # frontend tools (name + parameters) so the model can call them.
        _merge_frontend_tools(agent, state_writer_schemas or [], set(specs))

    return agent


def _merge_frontend_tools(agent, schemas: List[dict], names: set[str]) -> None:
    """Advertise frontend tool schemas to the model (name + real parameters)."""
    existing = list(agent.tools or [])
    existing_names = {(t.get("function") or {}).get("name") for t in existing if isinstance(t, dict)}
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
