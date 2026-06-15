"""DeepAgents runtime - LangGraph-backed agent backed by Hermes internals.

Removes all child AIAgent instantiation. Config is accepted directly from
the AIAgent facade parameters instead of delegating to a full runtime.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

try:
    from langchain_core.messages import (
        HumanMessage,
        AIMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.tools import StructuredTool
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.store.memory import InMemoryStore
    from deepagents.graph import create_deep_agent
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from deepagents.backends import langsmith

    DEEPAGENTS_AVAILABLE = True
except ImportError as e:
    logger.warning("DeepAgents SDK not available: %s", e)
    DEEPAGENTS_AVAILABLE = False
    FilesystemMiddleware = None  # type: ignore[misc,assignment]
    langsmith = None  # type: ignore[misc,assignment]
    MemorySaver = None  # type: ignore[misc,assignment]
    InMemoryStore = None  # type: ignore[misc,assignment]

# Langfuse (SDK v3) — optional, lazy-installed via the ``deepagents`` extra.
# In v3 the LangChain ``CallbackHandler`` carries no credentials: they live on
# a ``Langfuse`` client (singleton), and the handler reads from it. Both are
# imported at module level (not inside the method) so tests can patch
# ``Langfuse`` / ``CallbackHandler`` / ``LANGFUSE_AVAILABLE`` and so handler
# construction is gated on a single, inspectable flag. The langchain
# integration also needs the ``langchain`` package present, not just
# ``langchain-core``.
try:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    LANGFUSE_AVAILABLE = True
except ImportError:
    Langfuse = None  # type: ignore[misc,assignment]
    CallbackHandler = None  # type: ignore[misc,assignment]
    LANGFUSE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LANGCHAIN_FINAL_RESPONSE_KEY = "__hermes_final_response"


# ---------------------------------------------------------------------------
# Message Converters
# ---------------------------------------------------------------------------


def _convert_messages_to_langchain(messages):
    """Convert Hermes message list to LangChain messages."""
    if not messages:
        return []

    result = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            result.append(SystemMessage(content=content))
        elif role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            ai = AIMessage(content=content or "")
            if tool_calls:
                ai.tool_calls = tool_calls
            result.append(ai)
        elif role == "tool":
            tool_id = msg.get("tool_call_id", "")
            result.append(ToolMessage(content=content, tool_call_id=tool_id))
    return result


def _convert_langchain_to_hermes(messages):
    """Convert LangChain messages to Hermes message format."""
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            entry = {"role": "assistant", "content": msg.content or ""}
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                entry["tool_calls"] = msg.tool_calls
            result.append(entry)
        elif isinstance(msg, ToolMessage):
            result.append({
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id,
            })
    return result


# ---------------------------------------------------------------------------
# Result Parsing
# ---------------------------------------------------------------------------


def _parse_langgraph_result(result: dict, task_id: str = None) -> dict:
    """Parse LangGraph agent result dict into Hermes result dict shape."""
    raw_messages = result.get("messages", [])
    hermes_messages = _convert_langchain_to_hermes(raw_messages)

    final_response = ""
    last_reasoning = None
    for msg in reversed(raw_messages):
        if isinstance(msg, AIMessage):
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        final_response = part.get("text", "")
                        break
                    elif isinstance(part, str):
                        final_response = part
                        break
            elif msg.content:
                final_response = msg.content
            last_reasoning = getattr(msg, "reasoning", None) or getattr(
                msg, "additional_kwargs", {}
            ).get("reasoning")
            break

    api_calls = sum(
        1 for m in raw_messages if isinstance(m, AIMessage) and m.tool_calls
    )
    completed = bool(final_response)

    return {
        "final_response": final_response,
        "messages": hermes_messages,
        "api_calls": api_calls,
        "completed": completed,
        "failed": False,
        "interrupted": False,
        "partial": False,
        "turn_exit_reason": "completed" if completed else "no_response",
        "last_reasoning": last_reasoning,
        "model": "",
    }


def _parse_error_result(e: Exception) -> dict:
    """Return an error result dict matching run_conversation shape."""
    return {
        "final_response": f"Error: {e}",
        "messages": [],
        "api_calls": 0,
        "completed": False,
        "failed": True,
        "interrupted": False,
        "partial": False,
        "turn_exit_reason": f"deepagents_error: {e}",
    }


# ---------------------------------------------------------------------------
# Environment Injection
# ---------------------------------------------------------------------------


def _inject_provider_env(provider, base_url, api_key):
    """Set provider-specific env vars for LangChain auto-loading.

    LangChain model bindings read API keys from env vars based on provider.
    This maps Hermes' model/provider to the env var names LangChain expects.
    """
    env_map = {
        "": ("OPENAI_API_KEY", "OPENAI_API_BASE"),  # fallback to OpenAI
        "openai": ("OPENAI_API_KEY", "OPENAI_API_BASE"),
        "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_API_BASE"),
        "google": ("GOOGLE_API_KEY", "GOOGLE_API_BASE"),
        "xai": ("XAI_API_KEY", "XAI_API_BASE"),
        "cohere": ("COHERE_API_KEY", "COHERE_API_BASE"),
        "groq": ("GROQ_API_KEY", "GROQ_API_BASE"),
    }

    env_vars = env_map.get(provider, ("API_KEY", "BASE_URL"))

    if api_key:
        os.environ[env_vars[0]] = api_key

    if base_url:
        os.environ[env_vars[1]] = base_url


# ---------------------------------------------------------------------------
# Tool Adapters
# ---------------------------------------------------------------------------


class _HermesToolAdapter:
    """Adapts a Hermes tool to a LangChain StructuredTool.

    Invokes handle_function_call from model_tools atomically.
    """

    def __init__(self, tool_entry):
        self._entry = tool_entry
        self._tool = self._build_langchain_tool()

    def _build_langchain_tool(self):
        entry = self._entry
        schema = entry.schema
        func_name = schema.get("name", entry.name)
        func_desc = entry.description or schema.get("description", "")

        from model_tools import handle_function_call

        def _execute_sync(**kwargs):
            try:
                return handle_function_call(
                    function_name=entry.name, function_args=kwargs
                )
            except Exception as e:
                error_str = str(e)
                return json.dumps({"error": error_str}, ensure_ascii=False)

        return StructuredTool.from_function(
            name=func_name, description=func_desc, func=_execute_sync
        )

    @property
    def langchain_tool(self):
        return self._tool

    @property
    def name(self):
        return self._entry.name

    @property
    def schema(self):
        return self._entry.schema

    @property
    def toolset(self):
        return self._entry.toolset


def build_hermes_tools(enabled_toolsets, disabled_toolsets):
    """Build LangChain StructuredTool list from Hermes tool definitions."""
    from tools.registry import registry

    tools = []
    try:
        definitions = registry.get_definitions(
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
        )
    except Exception:
        definitions = []

    for tool_def in definitions:
        tool_name = tool_def.get("name", "")
        entry = registry.get_entry(tool_name) if tool_name else None
        if entry:
            adapter = _HermesToolAdapter(entry)
            tools.append(adapter.langchain_tool)
    return tools


# ---------------------------------------------------------------------------
# Streaming Bridge
# ---------------------------------------------------------------------------


class _HermesStreamingBridge:
    """Wires LangGraph stream events to Hermes callback forwarding.

    The bridge has access to all Hermes callbacks via the agent's
    ``_get_cap`` mechanism (populated by the gateway via __setattr__).
    It translates LangGraph events (AIMessageChunk, ToolCall, ToolResult)
    to the standard Hermes callback signatures so the gateway sees
    identical event streams regardless of runtime.
    """

    def __init__(
        self,
        agent,
        stream_delta=None,
        tool_progress=None,
        thinking=None,
        step=None,
    ):
        self._agent = agent
        self._stream_delta = (
            stream_delta
            or getattr(agent, "stream_delta_callback", None)
            or _noop_cb("stream_delta")
        )
        self._tool_progress = (
            tool_progress
            or getattr(agent, "tool_progress_callback", None)
            or _noop_cb("tool_progress")
        )
        self._thinking = (
            thinking
            or getattr(agent, "thinking_callback", None)
            or _noop_cb("thinking")
        )
        self._step = step or getattr(agent, "step_callback", None) or None

    def any_callbacks_set(self):
        return self._stream_delta is not None or self._tool_progress is not None

    def process_event(self, event):
        """Process a single LangGraph stream event and route to Hermes callbacks."""
        if not isinstance(event, dict):
            return

        if "events" in event:
            for sub_event in event["events"]:
                self._process_sub_event(sub_event)
        elif "output" in event:
            self._process_output(event["output"])

    def _process_sub_event(self, sub_event):
        """Process a LangGraph sub-event (message chunk, tool call, etc.)."""
        if not isinstance(sub_event, dict):
            return

        event_type = sub_event.get("type", "")
        data = sub_event.get("data", {})

        if event_type == "AIMessageChunk":
            self._handle_ai_message_chunk(data)
        elif event_type == "ToolCall":
            name = data.get("name", "")
            args = data.get("args", {})
            if self._tool_progress:
                self._tool_progress(
                    "tool.started", tool_name=name, preview=str(args)[:200]
                )

    def _handle_ai_message_chunk(self, data):
        """Handle an AIMessageChunk event (text streaming)."""
        content = data.get("content", "")

        # Stream text content through step_callback or stream_delta_callback
        if self._step:
            self._step(1, [])  # minimal step marker
        if self._stream_delta and content:
            self._stream_delta(content)

    def _process_output(self, output):
        """Handle node output (final result)."""
        if self._stream_delta and output:
            text = str(output)[:200]
            if isinstance(output, dict):
                text = output.get("final_response", str(output))
            self._stream_delta(text)


def _noop_cb(name):
    """Return a no-op callable for missing callbacks."""

    def noop(*args, **kwargs):
        pass

    return noop


# ---------------------------------------------------------------------------
# Main Agent Class
# ---------------------------------------------------------------------------


class DeepAgentsAIAgent:
    """Hermes agent backed by DeepAgents SDK (LangGraph).

    Accepts config directly — no child AIAgent instantiation.
    Public API matches run_agent.AIAgent:
      - run_conversation(user_message, system_message, conversation_history, task_id)
      - chat(message)

    Callback forwarding: known callback attributes are stored in ``_callbacks``
    and served via ``__getattr__`` so gateway attribute-setting still works.
    """

    _CAPTURED_NAMES = frozenset((
        "tool_progress_callback",
        "tool_start_callback",
        "tool_complete_callback",
        "thinking_callback",
        "reasoning_callback",
        "clarify_callback",
        "step_callback",
        "stream_delta_callback",
        "interim_assistant_callback",
        "tool_gen_callback",
        "status_callback",
        "notice_callback",
        "notice_clear_callback",
        "reasoning_config",
        "service_tier",
        "request_overrides",
        "background_review_callback",
        # Tracing / observability
        "debug",
        "langsmith_api_key",
        "langsmith_project",
        "langsmith_tags",
        # Langfuse credentials — set by the gateway (or seeded from
        # HERMES_LANGFUSE_* env at init) and read by _get_langfuse_handler.
        "langfuse_public_key",
        "langfuse_secret_key",
        "langfuse_base_url",
    ))

    # Attributes initialized during __init__ that must not raise
    # AttributeError when read on partially-constructed or mock agents.
    _DEFAULT_ATTRS = frozenset((
        "_quiet_mode",
        "_skip_memory",
        "_platform",
        "_session_id",
        "_max_iterations",
        "_langgraph_checkpointer",
        "_langgraph_store",
        "_langsmith_api_key",
        "_langsmith_project",
        "_langsmith_tags",
        "_ls_api_key",
        "_ls_project",
        "_ls_tags",
        "_checkpointer",
        "_store",
        "_agent",
        "_callbacks",
        "_langfuse_handler",
        "_debug",
    ))

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        provider: str = None,
        model: str = "",
        max_iterations: int = 90,
        enabled_toolsets: list = None,
        disabled_toolsets: list = None,
        quiet_mode: bool = False,
        skip_memory: bool = False,
        skip_context_files: bool = False,
        session_id: str = None,
        platform: str = None,
        runtime: str = "deepagents",
        langgraph_checkpointer: bool = False,  # Enable checkpointing / tracing
        langgraph_store: bool = False,  # Enable LangGraph store
        # (The gateway / AIAgent facade also passes the following via __setattr__)
        # - debug                               # Enable debug-level runnable traces
        # - langsmith_api_key                   # LangSmith project key (env)
        # - langsmith_project                   # LangSmith project name
        # - langsmith_tags                      # LangSmith tags list
        **kwargs,
    ):
        self.mode = "deepagents"
        self._quiet_mode = quiet_mode
        self._skip_memory = skip_memory
        self._platform = platform
        self._session_id = session_id or ""
        self._max_iterations = max_iterations
        self.provider = provider or ""
        self._base_url = base_url
        self._api_key = api_key
        self._langgraph_checkpointer = langgraph_checkpointer
        self._langgraph_store = langgraph_store
        self._debug: bool = False
        self._langsmith_api_key: str | None = None
        self._langsmith_project: str = "hermes"
        self._langsmith_tags: list[str] = ["hermes"]

        # Resolve model string (LangChain format – no provider prefix).
        # Keep the raw name too: OpenAI-compatible endpoints (vLLM) serve
        # models under their full id (e.g. "nvidia/Qwen…"), prefix included.
        self._model_raw = model or ""
        model_str = self._resolve_model(model)
        if not model_str:
            model_str = self._default_model()

        # Set env vars for LangChain model bindings
        _inject_provider_env(provider, base_url, api_key)

        # LangSmith tracing defaults (set before _build_langgraph_agent so
        # they exist even when __init__ is bypassed by test mocks).
        self._ls_api_key = None
        self._ls_project = "hermes"
        self._ls_tags = ["hermes"]

        # Langfuse tracing: memoized handler (None = not yet built,
        # False = checked-and-unavailable, otherwise the CallbackHandler).
        self._langfuse_handler = None
        # Seed credentials from env so the Docker / gateway path works
        # out of the box. The gateway may still override these afterwards
        # via attribute assignment (captured into _callbacks). Unit tests
        # bypass __init__, so the handler logic itself never reads env.
        self._seed_langfuse_from_env()

        # Build the LangGraph agent
        self._agent = self._build_langgraph_agent(
            model=model_str,
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            skip_context_files=skip_context_files,
            quiet_mode=quiet_mode,
            skip_memory=skip_memory,
            system_prompt=None,  # will be built lazily
        )

    def _resolve_model(self, model):
        """Resolve model string for LangChain / deep agents.

        DeepAgents uses ``create_deep_agent`` which internally calls
        LangChain's ``init_chat_model``.  That helper expects model names
        without the Hermes provider prefix – e.g. ``"gpt-4o"`` not
        ``"openai/gpt-4o"``.  The provider itself is injected via the
        ``_inject_provider_env`` side-channel (env vars).

        Also handles provider-specific aliases (e.g. "gemini" ->
        "google/gemini-pro").
        """
        if not model:
            return None
        # Strip provider prefix if present (provider/model -> model)
        if "/" in model and not model.startswith("http"):
            model = model.split("/", 1)[1]
        return model

    def _default_model(self, fallback="gpt-4o"):
        """Return a default model based on the configured provider."""
        p = (self.provider or "").lower()
        if "anthropic" in p:
            return "claude-sonnet-4-0"
        elif "google" in p:
            return "gemini-2.0-flash"
        return fallback

    def _build_langgraph_agent(
        self,
        model,
        enabled_toolsets,
        disabled_toolsets,
        skip_context_files,
        quiet_mode,
        skip_memory,
        system_prompt,
    ):
        """Construct the DeepAgents LangGraph agent with Hermes tools."""
        hermes_home = get_hermes_home()

        # Build system prompt lazily
        if system_prompt is None:
            try:
                from agent.system_prompt import get_system_prompt

                system_prompt = get_system_prompt(
                    skip_context_files=skip_context_files,
                    hermes_home=hermes_home,
                )
            except Exception:
                system_prompt = "You are a helpful AI assistant."

        # Middleware - pass through for create_deep_agent auto-stack
        # (FilesystemMiddleware, TodoListMiddleware, etc. are added by default).
        # User middleware is inserted between base and tail layers.
        middlewares = []

        # Memory middleware (Hermes-provided)
        if not skip_memory:
            from agent.deep_agents_middleware import _HermesMiddleware

            middlewares.append(_HermesMiddleware(self))

        # Build Hermes tools from registry
        tools = build_hermes_tools(
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
        )

        # Optional: LangGraph checkpointer for tracing / state persistence
        checkpointer = None
        if self._langgraph_checkpointer:
            checkpointer = MemorySaver()
            logger.info("checkpointer enabled")

        # Optional: LangGraph store for persistent memory
        store = None
        if self._langgraph_store:
            store = InMemoryStore()
            logger.info("store enabled")

        # Enable debug / LangSmith tracing if configured
        debug_val = self._get_cap("debug") or False
        if debug_val is True:
            logger.info("debug traces enabled")

        # --- LangSmith env setup ---------------------------------------------------
        ls_api_key = self._langsmith_api_key or os.environ.get("LANGSMITH_API_KEY")
        ls_project = self._langsmith_project or "hermes"
        ls_tags = self._langsmith_tags or ["hermes"]

        if ls_api_key:
            os.environ["LANGSMITH_API_KEY"] = ls_api_key
            os.environ.setdefault("LANGSMITH_TRACING_V2", "true")
            os.environ.setdefault("LANGSMITH_PROJECT", ls_project)
            logger.info("LangSmith tracing enabled (project='%s')", ls_project)
        else:
            ls_tags = []
            ls_project = ""

        # Custom / OpenAI-compatible endpoints (vLLM, llama.cpp, …) can't be
        # inferred by LangChain's ``init_chat_model`` from the bare model
        # name — build the ChatOpenAI client explicitly against the
        # configured base_url instead of passing a string through.
        if isinstance(model, str) and self._base_url:
            _p = (self.provider or "").lower()
            if _p in ("", "custom", "openai"):
                from langchain_openai import ChatOpenAI

                model = ChatOpenAI(
                    # The endpoint serves the model under its full id
                    # (prefix included) — don't use the stripped name.
                    model=getattr(self, "_model_raw", None) or model,
                    base_url=self._base_url,
                    api_key=self._api_key or "EMPTY",
                )

        # Create the LangGraph agent (recursion_limit set per-call via config)
        agent = create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            middleware=middlewares,
            checkpointer=checkpointer,
            store=store,
            debug=bool(debug_val),
            name="hermes-agent",
        )

        # Store refs for per-call tracing hooks
        self._checkpointer = checkpointer
        self._store = store
        self._ls_api_key = ls_api_key
        self._ls_project = ls_project
        self._ls_tags = ls_tags

        return agent

    # ------------------------------------------------------------------
    # Callback forwarding (__setattr__ / _get_cap / __getattr__)
    #
    # The gateway sets callback attributes on the agent after __init__,
    # e.g. ``agent.tool_progress_callback = cb``.  We capture those in
    # _callbacks and serve them on read so run_conversation can find them.
    # ------------------------------------------------------------------

    def __setattr__(self, name, value):
        """Forward known callback/config attributes to a stored dict."""
        if name in self._CAPTURED_NAMES:
            # ``hasattr`` is True even before initialization: ``_callbacks``
            # is in _DEFAULT_ATTRS, so __getattr__ returns None instead of
            # raising — check for None explicitly.
            if getattr(self, "_callbacks", None) is None:
                object.__setattr__(self, "_callbacks", {})
            self._callbacks[name] = value
        else:
            super().__setattr__(name, value)

    def _get_cap(self, name, default=None):
        """Read a forwarded callback / config attribute."""
        caps = getattr(self, "_callbacks", {})
        if caps is None:
            return default
        return caps.get(name, default)

    def __getattr__(self, name):
        if name in self._CAPTURED_NAMES:
            return self._get_cap(name)
        if name in self._DEFAULT_ATTRS:
            return None
        raise AttributeError(f"'{type(self).__name__}' object has no attr '{name}'")

    # ------------------------------------------------------------------
    # Conversation / Chat
    # ------------------------------------------------------------------

    def run_conversation(
        self,
        user_message: str,
        system_message: str = None,
        conversation_history: list = None,
        task_id: str = None,
        stream_callback=None,
    ) -> dict:
        """Run a single-turn conversation via DeepAgents/LangGraph.

        Returns the same dict shape as run_agent.AIAgent.run_conversation.
        Reads gateway-set callbacks (stream_delta_callback, etc.) from
        the forwarding dict so the gateway's per-turn attribute-setting
        pattern works without modification.
        """
        langchain_history = _convert_messages_to_langchain(conversation_history or [])
        if system_message:
            langchain_history = [
                SystemMessage(content=system_message)
            ] + langchain_history

        human = HumanMessage(content=user_message)
        messages = langchain_history + [human]

        config = {
            "configurable": {
                "session_id": self._session_id or "default",
                "task_id": task_id or "",
            },
        }

        # Langfuse tracing: the plugin hooks in conversation_loop.py never
        # fire under this runtime, so tracing rides LangChain's callback
        # system instead. Inert without Langfuse credentials. In v3 the
        # session is grouped via config metadata (``langfuse_session_id``),
        # which the CallbackHandler reads — the handler constructor no longer
        # accepts a session id.
        _lf_handler = self._get_langfuse_handler()
        if _lf_handler is not None:
            config["callbacks"] = [_lf_handler]
            if self._session_id:
                config.setdefault("metadata", {})["langfuse_session_id"] = (
                    self._session_id
                )

        # Add LangSmith tracing tags / metadata when tracing is enabled
        if self._ls_api_key:
            tags = list(self._ls_tags or [])
            if task_id:
                tags.append(f"task:{task_id}")
            if self._platform:
                metadata = {
                    "platform": self._platform,
                    "session_id": config["configurable"].get("session_id", ""),
                }
                config["metadata"] = metadata
            config["tags"] = tags

        # Collect forwarded callbacks so the streaming bridge can invoke them
        bridge = _HermesStreamingBridge(
            self,
            stream_delta=self._get_cap("stream_delta_callback"),
            tool_progress=self._get_cap("tool_progress_callback"),
            thinking=self._get_cap("thinking_callback"),
            step=self._get_cap("step_callback"),
        )

        # Tracing for the DeepAgents runtime rides LangChain's callback system
        # (the Langfuse CallbackHandler wired into ``config["callbacks"]`` above),
        # which is the v2-compatible path and captures tool calls + token usage
        # natively. The conversation_loop.py plugin hooks are intentionally NOT
        # emitted here: the bundled langfuse plugin targets the Langfuse v3 SDK
        # API, so it would no-op on the pinned v2 SDK while risking duplicate
        # traces alongside the CallbackHandler.
        if bridge.any_callbacks_set():
            result = self._run_streamed(messages, config, task_id, bridge)
        elif stream_callback:
            result = self._run_streamed(messages, config, task_id, bridge)
        else:
            result = self._run_sync(messages, config, task_id)

        return result

    def _seed_langfuse_from_env(self):
        """Populate Langfuse credential callbacks from ``HERMES_LANGFUSE_*`` env.

        This is the env/Docker sourcing layer: it runs only from ``__init__``
        (bypassed by unit tests). The handler logic in ``_get_langfuse_handler``
        reads exclusively from these captured callbacks, never from env, so it
        stays deterministic and injectable in tests.
        """
        public_key = os.environ.get("HERMES_LANGFUSE_PUBLIC_KEY") or os.environ.get(
            "LANGFUSE_PUBLIC_KEY"
        )
        secret_key = os.environ.get("HERMES_LANGFUSE_SECRET_KEY") or os.environ.get(
            "LANGFUSE_SECRET_KEY"
        )
        host = os.environ.get("HERMES_LANGFUSE_BASE_URL") or os.environ.get(
            "LANGFUSE_BASE_URL"
        )
        if public_key:
            self.langfuse_public_key = public_key
        if secret_key:
            self.langfuse_secret_key = secret_key
        if host:
            self.langfuse_base_url = host

    def _get_langfuse_handler(self):
        """Return a memoized Langfuse ``CallbackHandler``, or ``None``.

        Credentials come from gateway-set callbacks (``langfuse_public_key`` /
        ``langfuse_secret_key`` / ``langfuse_base_url``), seeded from
        ``HERMES_LANGFUSE_*`` env at init. Built once per agent (session-scoped)
        and cached; ``False`` marks "checked and unavailable" so construction
        is never retried.

        Langfuse v3: the credentials live on a ``Langfuse`` client (a process
        singleton); the LangChain ``CallbackHandler`` takes no creds and reads
        from that client. The client is what flushes (the handler has no
        ``flush``), so we keep a reference on ``self._langfuse_client``. session
        grouping is applied per-call via config metadata (``langfuse_session_id``
        in ``run_conversation``), not the constructor.
        """
        cached = getattr(self, "_langfuse_handler", None)
        if cached is not None:
            return cached or None

        handler = False
        self._langfuse_client = None
        public_key = self._get_cap("langfuse_public_key")
        secret_key = self._get_cap("langfuse_secret_key")
        if LANGFUSE_AVAILABLE and public_key and secret_key:
            host = self._get_cap("langfuse_base_url") or "https://cloud.langfuse.com"
            try:
                self._langfuse_client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
                handler = CallbackHandler()
                logger.info("Langfuse tracing enabled (host=%s)", host)
            except Exception as e:
                logger.warning("Langfuse callback unavailable: %s", e)
                handler = False
                self._langfuse_client = None

        self._langfuse_handler = handler
        return handler or None

    def _run_sync(self, messages, config, task_id):
        """Sync invocation."""
        result = None
        _flushed = False
        try:
            cfg = {**config, "recursion_limit": self._max_iterations}
            result = self._agent.invoke({"messages": messages}, config=cfg)
            self._maybe_flush_langfuse()
            _flushed = True
            return _parse_langgraph_result(result, task_id)
        except Exception as e:
            if not _flushed:
                self._maybe_flush_langfuse()
            logger.error("DeepAgents invoke failed: %s", e, exc_info=True)
            return _parse_error_result(e)

    def _run_streamed(self, messages, config, task_id, bridge):
        """Streaming invocation."""
        cfg = {**config, "recursion_limit": self._max_iterations}
        stream_result = self._agent.stream(
            {"messages": messages},
            config=cfg,
            stream_mode="updates",
            subgraphs=False,
        )

        # Accumulate messages from the streamed updates as we go. The stream is
        # the only reliable result source when no checkpointer is configured —
        # ``get_state()`` requires one. We prefer the streamed messages and fall
        # back to ``get_state()`` only when the stream surfaced no message
        # updates (a checkpointer IS configured, or a custom stream shape).
        streamed_messages: list = []
        for event in stream_result:
            bridge.process_event(event)
            if isinstance(event, dict):
                for node_update in event.values():
                    if isinstance(node_update, dict):
                        msgs = node_update.get("messages") or []
                        streamed_messages.extend(m for m in msgs if m is not None)

        # After streaming, parse final result from the last AIMessage.
        try:
            raw = streamed_messages
            if not raw:
                try:
                    raw = self._agent.get_state(config).values.get("messages", [])
                except Exception:
                    raw = []
            r = _parse_langgraph_result({
                "messages": raw if isinstance(raw, list) else list(raw)
            })
            self._maybe_flush_langfuse()
            return r
        except Exception:
            if streamed_messages:
                try:
                    r = _parse_langgraph_result({"messages": streamed_messages})
                    self._maybe_flush_langfuse()
                    return r
                except Exception:
                    pass
            r = _parse_error_result(Exception("Failed to get streaming final result"))
            self._maybe_flush_langfuse()
            return r

    def chat(self, message: str) -> str:
        """Simple chat interface — returns final response string."""
        result = self.run_conversation(message)
        return result.get("final_response", "")

    # ------------------------------------------------------------------
    # Observability — Langfuse tracing for deep agents.
    #
    # The deep agents runtime doesn't use run_agent.py or
    # conversation_loop.py, so the Hermes plugin hooks on `pre_api_request` /
    # `post_api_request` never fire here. Tracing instead rides LangChain's
    # native callback system: the Langfuse ``CallbackHandler`` is built from
    # the captured credentials and injected into ``config['callbacks']`` in
    # ``run_conversation``, capturing generations and tool spans directly from
    # LangGraph. This is the Langfuse v2-compatible path (the bundled
    # langfuse plugin targets the v3 SDK API, which is not installed).
    # ------------------------------------------------------------------

    def _maybe_flush_langfuse(self):
        """Flush the Langfuse client if tracing is active.

        v3 flushes on the ``Langfuse`` client, not the ``CallbackHandler``
        (which has no ``flush``). ``_langfuse_client`` is set alongside the
        handler in :meth:`_get_langfuse_handler`.
        """
        try:
            client = getattr(self, "_langfuse_client", None)
            if client is not None and hasattr(client, "flush"):
                client.flush()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Compatibility forwarders (matching run_agent.AIAgent attrs)
    # ------------------------------------------------------------------

    @property
    def model(self):
        return ""  # LangGraph tracks internally

    @property
    def iteration_budget(self):
        return self

    @property
    def valid_tool_names(self):
        # Compute from tools
        try:
            from model_tools import get_tool_definitions

            defs = get_tool_definitions()
            return [d.get("name", "") for d in defs]
        except Exception:
            return []

    @property
    def tools(self):
        return []  # LangGraph manages internally

    @property
    def quiet_mode(self):
        return self._quiet_mode

    @property
    def platform(self):
        return self._platform

    @property
    def skip_memory(self):
        return self._skip_memory

    @property
    def max_iterations(self):
        return self._max_iterations

    @property
    def has_checkpointer(self):
        """Return True if a LangGraph checkpointer is active."""
        return self._checkpointer is not None

    @property
    def has_store(self):
        """Return True if a LangGraph store is active."""
        return self._store is not None

    @property
    def has_langsmith_tracing(self):
        """Return True if LangSmith tracing is configured."""
        # Check both direct and callback-set values
        direct = self._ls_api_key or ""
        cb_key = self._get_cap("langsmith_api_key") or ""
        return bool(direct) or bool(cb_key) or bool(os.environ.get("LANGSMITH_API_KEY"))

    @property
    def has_langfuse_tracing(self):
        """Return True if Langfuse tracing is active (a handler was built)."""
        return self._get_langfuse_handler() is not None

    def get_tracing_config(self) -> dict:
        """Return the current tracing / observability config.

        Returns a dict of all tracing-related settings so gateway / external
        callers can inspect the active tracing configuration.
        """
        # read any gateway-set tracing values before returning
        ls_project = self._get_cap("langsmith_project") or self._ls_project or "hermes"
        ls_tags = self._get_cap("langsmith_tags") or self._ls_tags or ["hermes"]
        ls_api = self._get_cap("langsmith_api_key") or self._ls_api_key
        if isinstance(ls_tags, str):
            ls_tags = [ls_tags]

        return {
            "checkpointer": self._langgraph_checkpointer,
            "store": self._langgraph_store,
            "debug": self._debug is True,
            "langsmith_project": ls_project,
            "langsmith_tags": list(ls_tags),
            "langsmith_enabled": bool(ls_api),
            "langfuse_enabled": self.has_langfuse_tracing,
        }

    # ------------------------------------------------------------------
    # Memory delegation (pass-through)
    # ------------------------------------------------------------------

    def get_memory_context(self):
        return None

    def save_memory(self, messages):
        pass

    def interrupt(self):
        pass  # LangGraph doesn't have a direct interrupt
