"""
Bootstrap - Initialization sequence for AIAgent.

This class orchestrates the full initialization of an AIAgent by:
1. Loading environment variables from .env files
2. Creating the EventBus (via hermes.analytics)
3. Initializing memory (MemoryStore, MemoryManager)
4. Initializing tools (via model_tools)
5. Pre-warming the model metadata cache
6. Setting up signal handlers (SIGINT/SIGTERM)
7. Registering shutdown hooks

All initialization happens in initialize(), called once after AppState is created.
"""
import logging
import os
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from run_agent import AIAgent

logger = logging.getLogger(__name__)


class _TracingEventHandler:
    """
    Subscribes to EventBus and creates HermesTracer spans for key events.

    This bridges the EventBus analytics system with the distributed tracing
    system, ensuring all emitted events are visible in trace spans (OS: Observable State).
    """

    def __init__(self, tracer, event_bus):
        self.tracer = tracer
        self.event_bus = event_bus
        self._subscribe()

    def _subscribe(self):
        from agent.hermes.analytics import EventType
        self.event_bus.subscribe(EventType.TOOL_CALL, self.on_tool_call)
        self.event_bus.subscribe(EventType.TOOL_RESULT, self.on_tool_result)
        self.event_bus.subscribe(EventType.LLM_CALL, self.on_llm_call)
        self.event_bus.subscribe(EventType.LLM_RESPONSE, self.on_llm_response)
        self.event_bus.subscribe(EventType.ERROR, self.on_error)

    def on_tool_call(self, event):
        if self.tracer:
            try:
                with self.tracer.span(
                    "event.tool_call",
                    attributes={"tool.name": event.payload.get("tool", "unknown")}
                ):
                    pass
            except Exception:
                pass

    def on_tool_result(self, event):
        if self.tracer:
            try:
                success = not event.payload.get("error")
                self.tracer.record_span(
                    span_name="event.tool_result",
                    duration_seconds=0,
                    success=success,
                    attributes={"tool.name": event.payload.get("tool", "unknown")},
                )
            except Exception:
                pass

    def on_llm_call(self, event):
        if self.tracer:
            try:
                with self.tracer.span(
                    "event.llm_call",
                    attributes={"model": event.payload.get("model", "unknown")}
                ):
                    pass
            except Exception:
                pass

    def on_llm_response(self, event):
        if self.tracer:
            try:
                tokens_in = event.payload.get("input_tokens", 0)
                tokens_out = event.payload.get("output_tokens", 0)
                if tokens_in or tokens_out:
                    self.tracer.record_token_usage(input_tokens=tokens_in, output_tokens=tokens_out)
            except Exception:
                pass

    def on_error(self, event):
        if self.tracer:
            try:
                self.tracer.record_error(
                    error_type=event.payload.get("error_type", "Unknown"),
                    operation=event.payload.get("operation", "unknown"),
                )
            except Exception:
                pass


class Bootstrap:
    """
    Orchestrates the complete initialization sequence for an AIAgent.

    Args:
        app_state: AppState instance containing all agent configuration
        agent: AIAgent instance being bootstrapped
    """

    def __init__(self, app_state, agent: "AIAgent"):
        self.app_state = app_state
        self.agent = agent

    def initialize(self) -> None:
        """
        Run the complete initialization sequence.

        Called once after AIAgent.__init__ creates the AppState.
        The order matters: env must be loaded before other init steps.
        """
        self._load_environment()
        self._init_event_bus()
        self._init_memory()
        self._init_tools()
        self._init_model_cache()
        self._init_signal_handlers()
        self._register_shutdown_hooks()

    def _load_environment(self) -> None:
        """
        Load environment variables from .env files.

        This is called first so that environment variables are available
        for all subsequent initialization steps.
        """
        # The hermes_cli.env_loader is already invoked at module level in run_agent.py.
        # This method is here for any additional env processing needed
        # during bootstrap (e.g., validating required env vars).
        logger.debug("Bootstrap: environment loaded")

    def _init_event_bus(self) -> None:
        """
        Create and attach the EventBus.

        CRITICAL: EventBus is created ONLY here, not in run_agent.py.__init__.
        This fixes the double-init bug where Phase 1 created an EventBus
        that was then orphaned when Bootstrap created a second one.
        """
        from agent.hermes.analytics import EventBus
        self.agent._event_bus = EventBus()
        logger.debug("Bootstrap: EventBus initialized")

        # Add default telemetry backends based on config
        self._init_telemetry_backends()

        # Initialize CostAttributor and subscribe to events
        from agent.cost_attributor import CostAttributor
        self.agent._cost_attributor = CostAttributor(self.agent._event_bus)
        logger.debug("Bootstrap: CostAttributor initialized")

        # Initialize TaskRouter for intelligent routing
        from agent.task_router import TaskRouter
        self.agent._task_router = TaskRouter()
        self.app_state._task_router = self.agent._task_router
        logger.debug("Bootstrap: TaskRouter initialized")

        # Initialize LoadAwareRouter for load-based task routing
        from tools.delegate_tool import MAX_CONCURRENT_CHILDREN
        from agent.routing.load_aware_router import LoadAwareRouter
        self.agent._load_router = LoadAwareRouter(max_concurrent=MAX_CONCURRENT_CHILDREN)
        logger.debug("Bootstrap: LoadAwareRouter initialized")

        # Initialize LLMFallbackEngine for graceful degradation (GL: Generative Loop)
        # The fallback engine is a singleton, injected with the EventBus so all
        # fallback events are observable via the telemetry pipeline.
        from agent.llm_fallback_engine import get_fallback_engine
        self.agent._fallback_engine = get_fallback_engine(event_bus=self.agent._event_bus)
        logger.debug("Bootstrap: LLMFallbackEngine initialized")

        # Initialize LLM Circuit Breaker Manager for cascading failure prevention
        # Per-provider circuit breakers automatically detect failing endpoints and
        # circuit-trip to prevent wasted retries (OS: Observable State via EventBus).
        from agent.llm_circuit_breakers import get_circuit_breaker_manager
        self.agent._cb_manager = get_circuit_breaker_manager(event_bus=self.agent._event_bus)
        logger.debug("Bootstrap: LLMCircuitBreakerManager initialized")

        # Initialize OrphanCleanupManager for GL (Generative Loop) resource cleanup
        # This ensures orphan processes from crashed subagents are detected and cleaned up
        try:
            from agent.hermes.orphan_process_manager import OrphanCleanupManager
            manager = OrphanCleanupManager.get_instance()
            manager.set_event_bus(self.agent._event_bus)
            manager.register_with_shutdown_manager()
            manager.start_periodic_scan()
            logger.debug("Bootstrap: OrphanCleanupManager initialized")
        except ImportError:
            logger.debug("Bootstrap: OrphanCleanupManager not available (psutil may be missing)")
        except Exception as e:
            logger.debug("Bootstrap: OrphanCleanupManager init failed: %s", e)

        # Also initialize orphan cleanup on the SubagentCoordinator (if one exists)
        coord = getattr(self.agent, "_subagent_coordinator", None)
        if coord is not None:
            try:
                coord.initialize_orphan_cleanup()
            except Exception as e:
                logger.debug("SubagentCoordinator orphan cleanup init failed: %s", e)

        # Initialize GL Orchestrator for Generative Loop self-evolution
        # The orchestrator ties together FeedbackCollector, LearningEngine,
        # and AdaptiveController for the self-improvement feedback loop (GL principle).
        try:
            from agent.hermes.gl_orchestrator import GLOrchestrator
            gl_enabled = os.environ.get("HERMES_GL_ENABLED", "1").lower() in ("1", "true", "yes")
            if gl_enabled:
                orchestrator = GLOrchestrator(
                    event_bus=self.agent._event_bus,
                    task_router=self.agent._task_router,
                    load_router=getattr(self.agent, "_load_router", None),
                    auto_apply=True,
                    auto_evaluate=True,
                )
                # Defer full initialization (needs hermes_home from app_state)
                # The run_agent.py will call orchestrator.initialize() once
                # app_state is fully populated.
                self.agent._gl_orchestrator = orchestrator
                self.agent._gl_init_pending = True
                logger.debug("Bootstrap: GLOrchestrator initialized (pending hermes_home)")
            else:
                self.agent._gl_orchestrator = None
                logger.debug("Bootstrap: GL Orchestrator disabled via HERMES_GL_ENABLED=0")
        except ImportError as e:
            logger.debug("Bootstrap: GL components not available: %s", e)
            self.agent._gl_orchestrator = None
        except Exception as e:
            logger.debug("Bootstrap: GL Orchestrator init failed: %s", e)
            self.agent._gl_orchestrator = None

        # HermesTracer singleton initialization for distributed tracing (OS: Observable State)
        _otel_endpoint = os.environ.get("OTEL_EXPORTER_ENDPOINT", os.environ.get("HERMES_OTLP_ENDPOINT", ""))
        if _otel_endpoint:
            from agent.hermes.telemetry import HermesTracer, ExporterType
            _otel_type_str = os.environ.get("OTEL_EXPORTER_TYPE", "otlp_grpc")
            try:
                _exporter_type = ExporterType(_otel_type_str)
            except ValueError:
                _exporter_type = ExporterType.OTLP_GRPC
            tracer = HermesTracer.get(
                service_name=os.environ.get("OTEL_SERVICE_NAME", "hermes-agent"),
                endpoint=_otel_endpoint,
                exporter_type=_exporter_type,
            )
            self.agent._hermes_tracer = tracer
            logger.debug(f"Bootstrap: HermesTracer initialized: {tracer.service_name}")

            # TracingEventHandler: bridge EventBus → HermesTracer spans
            self.agent._tracing_handler = _TracingEventHandler(tracer, self.agent._event_bus)

    def _init_telemetry_backends(self) -> None:
        """
        Initialize default telemetry backends based on environment/config.

        Console backend is enabled by default in debug/verbose mode.
        File backend is enabled if HERMES_TELEMETRY_FILE is set.
        OTLP backend is enabled if HERMES_OTLP_ENDPOINT is set.
        """
        import os

        # Console backend (debug mode)
        verbose = getattr(self.app_state, 'verbose_logging', False) or os.environ.get("HERMES_TELEMETRY_CONSOLE", "").lower() in ("1", "true", "yes")
        if verbose:
            from agent.hermes.telemetry import ConsoleBackend
            self.agent._event_bus.add_backend(ConsoleBackend())
            logger.debug("Bootstrap: ConsoleBackend telemetry enabled")

        # File backend
        telemetry_file = os.environ.get("HERMES_TELEMETRY_FILE", "")
        if telemetry_file:
            from agent.hermes.telemetry import FileBackend
            self.agent._event_bus.add_backend(FileBackend(telemetry_file))
            logger.debug(f"Bootstrap: FileBackend telemetry enabled: {telemetry_file}")

        # OpenTelemetry backend
        otlp_endpoint = os.environ.get("HERMES_OTLP_ENDPOINT", "")
        if otlp_endpoint:
            try:
                from agent.hermes.telemetry import OpenTelemetryBackend
                self.agent._event_bus.add_backend(OpenTelemetryBackend(endpoint=otlp_endpoint))
                logger.debug(f"Bootstrap: OpenTelemetryBackend telemetry enabled: {otlp_endpoint}")
            except Exception as e:
                logger.warning(f"Bootstrap: Failed to initialize OpenTelemetryBackend: {e}")

    def _init_memory(self) -> None:
        """
        Initialize memory-related components.

        Sets up MemoryStore (persistent memory from disk) and
        MemoryManager (provider-based memory plugins).
        """
        # Memory initialization is handled in run_agent.py.__init__
        # after Bootstrap.initialize() returns. This method is a placeholder
        # for any memory-related setup that should happen during bootstrap.
        logger.debug("Bootstrap: memory subsystem ready")

    def _init_tools(self) -> None:
        """
        Initialize the tool system.

        This pre-warms tool metadata and validates toolset requirements.
        Tool definitions are already loaded in run_agent.py.__init__.
        """
        # Tool initialization (get_tool_definitions) is called in run_agent.py.__init__
        # during the attribute copy-back phase. This method is a placeholder
        # for any tool-related pre-warming needed during bootstrap.
        logger.debug("Bootstrap: tool system ready")

    def _init_model_cache(self) -> None:
        """
        Pre-warm the model metadata cache.

        For OpenRouter providers, fetch_model_metadata() is cached for 1 hour.
        Running it in a background thread avoids a blocking HTTP request
        on the first API response when pricing is estimated.
        """
        from agent.model_metadata import fetch_model_metadata

        # Only pre-warm for OpenRouter
        if self.app_state.provider == "openrouter" or self._is_openrouter_url():
            threading.Thread(
                target=lambda: fetch_model_metadata(),
                daemon=True,
            ).start()
            logger.debug("Bootstrap: model metadata cache pre-warming started")

    def _is_openrouter_url(self) -> bool:
        """Check if the base URL is OpenRouter."""
        base_url_lower = getattr(self.app_state, '_base_url_lower', '') or ''
        return "openrouter" in base_url_lower

    def _init_signal_handlers(self) -> None:
        """
        Set up SIGINT/SIGTERM signal handlers for graceful shutdown.

        Uses ShutdownManager singleton which supports prioritized hook execution:
        - Priority 40: credential cleanup
        - Priority 50: terminal sandbox cleanup
        - Priority 60: browser session cleanup
        """
        from agent.hermes.shutdown import ShutdownManager

        self.agent._shutdown_manager = ShutdownManager.get_instance()
        self.agent._shutdown_manager.setup_signal_handlers()
        logger.debug("Bootstrap: signal handlers registered")

    def _register_shutdown_hooks(self) -> None:
        """
        Migrate existing atexit handlers to ShutdownManager.

        Existing atexit handlers in terminal_tool.py, browser_tool.py,
        and credential_files.py are registered with ShutdownManager
        at their documented priority levels.
        """
        # The atexit handlers in terminal_tool.py, browser_tool.py, and
        # interrupt.py are already registered at module import time.
        # During Bootstrap, we ensure ShutdownManager is aware of them
        # if they haven't already been migrated.
        #
        # Migration of individual handlers to ShutdownManager priority system:
        # - terminal_tool._atexit_cleanup -> priority=50
        # - browser_tool._emergency_cleanup_all_sessions -> priority=60
        # - credential_files.clear_credential_files -> priority=40
        #
        # This is handled by the individual modules when they are imported,
        # as they now check for ShutdownManager availability.
        logger.debug("Bootstrap: shutdown hooks registered")
