try:
    from prometheus_client import Counter, Gauge, Histogram
except ModuleNotFoundError:  # pragma: no cover - optional instrumentation dep
    # Metrics are optional instrumentation. If prometheus_client isn't installed
    # we fall back to no-op stubs so importing this module can NEVER fail. A hard
    # failure here poisons the sys.modules entries of agent.display /
    # agent.tool_executor (which import this transitively via tools.delegate_tool)
    # and cascades into "cannot import name 'redact_tool_args_for_display'"
    # ImportErrors that take down every tool call. See incident 2026-06-30.
    class _NoopMetric:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            pass

        def dec(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

    Counter = Gauge = Histogram = _NoopMetric

# Subagent lifecycle metrics
SUBAGENT_ACTIVE_COUNT = Gauge(
    "hermes_subagent_active_count",
    "Number of currently active subagents."
)

SUBAGENT_ERRORS_TOTAL = Counter(
    "hermes_subagent_errors_total",
    "Total number of subagent tasks that ended in error.",
    ["model", "error_type"]
)

SUBAGENT_DURATION_SECONDS = Histogram(
    "hermes_subagent_duration_seconds",
    "Duration of subagent task execution.",
    ["model"]
)

SUBAGENT_STARTUP_LATENCY_SECONDS = Histogram(
    "hermes_subagent_startup_latency_seconds",
    "Latency from delegate_task call to first subagent tool/progress event.",
    ["model"]
)
