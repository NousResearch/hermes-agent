"""AIAgent → MultiStepLoop delegation example.

Shows how the existing ``AIAgent.run()`` in hermes-agent (currently a
~15.8k LOC class in ``run_agent.py``) delegates its inner loop to the
kernel without changing its public signature.

The pattern is intentionally minimal:

  1. Build the model shim (one per provider, lazily).
  2. Build the tool handler shim.
  3. Choose the governance policy (config-driven).
  4. Move hermes's scattered cross-cutting helpers
     (``IterationBudget``, ``_should_parallelize_tool_batch``,
     compression, redaction, telemetry) into **callbacks** registered
     against the relevant step type.
  5. Run the kernel; translate the typed ``RunResult`` back into
     hermes's existing return shape via ``RunResult.to_legacy_dict()``.

Existing callers of ``AIAgent.run()`` see no change in API.
"""

from __future__ import annotations

from typing import Any

from agent.runtime import (
    ActionStep,
    AllowListGovernance,
    CallbackRegistry,
    DenyAllGovernance,
    GovernanceProtocol,
    MultiStepLoop,
    RunResult,
    RunState,
    StepFailure,
)

from .hermes_model_shim import HermesModelShim
from .hermes_tool_handler_shim import HermesToolHandler


# ----- example callback adaptations ------------------------------------------
#
# In hermes-agent, these wrap real helpers. The bodies below show the
# CALLBACK SHAPE — the actual logic continues to live in run_agent.py
# until you migrate it.


def make_iteration_budget_callback(max_iterations: int):
    """Adapts hermes's ``IterationBudget`` (run_agent.py:287) as a callback.

    Hermes's existing class tracks iterations and may stop the run. As a
    callback we increment a counter and trip the cooperative interrupt
    when it exceeds the budget.
    """
    def _callback(step: ActionStep, state: RunState) -> None:
        n = state.increment("iterations", reason="action step consumed an iteration")
        if n >= max_iterations:
            state.set("__interrupt__", True, reason=f"iteration budget {max_iterations} reached")
    return _callback


def make_compression_callback():
    """Hook for ``agent/context_compressor.py``.

    The actual compression continues to run inside the adapter when it
    builds messages; this callback decides whether to *request* compression
    on the next turn by setting a state flag the model shim reads.
    """
    def _callback(step: ActionStep, state: RunState) -> None:
        if step.input_tokens > 100_000:  # TODO(hermes): use hermes's real threshold
            state.set("request_compression", True, reason="approaching context window")
    return _callback


def make_telemetry_callback(sink):
    """Send each step to a metrics sink (hermes uses ``agent/insights.py``).

    The sink is anything callable — pass in your existing telemetry
    pipeline. The callback is the only adapter needed.
    """
    def _callback(step, state: RunState) -> None:
        sink({"type": type(step).__name__, "step_number": step.step_number})
    return _callback


# ----- the delegation pattern ------------------------------------------------


def choose_governance(config: dict[str, Any]) -> GovernanceProtocol:
    """Pick a policy from ``~/.hermes/config.yaml``.

    Default is deny-all. Hermes's existing approval system
    (``tools/approval.py``) becomes a richer policy you can swap in here.
    """
    mode = (config.get("agent", {}).get("governance") or "deny-all").lower()
    if mode == "allow-all":
        from agent.runtime import AllowAllGovernance
        return AllowAllGovernance()
    if mode.startswith("allow-list:"):
        # ``allow-list: lookup, final_answer`` → ["lookup", "final_answer"].
        # Strip whitespace and drop empty entries so a stray comma or
        # trailing space doesn't silently exclude a tool.
        raw = mode.split(":", 1)[1]
        tools = [t.strip() for t in raw.split(",") if t.strip()]
        return AllowListGovernance(allowed=tools)
    if mode == "acgs":
        from .acgs_governance import build_acgs_governance_from_config
        return build_acgs_governance_from_config(config)
    return DenyAllGovernance()


class AIAgentKernelAdapter:
    """Drop-in shape for ``AIAgent.run()``'s inner loop.

    In hermes-agent, ``AIAgent.run(task)`` should construct one of these
    per request, drive ``adapter.run(task)``, and return the legacy dict.
    The remaining bulk of ``AIAgent`` (system prompt building, session
    management, history compaction, gateway emission) is unchanged.
    """

    def __init__(
        self,
        *,
        provider_adapter: Any,        # hermes adapter, e.g. AnthropicAdapter
        config: dict[str, Any],
        tools: list[dict[str, Any]],
        telemetry_sink=None,
        tool_dispatch=None,           # hermes ``handle_function_call`` or compat
    ) -> None:
        callbacks = CallbackRegistry()
        callbacks.register(
            ActionStep,
            make_iteration_budget_callback(
                max_iterations=config.get("agent", {}).get("max_iterations", 50),
            ),
        )
        callbacks.register(ActionStep, make_compression_callback())
        if telemetry_sink is not None:
            from agent.runtime import MemoryStep
            callbacks.register(MemoryStep, make_telemetry_callback(telemetry_sink))

        handler_kwargs: dict[str, Any] = {}
        if tool_dispatch is not None:
            handler_kwargs["dispatch"] = tool_dispatch

        self._loop = MultiStepLoop(
            model=HermesModelShim(provider_adapter),
            tool_handler=HermesToolHandler(**handler_kwargs),
            tools=tools,
            governance=choose_governance(config),
            callbacks=callbacks,
            max_steps=config.get("agent", {}).get("max_steps", 20),
            planning_interval=config.get("agent", {}).get("planning_interval"),
            continue_on_error=config.get("agent", {}).get("continue_on_error", False),
        )

    def run(self, task: str) -> dict[str, Any]:
        result: RunResult = self._loop.run(task)
        # Legacy callers consume dicts. The kernel returns a typed result;
        # ``to_legacy_dict()`` preserves the existing shape so callers of
        # ``AIAgent.run()`` don't need updating in the same PR.
        return result.to_legacy_dict()

    @property
    def loop(self) -> MultiStepLoop:
        """Exposed for callers that want the typed result or trace."""
        return self._loop


# ----- migration checklist (for the hermes-agent maintainer) -----------------
#
# Step 1. Land ``agent/runtime/`` + ``tests/runtime/``. Existing suite green.
# Step 2. Add ``HermesModelShim`` to ONE adapter (start with the simplest —
#         e.g. ``codex_responses_adapter`` rather than the 89 KB Anthropic).
# Step 3. Add ``HermesToolHandler`` wrapping ``handle_function_call``.
# Step 4. Add ``AIAgentKernelAdapter`` behind a config flag
#         ``agent.runtime: kernel | legacy``.
# Step 5. Pick a small reference task; run both paths; diff trajectories.
# Step 6. Move ``IterationBudget``, ``_should_parallelize_tool_batch``,
#         compression, redaction into callbacks one at a time. Each move
#         is one PR with a regression test.
# Step 7. When kernel is at parity for one provider, flip the default and
#         loop in the remaining providers.
