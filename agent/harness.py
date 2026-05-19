"""Unified Hermes harness facade.

``HermesHarness`` is the public harness shape for Hermes runtime concerns.  It
keeps the two important internal boundaries explicit:

* ``system_prompt`` owns model-visible prompt assembly.
* ``control_plane`` owns sidecar runtime evidence, harness runs, and health.

The facade is intentionally thin so existing module-level helpers remain easy
to patch in tests and easy to call from legacy code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class SystemPromptHarness:
    """Prompt-facing harness operations for a single agent instance."""

    agent: Any

    def _require_agent(self) -> Any:
        if self.agent is None:
            raise ValueError("SystemPromptHarness requires an agent instance")
        return self.agent

    def build_parts(self, system_message: Optional[str] = None) -> Dict[str, str]:
        from agent import system_prompt

        return system_prompt.build_system_prompt_parts(
            self._require_agent(),
            system_message=system_message,
        )

    def build(self, system_message: Optional[str] = None) -> str:
        from agent import system_prompt

        return system_prompt.build_system_prompt(
            self._require_agent(),
            system_message=system_message,
        )

    def invalidate(self) -> None:
        from agent import system_prompt

        system_prompt.invalidate_system_prompt(self._require_agent())

    def format_tools(self) -> str:
        from agent import system_prompt

        return system_prompt.format_tools_for_system_message(self._require_agent())


@dataclass(frozen=True)
class ControlPlaneHarness:
    """Runtime/control-plane harness operations.

    This side never contributes to the model-visible system prompt.  It records
    evidence and runs checks under the active Hermes profile.
    """

    profile: Optional[str] = None

    @property
    def core_name(self) -> str:
        from agent import harness_control_plane

        return harness_control_plane.CORE_HARNESS_NAME

    def core_suite(self) -> Dict[str, Any]:
        from agent import harness_control_plane

        if self.profile is None:
            return harness_control_plane.core_harness_suite()
        return harness_control_plane.core_harness_suite(self.profile)

    def core_status(self) -> Dict[str, Any]:
        from agent import harness_control_plane

        if self.profile is None:
            return harness_control_plane.core_harness_status()
        return harness_control_plane.core_harness_status(self.profile)

    def run_core(
        self,
        *,
        case_ids: Optional[Iterable[str]] = None,
        timeout_s: float = 600.0,
        runner: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        from agent import harness_control_plane

        kwargs: Dict[str, Any] = {"case_ids": case_ids}
        if self.profile is not None:
            kwargs["profile"] = self.profile
        if timeout_s != 600.0:
            kwargs["timeout_s"] = timeout_s
        if runner is not None:
            kwargs["runner"] = runner
        return harness_control_plane.run_core_harness(**kwargs)

    def learning_health(self) -> Dict[str, Any]:
        from agent import harness_control_plane

        return harness_control_plane.learning_health_summary()

    def learning_snapshot(self) -> Dict[str, Any]:
        from agent import harness_control_plane

        return harness_control_plane.learning_snapshot_summary()

    def replay_corpus(self) -> Dict[str, Any]:
        from agent import harness_control_plane

        return harness_control_plane.replay_corpus_summary()

    def promotion_gates(self) -> Dict[str, Any]:
        from agent import harness_control_plane

        return harness_control_plane.promotion_gate_summary()

    def context_hygiene(self) -> Dict[str, Any]:
        from agent import context_hygiene

        return context_hygiene.audit_context_hygiene()

    def skill_lifecycle(self) -> Dict[str, Any]:
        from agent import skill_lifecycle

        return skill_lifecycle.audit_skill_lifecycle()

    def autonomous_loops(self) -> Dict[str, Any]:
        from agent import autonomous_loops

        return autonomous_loops.audit_autonomous_loops()

    def learning_health_unavailable(self, error: Optional[str] = None) -> Dict[str, Any]:
        from agent import harness_control_plane

        return harness_control_plane.learning_health_unavailable_summary(error)

    def record_event(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        component: Optional[str] = None,
        runtime: Optional[str] = None,
    ) -> Dict[str, Any]:
        from agent import harness_control_plane

        return harness_control_plane.record_harness_event(
            event_type,
            payload,
            trace_id=trace_id,
            session_id=session_id,
            component=component,
            runtime=runtime,
        )

    def record_verification(
        self,
        *,
        name: str,
        status: str,
        command: Optional[str] = None,
        result: Optional[str] = None,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        from agent import harness_control_plane

        return harness_control_plane.record_verification_result(
            name=name,
            status=status,
            command=command,
            result=result,
            trace_id=trace_id,
            session_id=session_id,
        )

    def record_eval_suite(
        self,
        *,
        name: str,
        status: str,
        checks: Optional[List[str]] = None,
        result: Optional[str] = None,
    ) -> Dict[str, Any]:
        from agent import harness_control_plane

        return harness_control_plane.record_eval_suite(
            profile=self.profile,
            name=name,
            status=status,
            checks=checks,
            result=result,
        )


class HermesHarness:
    """One top-level Hermes harness with prompt and control-plane sides."""

    def __init__(self, agent: Any = None, *, profile: Optional[str] = None):
        self.agent = agent
        self.system_prompt = SystemPromptHarness(agent)
        self.control_plane = ControlPlaneHarness(profile=profile)

    @classmethod
    def for_agent(cls, agent: Any, *, profile: Optional[str] = None) -> "HermesHarness":
        return cls(agent, profile=profile)


__all__ = [
    "ControlPlaneHarness",
    "HermesHarness",
    "SystemPromptHarness",
]
