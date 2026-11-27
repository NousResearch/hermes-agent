#!/usr/bin/env python3
"""Smoke the real plugin controller with one isolated Hermes counterpoint child."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1] / "plugins" / "otto-counterpoint"
_RESPONSE = (
    "Architecture assessment: the admitted request is a design review, not an external operation.\n"
    "Evidence boundary: this isolated smoke has no source documents beyond the request metadata.\n"
    "Uncertainty: conclusions are provisional until independent sources are supplied.\n"
    "Recommendation: compare primary sources before implementation; no external effect was requested or performed."
)


def _workflow_smoke_passed(result: object) -> bool:
    """Return whether the full smoke reached a locally accepted terminal result."""
    return bool(
        getattr(result, "status", None) == "succeeded"
        and getattr(getattr(result, "decision", None), "verdict", None) == "accept_local"
    )


def _load_controller():
    namespace = types.ModuleType("hermes_plugins")
    namespace.__path__ = []
    sys.modules.setdefault("hermes_plugins", namespace)
    name = "hermes_plugins.otto_counterpoint"
    spec = importlib.util.spec_from_file_location(
        name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load plugin package")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = name
    module.__path__ = [str(PLUGIN_DIR)]
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.CounterpointController


class _State:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir


def main() -> int:
    CounterpointController = _load_controller()
    with tempfile.TemporaryDirectory(prefix="otto-counterpoint-smoke-") as directory:
        controller = CounterpointController(
            config={
                "mode": "shadow",
                "timeout_seconds": 180,
                "run_budget_seconds": 120,
                "counterpoint_route": {
                    "vendor": "anthropic",
                    "family": "anthropic",
                    "provider": "nous",
                    "model": "anthropic/claude-sonnet-5",
                    "reasoning_effort": "high",
                    "authenticated": True,
                    "accessible": True,
                    "smoke_tested": True,
                    "relative_load": 1.0,
                },
            },
            state=_State(Path(directory)),
        )
        controller.on_pre_llm_call(
            session_id="smoke-session",
            task_id="smoke-task",
            turn_id="smoke-turn",
            user_message="research the architecture and compare the evidence",
            conversation_history=[],
            is_first_turn=True,
            model="gpt-5.6-luna",
            provider="openai-codex",
            api_mode="codex_responses",
            platform="local-smoke",
        )
        if "--probe" in sys.argv:
            try:
                completion = controller._default_client().complete(
                    controller.pending("smoke-turn").counterpoint_route,
                    "Return exactly this JSON object and no other text: "
                    '{"status":"no_material_finding","findings":[],"coverage":{"criteria":["probe"]},"evidence_refs":["probe"]}',
                )
                print(
                    json.dumps(
                        {
                            "probe": "passed",
                            "provider": completion.provider,
                            "model": completion.model,
                            "tool_count": completion.tool_count,
                        },
                        sort_keys=True,
                    )
                )
            except Exception as exc:  # health: allow BLE001 -- diagnostic branch emits only bounded metadata
                error = "probe_failed"
                if "--debug" in sys.argv:
                    error = f"{type(exc).__name__}:{exc}"
                print(json.dumps({"probe": "failed", "error": error}, sort_keys=True))
                return 1
            return 0
        if "--critic-probe" in sys.argv:
            from hermes_plugins.otto_counterpoint.counterpoint import Artifact, HermesCounterpointCallbacks

            pending = controller.pending("smoke-turn")
            callbacks = HermesCounterpointCallbacks(
                client=controller._default_client(),
                generator_route=pending.generator_route,
                counterpoint_route=pending.counterpoint_route,
            )
            artifact = Artifact.from_content(
                run_id=pending.envelope.run_id,
                task_id=pending.envelope.task_id,
                artifact_id="smoke-artifact-v1",
                version=1,
                content=_RESPONSE,
                content_ref="memory://smoke/artifact-v1",
                claims=({"claim_id": "response", "text": "bounded response"},),
                evidence_refs=(f"request-sha256:{pending.envelope.request_sha256}",),
                tests=({
                    "test_id": "response-present",
                    "name": "response is non-empty",
                    "result": "pass",
                    "evidence_id": "gate:smoke:response-present",
                },),
            )
            callbacks._contents[artifact.content_ref] = _RESPONSE
            try:
                critique = callbacks.critic(
                    artifact,
                    (
                        "Answer the admitted request accurately.",
                        "Identify uncertainty and distinguish evidence from interpretation.",
                        "Do not authorize external effects.",
                    ),
                )
                print(
                    json.dumps(
                        {
                            "critic_probe": "passed",
                            "status": critique.status,
                            "finding_dispositions": [
                                item.get("disposition") for item in critique.findings
                            ],
                        },
                        sort_keys=True,
                    )
                )
            except Exception:  # health: allow BLE001 -- diagnostic branch emits only bounded metadata
                print(json.dumps({"critic_probe": "failed", "error": "critic_probe_failed"}, sort_keys=True))
                return 1
            return 0
        transformed = controller.on_transform_llm_output(
            response_text=_RESPONSE,
            session_id="smoke-session",
            model="gpt-5.6-luna",
            platform="local-smoke",
            turn_id="smoke-turn",
        )
        events = controller.ledger.list_run("cp-smoke-turn")
        pending = controller.pending("smoke-turn")
        result = controller.last_results.get("smoke-turn")
        smoke_passed = _workflow_smoke_passed(result)
        print(
            json.dumps(
                {
                    "decision": pending.outcome.decision.value,
                    "generator": f"{pending.generator_route.provider}/{pending.generator_route.model}",
                    "counterpoint": f"{pending.counterpoint_route.provider}/{pending.counterpoint_route.model}",
                    "workflow_status": getattr(result, "status", None),
                    "workflow_verdict": getattr(getattr(result, "decision", None), "verdict", None),
                    "critique_status": getattr(getattr(result, "critique", None), "status", None),
                    "correction_count": getattr(result, "correction_count", None),
                    "blocked_reason": getattr(result, "blocked_reason", None),
                    "finding_dispositions": [
                        getattr(item, "disposition", None)
                        for item in getattr(getattr(result, "critique", None), "findings", ())
                    ],
                    "ledger_events": [event["event_type"] for event in events],
                    "response_replaced": transformed is not None,
                    "tool_count": 0,
                },
                sort_keys=True,
            )
        )
        return 0 if smoke_passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
