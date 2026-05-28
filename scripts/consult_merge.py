#!/usr/bin/env python3
"""
HERMES CONSULT/MERGE ORCHESTRATOR
The main protocol implementation — ties together:
- Task classification
- Model routing
- Consult (sub-agent delegation)
- Merge/Become (persona swap)
- Quality gate (Ring review)
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

sys.path.insert(0, os.path.expanduser("~/.hermes/scripts"))


class ConsultMergeOrchestrator:
    def __init__(self):
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.context_window = []

    def load_architect_context(self) -> str:
        """Load the permanent architect context."""
        architect_path = os.path.expanduser("~/.hermes/context-architect.md")
        if os.path.exists(architect_path):
            with open(architect_path, "r") as f:
                return f.read()
        return ""

    def classify(self, user_message: str) -> str:
        """Classify task type."""
        from model_routing import classify_task
        return classify_task(user_message)

    def route(self, task_category: str, prompt: str,
              history_length: int = 0, budget: float = 1.0) -> Dict:
        """Select best model for this task."""
        from model_routing import select_model
        return select_model(task_category, prompt, history_length, budget)

    def consult(self, task: str, persona: str = "athena",
                context: str = "", budget: float = 1.0) -> Dict:
        """
        Delegate a sub-task to a consultant (sub-agent).

        In production, this would use delegate_task.
        Here we simulate the protocol and return the structured prompt
        that Hermes would send to the chosen model.
        """
        consultant_prompts = {
            "athena": """You are Athena — Strategic/Critical Analyst.
Analyze the following independently. Do NOT agree automatically.
Identify: gaps, risks, logical flaws, alternatives.
Format: FINDING | EVIDENCE | RECOMMENDATION | CONFIDENCE""",

            "hermes": """You are Hermes — Coordinator. Synthesize, plan,
and propose actionable next steps. Consider cost, time, and risk.""",

            "mnemosyne": """You are Mnemosyne — Memory Weaver. Connect
this task to past patterns, stored knowledge, and ongoing projects.""",
        }

        constitution = consultant_prompts.get(persona, consultant_prompts["athena"])

        # Task classification determines consultant choice
        task_category = self.classify(task)
        model_choice = self.route(task_category, task, budget=budget)

        return {
            "action": "consult",
            "consultant": persona,
            "task_category": task_category,
            "selected_model": model_choice,
            "constitution": constitution,
            "prompt": f"{constitution}\n\nTASK: {task}\n\nCONTEXT: {context}\n\nANALYSIS:",
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def merge(self, persona: str, task: str, context: str = "") -> Dict:
        """
        Merge — adopt a model's reasoning style for the current task.
        Prepends persona context to the active context window.
        """
        merge_prompts = {
            "athena_critique": "Adopt Athena's analytical mode. Critically evaluate all incoming information. Question assumptions.",
            "hermes_plan": "Adopt Hermes' coordination mode. Plan, orchestrate, and delegate efficiently.",
            "dionysus_creative": "Adopt Dionysus' creative mode. Think laterally, make unexpected connections, challenge conventions.",
        }

        persona_context = merge_prompts.get(
            persona,
            f"Adopt {persona} mode for the next interaction."
        )

        # Merge prepends persona to context
        merged_context = f"{persona_context}\n\nCURRENT TASK: {task}\n\n{context}"

        model_choice = self.route(self.classify(task), task)

        return {
            "action": "merge",
            "persona": persona,
            "model": model_choice,
            "context_to_prepend": persona_context,
            "merged_prompt": merged_context[:2000],  # Truncate for safety
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": "After merge, re-read context-architect.md to maintain coherence",
        }

    def become(self, persona: str, task: str) -> Dict:
        """
        Become — full persona swap. Use for one complete turn cycle.
        """
        return self.merge(persona, task, context="[FULL PERSONA SWAP — BECOME MODE]")

    def quality_gate(self, content: str, task_type: str) -> Dict:
        """
        Route through Ring for final quality verification.
        Always the last step before delivery.
        """
        route = self.route(task_type, content)
        return {
            "action": "quality_gate",
            "model": "openrouter:ring-2.6-1t",
            "route_detail": route,
            "content_preview": content[:200] + ("..." if len(content) > 200 else ""),
            "prompt": f"Verify and quality-check the following {task_type} output. Flag errors, inconsistencies, and suggest improvements. Rate quality 1-10:\n\n{content}",
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def full_cycle(self, task: str, context: str = "",
                   budget: float = 1.0) -> Dict:
        """
        Execute complete consult/merge/quality-gate cycle.

        1. Classify task
        2. Route to model
        3. Consult Athena if complex
        4. Merge if persona needed
        5. Quality gate through Ring
        6. Return structured result
        """
        result = {
            "cycle_start": datetime.now(timezone.utc).isoformat(),
            "task": task[:200],
            "session_id": self.session_id,
            "steps": [],
        }

        # Step 1: Classify
        category = self.classify(task)
        result["steps"].append({
            "step": "classify",
            "category": category,
        })

        # Step 2: Route
        model_choice = self.route(category, task, budget=budget)
        result["steps"].append({
            "step": "route",
            "selected_model": model_choice,
        })

        # Step 3: Consult if complex
        task_size = len(task.split())
        if task_size > 50:  # Complex task threshold
            consult = self.consult(task, "athena", context, budget)
            result["steps"].append({
                "step": "consult",
                "detail": consult,
            })

        # Step 4: Quality gate
        qg = self.quality_gate(task, category)
        result["steps"].append({
            "step": "quality_gate",
            "detail": qg,
        })

        result["cycle_end"] = datetime.now(timezone.utc).isoformat()
        result["total_steps"] = len(result["steps"])

        return result


# ─── PROTOCOL STATE MACHINE ────────────────────────────────────

PROTOCOL_STATES = {
    "IDLE": "Waiting for task",
    "CLASSIFIED": "Task type identified, selecting model",
    "ROUTED": "Model selected, preparing execution",
    "CONSULTING": "Sub-agent engaged for complex analysis",
    "MERGING": "Persona adopted for task execution",
    "QUALITY_GATE": "Ring verification in progress",
    "DELIVERING": "Final output prepared",
    "ERROR": "Protocol failed at some stage",
}


def get_protocol_state_description(state: str) -> str:
    return PROTOCOL_STATES.get(state, "Unknown state")


if __name__ == "__main__":
    print("Consult/Merge Orchestrator self-test...")
    print("=" * 50)

    orch = ConsultMergeOrchestrator()

    # Test classification
    test_tasks = [
        "Write a Python function to sort a list",
        "Analyze the tradeoffs between microservices and monoliths",
        "Research the latest developments in quantum computing",
        "Write a poem about artificial intelligence",
        "Review this code for security vulnerabilities",
    ]

    for task in test_tasks:
        cat = orch.classify(task)
        route = orch.route(cat, task)
        print(f"  '{task[:45]}...' → {cat} → {route['provider']}/{route['model']}")

    print()

    # Test full cycle
    complex_task = """Design a distributed caching layer for our microservices
    architecture that handles 100K requests/second with sub-millisecond
    latency. The system needs to be fault-tolerant, support cache invalidation,
    and work across multiple data centers."""

    cycle = orch.full_cycle(complex_task, budget=5.0)
    print(f"Full cycle for complex task:")
    for step in cycle["steps"]:
        print(f"  Step {step['step']}: ", end="")
        if "detail" in step:
            detail = step["detail"]
            if isinstance(detail, dict) and "selected_model" in detail:
                model = detail.get("model", {})
                print(f"{detail.get('action', '?')} → {model}")
            else:
                print(f"{step.get('category', '?')}")
        else:
            print(f"{step.get('category', step.get('selected_model', '?'))}")

    print(f"\n  Total steps: {cycle['total_steps']}")
    print("Consult/Merge Orchestrator ready. ✅")