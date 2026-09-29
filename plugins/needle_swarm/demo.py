"""demo.py - Runnable end-to-end example demonstration for Needle Swarm.

Registers coding and security clusters, runs single-dispatch and swarm fan-out,
then triggers a growth cycle that creates/retrains skills from escalations.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from plugins.needle_swarm.orchestrator import Orchestrator
from plugins.needle_swarm.registry import SkillMeta


def run_demo():
    print("=== Needle Swarm End-to-End Demo ===")

    temp_dir = Path(tempfile.mkdtemp(prefix="needle_swarm_demo_"))
    try:
        # Mock LLM Callback simulating cloud / Bonsai LLM
        def mock_llm_brain(prompt: str) -> str:
            return f"[Cloud Brain LLM Resolution] Synthesized response for prompt: '{prompt[:40]}...'"

        orchestrator = Orchestrator(
            data_dir=temp_dir,
            brain_llm_callback=mock_llm_brain,
            confidence_threshold=0.6,
        )

        # 1. Register Coding Skills
        orchestrator.registry.register(SkillMeta(
            skill_id="code_formatter",
            name="Code Formatter",
            description="Formats Python and JavaScript code blocks according to PEP8 / Prettier.",
            kind="ACTION",
            tools=["format_code"],
            clusters=["coding"],
        ))

        orchestrator.registry.register(SkillMeta(
            skill_id="linter_fixer",
            name="Linter Fixer",
            description="Fixes common flake8 and eslint syntax errors and unused imports.",
            kind="ACTION",
            tools=["fix_lint_errors"],
            clusters=["coding"],
        ))

        # 2. Register Security Swarm Skills
        orchestrator.registry.register(SkillMeta(
            skill_id="sec_ip_reputation",
            name="IP Reputation Checker",
            description="Checks incoming IP address against threat intelligence lists.",
            kind="ACTION",
            tools=["check_ip"],
            clusters=["security"],
        ))

        orchestrator.registry.register(SkillMeta(
            skill_id="sec_playbook_ddos",
            name="DDoS Mitigation Playbook",
            description="Matching situation to DDoS incident response checklist and mitigation procedures.",
            kind="PLAYBOOK",
            tools=["execute_ddos_mitigation"],
            clusters=["security"],
        ))

        orchestrator.registry.register(SkillMeta(
            skill_id="sec_memory_logs",
            name="Security Log Corpus RAG",
            description="Retrieves relevant security log entries and threat reports.",
            kind="MEMORY",
            clusters=["security"],
            corpus_path=str(temp_dir / "sec_logs.txt"),
        ))

        # 3. Test Single Dispatch in Coding Mode
        print("\n--- 1. Single Dispatch (Coding Mode) ---")
        loaded_coding = orchestrator.enter_mode("coding", exclusive=True)
        print(f"Loaded Coding Cluster Skills: {loaded_coding}")

        res1 = orchestrator.handle_single("Please format this Python script according to PEP8")
        print(f"Result 1 (Single Dispatch): {res1}")

        # Unmatched utterance -> Escalation
        print("\n--- 2. Unmatched Request -> Cloud/Bonsai Escalation ---")
        for i in range(5):
            orchestrator.handle_single(f"Deploy microservice container to Kubernetes cluster attempt {i+1}")
        print("Logged 5 unhandled escalation events.")

        # 4. Test Swarm Fan-Out in Security Mode
        print("\n--- 3. Swarm Fan-Out (Security Cluster) ---")
        loaded_sec = orchestrator.enter_mode("security", exclusive=True)
        print(f"Loaded Security Cluster Skills: {loaded_sec}")

        swarm_res = orchestrator.handle_swarm("Surge in traffic detected from IP 192.168.1.100 targeting port 443")
        print("\n[Swarm Merged Context for Brain LLM]:")
        print(swarm_res.as_bonsai_context())

        # 5. Hindsight Recall
        print("\n--- 4. Hindsight Memory Recall ---")
        hindsight_res = orchestrator.hindsight_store.recall("DDoS traffic surge")
        print(f"Hindsight Recall Results: {hindsight_res}")

        # 6. Growth Cycle with Dreaming
        print("\n--- 5. Triggering Growth Cycle (Dreaming & Fine-Tuning) ---")
        growth_summary = orchestrator.run_growth_cycle()
        print(f"Growth Cycle Summary: {growth_summary}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("\n=== Demo Complete ===")


if __name__ == "__main__":
    run_demo()
