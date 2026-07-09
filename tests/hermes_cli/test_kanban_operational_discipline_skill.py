from __future__ import annotations

import os
from pathlib import Path

import pytest


def _local_hermes_root() -> Path:
    explicit = os.environ.get("HERMES_LOCAL_ROOT")
    if explicit:
        return Path(explicit)
    local_appdata = os.environ.get("LOCALAPPDATA")
    if not local_appdata:
        pytest.skip("LOCALAPPDATA is not available for local Hermes checks")
    return Path(local_appdata) / "hermes"


def _read_local_skill(relative: str) -> str:
    root = _local_hermes_root() / "skills"
    if not root.exists():
        pytest.skip(f"local Hermes skills root is not present: {root}")
    path = _local_hermes_root() / "skills" / relative / "SKILL.md"
    assert path.exists(), f"missing local Hermes skill: {path}"
    return path.read_text(encoding="utf-8")


def _read_profile_config(profile: str) -> str:
    root = _local_hermes_root() / "profiles"
    if not root.exists():
        pytest.skip(f"local Hermes profiles root is not present: {root}")
    path = _local_hermes_root() / "profiles" / profile / "config.yaml"
    assert path.exists(), f"missing local Hermes profile: {path}"
    return path.read_text(encoding="utf-8")


def test_kanban_worker_operational_discipline_skill_contains_all_lessons():
    text = _read_local_skill("kanban-worker-operational-discipline")

    required_phrases = [
        "after complete: final response and exit",
        "after block: final response and exit",
        "no further tools after a terminal Kanban action",
        "host-visible artifact verification",
        "compact handoff packet",
        "usage-guard compact packet",
        "reviewer bounded verdict",
        "actionable",
        "trade-off",
        "contract-misread",
        "noise",
        "synthesizer-only mode",
        "missing-evidence request",
        "scoped acceptance",
        "unrelated full-suite failures",
        "site-factory-operational-safety",
        "Publisher readiness/upload CLI is authoritative",
        "closure artifact",
        "closure resume",
    ]

    missing = [phrase for phrase in required_phrases if phrase not in text]
    assert not missing, "missing operational discipline phrases: " + ", ".join(missing)


def test_core_kanban_skills_reference_operational_discipline():
    worker = _read_local_skill("kanban-worker")
    synthesizer = _read_local_skill("kanban-synthesizer")
    autopilot = _read_local_skill("devops/hermes-kanban-autopilot")

    assert "kanban-worker-operational-discipline" in worker
    assert "kanban-worker-operational-discipline" in synthesizer
    assert "kanban-worker-operational-discipline" in autopilot


def test_builder_reviewer_synthesizer_profiles_hint_operational_skills():
    expected = {
        "builder": [
            "kanban-worker-operational-discipline",
            "hermes-kanban-autopilot",
            "site-factory-operational-safety",
        ],
        "reviewer": [
            "kanban-worker-operational-discipline",
            "reviewer bounded verdict",
            "requesting-code-review",
        ],
        "synthesizer": [
            "kanban-worker-operational-discipline",
            "kanban-synthesizer",
            "synthesizer-only mode",
        ],
    }

    for profile, phrases in expected.items():
        text = _read_profile_config(profile)
        assert "environment_hint:" in text
        assert "max_concurrent_children: 11" in text
        for phrase in phrases:
            assert phrase in text
