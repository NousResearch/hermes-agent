from __future__ import annotations

import os
from pathlib import Path

import pytest


def _local_skills_root() -> Path:
    explicit = os.environ.get("HERMES_LOCAL_SKILLS_DIR")
    if explicit:
        return Path(explicit)
    local_appdata = os.environ.get("LOCALAPPDATA")
    if not local_appdata:
        pytest.skip("LOCALAPPDATA is not available for local Hermes skill checks")
    return Path(local_appdata) / "hermes" / "skills"


def _read_local_skill(relative: str) -> str:
    root = _local_skills_root()
    if not root.exists():
        pytest.skip(f"local Hermes skills root is not present: {root}")
    path = root / relative / "SKILL.md"
    assert path.exists(), f"missing local Hermes skill: {path}"
    return path.read_text(encoding="utf-8")


def _read_local_skill_file(relative: str) -> str:
    root = _local_skills_root()
    if not root.exists():
        pytest.skip(f"local Hermes skills root is not present: {root}")
    path = root / relative
    assert path.exists(), f"missing local Hermes skill file: {path}"
    return path.read_text(encoding="utf-8")


def test_site_factory_operational_safety_skill_contains_all_hard_gates():
    text = _read_local_skill(
        "software-development/site-factory-operational-safety"
    )

    required_phrases = [
        "adjacent idempotency tests",
        "hashes from the original canonical payload",
        "value-free evidence only",
        "checkout draft metadata",
        "runtime request metadata",
        "endpoint parser",
        "bridge parser",
        "OVH live gate",
        "Cloudflare DNS boundary",
        "Publisher readiness/upload CLI is authoritative",
        "do not manually FTPS-upload around the gate",
        "preview token generation must satisfy current Publisher readiness token policy",
        "SiteFactoryAgentRun",
        "SiteFactoryAgentStep",
        "queued_for_factory_runner",
        "latest created artifacts",
        "site_content",
        "brand_kit",
        "site_plan",
        "template_build_report",
        "Vite/SPA token preview",
        "rendered browser/public asset verification",
        "newly-created webhook signing secret is available only at creation time",
        "stored locally and in Vercel Production",
        "redeploy production before claiming runtime updated",
        "migrations actually applied/up to date",
        "normalize UTF-16/NUL output",
        "buildFinalHandoffActivationReadinessReport",
        "reject raw destination/provider fields",
        ".invalid",
        ".test",
        ".example",
    ]

    missing = [phrase for phrase in required_phrases if phrase not in text]
    assert not missing, "missing Site Factory safety checklist phrases: " + ", ".join(missing)


def test_kanban_worker_and_autopilot_reference_site_factory_safety_skill():
    worker = _read_local_skill("kanban-worker")
    autopilot = _read_local_skill("devops/hermes-kanban-autopilot")

    assert "site-factory-operational-safety" in worker
    assert "site-factory-operational-safety" in autopilot


def test_site_factory_dry_run_kanban_card_includes_safety_prompt():
    card = _read_local_skill_file(
        "software-development/site-factory-operational-safety/"
        "references/dry-run-kanban-card.md"
    )

    assert "Load/use site-factory-operational-safety" in card
    assert "Publisher readiness/upload CLI is authoritative" in card
    assert "No live provider/deploy/payment/DNS/email/customer-data action" in card
    assert "reject raw destination/provider fields" in card
    assert "remaining-work packet" in card
