"""Docs invariants for AgentCyber Live USB safety gates."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"


def _live_usb_section() -> str:
    text = README.read_text(encoding="utf-8")
    start = text.index("## Live USB")
    end = text.index("## Quick Install", start)
    return text[start:end]


def test_readme_live_usb_section_documents_agent_tool_gates() -> None:
    section = _live_usb_section()
    lowered = section.lower()

    for phrase in (
        "disabled by default",
        "status",
        "list_usb",
        "read-only",
        "build",
        "write",
        "provision",
        "root",
        "exact operator approval",
        "hermes_agentcyber_live_usb_approval",
        "operator_approval",
        "removable linux block-device metadata",
        "canonical",
        "/dev/",
    ):
        assert phrase in lowered


def test_readme_live_usb_examples_do_not_imply_root_alone_is_enough() -> None:
    section = _live_usb_section()
    lowered = section.lower()

    assert "root/sudo alone is not sufficient" in lowered
    assert "unattended cron lanes must not" in lowered
    assert "build`, `write`, and `provision` require" in lowered
    assert "operator_approval=\"<matching one-time token>\"" in section

    obsolete_root_only_claims = (
        "build` and `write` require the agent session to run as root",
        "need no root. `build` and `write` require",
    )
    for obsolete_claim in obsolete_root_only_claims:
        assert obsolete_claim not in lowered
