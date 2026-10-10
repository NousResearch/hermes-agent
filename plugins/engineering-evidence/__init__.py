"""Hermes engineering-evidence plugin entry point."""

from __future__ import annotations

from .engineering_evidence.cli import engineering_evidence_command, register_cli


_SYSTEM_PROMPT = (
    "Engineering evidence is available through `hermes engineering-evidence`. "
    "For repository tasks, prefer an exact-HEAD read-only snapshot before editing. "
    "Use `experience-search` only for exact-repository/exact-HEAD validated candidates; "
    "proposed candidates require the memory-validator role before retrieval. Record test "
    "discovery and validated fixes as diagnostic-only receipts with source references. "
    "Never treat a receipt, model summary, or memory candidate as merge, release, trading, "
    "credential, or deployment authority."
)


def register(ctx) -> None:
    """Expose the bounded CLI and the agent-facing evidence contract."""

    ctx.register_cli_command(
        name="engineering-evidence",
        help="Create and validate exact-head engineering evidence",
        setup_fn=register_cli,
        handler_fn=engineering_evidence_command,
        description=(
            "Read-only codebase snapshots plus diagnostic test and experience "
            "receipts for governed Hermes handoffs."
        ),
    )
    ctx.register_system_prompt_section(
        "engineering-evidence",
        _SYSTEM_PROMPT,
        position="after_memory",
        max_chars=1200,
    )
