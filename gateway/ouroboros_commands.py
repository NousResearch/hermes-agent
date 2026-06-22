"""Shared command metadata for the gateway /ooo Ouroboros namespace."""

from __future__ import annotations

OOO_SUBCOMMAND_DEFINITIONS: tuple[tuple[str, str], ...] = (
    ("help", "Show Ouroboros command reference"),
    ("interview", "Clarify requirements"),
    ("seed", "Generate a Seed spec"),
    ("run", "Execute a Seed/workflow"),
    ("evaluate", "Run three-stage verification"),
    ("status", "Show session/job status"),
    ("pm", "Run a PM/PRD interview"),
    ("qa", "Run a quality verdict"),
    ("unstuck", "Use lateral thinking personas"),
    ("evolve", "Run one evolutionary loop step"),
    ("ralph", "Iterate until verified or bounded"),
    ("auto", "Run the full auto pipeline"),
    ("cancel", "Cancel an explicit execution/job"),
    ("resume-session", "Show resume/session guidance"),
    ("setup", "Blocked in Discord: setup changes local config"),
    ("config", "Blocked in Discord: settings changes need approval"),
    ("brownfield", "Query brownfield repo defaults read-only"),
    ("publish", "Blocked in Discord: would publish externally"),
    ("welcome", "Show first-touch guide"),
    ("tutorial", "Show hands-on tutorial steps"),
    ("update", "Blocked in Discord: upgrades need operator control"),
)

OOO_SUBCOMMANDS: tuple[str, ...] = tuple(name for name, _ in OOO_SUBCOMMAND_DEFINITIONS)
OOO_SUBCOMMAND_DESCRIPTIONS: dict[str, str] = dict(OOO_SUBCOMMAND_DEFINITIONS)

# Accepted by the native router for typed text, but intentionally not exposed as
# a Discord slash-picker child. `/ooo status --job <id>` is the public picker path.
OOO_ROUTER_ONLY_SUBCOMMANDS: tuple[str, ...] = ("job",)
OOO_NATIVE_COMMANDS: tuple[str, ...] = OOO_SUBCOMMANDS + OOO_ROUTER_ONLY_SUBCOMMANDS
