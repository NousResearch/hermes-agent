"""Temporary strangler seam for CLI commands that have not migrated yet."""

from __future__ import annotations


def dispatch_legacy():
    """Run the existing CLI implementation until command ownership migrates."""
    from hermes_cli.main import main as legacy_main

    return legacy_main()


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Temporary presentation shim until setup prompts migrate into nous_cli."""
    from hermes_cli.setup import prompt_yes_no as legacy_prompt_yes_no

    return legacy_prompt_yes_no(question, default)
