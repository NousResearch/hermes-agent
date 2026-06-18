"""Regression coverage for Codex gpt-5.5 autoraise notice routing.

The autoraise is still useful, but its explanatory notice must not be replayed
through gateway ``status_callback``. Gateway agent objects are rebuilt often, so
"one-time per agent" becomes repeated Discord/Telegram noise.
"""
from __future__ import annotations

import ast
from pathlib import Path


AGENT_INIT = Path(__file__).resolve().parents[2] / "agent" / "agent_init.py"


def _agent_init_tree() -> ast.Module:
    return ast.parse(AGENT_INIT.read_text(encoding="utf-8"))


def test_codex_gpt55_autoraise_notice_is_not_stashed_for_gateway_replay() -> None:
    """Do not assign the informational autoraise notice to _compression_warning.

    ``_compression_warning`` is replayed to messaging platforms on the first
    turn. The Codex gpt-5.5 autoraise message is informational only and repeats
    whenever the gateway rebuilds an agent, so it should stay CLI-only.
    """
    tree = _agent_init_tree()
    offenders: list[int] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        assigns_compression_warning = any(
            isinstance(target, ast.Attribute)
            and target.attr == "_compression_warning"
            for target in node.targets
        )
        if not assigns_compression_warning:
            continue
        calls_notice_builder = any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "_build_codex_gpt55_autoraise_notice"
            for child in ast.walk(node.value)
        )
        if calls_notice_builder:
            offenders.append(node.lineno)

    assert offenders == [], (
        "Codex gpt-5.5 autoraise notice must not be stored in "
        f"agent._compression_warning for gateway replay; offending lines: {offenders}"
    )


def test_codex_gpt55_autoraise_notice_remains_available_for_cli_startup() -> None:
    source = AGENT_INIT.read_text(encoding="utf-8")
    assert "_build_codex_gpt55_autoraise_notice(_autoraise)" in source
    assert "Opt back out: hermes config set compression.codex_gpt55_autoraise false" in source
