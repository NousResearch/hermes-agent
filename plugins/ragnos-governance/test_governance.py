"""Hermetic tests for the RAGnos governance policy (Sprint 2).

Imports governance.py by path so the hyphenated plugin dir name does not block
a normal ``import``.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "ragnos_governance_governance", Path(__file__).parent / "governance.py"
)
governance = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(governance)


def test_observe_mode_never_blocks() -> None:
    # Default (no enforcement): no tool is ever blocked.
    assert governance.evaluate_tool("delete_everything", {}, enforce=False, forbidden=frozenset({"delete_everything"})) is None


def test_enforced_forbidden_tool_is_blocked() -> None:
    verdict = governance.evaluate_tool(
        "send_money", {}, enforce=True, forbidden=frozenset({"send_money"})
    )
    assert verdict is not None
    assert verdict["action"] == "block"
    assert "send_money" in verdict["message"]
    assert "Hermes Hub" in verdict["message"]


def test_enforced_but_unlisted_tool_is_allowed() -> None:
    assert governance.evaluate_tool("read_file", {}, enforce=True, forbidden=frozenset({"send_money"})) is None


def test_is_enforcing_reads_env() -> None:
    assert governance.is_enforcing({"RAGNOS_GOVERNANCE_ENFORCE": "1"}) is True
    assert governance.is_enforcing({"RAGNOS_GOVERNANCE_ENFORCE": "true"}) is True
    assert governance.is_enforcing({}) is False
    assert governance.is_enforcing({"RAGNOS_GOVERNANCE_ENFORCE": "0"}) is False


def test_forbidden_tools_parses_env_list() -> None:
    got = governance.forbidden_tools({"RAGNOS_GOVERNANCE_FORBIDDEN_TOOLS": "a, b ,c"})
    assert got == frozenset({"a", "b", "c"})
    assert governance.forbidden_tools({}) == frozenset()


def test_record_event_appends_jsonl(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    env = {"RAGNOS_GOVERNANCE_LEDGER": str(ledger)}
    governance.record_event("pre_tool_call", {"tool": "x", "blocked": False}, env=env)
    governance.record_event("post_tool_call", {"tool": "x"}, env=env)
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["event_type"] == "pre_tool_call"
    assert rows[0]["tool"] == "x"
    assert rows[1]["event_type"] == "post_tool_call"
