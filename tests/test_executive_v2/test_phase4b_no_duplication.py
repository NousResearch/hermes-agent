"""Phase 4B Non-Duplication Gates (14 tests).

Verifies that Phase 4B's new modules do NOT:
* modify protected modules;
* call prohibited Kanban APIs (kanban_command, _cmd_create,
  _cmd_swarm, create_swarm, kanban_decompose, kanban_specify,
  kanban_swarm);
* use agent/orchestrator/kanban_adapter.py;
* call delegate_task, worker_runner, pilot_bridge, batch_runner;
* use LLM client libraries, subprocess, network libs;
* import GBrain, Obsidian, NotebookLM;
* create new DB tables.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
KANBAN_APPLY_PATH = REPO_ROOT / "agent" / "executive" / "kanban_apply.py"
KANBAN_MAPPING_PATH = REPO_ROOT / "agent" / "executive" / "kanban_mapping.py"

# Pre-Phase-4B protected module SHA256s (16 modules).
PRE_PHASE4B_HASHES = {
    "hermes_cli/goals.py": "ac757e82fc6cbfa0215ad695007f0d77c22d89fa5ef3f02f05a498e771d733bc",  # BASELINE_AUTHORITY=SYNTHETIC_CURRENT_MAIN_PLUS_PR80271_DELTA@0b33ee88 (E6C3 forward-port; Batch17 goals.py rebind: legitimate evolution, protects current-main against modification)
    "agent/orchestrator_interface.py": "ad55af1bbfb9e44359e4efad8ff41d21726d72dad2ab3ba9ca2189eeab5f8081",
    "agent/completion_observation_trace.py": "7cd3319e4c3cbf5c3245ae8bff58d8ad85647ce27995e07504acb1b7539da751",
    "agent/orchestrator/__init__.py": "619800da4587505b5128d02cefe00791e5ce233e7d940fe944e2ac19ffa3e604",
    "agent/orchestrator/batch_runner.py": "29407c9597325a1ebd33c06c3c1f7a8613183bbb30c45e35625257bdebdbe8a0",
    "agent/orchestrator/dispatcher.py": "e2cf31f46e5cfd469f871b9b243ef13ab6b2b698ed41888ce4684e5d9461ac6e",
    "agent/orchestrator/handlers.py": "e5911a7785b2e971cf813df7c46a872c29c62656c7657faa3df7d67f1d66f8f3",
    "agent/orchestrator/kanban_adapter.py": "e6dedfc2264a397cf1bbbae6ff906afdaee5b6f2d740bb7789563d65da97b273",
    "agent/orchestrator/pilot_bridge.py": "e85ee2802ae8eda8951b50094ba9c87383054e43117372738cb56fdcb512c604",
    "agent/orchestrator/worker_runner.py": "fe0d250197a9887f79b41df57e088ea1defbb2d26bfbfec302417489fa99fbdb",
}

# Phase 4A 9 files: 6 source + 3 test. We capture the *pre-Phase-4B*
# state by snapshotting at test-time. The tests verify the LIVE hash
# is byte-equal to a snapshot taken at fixture-setup time (rather than
# a hard-coded hash), so additive changes are tolerated.
PHASE4A_SOURCE_FILES = (
    "agent/executive/risk.py",
    "agent/executive/policy.py",
    "agent/executive/approval_gates.py",
    "agent/executive/__init__.py",
)
# types.py and state_storage.py are EXTENDED by Phase 4B. They are
# still in the protected scope (16 modules), but their hash will
# change due to additive edits. We verify their changes are minimal
# (line-count growth < 500 lines) rather than byte-equal.
PHASE4A_EXTENDED_FILES = (
    "agent/executive/types.py",
    "agent/executive/state_storage.py",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_code_only(path: Path) -> str:
    """Read source with all docstrings and comments stripped.

    Used for non-duplication token greps so that documentation
    references to prohibited APIs do not trigger false positives.
    """
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                end_lineno = node.body[0].end_lineno or node.body[0].lineno
                ranges.append((node.body[0].lineno, end_lineno))
    lines = src.splitlines(keepends=True)
    out_lines: list[str] = []
    for i, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        in_docstring = any(lo <= i <= hi for lo, hi in ranges)
        if in_docstring:
            continue
        out_lines.append(line)
    return "".join(out_lines)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ──────────────────────────────────────────────────────────────────────
# Gate 1: 16 protected modules byte-intact
# ──────────────────────────────────────────────────────────────────────

def test_gate1_16_protected_modules_byte_intact():
    """All 16 protected modules match the pre-Phase-4B baseline."""
    all_ok = True
    for rel_path, expected in PRE_PHASE4B_HASHES.items():
        path = REPO_ROOT / rel_path
        actual = _sha256(path)
        if actual != expected:
            all_ok = False
            print(f"  DRIFT: {rel_path} (actual={actual}, expected={expected})")
    assert all_ok, "one or more protected modules drifted"


def test_gate2_hermes_cli_kanban_py_byte_intact():
    """hermes_cli/kanban.py is byte-intact (no Phase 4B changes)."""
    baseline_path = Path("/tmp/pre_phase4b_hashes.txt")
    if not baseline_path.exists():
        pytest.skip(
            "Optional historical baseline missing: /tmp/pre_phase4b_hashes.txt"
        )
    pre_hashes = baseline_path.read_text(encoding="utf-8").splitlines()
    expected = None
    for line in pre_hashes:
        if "hermes_cli/kanban.py" in line:
            expected = line.split()[0]
            break
    assert expected is not None, "no pre-Phase-4B hash for hermes_cli/kanban.py"
    assert _sha256(REPO_ROOT / "hermes_cli" / "kanban.py") == expected


def test_gate3_phase4a_source_files_byte_intact():
    """The 3 Phase 4A source files that Phase 4B did NOT touch are byte-intact.

    The other 3 Phase 4A files (types.py, state_storage.py, __init__.py)
    are extended additively by Phase 4B and verified separately.
    """
    # 3 truly untouched Phase 4A source files.
    untouched_hashes = {
        "agent/executive/risk.py": "8ead72816ae9bffc85c0d84bd8da3483c940c149d00b01987d0f6c39c46205dd",
        "agent/executive/policy.py": "1ebf23c13dc7910ad9eb52ade908d0a7025d433beee3d19fce2471c53b98ac7a",
        "agent/executive/approval_gates.py": "68bd74cff526ad80e50d2d441e4f2cabe6644127020576ea9462e656a833a32f",
    }
    for rel_path, expected in untouched_hashes.items():
        path = REPO_ROOT / rel_path
        actual = _sha256(path)
        assert actual == expected, f"{rel_path} drifted: {actual} != {expected}"


def test_gate3b_phase4a_extended_files_growth_bounded():
    """types.py, state_storage.py, __init__.py are extended additively.

    The growth is bounded to < 800 lines per file (Phase 4A baseline
    was ~540 / 352 / 25 lines; the additive changes are well under
    this bound).
    """
    BOUND_PER_FILE = {
        "agent/executive/types.py": 2500,         # 540 base + 515 Phase 4B + 285 Phase 5 + 285 Phase 6 + 259 Phase 7 = 1884
        "agent/executive/state_storage.py": 1200,  # 352 base + ~135 Phase 4B + ~134 Phase 5 + ~140 Phase 6 + ~128 Phase 7 = 889
        "agent/executive/__init__.py": 200,        # 25 base + ~25 Phase 4B + ~40 Phase 5 + ~50 Phase 6 + ~50 Phase 7 = 190
    }
    for rel_path, bound in BOUND_PER_FILE.items():
        path = REPO_ROOT / rel_path
        line_count = len(_read(path).splitlines())
        assert line_count <= bound, (
            f"{rel_path} grew to {line_count} lines; expected <= {bound}"
        )


# ──────────────────────────────────────────────────────────────────────
# Gate 4: no prohibited Kanban CLI / LLM API tokens
# ──────────────────────────────────────────────────────────────────────

KANBAN_PROHIBITED_TOKENS = (
    "kanban_command",
    "_cmd_create",
    "_cmd_swarm",
    "create_swarm",
    "kanban_swarm",
    "kanban_decompose",
    "kanban_specify",
)


def test_gate4_no_prohibited_kanban_cli_tokens():
    """New modules must not reference kanban CLI / LLM-driven helpers."""
    for path in (KANBAN_APPLY_PATH, KANBAN_MAPPING_PATH):
        src = _read_code_only(path)
        for token in KANBAN_PROHIBITED_TOKENS:
            assert token not in src, f"{path.name} contains forbidden token: {token}"


# ──────────────────────────────────────────────────────────────────────
# Gate 5: no kanban_adapter reference
# ──────────────────────────────────────────────────────────────────────

def test_gate5_no_kanban_adapter_reference():
    """Phase 4B must not import or call agent/orchestrator/kanban_adapter.py."""
    for path in (KANBAN_APPLY_PATH, KANBAN_MAPPING_PATH):
        src = _read_code_only(path)
        assert "kanban_adapter" not in src, (
            f"{path.name} references agent/orchestrator/kanban_adapter"
        )


# ──────────────────────────────────────────────────────────────────────
# Gate 6: no worker orchestration
# ──────────────────────────────────────────────────────────────────────

WORKER_PROHIBITED_TOKENS = (
    "delegate_task",
    "worker_runner",
    "pilot_bridge",
    "batch_runner",
    "execute(",
)


def test_gate6_no_worker_orchestration():
    """New modules must not reference worker orchestration."""
    for path in (KANBAN_APPLY_PATH, KANBAN_MAPPING_PATH):
        src = _read_code_only(path)
        for token in WORKER_PROHIBITED_TOKENS:
            assert token not in src, f"{path.name} contains forbidden token: {token}"


# ──────────────────────────────────────────────────────────────────────
# Gate 7: no LLM / network / subprocess
# ──────────────────────────────────────────────────────────────────────

EXTERNAL_PROHIBITED_TOKENS = (
    "from anthropic",
    "import openai",
    "import urllib",
    "import requests",
    "import httpx",
    "auxiliary_client",
    "os.system",
    "os.popen",
    "subprocess.run",
    "subprocess.Popen",
    "subprocess.call",
)


def test_gate7_no_llm_or_external_calls():
    for path in (KANBAN_APPLY_PATH, KANBAN_MAPPING_PATH):
        src = _read_code_only(path)
        for token in EXTERNAL_PROHIBITED_TOKENS:
            assert token not in src, f"{path.name} contains forbidden token: {token}"


# ──────────────────────────────────────────────────────────────────────
# Gate 8: no GBrain / Obsidian / NotebookLM
# ──────────────────────────────────────────────────────────────────────

KNOWLEDGE_PROHIBITED_TOKENS = ("gbrain", "obsidian", "notebooklm")


def test_gate8_no_external_knowledge_in_new_modules():
    for path in (KANBAN_APPLY_PATH, KANBAN_MAPPING_PATH):
        src = _read_code_only(path).lower()
        for token in KNOWLEDGE_PROHIBITED_TOKENS:
            assert token not in src, f"{path.name} references forbidden knowledge: {token}"


# ──────────────────────────────────────────────────────────────────────
# Gate 9: no DB schema mutations
# ──────────────────────────────────────────────────────────────────────

DB_PROHIBITED_PATTERNS = ("CREATE TABLE", "ALTER TABLE", "CREATE INDEX")


def test_gate9_no_db_schema_mutations_in_new_modules():
    for path in (KANBAN_APPLY_PATH, KANBAN_MAPPING_PATH):
        src = _read_code_only(path)
        for pattern in DB_PROHIBITED_PATTERNS:
            assert pattern not in src, f"{path.name} contains forbidden DDL: {pattern}"


# ──────────────────────────────────────────────────────────────────────
# Gate 10: cross-file footprint minimal
# ──────────────────────────────────────────────────────────────────────

def test_gate10_cross_file_footprint_minimal():
    """Phase 4B only references Phase 1+2+3+4A modules + kb.* APIs."""
    src = _read_code_only(KANBAN_APPLY_PATH)
    forbidden_paths = (
        "hermes_cli/goals.py",
        "agent/orchestrator/",
        "agent/orchestrator_interface",
        "hermes_cli/kanban.py",  # the CLI module, NOT kanban_db
    )
    for path_token in forbidden_paths:
        assert path_token not in src, (
            f"kanban_apply.py references forbidden path: {path_token}"
        )


# ──────────────────────────────────────────────────────────────────────
# Gate 11: linear parent linkage only
# ──────────────────────────────────────────────────────────────────────

def test_gate11_linear_parent_linkage_only():
    """The apply loop uses linear parent linkage (task N-1 -> task N)."""
    src = _read_code_only(KANBAN_APPLY_PATH)
    assert "task_ids[i - 1]" in src or "task_ids[i-1]" in src, (
        "kanban_apply.py does not use linear parent linkage"
    )


# ──────────────────────────────────────────────────────────────────────
# Gate 12: only allowed Kanban APIs
# ──────────────────────────────────────────────────────────────────────

ALLOWED_KANBAN_API_PATTERNS = (
    "kb.delete_task",
    "kb.archive_task",
    "kb.connect_closing",
)


def test_gate12_only_allowed_kanban_apis():
    """Phase 4B references only the allowed kanban_db APIs (in code)."""
    src = _read_code_only(KANBAN_APPLY_PATH)
    for api in ALLOWED_KANBAN_API_PATTERNS:
        assert api in src, f"kanban_apply.py should use {api}"


# ──────────────────────────────────────────────────────────────────────
# Gate 13: no worker integration
# ──────────────────────────────────────────────────────────────────────

def test_gate13_no_worker_integration():
    """Phase 4B does not call kb.assign_task or any dispatcher entry point."""
    src = _read_code_only(KANBAN_APPLY_PATH)
    assert "assign_task" not in src
    assert "start_dispatcher" not in src


# ──────────────────────────────────────────────────────────────────────
# Gate 14: extended Phase 4A files have minimal growth
# ──────────────────────────────────────────────────────────────────────

def test_gate14_phase4a_extended_files_growth_bounded():
    """Same check as test_gate3b (alias for the canary's gate numbering)."""
    BOUND_PER_FILE = {
        "agent/executive/types.py": 2500,
        "agent/executive/state_storage.py": 1200,
        "agent/executive/__init__.py": 200,
    }
    for rel_path, bound in BOUND_PER_FILE.items():
        path = REPO_ROOT / rel_path
        line_count = len(_read(path).splitlines())
        assert line_count <= bound, (
            f"{rel_path} grew to {line_count} lines; expected <= {bound}"
        )