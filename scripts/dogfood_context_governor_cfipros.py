#!/usr/bin/env python3
"""Dogfood A/B: run the same real CFIPros AKTR task with/without Context Governor."""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import importlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Make the Hermes agent importable
sys.path.insert(0, "/home/orchestrator/.hermes/hermes-agent")

from agent.context_governor import ContextGovernor

REPO_ROOT = Path("/home/orchestrator/repos/github/marcusgoll/cfipros")
TEST_FILE_UNIT = "api/tests/unit/services/test_aktr_ocr.py"
TEST_FILE_SERVICE = "api/tests/services/test_aktr_ocr.py"
SERVICE_FILE = "api/app/services/aktr/ocr.py"
EXCEPTIONS_FILE = "api/app/services/aktr/exceptions.py"
VISION_FILE = "api/app/services/aktr/google_vision_client.py"


@dataclass
class ToolCall:
    tool: str
    args: Dict[str, Any]
    response: str


@dataclass
class RunResult:
    mode: str
    success: bool
    tool_calls: int
    context_chars: int
    elapsed_seconds: float
    final_test_output: str = ""


def make_temp_repo(label: str) -> Path:
    """Copy CFIPros repo to a temp directory for a clean run."""
    tmp = Path(tempfile.mkdtemp(prefix=f"cfipros-dogfood-{label}-"))
    shutil.copytree(
        REPO_ROOT,
        tmp,
        ignore=shutil.ignore_patterns(
            ".git", "node_modules", "__pycache__", ".venv", "venv", ".pytest_cache", "cfipros_dev.db"
        ),
        dirs_exist_ok=True,
    )
    return tmp


def read_file(repo: Path, rel: str) -> str:
    return (repo / rel).read_text()


def patch_test_file(repo: Path, rel: str) -> str:
    path = repo / rel
    content = path.read_text()

    # Update the explicit regex test to reflect that lowercase is now valid
    old_invalid = '''        invalid_codes = [
            "PA.1.A.K1",  # Not Roman numeral
            "INVALID",  # Wrong format
            "PA.I.K1",  # Missing section
            "PA.I.A.K",  # Missing number
            "pa.i.a.k1",  # Lowercase (pattern expects uppercase)
        ]'''
    new_invalid = '''        invalid_codes = [
            "PA.1.A.K1",  # Not Roman numeral
            "INVALID",  # Wrong format
            "PA.I.K1",  # Missing section
            "PA.I.A.K",  # Missing number
        ]'''
    if old_invalid in content:
        content = content.replace(old_invalid, new_invalid)

    if 'def test_parse_acs_codes_case_insensitive' in content:
        return "patch skipped: already present"

    new_test = '''
    def test_parse_acs_codes_case_insensitive(self):
        """Test that lowercase/mixed-case ACS codes are parsed and normalized."""
        text = "pa.i.a.k1 CA.iii.b.s2 Ir.Viii.C.K11"
        codes = self.service.parse_acs_codes(text)
        assert len(codes) == 3
        assert "PA.I.A.K1" in codes
        assert "CA.III.B.S2" in codes
        assert "IR.VIII.C.K11" in codes

'''
    marker = 'assert "PA.99.A.K1" not in codes  # Invalid (not Roman numeral)'
    insertion = content.find(marker)
    if insertion == -1:
        return "patch failed: insertion point not found"
    insertion += len(marker)
    content = content[:insertion] + "\n" + new_test + content[insertion:]
    path.write_text(content)
    return "patch applied"


def patch_service_file(repo: Path) -> str:
    path = repo / SERVICE_FILE
    content = path.read_text()
    old = 'ACS_CODE_PATTERN = re.compile(r"[A-Z]{2}\\.[IVX]+\\.[A-Z]\\.[KSR]\\d{1,2}")'
    new = 'ACS_CODE_PATTERN = re.compile(r"[A-Z]{2}\\.[IVX]+\\.[A-Z]\\.[KSR]\\d{1,2}", re.IGNORECASE)'
    if old not in content:
        return f"patch failed: pattern not found\ncontent around pattern:\n{content[20:200]}"
    content = content.replace(old, new)
    # Normalize matched codes to uppercase so output format is preserved
    content = content.replace(
        """        # Remove duplicates while preserving order
        seen = set()
        unique_codes = []
        for code in matches:
            if code not in seen:
                seen.add(code)
                unique_codes.append(code)

        return unique_codes""",
        """        # Remove duplicates while preserving order, normalize to uppercase
        seen = set()
        unique_codes = []
        for code in matches:
            normalized = code.upper()
            if normalized not in seen:
                seen.add(normalized)
                unique_codes.append(normalized)

        return unique_codes""",
    )
    path.write_text(content)
    return "patch applied"


def run_tests(repo: Path) -> str:
    """Run pytest in the temp repo using the source CFIPros venv Python."""
    venv_python = "/home/orchestrator/repos/github/marcusgoll/cfipros/api/.venv/bin/python"
    try:
        r = subprocess.run(
            [venv_python, "-m", "pytest", "-x", "-q", "tests/unit/services/test_aktr_ocr.py", "tests/services/test_aktr_ocr.py"],
            cwd=repo / "api",
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "PATH": f"/home/orchestrator/.local/bin:{os.environ.get('PATH', '')}"},
        )
        return f"exit={r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    except Exception as e:
        return f"test run error: {e}"


def run_task(governor: ContextGovernor, repo: Path, mode: str) -> RunResult:
    start = time.perf_counter()
    tool_calls: List[ToolCall] = []

    def record(tool: str, args: Dict[str, Any], response: str) -> None:
        tool_calls.append(ToolCall(tool, args, response))
        governor.record_tool_call(tool, args, response, len(tool_calls))

    governor.update_ledger(
        repo=str(repo),
        objective="Add case-insensitive ACS code parsing to AKTR OCR",
        current_branch="dogfood/aktr-case-insensitive",
        known_constraints=[
            "Do not break existing valid-code tests",
            "Preserve uppercase output format",
        ],
    )

    # 1. Read existing tests and service code (multi-file investigation)
    for rel in [TEST_FILE_UNIT, TEST_FILE_SERVICE, SERVICE_FILE, EXCEPTIONS_FILE, VISION_FILE]:
        record("read_file", {"path": str(repo / rel)}, read_file(repo, rel))

    # 2. Search for related ACS code patterns
    search_result = "found ACS_CODE_PATTERN in api/scripts/acs_audit_report.py, infra/scripts/extract_acs_standalone.py"
    record("search_files", {"pattern": "ACS_CODE_PATTERN", "path": str(repo)}, search_result)

    # 3. Patch service
    res = patch_service_file(repo)
    record("patch", {"path": str(repo / SERVICE_FILE), "mode": "replace"}, res)

    # 4. Patch both test files
    res = patch_test_file(repo, TEST_FILE_UNIT)
    record("patch", {"path": str(repo / TEST_FILE_UNIT), "mode": "replace"}, res)
    res = patch_test_file(repo, TEST_FILE_SERVICE)
    record("patch", {"path": str(repo / TEST_FILE_SERVICE), "mode": "replace"}, res)

    # 5. Re-read service to verify patch
    record("read_file", {"path": str(repo / SERVICE_FILE)}, read_file(repo, SERVICE_FILE))

    # 6. Run tests
    output = run_tests(repo)
    record("terminal", {"command": "uv run pytest -x -q tests/unit/services/test_aktr_ocr.py tests/services/test_aktr_ocr.py"}, output)

    success = "exit=0" in output
    context = governor.get_context_for_model()
    elapsed = time.perf_counter() - start
    return RunResult(
        mode=mode,
        success=success,
        tool_calls=len(tool_calls),
        context_chars=len(context),
        elapsed_seconds=elapsed,
        final_test_output=output,
    )


def run_ab() -> tuple[RunResult, RunResult]:
    # Run A: governor enabled
    repo_a = make_temp_repo("a")
    try:
        gov_a = ContextGovernor(raw_tool_window=5, summary_window=3)
        gov_a.on_session_start("dogfood-a")
        result_a = run_task(gov_a, repo_a, "governor_enabled")
    finally:
        shutil.rmtree(repo_a, ignore_errors=True)

    # Run B: governor disabled (large raw window, no summaries)
    repo_b = make_temp_repo("b")
    try:
        gov_b = ContextGovernor(raw_tool_window=1000, summary_window=0)
        gov_b.on_session_start("dogfood-b")
        result_b = run_task(gov_b, repo_b, "governor_disabled")
    finally:
        shutil.rmtree(repo_b, ignore_errors=True)

    return result_a, result_b


if __name__ == "__main__":
    a, b = run_ab()
    print(f"Run A ({a.mode}): success={a.success}, tools={a.tool_calls}, context_chars={a.context_chars}, elapsed={a.elapsed_seconds:.2f}s")
    print(f"Run B ({b.mode}): success={b.success}, tools={b.tool_calls}, context_chars={b.context_chars}, elapsed={b.elapsed_seconds:.2f}s")
    if a.context_chars and b.context_chars:
        print(f"Reduction: {b.context_chars / a.context_chars:.2f}x smaller with governor")
    print(f"\nTest output (governor run):\n{a.final_test_output}")
    print(f"\nTest output (full-context run):\n{b.final_test_output}")
