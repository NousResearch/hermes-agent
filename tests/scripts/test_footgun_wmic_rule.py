"""Tests for the ``wmic invocation without shutil.which guard`` footgun rule
in ``scripts/check-windows-footguns.py``.

wmic was removed as part of the WMIC deprecation on modern Windows 11 / late
Win 10 builds, so every spawn has to be gated on ``shutil.which("wmic")`` with
a PowerShell ``Get-CimInstance`` fallback. The rule is enforced by the blocking
``Windows footguns`` CI job.

It used to anchor on ``subprocess.``, which meant it matched nothing at all:
the Windows process scans were moved onto ``bounded_probe_run`` for the
deadlock-safe post-timeout cleanup (#87134) and every wmic call in the tree
went with them, out of the rule's sight. A rule that matches nothing reads
exactly like a rule that is being obeyed.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LINTER_PATH = REPO_ROOT / "scripts" / "check-windows-footguns.py"

RULE_NAME = "wmic invocation without shutil.which guard"


def _load_linter_module():
    spec = importlib.util.spec_from_file_location("check_windows_footguns", LINTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_windows_footguns"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def linter():
    return _load_linter_module()


def _scan_line(linter, line: str) -> bool:
    """Run the linter's own detection path over a single line."""
    for fg in linter.FOOTGUNS:
        if fg.name == RULE_NAME:
            break
    else:  # pragma: no cover - the rule is the subject of this file
        pytest.fail(f"Footgun rule '{RULE_NAME}' not found in FOOTGUNS")
    if linter.SUPPRESS_MARKER.search(line):
        return False
    if any(hint in line for hint in linter.GUARD_HINTS):
        return False
    code = linter._strip_code(line)
    if not code.strip():
        return False
    match = fg.pattern.search(code)
    if not match:
        return False
    if fg.post_filter is not None:
        try:
            if not fg.post_filter(match, line):
                return False
        except (IndexError, AttributeError):
            return False
    return True


class TestDetection:
    """Spawning wmic by name, whoever does the spawning."""

    @pytest.mark.parametrize("line", [
        # The shape that actually existed in the tree and went unseen.
        '    result = bounded_probe_run(["wmic", "process", "get", "ProcessId"],',
        # The shape the old anchor was written for.
        '    subprocess.run(["wmic", "process", "get", "ProcessId"])',
        '    subprocess.Popen(["wmic", "process", "list"])',
        '    subprocess.check_output(["wmic", "os", "get", "Caption"])',
        # Any other helper, including ones that do not exist yet.
        '    out = _run(["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"])',
        '    spawn([ "wmic", "process" ])',
        # Explicit .exe anywhere.
        '    cmd = "wmic.exe"',
    ])
    def test_flags_an_unguarded_spawn(self, linter, line):
        assert _scan_line(linter, line) is True


class TestTheGuardedFormIsNotFlagged:
    """The fix must be recognisable, or the rule cannot be satisfied."""

    @pytest.mark.parametrize("line", [
        # shutil.which's result is a variable, so no quoted literal follows.
        '                        wmic_path,',
        '    result = bounded_probe_run([wmic_path, "process", "get", "ProcessId"],',
        # The guard itself has no bracket in front of it.
        '    wmic_path = shutil.which("wmic")',
        '    if shutil.which("wmic") is not None:',
        # The PowerShell fallback.
        '    ps = shutil.which("powershell") or shutil.which("pwsh")',
        # Prose.
        '    # wmic was removed in Windows 10 21H1',
    ])
    def test_does_not_flag(self, linter, line):
        assert _scan_line(linter, line) is False

    def test_the_inline_marker_still_suppresses(self, linter):
        line = '    _run(["wmic", "os", "get"])  # windows-footgun: ok — POSIX-gated'
        assert _scan_line(linter, line) is False


class TestTheRuleReachesTheCodeItPolices:
    """A rule that matches nothing is indistinguishable from a rule that is
    being obeyed. This is the regression guard for that."""

    def test_the_rule_matches_the_call_shape_this_repo_writes(self, linter):
        """`bounded_probe_run` is what the Windows scans are required to use
        (#87134). A rule that cannot see through it cannot see the codebase."""
        assert _scan_line(
            linter,
            '                ["wmic", "process", "get", "ProcessId,CommandLine"],',
        ) is True

    def test_the_full_repo_scan_is_clean(self):
        """`--all` is what the blocking CI job runs. It must pass with the
        widened rule — if a real violation is added, this goes red."""
        proc = subprocess.run(
            [sys.executable, str(LINTER_PATH), "--all"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(REPO_ROOT), timeout=300,
        )
        assert proc.returncode == 0, (
            "check-windows-footguns.py --all reported violations:\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
