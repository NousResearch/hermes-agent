"""Tests for the ``PowerShell shell-out without -NoProfile`` footgun rule in
``scripts/check-windows-footguns.py``.

The rule catches internal PowerShell shell-outs that don't pass ``-NoProfile``.
Without it the user's ``$PROFILE`` runs before our command: a profile that
blocks on stdin (the common "Windows PowerShell 5.1 hands off to an interactive
pwsh 7" snippet does exactly that) hangs the child forever, and because the
caller usually captures output, it reads as a silent stall rather than an error
— the shape reported in #108735, where ``hermes update`` looped on the uv
install for hours.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LINTER_PATH = REPO_ROOT / "scripts" / "check-windows-footguns.py"

RULE_NAME = "PowerShell shell-out without -NoProfile"


def _load_linter_module():
    """Import the linter script as a module (it's not a package)."""
    spec = importlib.util.spec_from_file_location("check_windows_footguns", LINTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_windows_footguns"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def linter():
    return _load_linter_module()


def _find_footgun(linter, name: str):
    for fg in linter.FOOTGUNS:
        if fg.name == name:
            return fg
    pytest.fail(f"Footgun rule '{name}' not found in FOOTGUNS")


def _scan_line(linter, line: str, footgun_name: str) -> bool:
    """Return True if the given line triggers the named footgun rule.

    Mirrors ``scan_file``'s checks (suppression marker, guard hints, pattern,
    post_filter) so the test exercises the real detection path.
    """
    fg = _find_footgun(linter, footgun_name)
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


# ---------------------------------------------------------------------------
# Detection — these SHOULD be flagged
# ---------------------------------------------------------------------------


class TestDetection:
    def test_flags_quoted_powershell_with_command(self, linter):
        line = '    subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-Command", cmd])'
        assert _scan_line(linter, line, RULE_NAME)

    def test_flags_powershell_exe_with_file(self, linter):
        line = '    subprocess.run(["powershell.exe", "-File", "install.ps1"], check=True)'
        assert _scan_line(linter, line, RULE_NAME)

    def test_flags_pwsh_host(self, linter):
        line = '    subprocess.run(["pwsh", "-c", script], capture_output=True)'
        assert _scan_line(linter, line, RULE_NAME)

    def test_flags_argv_built_around_ps_bin(self, linter):
        line = '    cmd = [ps_bin, "-ExecutionPolicy", "Bypass", "-File", str(script)]'
        assert _scan_line(linter, line, RULE_NAME)

    def test_flags_ps_exe_variable_with_encoded_command(self, linter):
        line = '    args = [ps_exe, "-EncodedCommand", payload]'
        assert _scan_line(linter, line, RULE_NAME)


# ---------------------------------------------------------------------------
# Suppression — these should NOT be flagged
# ---------------------------------------------------------------------------


class TestSuppression:
    def test_does_not_flag_when_noprofile_present(self, linter):
        line = (
            '    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", '
            '"-ExecutionPolicy", "Bypass", "-Command", cmd])'
        )
        assert not _scan_line(linter, line, RULE_NAME)

    def test_does_not_flag_noprofile_before_file(self, linter):
        line = '    cmd = [ps_bin, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]'
        assert not _scan_line(linter, line, RULE_NAME)

    def test_does_not_flag_availability_probe(self, linter):
        # Bare presence probe: no switch, and GUARD_HINTS covers the which() case.
        assert not _scan_line(linter, '    ps_bin = shutil.which("powershell")', RULE_NAME)

    def test_does_not_flag_host_name_without_switch(self, linter):
        assert not _scan_line(linter, '    label = "powershell"', RULE_NAME)

    def test_does_not_flag_comment_only_line(self, linter):
        line = '    # subprocess.run(["powershell", "-Command", cmd]) — example'
        assert not _scan_line(linter, line, RULE_NAME)

    def test_does_not_flag_explicit_suppression(self, linter):
        line = '    subprocess.run(["powershell", "-Command", cmd])  # windows-footgun: ok'
        assert not _scan_line(linter, line, RULE_NAME)


# ---------------------------------------------------------------------------
# Full-repo scan — the tree must stay clean under this rule, otherwise adding
# it turns CI red for everyone else.
# ---------------------------------------------------------------------------


class TestFullRepoScan:
    def test_no_unsuppressed_violations_in_tree(self, linter):
        roots = [
            REPO_ROOT / top
            for top in ("hermes_cli", "gateway", "tools", "cron", "agent", "plugins", "scripts", "acp_adapter")
        ]
        roots = [r for r in roots if r.exists()]
        footgun = _find_footgun(linter, RULE_NAME)

        hits = []
        for path in linter.iter_files(roots):
            for lineno, line, fg in linter.scan_file(path, [footgun]):
                hits.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{lineno}: {line.strip()}")

        assert not hits, "PowerShell shell-outs missing -NoProfile:\n" + "\n".join(hits)
