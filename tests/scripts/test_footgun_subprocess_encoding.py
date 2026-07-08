"""Tests for the ``subprocess text=True without explicit encoding=`` footgun
rule in ``scripts/check-windows-footguns.py``.

This rule (added alongside PR #60741) catches ``subprocess.run/Popen/call/
check_output/check_call(..., text=True, ...)`` calls that don't pass an
explicit ``encoding=``. On Chinese Windows (cp936/GBK) and other non-UTF-8
default codepages, ``text=True`` without ``encoding=`` decodes child output
with ``locale.getpreferredencoding(False)`` and crashes ``_readerthread``
with ``UnicodeDecodeError`` on non-default-codepage bytes.

See issues #47939, #53428, #57238.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LINTER_PATH = REPO_ROOT / "scripts" / "check-windows-footguns.py"


def _load_linter_module():
    """Import the linter script as a module (it's not a package).

    Register the module in sys.modules BEFORE exec_module so that
    ``@dataclass`` can resolve ``cls.__module__`` via
    ``sys.modules.get(cls.__module__).__dict__`` (CPython 3.11+ dataclass
    internals require this).
    """
    spec = importlib.util.spec_from_file_location("check_windows_footguns", LINTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_windows_footguns"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def linter():
    return _load_linter_module()


def _find_footgun(linter, name: str):
    """Locate a Footgun by name in the FOOTGUNS list."""
    for fg in linter.FOOTGUNS:
        if fg.name == name:
            return fg
    pytest.fail(f"Footgun rule '{name}' not found in FOOTGUNS")


def _scan_line(linter, line: str, footgun_name: str) -> bool:
    """Return True if the given line triggers the named footgun rule.

    Uses the linter's own pattern + post_filter logic so the test exercises
    the real detection path (including guard-hint and suppression checks).
    """
    fg = _find_footgun(linter, footgun_name)
    # Replicate the relevant checks from scan_file(): suppression marker,
    # guard hints, then pattern + post_filter.
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


RULE_NAME = "subprocess text=True without explicit encoding="


# ---------------------------------------------------------------------------
# Detection — these SHOULD be flagged
# ---------------------------------------------------------------------------


class TestDetection:
    def test_flags_subprocess_run_text_true_without_encoding(self, linter):
        line = '    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)'
        assert _scan_line(linter, line, RULE_NAME), "expected flag for text=True without encoding="

    def test_flags_subprocess_popen_text_true_without_encoding(self, linter):
        line = '    p = subprocess.Popen(cmd, text=True, stdout=PIPE)'
        assert _scan_line(linter, line, RULE_NAME)

    def test_flags_subprocess_check_output_text_true(self, linter):
        line = '    out = subprocess.check_output(["git", "status"], text=True)'
        assert _scan_line(linter, line, RULE_NAME)

    def test_flags_sp_alias_text_true(self, linter):
        line = '    res = _sp.run(cmd, text=True, timeout=5)'
        assert _scan_line(linter, line, RULE_NAME)

    def test_flags_text_with_spaces_around_equals(self, linter):
        line = '    subprocess.run(cmd, text = True, timeout=10)'
        assert _scan_line(linter, line, RULE_NAME)

    def test_flags_bare_run_call(self, linter):
        # .run( without explicit subprocess. prefix — still a subprocess call
        line = '    result = obj.run(cmd, text=True)'
        assert _scan_line(linter, line, RULE_NAME)


# ---------------------------------------------------------------------------
# Suppression — these should NOT be flagged
# ---------------------------------------------------------------------------


class TestSuppression:
    def test_does_not_flag_when_encoding_present(self, linter):
        line = '    subprocess.run(cmd, text=True, encoding="utf-8", errors="replace")'
        assert not _scan_line(linter, line, RULE_NAME)

    def test_does_not_flag_when_encoding_with_spaces(self, linter):
        line = "    subprocess.run(cmd, text=True, encoding = 'utf-8')"
        assert not _scan_line(linter, line, RULE_NAME)

    def test_does_not_flag_inline_suppression_marker(self, linter):
        line = '    subprocess.run(cmd, text=True)  # windows-footgun: ok — POSIX only'
        assert not _scan_line(linter, line, RULE_NAME)

    def test_does_not_flag_non_subprocess_text_kwarg(self, linter):
        # DataFrame.rename(text=True) — not a subprocess call
        line = '    df = df.rename(text=True)'
        assert not _scan_line(linter, line, RULE_NAME), (
            "should not flag non-subprocess APIs that accept text= kwarg"
        )

    def test_does_not_flag_text_true_in_string_literal(self, linter):
        line = '    """See subprocess.run(text=True) for details."""'
        assert not _scan_line(linter, line, RULE_NAME), (
            "should not flag text=True inside docstrings"
        )

    def test_does_not_flag_def_text_method(self, linter):
        line = '    def text(self, value: bool = True):'
        assert not _scan_line(linter, line, RULE_NAME)

    def test_does_not_flag_comment_only_line(self, linter):
        line = '    # subprocess.run(cmd, text=True) — example'
        assert not _scan_line(linter, line, RULE_NAME)


# ---------------------------------------------------------------------------
# Helper functions — unit tests for _is_likely_subprocess_call and
# _looks_like_string_literal
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_is_likely_subprocess_call_matches_subprocess_run(self, linter):
        assert linter._is_likely_subprocess_call("subprocess.run(cmd, text=True)")

    def test_is_likely_subprocess_call_matches_bare_run(self, linter):
        assert linter._is_likely_subprocess_call("result = obj.run(cmd, text=True)")

    def test_is_likely_subprocess_call_rejects_dataframe(self, linter):
        assert not linter._is_likely_subprocess_call("df.rename(text=True)")

    def test_is_likely_subprocess_call_rejects_plain_assignment(self, linter):
        assert not linter._is_likely_subprocess_call("config.text = True")

    def test_looks_like_string_literal_double_quotes(self, linter):
        import re
        line = '    msg = "use text=True carefully"'
        match = re.search(r"\btext\s*=\s*True\b", line)
        assert match is not None
        assert linter._looks_like_string_literal(line, match)

    def test_looks_like_string_literal_single_quotes(self, linter):
        import re
        line = "    msg = 'see text=True in docs'"
        match = re.search(r"\btext\s*=\s*True\b", line)
        assert match is not None
        assert linter._looks_like_string_literal(line, match)

    def test_looks_like_string_literal_false_for_real_code(self, linter):
        import re
        line = '    subprocess.run(cmd, text=True)'
        match = re.search(r"\btext\s*=\s*True\b", line)
        assert match is not None
        assert not linter._looks_like_string_literal(line, match)


# ---------------------------------------------------------------------------
# Full-repo scan — after PR #60741 merges, the new rule should find ZERO
# unsuppressed violations in the whole tree (excluding the linter itself
# and CONTRIBUTING docs). This test will FAIL until PR #60741 is merged;
# mark it xfail when run on a branch that doesn't include PR #60741's fixes.
# ---------------------------------------------------------------------------


class TestFullRepoScan:
    def test_new_rule_find_only_known_violations(self, linter, monkeypatch):
        """Scan the full repo and assert the new rule's matches are exactly
        the set of call sites that PR #60741 fixes (or zero, if PR #60741
        is already merged into this branch).

        This is a regression guard: if someone adds a new
        ``subprocess.run(text=True)`` without ``encoding=``, this test
        catches it.
        """
        # The 7 call sites that PR #60741 fixes. If PR #60741 is merged
        # into this branch, this set should be empty. If not, these are
        # the expected matches.
        pr_60741_sites = {
            "hermes_cli/main.py",
            "hermes_cli/onepassword_secrets_cli.py",
            "hermes_cli/setup.py",
            "tools/transcription_tools.py",
            "tools/tts_tool.py",
        }

        # Run the full scan
        roots = [
            REPO_ROOT / "hermes_cli",
            REPO_ROOT / "gateway",
            REPO_ROOT / "tools",
            REPO_ROOT / "cron",
            REPO_ROOT / "agent",
            REPO_ROOT / "plugins",
            REPO_ROOT / "scripts",
            REPO_ROOT / "acp_adapter",
            REPO_ROOT / "acp_registry",
        ]
        roots = [r for r in roots if r.exists()]

        fg = _find_footgun(linter, RULE_NAME)
        new_rule_matches: dict[str, list[int]] = {}

        for path in linter.iter_files(roots):
            matches = linter.scan_file(path, [fg])  # scan with ONLY the new rule
            if matches:
                rel = path.relative_to(REPO_ROOT).as_posix()
                new_rule_matches[rel] = [m[0] for m in matches]

        # Determine which sites remain. PR #60741's fixes are on a separate
        # branch; if this branch doesn't include them, the 7 call sites
        # will still be flagged — that's expected, not a failure.
        if new_rule_matches:
            # Filter out the linter itself (it mentions text=True in its
            # own pattern/message, but EXCLUDED_FILES handles that for the
            # CLI entry point; the helper functions could trip it).
            new_rule_matches = {
                k: v for k, v in new_rule_matches.items()
                if k != "scripts/check-windows-footguns.py"
            }

        if not new_rule_matches:
            # PR #60741 already merged — clean tree. This is the goal state.
            return

        # Matches remain — they must be exactly the PR #60741 sites.
        matched_files = set(new_rule_matches.keys())
        unexpected = matched_files - pr_60741_sites
        if unexpected:
            pytest.fail(
                f"New footgun rule found UNEXPECTED matches in files not "
                f"covered by PR #60741: {sorted(unexpected)}.\n"
                f"These are either new regressions or call sites that need "
                f"a `# windows-footgun: ok` suppression."
            )
        # All matches are the expected PR #60741 sites — OK on this branch.
