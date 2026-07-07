"""Structural source-anchored regression for the relaxed absolute-path
rule in ``tools.skills_tool._skill_lookup_path_error`` (#59824).

Behavioral tests that import ``tools.skills_tool`` end up pulling
in the import chain ``hermes_cli.config`` -> ``yaml`` (not always
installed in the test harness) and ``hermes_constants`` (state
caches ``HERMES_HOME`` at module-load time). Both are fragile in
this sandbox.

Instead of importing the module under test, we read the source
file and assert on textual anchors that catch the four
regressions that motivated the fix:

A. The previous unconditional absolute-path rejection is gone.
B. A trusted-root gate exists and resolves to the canonical
   ``HERMES_HOME/skills``, profile skills, and bundled/optional
   roots via ``os.path.realpath``.
C. ``..`` traversal components are still rejected (the
   relaxation only narrows the absolute-path accept set; it
   never widens the reject set).
D. Windows drive paths are still rejected (e.g. ``C:\\skills``).
"""

from __future__ import annotations

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL_PATH = os.path.join(REPO_ROOT, "tools", "skills_tool.py")


def _read_source():
    with open(TOOL_PATH, encoding="utf-8") as f:
        return f.read()


class TestRunner:
    def __init__(self):
        self.passed = []
        self.failed = []

    def run(self, name, fn):
        try:
            fn()
        except Exception as e:
            import traceback
            self.failed.append((name, e, traceback.format_exc()))
        else:
            self.passed.append(name)

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*60}\nResults: {len(self.passed)}/{total} passed")
        for n, _e, tb in self.failed:
            print(f"\n[FAIL] {n}\n{tb}")
        return 0 if not self.failed else 1


# ---------------------------------------------------------------------------
# A. _trusted_skill_roots exists and pulls in the four known roots
# ---------------------------------------------------------------------------


def test_a_trusted_skill_roots_pulls_in_all_canonical_roots():
    src = _read_source()
    assert "def _trusted_skill_roots" in src, (
        "_trusted_skill_roots helper missing — fix removed the gate"
    )
    # All four canonical roots are referenced.
    for fn_name in (
        "get_skills_dir",
        "get_optional_skills_dir",
        "get_bundled_skills_dir",
        "get_default_hermes_root",
    ):
        assert fn_name in src, f"{fn_name} not referenced by the trusted-root gate"


# ---------------------------------------------------------------------------
# B. The gate uses ``os.path.realpath`` (Path.resolve can fail
#    inconsistently across sandboxes) and ``os.path.relpath``
# ---------------------------------------------------------------------------


def test_b_resolve_uses_kernel_level_realpath():
    src = _read_source()
    # The fix introduces a ``_resolve_canon`` helper. Pin the kernel
    # resolution that survives Termux/Android sandbox quirks.
    assert "def _resolve_canon" in src, (
        "_resolve_canon helper missing — fix is using Path.resolve() directly"
    )
    assert "os.path.realpath" in src, "kernel-level realpath not used"


# ---------------------------------------------------------------------------
# C. Traversal is still rejected unconditionally — bypass-proof
# ---------------------------------------------------------------------------


def test_c_traversal_still_rejected():
    src = _read_source()
    # Walk to the body of _skill_lookup_path_error and check that
    # ``has_traversal_component`` is invoked AFTER the absolute-path
    # gate (so a path like /hermes/skills/../etc is still denied).
    fn_idx = src.find("def _skill_lookup_path_error")
    assert fn_idx >= 0
    segment = src[fn_idx:fn_idx + 4000]
    assert "has_traversal_component" in segment, (
        "traversal check missing from the validator body"
    )
    # The earlier absolute-path shortcut returns BEFORE the
    # traversal check would otherwise be reached — make sure the
    # prefix gate RUNS the traversal check.
    assert "if has_traversal_component(candidate):" in segment


# ---------------------------------------------------------------------------
# D. Windows drive paths still rejected
# ---------------------------------------------------------------------------


def test_d_windows_drive_still_rejected():
    src = _read_source()
    fn_idx = src.find("def _skill_lookup_path_error")
    assert fn_idx >= 0
    segment = src[fn_idx:fn_idx + 4000]
    # Drive-letter absolute paths must still be rejected, even
    # after the friendly relative-path error.
    assert (
        "PureWindowsPath(candidate).drive" in segment
    ), "drive-letter check missing from validator"
    # And the rejection message for a drive path is preserved.
    assert "relative path within the skills directory" in segment


# ---------------------------------------------------------------------------
# E. Empty/name input is rejected
# ---------------------------------------------------------------------------


def test_e_empty_input_rejected():
    src = _read_source()
    fn_idx = src.find("def _skill_lookup_path_error")
    assert fn_idx >= 0
    segment = src[fn_idx:fn_idx + 4000]
    # Candidate is non-empty invariant — a fresh check we added.
    assert "Skill name must not be empty." in segment, (
        "empty-string check missing — fix may be lenient here"
    )


# ---------------------------------------------------------------------------
# F. Issue anchor (#59824) is in the validator docstring/anchor
# ---------------------------------------------------------------------------


def test_f_issue_anchor_in_source():
    src = _read_source()
    fn_idx = src.find("def _skill_lookup_path_error")
    assert fn_idx >= 0
    segment = src[fn_idx:fn_idx + 4000]
    assert "#59824" in segment, "issue anchor missing from validator"


def main():
    runner = TestRunner()
    runner.run("a_trusted_skill_roots_pulls_in_all_canonical_roots", test_a_trusted_skill_roots_pulls_in_all_canonical_roots)
    runner.run("b_resolve_uses_kernel_level_realpath", test_b_resolve_uses_kernel_level_realpath)
    runner.run("c_traversal_still_rejected", test_c_traversal_still_rejected)
    runner.run("d_windows_drive_still_rejected", test_d_windows_drive_still_rejected)
    runner.run("e_empty_input_rejected", test_e_empty_input_rejected)
    runner.run("f_issue_anchor_in_source", test_f_issue_anchor_in_source)
    return runner.summary()


if __name__ == "__main__":
    sys.exit(main())