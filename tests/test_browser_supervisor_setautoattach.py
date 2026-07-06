"""Structural + source-anchored regression tests for the
``Target.setAutoAttach`` best-effort fix in
``tools.browser_supervisor._attach_initial_page`` (#59797).

The supervisor module pulls in optional dependencies (``websockets``,
etc.) that may not be installed in every test harness. Instead of
importing the module — which would force a websockets install just
for this test — we assert textual anchors in the source so a
regression that silently removes or misorders the try/except is
caught at unit-test time. The behaviour under live CDP supervision
is exercised manually by the reporter on the Brave/Windows + local
CDP environment described in the issue.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPERVISOR_PATH = os.path.join(
    REPO_ROOT, "tools", "browser_supervisor.py"
)


def _read_source():
    with open(SUPERVISOR_PATH, encoding="utf-8") as f:
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
# A. Fix is present — try/except wraps Target.setAutoAttach
# ---------------------------------------------------------------------------


def test_a_setautoattach_wrapped_in_try_except():
    src = _read_source()
    method_segment = src.split('async def _attach_initial_page', 1)[1].split(
        'async def ', 1
    )[0]
    # The Target.setAutoAttach call must appear inside a try and any
    # exception must be logged, not re-raised. We search a constrained
    # substring (the method body) so a different setAutoAttach call
    # elsewhere doesn't muddy the assertion.
    assert (
        "try:" in method_segment
        and "Target.setAutoAttach" in method_segment
    ), "method body missing try/setAutoAttach wrapper"
    # Best-effort log line — the exact wording carries the host-name
    # failure rationale (clarifying why flattening-only is OK).
    assert (
        "setAutoAttach declined by host" in method_segment
    ), "expected best-effort log line; if removed, the fix lost its rationale"
    assert (
        "#59797" in method_segment
    ), "issue reference missing in the method docstring/anchor"


# ---------------------------------------------------------------------------
# B. Flatten attach still runs BEFORE setAutoAttach in the fix
# ---------------------------------------------------------------------------


def test_b_attach_before_setautoattach_ordering():
    """The fix must NOT change the call ordering relative to flatten
    attach. Regression guard against an accidental reordering that
    would still flatten-attach successfully for some hosts but break
    hosts that DO expect setAutoAttach to fire.
    """
    src = _read_source()
    method_segment = src.split('async def _attach_initial_page', 1)[1].split(
        'async def ', 1
    )[0]
    flatten_attach_idx = method_segment.find('"Target.attachToTarget"')
    setautoattach_idx = method_segment.find('"Target.setAutoAttach"')
    page_enable_idx = method_segment.find('"Page.enable"')
    runtime_enable_idx = method_segment.find('"Runtime.enable"')
    # All four must be present.
    for name, idx in (
        ("attachToTarget", flatten_attach_idx),
        ("setAutoAttach", setautoattach_idx),
        ("Page.enable", page_enable_idx),
        ("Runtime.enable", runtime_enable_idx),
    ):
        assert idx >= 0, f"{name} missing from _attach_initial_page"
    # Documented ordering preserved.
    assert (
        flatten_attach_idx
        < page_enable_idx
        < runtime_enable_idx
        < setautoattach_idx
    ), (
        f"ordering broken: attach={flatten_attach_idx} "
        f"Page={page_enable_idx} Runtime={runtime_enable_idx} "
        f"setAutoAttach={setautoattach_idx}"
    )


# ---------------------------------------------------------------------------
# C. setAutoAttach exception is logged with logger.warning not swallowed
# ---------------------------------------------------------------------------


def test_c_best_effort_logged_at_warning_level():
    """The swallowed exception must produce a ``logger.warning`` so
    operators see setAutoAttach rejections in their logs even though
    the supervisor continues. A regression that downgrades the log to
    debug (silent) defeats the observability purpose.
    """
    src = _read_source()
    method_segment = src.split('async def _attach_initial_page', 1)[1].split(
        'async def ', 1
    )[0]
    # Locate the try/except block by anchoring on the call, then take
    # a 600-char window afterwards to inspect what comes after the
    # except clause. Big enough to cover the warning block without
    # picking up later methods.
    setauto_idx = method_segment.find('"Target.setAutoAttach"')
    assert setauto_idx >= 0
    after = method_segment[setauto_idx:setauto_idx + 1200]
    assert "except Exception as exc:" in after, (
        "expected `except Exception as exc:` guarding setAutoAttach"
    )
    assert "logger.warning" in after, (
        "logger.warning must follow the setAutoAttach exception handler"
    )


# ---------------------------------------------------------------------------
# D. Issue-anchor comment (#59797) referenced inline (helps a maintainer
#    who greps for cause when triaging regressions)
# ---------------------------------------------------------------------------


def test_d_issue_reference_in_source():
    src = _read_source()
    method_segment = src.split('async def _attach_initial_page', 1)[1].split(
        'async def ', 1
    )[0]
    assert (
        "#59797" in method_segment
    ), "issue anchor (#59797) missing from _attach_initial_page"


def main():
    runner = TestRunner()
    runner.run("a_setautoattach_wrapped_in_try_except", test_a_setautoattach_wrapped_in_try_except)
    runner.run("b_attach_before_setautoattach_ordering", test_b_attach_before_setautoattach_ordering)
    runner.run("c_best_effort_logged_at_warning_level", test_c_best_effort_logged_at_warning_level)
    runner.run("d_issue_reference_in_source", test_d_issue_reference_in_source)
    return runner.summary()


if __name__ == "__main__":
    sys.exit(main())