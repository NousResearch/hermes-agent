"""
Regression test for issue #62175 - Dashboard leaks CLOSE_WAIT sockets
to Nous cloud backend at ~40/day, EMFILE in ~5 days.

This is a defensive observability fix, NOT a fix for the underlying leak
(which the reporter could not pin down from static reading — they asked
for live py-spy + lsof data, and the canonical socket-owner analysis
remains open).

What this fix does: log a warning at dashboard startup if fd count
exceeds 80% of the soft RLIMIT_NOFILE. This gives operators lead time
to investigate BEFORE the dashboard crashes with EMFILE.

Tests:
  1. test_static_warning_at_startup - source-level tripwire: the warning
     code must be present in web_server.py near the dashboard startup.
  2. test_behavioral_rlimit_no_file_readable - functional check that
     resource.getrlimit returns a usable value on Linux.
"""

import re
import resource
from pathlib import Path


def test_static_warning_at_startup():
    """Source-level tripwire: the fd-count warning must exist in
    web_server.py near the dashboard startup logic."""
    worktree = Path("/tmp/hermes-pr-work-60859/hermes-agent")
    src = (worktree / "hermes_cli" / "web_server.py").read_text()

    # Find the area around start_nous_auth_keepalive
    m = re.search(
        r"start_nous_auth_keepalive\(\)\s*\n\s*except.*?as exc:.*?\n.*?_log\.debug.*?Nous auth keepalive did not start:.*?\n",
        src, re.DOTALL,
    )
    assert m, "Could not find start_nous_auth_keepalive block in web_server.py"
    startup_area = m.end()
    context = src[startup_area:startup_area + 2000]

    # The fd warning must be present after startup
    assert "soft RLIMIT_NOFILE" in context or "RLIMIT_NOFILE" in context, (
        "#62175 regression: fd-count warning is missing from dashboard startup. "
        "Without this warning, operators get no lead time before the dashboard "
        "crashes with EMFILE."
    )

    # The 80% threshold must be present
    assert "0.8" in context, (
        "#62175: the 80% threshold for fd-count warning is missing."
    )


def test_behavioral_rlimit_no_file_readable():
    """Functional check: resource.getrlimit(RLIMIT_NOFILE) returns a
    usable soft limit on Linux. This is the precondition for the warning
    to actually fire on the dashboard process.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # On Linux, this is typically 1024 (soft) / 1048576 (hard) or similar
    assert soft > 0, (
        f"#62175: resource.getrlimit returned non-positive soft limit: {soft}. "
        f"Cannot compute fd-count warning threshold."
    )
    assert hard >= soft, (
        f"#62175: hard limit ({hard}) is less than soft limit ({soft}). "
        f"OS configuration is broken — fd-count warning may mis-fire."
    )


def test_static_references_issue_number():
    """Defensive check: the new code path must reference the issue number
    so future maintainers can find the discussion."""
    worktree = Path("/tmp/hermes-pr-work-60859/hermes-agent")
    src = (worktree / "hermes_cli" / "web_server.py").read_text()

    # Find any comment that includes the issue number
    assert "62175" in src, (
        "#62175: the fd-count warning code path should reference the issue "
        "number in a comment so future maintainers can find this discussion."
    )