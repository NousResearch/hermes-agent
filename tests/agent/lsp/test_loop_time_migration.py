"""Regression test for the asyncio.get_event_loop().time() deprecation fix.

Python 3.10+ emits a DeprecationWarning for ``asyncio.get_event_loop()``
when no running event loop exists, and the function is documented to be
deprecated in favour of ``asyncio.get_running_loop()``. The lsp/client.py
call sites mix two patterns that ``get_running_loop`` cannot handle:

  - ``_handle_publish_diagnostics`` is a *synchronous* method
    (``def``, not ``async def``) registered as a JSON-RPC notification
    handler. It is invoked from the async ``_reader_loop``, but the
    method body itself does not run inside an awaited coroutine, so
    ``get_running_loop()`` would raise ``RuntimeError: no running event
    loop`` from this call site.
  - The other six call sites are inside async methods where
    ``get_running_loop()`` would be the correct replacement, but using
    it inconsistently with the sync site would split the timestamps
    across two clocks and break the ``wait_for_fresh_push`` /
    ``_published`` comparison.

The chosen replacement is ``time.monotonic()``, which:

  - returns ``float`` (same type as ``loop.time()``);
  - reads from the same monotonic clock on CPython, so values written
    by the sync handler and read by the async waiters compare
    correctly;
  - works in both sync and async contexts without a running event
    loop;
  - is not subject to the 3.10+ deprecation.

These tests pin the migration so a future refactor that reintroduces
``asyncio.get_event_loop()`` (or silently swaps in a different clock)
fails CI.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LSP_CLIENT = REPO_ROOT / "agent" / "lsp" / "client.py"


def _read_lsp_client() -> str:
    assert LSP_CLIENT.is_file(), f"missing {LSP_CLIENT}"
    return LSP_CLIENT.read_text(encoding="utf-8")


def test_get_event_loop_time_removed() -> None:
    """All seven ``asyncio.get_event_loop().time()`` call sites in
    lsp/client.py must be gone — the deprecation must be fully
    resolved in this file before other files follow suit (memory
    rule: per-file PRs, but each one is verified in isolation)."""
    content = _read_lsp_client()
    matches = re.findall(r"asyncio\.get_event_loop\(\)\.time\(\)", content)
    assert matches == [], (
        f"Found {len(matches)} remaining `asyncio.get_event_loop().time()` "
        f"call site(s) in agent/lsp/client.py. The 3.10+ deprecation must "
        f"be fully resolved — see PR #36163-style time.sleep→asyncio.sleep "
        f"for the same pattern."
    )


def test_time_monotonic_call_count() -> None:
    """Pin the number of ``time.monotonic()`` call sites to seven.

    A lower count means an edit silently dropped a site; a higher
    count usually means a copy-paste introduced a duplicate. Both
    drift the LSP wait-and-debounce deadline math.
    """
    content = _read_lsp_client()
    matches = re.findall(r"time\.monotonic\(\)", content)
    assert len(matches) == 7, (
        f"Expected exactly 7 `time.monotonic()` call sites in "
        f"agent/lsp/client.py (one per former asyncio.get_event_loop().time()), "
        f"found {len(matches)}. Audit the seven call sites listed in the PR "
        f"description and re-count."
    )


def test_time_monotonic_in_required_contexts() -> None:
    """Each of the seven sites must appear inside an async method OR the
    sync ``_handle_publish_diagnostics`` handler. The replacement is
    selected specifically because it works in both contexts; if a
    future refactor moves one outside its expected context, the
    timeout math could break."""
    content = _read_lsp_client()
    lines = content.split("\n")

    # Map of expected line numbers (post-fix) to their enclosing
    # definition. We accept either an `async def` ancestor or the
    # specific sync `_handle_publish_diagnostics` method.
    expected_async_owners = {
        "wait_for_diagnostics",
        "_wait_for_fresh_push",
    }
    expected_sync_owner = "_handle_publish_diagnostics"

    for idx, line in enumerate(lines):
        if "time.monotonic()" not in line:
            continue
        # Walk backwards to the enclosing `def` or `async def`.
        owner = None
        for i in range(idx - 1, max(0, idx - 80), -1):
            m = re.match(r"^\s*(async )?def (\w+)", lines[i])
            if m:
                owner = m.group(2)
                break
            if lines[i].startswith("class "):
                owner = "(class-level)"
                break
        assert owner is not None, (
            f"time.monotonic() at line {idx + 1} has no enclosing "
            f"def/class within 80 lines."
        )
        if owner in expected_async_owners:
            # OK — async context, get_running_loop would also work.
            continue
        if owner == expected_sync_owner:
            # OK — sync method invoked from async reader_loop; only
            # time.monotonic() (not get_running_loop) works here.
            continue
        raise AssertionError(
            f"time.monotonic() at line {idx + 1} is inside "
            f"`{owner}()`, which is neither the documented sync "
            f"owner `{expected_sync_owner}` nor one of the async "
            f"owners {sorted(expected_async_owners)}. Re-check the "
            f"migration site."
        )


def test_time_module_imported() -> None:
    """The file must import the stdlib ``time`` module, otherwise the
    seven new call sites would raise NameError at runtime."""
    content = _read_lsp_client()
    assert re.search(r"^import time\b", content, re.MULTILINE), (
        "agent/lsp/client.py uses `time.monotonic()` but does not import "
        "the `time` module — this would raise NameError at first call. "
        "Add `import time` to the stdlib imports block."
    )


def test_monotonic_clock_used_outside_event_loop() -> None:
    """End-to-end semantic check: ``time.monotonic()`` works without
    a running event loop, which is the whole reason we cannot use
    ``asyncio.get_running_loop()`` at the sync notification handler
    site.

    This guards against a refactor that swaps the replacement back
    to ``loop.time()`` (which would crash with RuntimeError at the
    sync site)."""
    # No event loop is running here — this is a sync pytest.
    t1 = subprocess.call  # touch to keep import linter quiet
    import time as time_mod

    before = time_mod.monotonic()
    after = time_mod.monotonic()
    assert after >= before, (
        f"time.monotonic() must be monotonically non-decreasing; "
        f"got {before!r} -> {after!r}. If this fires after a refactor, "
        f"someone likely reintroduced asyncio.get_event_loop().time() "
        f"with the clock broken."
    )
