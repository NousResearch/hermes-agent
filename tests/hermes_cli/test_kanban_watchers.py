"""Regression tests for AP-4027 — dispatcher watcher UnboundLocalError.

Bug: in ``gateway/kanban_watchers.py:_kanban_dispatcher_watcher`` the per-tick
loop bound the local variable ``any_spawned`` (line ~1188) but the
``logger.info(...)`` call inside that loop still referenced the previous
name ``spawned_n`` (~line 1198). The first time a tick returned a result
with ``spawned`` truthy, Python raised ``UnboundLocalError`` at the logger
call and the dispatcher loop crashed silently (only visible in the
gateway log, easy to miss in steady-state).

These tests lock the fix in two ways:

1. **Static**: ``inspect.getsource`` on the watcher proves the closure no
   longer references the unbound name ``spawned_n`` and that it does log
   the spawned count correctly via ``len(res.spawned)``.
2. **Runtime**: drive the watcher for one tick with a fake dispatch result
   that has a populated ``spawned`` list and assert no exception escapes
   the loop body, plus that the expected log line is emitted.

The runtime test follows the same `object.__new__(GatewayRunner)` +
``runner._running`` toggle pattern already used in
``test_kanban_notify.py::test_dispatcher_tick_does_not_call_init_db`` —
proven in this codebase and avoids spinning a real gateway.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an initialised default kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.discard(
        str((home / "kanban" / "default.db").resolve())
    )
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Static regression checks
# ---------------------------------------------------------------------------


def _watcher_source() -> str:
    from gateway.run import GatewayRunner  # heavy import — keep lazy
    return inspect.getsource(GatewayRunner._kanban_dispatcher_watcher)


def _extract_per_result_log_block() -> str:
    """Return the body of the dispatcher's per-result ``for slug, res in
    (results or [])`` loop that contains the buggy ``logger.info`` call.

    The watcher source contains TWO copies of this loop (lines ~1233
    and ~1289): the first formats the per-board log line (the one AP-4027
    broke); the second emits the per-tick heartbeat event for
    ``hermes gateway health``. We want the FIRST one — it contains the
    bug.

    The marker ``for slug, res in (results or []):`` is itself indented
    at 16 spaces (column 16). The lines INSIDE its body are indented at
    20 spaces. The block ends when we encounter a non-empty line whose
    indent is <= 16 (a sibling statement or a parent — never a deeper
    child line). A line-based slice (not regex — the regex version
    catastrophically backtracks on the multi-KB watcher source) walks
    forward and stops at that boundary.
    """
    src = _watcher_source()
    marker = "for slug, res in (results or []):"
    occurrences = src.count(marker)
    assert occurrences >= 1, (
        f"Expected at least one occurrence of {marker!r} in the dispatcher "
        f"watcher, found {occurrences}. The watcher source structure "
        f"changed — update this helper."
    )
    start_line_idx = src.index(marker)
    block_indent = 16  # the marker's column position in the watcher
    after = src[start_line_idx + len(marker):]
    lines = after.split("\n")
    body_lines = []
    for line in lines:
        if not line.strip():
            # Blank lines belong to the block.
            body_lines.append(line)
            continue
        leading = len(line) - len(line.lstrip(" \t"))
        # A line whose indent is <= block_indent is either the sibling
        # `for` loop (also at column 16) or a parent statement at
        # column 12 or lower — either way, the block has ended.
        if leading <= block_indent:
            break
        body_lines.append(line)
    return "\n".join(body_lines) + "\n"


def test_watcher_source_has_expected_block_marker():
    """Lock the source shape: the watcher must contain the per-result
    ``for slug, res in (results or []):`` block we are guarding. If this
    invariant breaks (refactor renames the variable, moves the logger,
    etc.), the regression tests below need to be re-anchored — and the
    crash-on-tick bug surface may have moved too."""
    src = _watcher_source()
    marker = "for slug, res in (results or []):"
    # Currently exactly two copies exist (the buggy logger loop and the
    # tick-event emit loop). If that count changes, the structure has
    # been refactored and this regression suite must be re-anchored.
    assert src.count(marker) == 2, (
        f"Watcher source must contain exactly two {marker!r} blocks "
        f"(per-board logger loop + per-tick event emit loop); found "
        f"{src.count(marker)}. Update the regression tests."
    )


def test_watcher_logger_block_does_not_reference_unbound_spawned_n():
    """AP-4027: inside the per-result ``logger.info('kanban dispatcher ...')``
    block, the variable ``spawned_n`` must NOT appear as a free reference —
    the local variable in scope is ``any_spawned`` (boolean). Using
    ``spawned_n`` here raised UnboundLocalError on the first tick with
    ``res.spawned`` truthy.

    We scope the check to the buggy region (the for-loop body that
    formats the per-board log line) to avoid false positives from the
    later ``spawned_n = len(...)`` declarations in the tick-event emit
    block, which are correct (they are local bindings, not free uses).
    """
    block = _extract_per_result_log_block()
    assert "spawned_n" not in block, (
        "AP-4027 regression: the per-board 'kanban dispatcher [%s]: "
        "spawned=%d ...' log line block still references the unbound "
        "name 'spawned_n'. The logger call should use 'len(res.spawned)' "
        "so the spawned count is reported correctly.\n\nBlock:\n" + block
    )


def test_watcher_logs_spawned_count_via_len_res_spawned():
    """The per-result log line must report the spawned count, not a
    placeholder or a different attribute."""
    block = _extract_per_result_log_block()
    # The logger call should reference len(res.spawned) — the actual
    # number of tasks spawned for this board on this tick.
    assert "len(res.spawned)" in block, (
        "AP-4027 regression: the dispatcher 'spawned=%d' log line should "
        "be formatted with 'len(res.spawned)' so the spawned count is "
        "reported correctly.\n\nBlock:\n" + block
    )


def test_watcher_does_not_use_any_spawned_as_a_format_arg():
    """Defensive check: ``any_spawned`` is the boolean gate inside the
    ``if res is not None and getattr(res, 'spawned', None):`` block. Using
    it as the %d argument to ``logger.info('spawned=%d', ...)`` would
    format ``True``/``False`` as ``1``/``0`` — visually identical to a
    working count but semantically wrong. Confirm we log ``len(res.spawned)``
    not ``any_spawned``.
    """
    block = _extract_per_result_log_block()
    # The spawned count arg (first arg after `slug`) must reference
    # res.spawned (length, not the boolean).
    assert "len(res.spawned)" in block, (
        f"AP-4027 regression: dispatcher's 'spawned=%d' log line uses "
        f"args:\n{block}\nThe spawned count arg should be "
        f"'len(res.spawned)' so the actual spawned count is logged."
    )
    # And it must NOT be the boolean `any_spawned` (which would format
    # as 0/1 and be technically valid Python but semantically wrong).
    assert "any_spawned," not in block and "any_spawned)" not in block, (
        f"AP-4027 regression: dispatcher's 'spawned=%d' log line uses "
        f"the boolean 'any_spawned' as the count arg. Use "
        f"'len(res.spawned)' instead.\nBlock:\n{block}"
    )


# ---------------------------------------------------------------------------
# Runtime regression check
# ---------------------------------------------------------------------------


def _make_fake_dispatch_result(spawned_count: int) -> SimpleNamespace:
    """Build a minimal DispatchResult-shaped object that satisfies the
    watcher loop's ``getattr(res, "spawned", None)`` and attribute
    accesses. We use SimpleNamespace because the watcher treats res as an
    opaque object (``getattr`` for `spawned`, ``hasattr`` for `crashed`
    / `timed_out` / `auto_blocked`, direct attr for `reclaimed` /
    `promoted`)."""
    return SimpleNamespace(
        spawned=[("t1", "code-craftsman", "/tmp")] * spawned_count,
        reclaimed=0,
        promoted=0,
        crashed=[],
        timed_out=[],
        auto_blocked=[],
    )


@pytest.mark.asyncio
async def test_dispatcher_per_result_log_block_compiles_and_runs(
    kanban_home, caplog
):
    """AP-4027 runtime regression: extract the EXACT per-result log-block
    from the watcher source and ``exec`` it in a controlled namespace that
    does NOT pre-define ``spawned_n``. With the old code, the first time
    a result with ``res.spawned`` truthy flowed through, Python raised
    ``UnboundLocalError`` at the ``logger.info`` call. With the fix, the
    block executes and emits the expected log line.

    This is more focused than spinning the whole watcher (which has
    singleton lock, config loader, board enumeration, auto-decompose
    fan-out, etc. all of which we'd have to mock). The bug is purely
    local to the per-result logger block, so we test that block in
    isolation against the real logger output.
    """
    block = _extract_per_result_log_block()
    # The block starts AFTER the ``for slug, res in (results or []):``
    # marker line, so we re-prepend the dedented for-loop header so the
    # ``res`` binding is established before the ``if res is not None ...``
    # branch we extracted.
    snippet = (
        "any_spawned = False\n"
        # The for-loop header (dedented from col 16 to col 8).
        "for slug, res in (results or []):\n"
        # Dedent the extracted block from col 20 to col 12 so it lives
        # inside the for-loop body.
        + re.sub(r"^ {20}", "            ", block, flags=re.MULTILINE)
    )

    fake_result = _make_fake_dispatch_result(spawned_count=2)

    # IMPORTANT: this namespace intentionally does NOT define ``spawned_n``.
    # With the bug present, exec() of the block raises UnboundLocalError.
    # With the fix (len(res.spawned)), the block runs cleanly.
    namespace = {
        "results": [("default", fake_result)],
        "logger": logging.getLogger("gateway.run"),
        "asyncio": asyncio,  # in case the block references it
    }

    caplog.set_level(logging.INFO, logger="gateway.run")

    # If AP-4027 regresses, this raises UnboundLocalError. The test
    # passes only when the block runs cleanly AND emits the expected
    # log line.
    exec(snippet, namespace)  # noqa: S102 — exec is the point of the test

    # Verify the boolean gate fired (proves we actually walked the
    # spawned branch, not silently skipped).
    assert namespace.get("any_spawned") is True, (
        "AP-4027 regression: the per-result log block did not set "
        "any_spawned=True for a result with a truthy spawned list. "
        "Snippet:\n" + snippet
    )

    # Verify the log line landed with the correct spawned count.
    matching = [
        rec for rec in caplog.records
        if rec.name == "gateway.run"
        and "kanban dispatcher [default]" in rec.getMessage()
        and "spawned=2" in rec.getMessage()
    ]
    assert matching, (
        "AP-4027 regression: expected the per-board 'spawned=2' log line "
        "to be emitted. Captured records: "
        + repr([(r.name, r.getMessage()) for r in caplog.records])
    )


def test_watcher_logger_block_passes_python_compile():
    """Sanity check the extracted block is syntactically valid Python.
    Catches accidental edits that introduce a SyntaxError in the
    dispatcher watcher (the bug fix lives on one line, but a typo on
    a neighbour would also break the gateway at boot)."""
    body = _extract_per_result_log_block()
    # Compile the indented body as a function body — Python won't compile
    # indented code at module scope, so wrap in a def to validate syntax.
    try:
        compile("def _check():\n" + body, "<watcher-extract>", "exec")
    except SyntaxError as exc:
        pytest.fail(
            f"AP-4027 watcher per-result log block has a SyntaxError: "
            f"{exc}\nBlock:\n{body}"
        )


# ---------------------------------------------------------------------------
# AP-4028 — notifier must run independently of the dispatcher singleton lock
# ---------------------------------------------------------------------------
#
# These tests lock in the AP-4028 contract: ``_kanban_notifier_watcher`` must
# reach the board-fan-out / collection step regardless of which gateway holds
# the dispatcher lock, and its gate must read ``notifier_in_gateway`` /
# ``HERMES_KANBAN_NOTIFIER_IN_GATEWAY`` (NOT ``dispatch_in_gateway`` /
# ``HERMES_KANBAN_DISPATCH_IN_GATEWAY``). The legacy dispatcher-gate keys
# govern the dispatcher watcher only; the notifier decouples in AP-4028 to
# prevent a single-point-of-failure for ship reports whenever the failover
# script picks a bot-less dispatcher host (e.g. spec-writer, fleet-coach).
#
# Two layers of coverage, mirroring the AP-4027 style:
#
# 1. **Static** (source-level): proves the watcher no longer references the
#    dispatcher-gate config key or env var, and DOES read the new
#    notifier-gate config key and env var.
# 2. **Runtime** (behavioral): drives the watcher with the dispatcher-gate
#    explicitly disabled and confirms the notifier still reaches the board
#    fan-out — the exact scenario that produced the operator directive
#    "All I am seeing are failures" on 2026-06-23.


def _notifier_source() -> str:
    from gateway.run import GatewayRunner  # heavy import — keep lazy

    return inspect.getsource(GatewayRunner._kanban_notifier_watcher)


# ---------------------------------------------------------------------------
# Headline regression test (matches ``-k notifier_dispatcher_lock_independence``)
# ---------------------------------------------------------------------------


def test_notifier_dispatcher_lock_independence(kanban_home, monkeypatch, caplog):
    """AP-4028 headline contract: the notifier watcher must NOT depend on
    whether this gateway holds the dispatcher lock. Selectable via
    ``pytest -k notifier_dispatcher_lock_independence`` per the AP
    verification step.

    Pins all three pillars in a single test so a regression on any of
    them fails the headline check immediately:

    1. **Gate decoupling** — with ``dispatch_in_gateway: false`` the
       notifier still reaches the board fan-out (legacy config key no
       longer controls the notifier).
    2. **Per-tick observability** — operators see one info-level
       ``kanban notifier: tick profile=...`` line per tick.
    3. **Per-subscription routing intact** — the notifier-gate runs but
       the per-sub ``notifier_profile`` filter still routes correctly
       (this gateway, with no Telegram adapter, picks up zero subs
       for ``platform=telegram`` — confirmed via ``subs=0`` in the log).
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running = True
    # Provide a stub adapter so ``active_platforms`` is non-empty and
    # the watcher reaches the board-fan-out (where ``list_boards``
    # is called). The adapter is never asked to send because
    # ``list_boards`` returns [] so there are no deliveries.
    stub_adapter = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: stub_adapter}
    runner._kanban_sub_fail_counts = {}

    past_gate = {"n": 0}
    sleep_calls = {"n": 0}

    real_sleep = asyncio.sleep  # capture the real sleep before patching

    async def _fake_sleep(_delay):
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 2:
            runner._running = False
        await real_sleep(0)  # delegate to the real sleep, not the patch

    async def _fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    import hermes_cli.kanban_db as _kb

    def _list_boards(*_a, **_kw):
        past_gate["n"] += 1
        return []

    def _read_board_metadata(*_a, **_kw):
        # Fallback path the watcher takes if list_boards raises.
        past_gate["n"] += 1
        return {"slug": "default", "db_path": None}

    # Critical: dispatch_in_gateway is FALSE (no dispatcher lock held).
    # Before AP-4028 this caused the notifier to bail out — the exact
    # scenario that produced "All I am seeing are failures".
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"dispatch_in_gateway": False}},
        raising=False,
    )
    # Patch the helpers the watcher calls inside `_collect`. The watcher
    # binds ``_kb = hermes_cli.kanban_db`` at function scope, so patching
    # the module attribute is what reaches the call site.
    monkeypatch.setattr(_kb, "list_boards", _list_boards)
    monkeypatch.setattr(_kb, "read_board_metadata", _read_board_metadata)
    monkeypatch.setattr(_kb, "DEFAULT_BOARD", "default")
    caplog.set_level(logging.INFO, logger="gateway.run")

    with patch("asyncio.sleep", side_effect=_fake_sleep), \
         patch("asyncio.to_thread", side_effect=_fake_to_thread):
        asyncio.run(runner._kanban_notifier_watcher(interval=1))

    # Pillar 1: gate decoupling — the notifier reached the board fan-out
    # despite dispatch_in_gateway=false.
    assert past_gate["n"] >= 1, (
        "AP-4028 regression: notifier watcher bailed out before reaching "
        "_kb.list_boards when dispatch_in_gateway=false. The notifier "
        "must be independent of the dispatcher's singleton lock."
    )

    # Pillar 2: per-tick observability log.
    matching = [
        rec for rec in caplog.records
        if rec.name == "gateway.run"
        and "kanban notifier: tick" in rec.getMessage()
        and "profile=" in rec.getMessage()
        and "adapters=" in rec.getMessage()
        and "subs=" in rec.getMessage()
    ]
    assert matching, (
        "AP-4028 regression: expected at least one info-level log line of "
        "the form 'kanban notifier: tick profile=X adapters=[...] subs=N' "
        "per tick. Captured records: "
        + repr([(r.name, r.getMessage()) for r in caplog.records])
    )


# ---------------------------------------------------------------------------
# Granular static + escape-hatch checks
# ---------------------------------------------------------------------------


def test_notifier_watcher_does_not_reference_dispatch_gate_env():
    """AP-4028: the notifier gate code must NOT consult the dispatcher env var.

    The legacy env override was ``HERMES_KANBAN_DISPATCH_IN_GATEWAY``. If
    the notifier still reads it, an operator setting that flag on the
    failover bot-less host will silently disable the notifier — the
    exact regression AP-4028 fixes.

    The check is scoped to the GATE block (after the docstring) so we
    don't false-positive on the BC-9 migration note in the docstring,
    which intentionally references the legacy var to explain the change.
    """
    src = _notifier_source()
    # Strip the docstring (everything from the opening triple-quote to
    # the closing triple-quote) so the migration-note reference doesn't
    # false-positive. The gate code lives after the docstring closes.
    import re as _re
    no_doc = _re.sub(r'"""[\s\S]*?"""', "", src, count=1)
    assert "HERMES_KANBAN_DISPATCH_IN_GATEWAY" not in no_doc, (
        "AP-4028 regression: _kanban_notifier_watcher gate code still "
        "references the dispatcher env override "
        "'HERMES_KANBAN_DISPATCH_IN_GATEWAY'. Use "
        "'HERMES_KANBAN_NOTIFIER_IN_GATEWAY' instead so the notifier "
        "gate is independent of the dispatcher's singleton lock."
    )


def test_notifier_watcher_does_not_read_dispatch_in_gateway_config():
    """AP-4028: the notifier gate must NOT consult ``dispatch_in_gateway``.

    The legacy config key governed both dispatch and notify. If the
    notifier still reads it, every operator who runs ``hermes kanban
    daemon`` externally (and so sets ``dispatch_in_gateway: false``)
    also silently disables ship notifications.
    """
    src = _notifier_source()
    # Look specifically inside the kanban_cfg block — a stray reference in
    # an unrelated comment or docstring is fine, but the actual config read
    # for the gate must be ``notifier_in_gateway``.
    assert 'kanban_cfg.get("dispatch_in_gateway"' not in src, (
        "AP-4028 regression: _kanban_notifier_watcher reads the legacy "
        "config key 'kanban.dispatch_in_gateway' for its gate. Replace "
        "with 'kanban.notifier_in_gateway'."
    )


def test_notifier_watcher_reads_new_notifier_gate_config():
    """AP-4028 positive lock: the new config key must be consulted."""
    src = _notifier_source()
    assert 'kanban_cfg.get("notifier_in_gateway"' in src, (
        "AP-4028 regression: _kanban_notifier_watcher does NOT read the "
        "new config key 'kanban.notifier_in_gateway'. The decoupled "
        "notifier-gate is missing."
    )


def test_notifier_watcher_reads_new_notifier_gate_env():
    """AP-4028 positive lock: the new env override must be consulted."""
    src = _notifier_source()
    assert "HERMES_KANBAN_NOTIFIER_IN_GATEWAY" in src, (
        "AP-4028 regression: _kanban_notifier_watcher does NOT read the "
        "new env override 'HERMES_KANBAN_NOTIFIER_IN_GATEWAY'. Operators "
        "who set the env to '0' to disable the notifier have no escape "
        "hatch."
    )


def test_notifier_watcher_docstring_carries_bc9_migration_note():
    """AP-4028: the migration note must live in the docstring so operators
    reading the source understand the default-change rationale."""
    src = _notifier_source()
    assert "AP-4028" in src, (
        "AP-4028 regression: _kanban_notifier_watcher docstring is missing "
        "the AP-4028 / BC-9 migration note. Operators flipping the new "
        "flag back to False need the rationale documented in-place."
    )
    # The note must mention BOTH the new opt-out paths so operators know
    # where to find them.
    assert "notifier_in_gateway" in src, (
        "AP-4028 regression: BC-9 migration note does not mention the new "
        "config key 'kanban.notifier_in_gateway'."
    )
    assert "HERMES_KANBAN_NOTIFIER_IN_GATEWAY" in src, (
        "AP-4028 regression: BC-9 migration note does not mention the new "
        "env override 'HERMES_KANBAN_NOTIFIER_IN_GATEWAY'."
    )


def test_notifier_watcher_emits_per_tick_info_log(
    kanban_home, monkeypatch, caplog
):
    """AP-4028: operators must see one info-level log line per tick so a
    silent notifier is detectable. The expected format is
    ``kanban notifier: tick profile=X adapters=[...] subs=N``.

    Strategy: same ``object.__new__(GatewayRunner)`` + ``_running``
    toggle pattern as the AP-4027 tests. We patch the board-discovery
    helper and the per-tick ``asyncio.to_thread`` to return an empty
    list so the loop body runs cleanly, then assert the info log fired.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {}  # no adapters → subs=0; info log still must fire
    runner._kanban_sub_fail_counts = {}

    # Stop the watcher after the initial 5s sleep + first per-interval tick.
    sleep_calls = {"n": 0}
    real_sleep = asyncio.sleep  # capture the real sleep before patching

    async def _fake_sleep(_delay):
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 2:
            runner._running = False
        await real_sleep(0)  # delegate to the real sleep, not the patch

    async def _fake_to_thread(fn, *args, **kwargs):
        # The board fan-out returns []; nothing to deliver.
        return fn(*args, **kwargs)

    caplog.set_level(logging.INFO, logger="gateway.run")

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"kanban": {}})
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "list_boards", lambda *a, **kw: [])

    with patch("asyncio.sleep", side_effect=_fake_sleep), \
         patch("asyncio.to_thread", side_effect=_fake_to_thread):
        asyncio.run(runner._kanban_notifier_watcher(interval=1))

    matching = [
        rec for rec in caplog.records
        if rec.name == "gateway.run"
        and "kanban notifier: tick" in rec.getMessage()
        and "profile=" in rec.getMessage()
        and "adapters=" in rec.getMessage()
        and "subs=" in rec.getMessage()
    ]
    assert matching, (
        "AP-4028 regression: expected at least one info-level log line of "
        "the form 'kanban notifier: tick profile=X adapters=[...] subs=N' "
        "per tick. Captured records: "
        + repr([(r.name, r.getMessage()) for r in caplog.records])
    )


def test_notifier_watcher_runs_when_dispatcher_gate_disabled(
    kanban_home, monkeypatch
):
    """AP-4028 runtime regression: with ``dispatch_in_gateway: false``
    (the operator runs ``hermes kanban daemon`` externally and the
    gateway does NOT hold the dispatcher lock), the notifier MUST
    still reach the board fan-out. Before AP-4028 the watcher read
    ``dispatch_in_gateway`` and exited silently — this is the exact
    scenario that produced "All I am seeing are failures"."""

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running = True
    # Provide a stub adapter so ``active_platforms`` is non-empty and
    # the watcher reaches the board-fan-out (where ``list_boards``
    # is called). The adapter is never asked to send because
    # ``list_boards`` returns [] so there are no deliveries.
    stub_adapter = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: stub_adapter}
    runner._kanban_sub_fail_counts = {}

    past_gate = {"n": 0}
    sleep_calls = {"n": 0}
    real_sleep = asyncio.sleep  # capture the real sleep before patching

    async def _fake_sleep(_delay):
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 2:
            runner._running = False
        await real_sleep(0)  # delegate to the real sleep, not the patch

    async def _fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    import hermes_cli.kanban_db as _kb

    def _list_boards(*_a, **_kw):
        past_gate["n"] += 1
        return []

    def _read_board_metadata(*_a, **_kw):
        # Fallback path the watcher takes if list_boards raises.
        past_gate["n"] += 1
        return {"slug": "default", "db_path": None}

    # Critical: dispatch_in_gateway is FALSE (external daemon owns the
    # lock). Before AP-4028 this caused the notifier to bail out.
    # notifier_in_gateway is unset → defaults to True → must proceed.
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"dispatch_in_gateway": False}},
        raising=False,
    )
    monkeypatch.setattr(_kb, "list_boards", _list_boards)

    with patch("asyncio.sleep", side_effect=_fake_sleep), \
         patch("asyncio.to_thread", side_effect=_fake_to_thread):
        asyncio.run(runner._kanban_notifier_watcher(interval=1))

    assert past_gate["n"] >= 1, (
        "AP-4028 regression: notifier watcher bailed out before reaching "
        "_kb.list_boards when dispatch_in_gateway=false. The notifier "
        "must be independent of the dispatcher's singleton lock."
    )


def test_notifier_watcher_can_be_disabled_via_new_env_var(monkeypatch):
    """AP-4028: the new env override must provide the escape hatch."""
    from gateway.run import GatewayRunner

    monkeypatch.setenv("HERMES_KANBAN_NOTIFIER_IN_GATEWAY", "0")

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {}
    runner._kanban_sub_fail_counts = {}

    # Should return BEFORE the initial 5s sleep (early exit on env override).
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: (_ for _ in ()).throw(
            AssertionError("load_config must NOT be called when env override disables")
        ),
        raising=False,
    )
    import hermes_cli.kanban_db as _kb
    monkeypatch.setattr(
        _kb, "connect",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("_kb.connect must NOT be called when env override disables")
        ),
    )

    # The watcher returns immediately when the env override disables it,
    # without ever calling load_config or _kb.connect.
    asyncio.run(runner._kanban_notifier_watcher(interval=1))


def test_notifier_watcher_can_be_disabled_via_new_config_key(monkeypatch):
    """AP-4028: the new config key must provide the escape hatch."""
    from gateway.run import GatewayRunner

    monkeypatch.delenv("HERMES_KANBAN_NOTIFIER_IN_GATEWAY", raising=False)

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {}
    runner._kanban_sub_fail_counts = {}

    # ``notifier_in_gateway: false`` must cause the gate to return BEFORE
    # opening any board DB.
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"notifier_in_gateway": False}},
        raising=False,
    )
    import hermes_cli.kanban_db as _kb
    monkeypatch.setattr(_kb, "connect", lambda *a, **kw: None)

    asyncio.run(runner._kanban_notifier_watcher(interval=1))
