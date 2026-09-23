"""Entrypoint parity matrix (issue classes C19 + C15).

One fixture HERMES_HOME (shell hook + plugin hook on ``pre_llm_call``, AGENTS.md,
a skill, a memory entry, a stdio MCP server that spawns a grandchild, the custom
provider pointed at the recording fake, a disabled toolset) and ONE scripted turn
per entrypoint: the fake model calls the MCP canary tool, then answers.

For every entrypoint the same invariants must hold on the FIRST turn after a cold
start — any red cell is a wiring-parity bug of the "works in the CLI, missing in
Desktop/gateway/ACP/cron" class:

* the request carries the context file, the skill index and the memory entry;
* both hooks fired AND their injected context reached the model;
* the MCP tool was offered and a REAL call returned the fixture's canary (the
  inverted ``_stdio_children_dead`` burst failed exactly here on every surface);
* the documented toolset's core feature tools are present, the disabled toolset
  is absent;
* the surface delivered the model's answer to its client;
* after the surface's normal shutdown no MCP server / grandchild survives.

``PARITY_TABLE_OUT=<path>`` appends a markdown row per entrypoint (REPORT.md table).
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import pytest

from tests.e2e.core.parity import _drive_acp, _drive_cli, _drive_cron, _drive_gateway, _drive_rpc
from tests.e2e.core.parity._helpers import (
    FINAL_ANSWER,
    DriveResult,
    ParityHome,
    build_parity_home,
    check_invariants,
    collect_observation,
    describe_pids,
    kill_tagged,
    mcp_pids,
    start_provider,
    wait_no_orphans,
)


# Linux-only (/proc process-tree scans). The live-system guard bypass is needed
# ONLY for the orphan sweep: orphans reparented to init (the exact failure this suite
# hunts) sit outside the pytest subtree, and kill_tagged() signals nothing but
# PIDs carrying this run's unique PARITY_TREE_TAG. Children run with a tmp
# HOME/HERMES_HOME (asserted in ParityHome.env), so no real state is reachable.
pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="process-tree checks use /proc"),
    pytest.mark.live_system_guard_bypass,
]

PROMPT = "Call the parity canary tool, then report."

Driver = Callable[..., DriveResult]

DRIVERS: dict[str, Driver] = {
    "oneshot (-z)": _drive_cli.drive_oneshot,
    "chat -q": _drive_cli.drive_chat_q,
    "tui_gateway (stdio)": _drive_rpc.drive_tui_gateway,
    "serve (Desktop WS)": _drive_rpc.drive_serve,
    "gateway (fake adapter)": _drive_gateway.drive_gateway,
    "api_server": _drive_gateway.drive_api_server,
    "acp (stdio)": _drive_acp.drive_acp,
    "cron (run-now)": _drive_cron.drive_cron,
}

# Cells that are red on current main for a tracked, open bug. Strict: the test
# FAILS as soon as the cell turns green, so the entry is removed with the fix
# instead of silently masking a later regression of the same cell.
KNOWN_RED: dict[tuple[str, str], str] = {
    # ACP builds its AIAgent without the configured agent.disabled_toolsets.
    ("acp (stdio)", "toolset_restriction"): "#74582",
}


@dataclass
class Row:
    entrypoint: str
    cells: dict[str, bool] = field(default_factory=dict)
    error: str | None = None
    detail: str = ""
    toolset: str | None = None
    cwd_channel: str | None = None
    other_survivors: list[str] = field(default_factory=list)
    wall_s: float = 0.0


def _run_row(entrypoint: str, root: Path) -> Row:
    """Drive one entrypoint end to end and evaluate every cell. Never raises: the
    row carries the error, and its own process tree is swept before returning."""
    row = Row(entrypoint)
    started = time.monotonic()
    srv = start_provider(uuid.uuid4().hex[:8])
    ph: ParityHome | None = None
    try:
        ph = build_parity_home(root, srv.base_url)
        result = DRIVERS[entrypoint](ph, srv, PROMPT)
        row.toolset, row.cwd_channel = result.toolset, result.cwd_channel
        obs = collect_observation(entrypoint, ph, srv, result.final_text)
        cells = check_invariants(obs, ph, toolset=result.toolset)
        cells["answer_delivered"] = FINAL_ANSWER in (result.final_text or "")
        cells["graceful_exit"] = result.graceful_exit
        pids = mcp_pids(ph)
        # Vacuity guard: the orphan check only means something if the tree existed.
        cells["mcp_tree_spawned"] = bool(pids.get("server")) and bool(pids.get("grandchild"))
        survivors = wait_no_orphans(ph, timeout=30.0)
        cells["zero_mcp_orphans"] = not survivors
        # Report-only: non-MCP descendants still alive (timing-dependent, e.g. a
        # picker-prewarm `gh auth token` orphaned by a fast host exit).
        row.other_survivors = describe_pids(wait_no_orphans(ph, timeout=10.0, mcp_only=False))
        row.cells = cells
        stderr_tail = str(result.extra.get("stderr_tail", ""))
        log = result.extra.get("stderr_log")
        if log and Path(log).exists():
            stderr_tail = Path(log).read_text(encoding="utf-8", errors="replace")
        row.detail = (
            f"  tools offered: {sorted(obs.tool_names)}\n"
            f"  tool results: {[r[-300:] for r in obs.tool_results]}\n"
            f"  surviving MCP-tree pids: {describe_pids(survivors)} (mcp pid log {pids})\n"
            f"  final text: {(result.final_text or '')[-300:]!r}\n"
            f"  host exit code: {result.extra.get('exit_code')}\n"
            f"  host stderr tail: {stderr_tail[-1500:]}"
        )
    except Exception as exc:  # noqa: BLE001 - reported per row
        row.error = f"{type(exc).__name__}: {exc}"[:4000]
    finally:
        if ph is not None:
            kill_tagged(ph)
        srv.stop()
        row.wall_s = round(time.monotonic() - started, 1)
    return row


@pytest.fixture(scope="module")
def matrix(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Row]:
    """All entrypoints driven CONCURRENTLY (each in its own home, provider and
    process tree): the file's wall time is the slowest cold start, not the sum.
    Everything, including the orphan sweep, completes during setup, while the
    first test's live-guard bypass is in effect."""
    roots = {ep: tmp_path_factory.mktemp(f"parity{i}") for i, ep in enumerate(DRIVERS)}
    with ThreadPoolExecutor(max_workers=len(DRIVERS), thread_name_prefix="parity") as pool:
        futures = {ep: pool.submit(_run_row, ep, roots[ep]) for ep in DRIVERS}
        rows = {ep: fut.result() for ep, fut in futures.items()}
    out = os.environ.get("PARITY_TABLE_OUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            for row in rows.values():
                fh.write(json.dumps({
                    **asdict(row), "detail": None,
                    "known_red": {c: ref for (ep, c), ref in KNOWN_RED.items() if ep == row.entrypoint},
                }) + "\n")
    return rows


@pytest.mark.parametrize("entrypoint", list(DRIVERS))
def test_entrypoint_parity(entrypoint: str, matrix: dict[str, Row]) -> None:
    row = matrix[entrypoint]
    assert row.error is None, f"{entrypoint}: turn failed before the cells could be evaluated:\n{row.error}"
    known = {cell for (ep, cell) in KNOWN_RED if ep == entrypoint}
    fixed = sorted(cell for cell in known if row.cells.get(cell))
    assert not fixed, (
        f"{entrypoint}: {fixed} now green — drop the KNOWN_RED entry "
        f"({[KNOWN_RED[(entrypoint, c)] for c in fixed]}) so the cell is enforced again")
    failed = sorted(k for k, ok in row.cells.items() if not ok and k not in known)
    assert not failed, f"{entrypoint}: parity cells red: {failed}\n{row.detail}"
