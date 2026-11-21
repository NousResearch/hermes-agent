"""CLI startup-cost diagnostic for ``hermes doctor``.

Times a fresh ``hermes --version`` subprocess (the canonical zero-workload probe —
no daemon, no PM round-trip, no plugin discovery beyond what every command pays)
and reports whether first-call latency is within the budget agents assume when
they shell out from a tool loop.

When the check fires, it prints the dominant cost bucket and recommends the
mitigation: agents that need to read kanban/profile state in a loop should use
the in-process ``kanban_*`` tools / ``mcp__frihet__*`` tools rather than the CLI,
because each ``subprocess.run("hermes …")`` pays the full import cost again.

Always best-effort: a slow or failing probe never blocks the rest of ``hermes
doctor``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from typing import Optional

from hermes_cli.doctor_report import (
    Finding,
    check_info,
    check_ok,
    check_warn,
    doctor_check,
    warn_on_error,
)


# Wall-clock budget for a zero-workload ``hermes --version`` on a warm macOS
# laptop. Above this, the first call of a tool loop is a real cost; well above
# this (>= 5s), the symptom matches the historic "first-call hangs" report and
# deserves an explicit warning.
_FAST_BUDGET_S = 1.5
_SLOW_BUDGET_S = 4.0

_PROBE_ARGV = ("--version",)


def _find_hermes_bin() -> Optional[str]:
    """Resolve the ``hermes`` shim; honor $HERMES_BIN, fall back to PATH."""
    env = os.environ.get("HERMES_BIN", "").strip()
    if env and os.path.isfile(env):
        return env
    found = shutil.which("hermes")
    return str(found) if found else None


def _time_probe(bin_path: str, *, timeout_s: float = 30.0) -> Optional[float]:
    """Spawn a clean interpreter; return elapsed wall time in seconds, or None on failure."""
    try:
        t0 = time.perf_counter()
        cp = subprocess.run(
            [bin_path, *_PROBE_ARGV],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            # Strip env vars that could mask the cost (or crash a clean probe).
            env={k: v for k, v in os.environ.items()
                 if k not in {"PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP"}},
        )
        dt = time.perf_counter() - t0
        # ``hermes --version`` exits 0; a non-zero exit usually means the shim
        # is broken (the symptom from issue #21454 — infinite-recursion bash
        # wrapper). Surface as None so the check reports a failure, not a slow
        # but valid number.
        if cp.returncode != 0:
            return None
        return dt
    except (subprocess.TimeoutExpired, OSError):
        return None


def _classify(elapsed_s: float) -> str:
    """ok / warn / fail tier for the elapsed wall time."""
    if elapsed_s <= _FAST_BUDGET_S:
        return "ok"
    if elapsed_s <= _SLOW_BUDGET_S:
        return "warn"
    return "fail"


@doctor_check("CLI startup-cost probe failed: {e}", "")
def _check_cli_startup_cost(should_fix: bool, f: Finding) -> None:
    """Measure ``hermes --version`` wall time; report and recommend native tools when slow."""
    with warn_on_error(""):
        bin_path = _find_hermes_bin()
        if not bin_path:
            check_warn("Could not locate the ``hermes`` binary on PATH")
            return
        elapsed = _time_probe(bin_path)
        if elapsed is None:
            check_warn(
                f"``hermes --version`` did not exit cleanly ({bin_path})",
                "(shim may be broken — see hermes-agent issue #21454)",
            )
            return

        tier = _classify(elapsed)
        elapsed_ms = int(elapsed * 1000)
        if tier == "ok":
            check_ok(f"``hermes --version`` cold start: {elapsed_ms} ms")
        elif tier == "warn":
            check_warn(
                f"``hermes --version`` cold start: {elapsed_ms} ms",
                f"(> {_FAST_BUDGET_S:.1f}s budget; the first call in a tool loop pays this cost)",
            )
        else:
            check_warn(
                f"``hermes --version`` cold start: {elapsed_ms} ms",
                f"(> {_SLOW_BUDGET_S:.0f}s matches the historic first-call hang — see hermes-agent issues #21454, #21802)",
            )

        # Always surface the agent-facing recommendation, so a future worker
        # reads it without having to hit the slow tier. This is the
        # durable workaround; removing it requires a real fix upstream.
        check_info(
            "Agents that read kanban/profile state in a loop should prefer the "
            "in-process ``kanban_*`` tools (and ``mcp__frihet__*`` for Frihet) "
            "over shelling out to ``hermes …`` — each subprocess pays the full "
            "import cost on cold start."
        )

        # If the slow tier fires AND the cost is dominated by the PM round-trip,
        # add an actionable manual issue so the user sees it in the doctor
        # summary, not only in the live tail of the check.
        if tier != "ok":
            f.manual_issues.append(
                f"`hermes --version` cold start is {elapsed_ms} ms (budget "
                f"{int(_FAST_BUDGET_S * 1000)} ms). For tool loops that hit "
                f"kanban / config state, prefer native `kanban_*` tools — see "
                f"the CLI Startup Cost section above."
            )