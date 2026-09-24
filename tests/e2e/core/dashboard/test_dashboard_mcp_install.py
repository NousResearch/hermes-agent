"""#120527: a dashboard MCP catalog install must never wedge the web server.

Real ``hermes dashboard`` launched the way the reporter launched ``hermes serve``: from an
interactive terminal, so the process's stdin IS a TTY (a pty slave here; the master is held open and
never written — nobody is watching that console). The MCP catalog is sourced through its supported
package-manager override (``HERMES_OPTIONAL_MCPS``) from a directory holding one manifest for a
local offline stdio MCP server (``fixture_catalog_mcp.py``), whose tool probe succeeds.

User-visible contract (what the Connectors page needs):
* ``POST /api/mcp/catalog/install`` answers — success or a clear 4xx — within a bounded deadline;
* afterwards an unrelated profile-scoped request (``GET /api/config``) and a ``/api/ws`` upgrade
  still complete within a few seconds.

Before the fix the install reaches the interactive tool checklist on the serve worker thread, blocks
forever on a console read, and keeps holding the process-global ``_SKILLS_PROFILE_LOCK``, so every
later profile-scoped request parks behind it.
"""

from __future__ import annotations

import concurrent.futures
import os
import sys
import time
from pathlib import Path

import pytest
import yaml

from tests.e2e.core.dashboard._helpers import make_sandbox
from tests.e2e.core.dashboard._issue_helpers import Issue120527, PtyDashboard, xfail_known

INSTALL_DEADLINE_S = 30.0  # a healthy install (write config + spawn/probe a local server) takes ~2 s
FOLLOWUP_DEADLINE_S = 5.0
FIXTURE = Path(__file__).with_name("fixture_catalog_mcp.py")

# Open issues this file encodes. A strict xfail XPASSes (and fails) the moment the fix lands, so the
# entry is removed with the fix; ``raises`` excuses only the issue's own assertion class.
KNOWN: dict[str, tuple[str, type]] = {
    "test_catalog_install_from_tty_launched_dashboard_never_wedges_the_server": (
        "#120527 dashboard MCP catalog install reaches the interactive tool checklist on the serve "
        "worker thread when stdin is a TTY and wedges _SKILLS_PROFILE_LOCK", Issue120527),
}


def _write_catalog(root: Path, name: str) -> Path:
    catalog = root / "optional-mcps"
    (catalog / name).mkdir(parents=True)
    (catalog / name / "manifest.yaml").write_text(yaml.safe_dump({
        "manifest_version": 1,
        "name": name,
        "description": "Offline stdio fixture MCP for the dashboard e2e lane.",
        "source": "tests/e2e/core/dashboard/fixture_catalog_mcp.py",
        "transport": {"type": "stdio", "command": sys.executable, "args": [str(FIXTURE)]},
        "auth": {"type": "none"},
    }, sort_keys=False), encoding="utf-8")
    return catalog


def _ws_opens(url: str, timeout: float) -> str:
    from websockets.sync.client import connect
    try:
        with connect(url, open_timeout=timeout, max_size=None):
            return "accepted"
    except TimeoutError:
        return f"timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001 - reported verbatim in the assertion
        return f"{type(exc).__name__}: {exc}"


def _install_and_probe(tmp_path: Path, *, tty: bool, wedge_exc: type[AssertionError]) -> dict:
    """Install the fixture entry through the dashboard, then probe the server's liveness.

    Raises ``wedge_exc`` when the install misses its deadline or a follow-up request stalls;
    returns the installed ``mcp_servers.<name>`` block when the install answered 200."""
    sb = make_sandbox(tmp_path)
    name = f"dashfix-{sb.profiles['default'].tag}"
    catalog = _write_catalog(tmp_path, name)
    dash = PtyDashboard(sb, tmp_path / "dashboard.log", extra_env={"HERMES_OPTIONAL_MCPS": str(catalog)}, tty=tty)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="e2e-120527")
    try:
        stdin = dash.stdin_target()
        assert stdin.startswith("/dev/pts/") if tty else stdin == os.devnull, f"precondition: stdin is {stdin}"
        listed = dash.ok("GET", "/api/mcp/catalog")
        entries = {e["name"]: e for e in listed["entries"]}
        assert name in entries and not entries[name]["installed"], f"fixture entry not listed: {sorted(entries)}"
        # Sanity before the install: the follow-up probe is fast on an idle server.
        assert dash.request("GET", "/api/config", timeout=FOLLOWUP_DEADLINE_S * 4).status_code == 200

        t0 = time.monotonic()
        install = pool.submit(dash.request, "POST", "/api/mcp/catalog/install",
                              json={"name": name, "enable": True}, timeout=INSTALL_DEADLINE_S)
        try:
            resp = install.result(timeout=INSTALL_DEADLINE_S + 5)
            install_outcome = f"HTTP {resp.status_code} {resp.text[:300]}"
        except Exception as exc:  # noqa: BLE001 - httpx.ReadTimeout on the bug
            resp, install_outcome = None, f"{type(exc).__name__} after {time.monotonic() - t0:.1f}s"

        # Both follow-ups run concurrently, after the install either answered or timed out.
        config = pool.submit(dash.request, "GET", "/api/config", timeout=FOLLOWUP_DEADLINE_S)
        ws = pool.submit(_ws_opens, dash.ws_url("/api/ws", token=dash.token), FOLLOWUP_DEADLINE_S)
        try:
            config_outcome = f"HTTP {config.result(timeout=FOLLOWUP_DEADLINE_S + 5).status_code}"
        except Exception as exc:  # noqa: BLE001
            config_outcome = type(exc).__name__
        ws_outcome = ws.result(timeout=FOLLOWUP_DEADLINE_S + 10)

        if resp is None or config_outcome != "HTTP 200" or ws_outcome != "accepted":
            raise wedge_exc(
                f"dashboard wedged by an MCP catalog install (stdin={stdin}): "
                f"install -> {install_outcome} (deadline {INSTALL_DEADLINE_S}s); "
                f"GET /api/config -> {config_outcome}; /api/ws -> {ws_outcome}\n"
                f"--- dashboard log tail ---\n{dash.log_tail(1500)}")
        # The install answered: success installs the entry, an error must be a clear 4xx (never 5xx).
        assert resp.status_code == 200 or 400 <= resp.status_code < 500, install_outcome
        if resp.status_code != 200:
            assert resp.json().get("detail"), f"4xx without a detail message: {install_outcome}"
            return {}
        servers = sb.profiles["default"].config().get("mcp_servers") or {}
        assert name in servers, f"install reported ok but config.yaml has no {name}: {sorted(servers)}"
        listed = {e["name"]: e for e in dash.ok("GET", "/api/mcp/catalog")["entries"]}
        assert listed[name]["installed"] and listed[name]["enabled"], listed[name]
        return servers[name]
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        dash.close()
        sb.stop()


def test_catalog_install_from_headless_dashboard_completes_with_all_probed_tools(tmp_path: Path) -> None:
    """Control (enforced): the same install from a stdin=/dev/null dashboard answers 200 promptly,
    leaves the server responsive, and — the manifest declaring no default_enabled — writes no tool
    filter, so both probed fixture tools stay enabled."""
    block = _install_and_probe(tmp_path, tty=False, wedge_exc=AssertionError)
    assert block, "headless install did not answer 200"
    assert block.get("enabled") is True and "tools" not in block, f"unexpected tool filter: {block}"


@xfail_known(KNOWN, "test_catalog_install_from_tty_launched_dashboard_never_wedges_the_server")
def test_catalog_install_from_tty_launched_dashboard_never_wedges_the_server(tmp_path: Path) -> None:
    _install_and_probe(tmp_path, tty=True, wedge_exc=Issue120527)
