"""Backend ownership must not follow the UI's initial profile (#133922)."""

from __future__ import annotations

import asyncio
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import process_identity, update_inventory, web_server
from hermes_constants import hermes_home_key


@pytest.mark.parametrize("headless", [False, True], ids=["dashboard", "serve"])
@pytest.mark.parametrize(
    ("owner", "selected"),
    [
        ("profile-a", ""),
        ("profile-a", "profile-a"),
        ("profile-a", "profile-b"),
        ("default", ""),
        ("default", "default"),
        ("default", "profile-a"),
    ],
)
def test_post_bind_ledger_and_update_plan_keep_backend_owner(
    tmp_path,
    monkeypatch,
    capsys,
    headless,
    owner,
    selected,
):
    home = tmp_path / "home"
    root = home / ".hermes"
    root.mkdir(parents=True)
    (root / "config.yaml").write_text("{}", encoding="utf-8")
    for name in ("profile-a", "profile-b"):
        profile = root / "profiles" / name
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text("{}", encoding="utf-8")
    backend_home = root if owner == "default" else root / "profiles" / owner
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(backend_home))
    for name in (
        "HERMES_DESKTOP",
        "HERMES_PARENT_PID",
        "HERMES_PARENT_START_MARKER",
        "HERMES_SPAWN",
    ):
        monkeypatch.delenv(name, raising=False)

    # Keep real registration, disk I/O and process verification; unrelated boot
    # services must not touch the developer's running backends or open a browser.
    monkeypatch.setattr(process_identity, "reap_orphaned_mcp_helpers", lambda: None)
    monkeypatch.setattr(
        process_identity, "attach_self_to_kill_on_close_job", lambda: None
    )
    monkeypatch.setattr(web_server, "_start_parent_death_watchdog", lambda: None)
    monkeypatch.setattr(web_server, "_publish_host_rendezvous", lambda *_: None)
    ready = Mock()
    browser = Mock()
    monkeypatch.setattr(web_server, "_write_dashboard_ready_file", ready)
    monkeypatch.setattr(web_server, "_write_machine_sentinel_line", lambda *_: None)
    monkeypatch.setattr(web_server, "_maybe_open_browser", browser)
    state = SimpleNamespace(initial_profile=selected)
    monkeypatch.setattr(web_server.app, "state", state)

    async def start(sock):
        server = SimpleNamespace(servers=[SimpleNamespace(sockets=[sock])])
        web_server._on_server_started(
            server,
            host="127.0.0.1",
            port=0,
            headless=headless,
            isolated=True,
            open_browser=True,
            initial_profile=selected,
            start_mcp_discovery_after_bind=False,
        )

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        asyncio.run(start(sock))

    entries = process_identity.ledger_entries(verified_only=True)
    own = [entry for entry in entries if entry["pid"] == os.getpid()]
    assert len(own) == 1
    entry = own[0]
    purpose = "serve" if headless else "dashboard"
    assert entry["purpose"] == purpose
    assert entry["hermes_home"] == hermes_home_key(backend_home)
    assert (entry["host"], entry["port"], entry["isolated"]) == (
        "127.0.0.1",
        port,
        True,
    )
    assert (entry["profile"] or "default") == owner
    assert process_identity._ledger_path().is_relative_to(root)
    assert state.initial_profile == selected
    assert state.serves_spa is not headless
    ready.assert_called_once_with(port)
    browser.assert_called_once_with("127.0.0.1", port, True, selected)

    plan = update_inventory.UpdatePlan()
    update_inventory._collect_ledger_runtimes(plan, set())
    [runtime] = [runtime for runtime in plan.runtimes if runtime.pid == os.getpid()]
    assert (runtime.kind, runtime.profile, runtime.supervisor) == (
        purpose,
        owner,
        "manual-serve",
    )
    assert runtime.detail["port"] == port
    capsys.readouterr()
    update_inventory.print_update_plan(plan)
    assert f"{purpose} [{owner}] pid {os.getpid()}" in capsys.readouterr().out
