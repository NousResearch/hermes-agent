"""``presence_for(names)``: catalog plugins by name for the desktop questionnaire — unknown and removed
names dropped, other-OS entries dropped, order kept, manifests read in parallel, ``unknown`` when the
pinned manifest cannot be read."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import httpx
import pytest

from hermes_cli import plugin_catalog as pc
from hermes_cli import plugin_catalog_presence as presence_mod

SHA = "0" * 40
NEEDS_APP = {"extensions": {"com.nousresearch.hermes": {"servers": {"srv": {
    "app": {"darwin": {"presence": "executable", "location": "/nonexistent/fx-app"},
            "linux": {"presence": "executable", "location": "/nonexistent/fx-app"},
            "win32": {"presence": "executable", "location": "C:/nonexistent/fx-app.exe"}},
    "requires": {"app": True}}}}}}


def _entry(name, *, platforms=(), title="", description=""):
    return pc.entry_from_mapping({"name": name, "repo": f"https://github.com/fx/{name}", "sha": SHA,
                                  "description": description or f"{name} does things.", "maintainer": "fx",
                                  "tier": "official", "category": "tools", "platforms": list(platforms),
                                  "title": title}, name)


@pytest.fixture
def catalog(monkeypatch):
    from hermes_platform.host import facts

    here = {"darwin": "macos", "win32": "windows"}.get(facts.os_family(), "linux")
    other = "windows" if here != "windows" else "macos"
    entries = [
        _entry("app", title="Fx App", platforms=[here],
               description="Drives Fx. Disclosure: the tools change Fx settings; nothing leaves loopback. Needs Fx 2."),
        _entry("dashed", description="Pane. Disclosure \u2014 reads the cron file under HERMES_HOME."),
        _entry("broken", title="Broken"),
        _entry("elsewhere", platforms=[other]),
    ]
    monkeypatch.setattr(pc, "load_catalog_live", lambda: entries)
    monkeypatch.setattr(presence_mod, "_manifests", {})
    # Every fetch waits for the other two: a serial reader would break the barrier and read nothing.
    barrier = threading.Barrier(3, timeout=5)
    manifests = {"app": NEEDS_APP, "dashed": {"name": "dashed"}}

    def get(url, **kwargs):
        barrier.wait()
        repo = url.split("/")[4]
        if repo not in manifests:
            raise httpx.ConnectError("unreachable")
        return SimpleNamespace(status_code=200, content=json.dumps(manifests[repo]).encode())

    monkeypatch.setattr(httpx, "get", get)
    return entries


def test_rows_in_the_order_asked_without_removed_or_other_os_names(catalog):
    rows = presence_mod.presence_for(["broken", "removed-from-catalog", "elsewhere", "app", "dashed", "app"])

    assert [row["name"] for row in rows] == ["broken", "app", "dashed"]


def test_row_shape_state_and_disclosure(catalog):
    rows = {row["name"]: row for row in presence_mod.presence_for(["app", "dashed", "broken"])}

    assert rows["app"] == {"name": "app", "title": "Fx App", "state": "missing_app", "sentence": "needs Fx App",
                           "disclosure": "the tools change Fx settings; nothing leaves loopback."}
    assert rows["dashed"]["title"] == "dashed"
    assert rows["dashed"]["disclosure"] == "reads the cron file under HERMES_HOME."
    assert rows["dashed"]["state"] == "unknown"  # a manifest without a declared app
    assert rows["broken"]["state"] == "unknown" and rows["broken"]["sentence"] == ""  # manifest unreadable
    assert rows["broken"]["disclosure"] == ""


def test_plugins_manage_presence_over_the_wire(catalog):
    import tui_gateway.server as server

    response = server.handle_request({"jsonrpc": "2.0", "id": 9, "method": "plugins.manage",
                                      "params": {"action": "presence", "names": ["dashed", "app", "broken"]}})

    assert [row["name"] for row in response["result"]["presence"]] == ["dashed", "app", "broken"]
