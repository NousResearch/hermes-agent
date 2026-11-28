"""Plugin events socket ticket semantics.

Run:  PYTHONPATH=<repo> python3 tests/hermes_cli/test_live_sync_tickets.py
   or: scripts/run_tests.sh tests/hermes_cli/test_live_sync_tickets.py

Covers the server half of the kanban live-sync fix: a ``plugin-events/<id>`` socket opens with a
ticket minted for exactly that name — never with another plugin's ticket or a core-socket ticket
(and vice versa) — while an unnamed legacy ticket keeps opening it for the classic dashboard
client. Name construction confines plugin ids so a ticket's bound path can never smuggle a
separator or traversal.
"""

import sys

from hermes_cli.web_session_sockets import (
    is_plugin_events_socket,
    plugin_events_socket,
    ticket_refusal,
)


def test_socket_names() -> None:
    assert plugin_events_socket("kanban") == "plugin-events/kanban"
    for bad in ("../ws", "a/b", "/kanban", "-x", "_x", "", None, "Kanban"):
        try:
            plugin_events_socket(bad)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"plugin_events_socket accepted {bad!r}")
    assert is_plugin_events_socket("plugin-events/kanban")
    assert is_plugin_events_socket("plugin-events/team_hub-2")
    for no in ("gateway", "plugin-events/", "plugin-events/a/b", "plugin-events/../ws",
               "plugin-events", "Plugin-Events/kanban", "", None):
        assert not is_plugin_events_socket(no), no


def test_named_ticket_opens_only_its_socket() -> None:
    named = {"socket": "plugin-events/kanban"}
    assert ticket_refusal(named, "plugin-events/kanban", auth_required=False) is None
    assert ticket_refusal(named, "plugin-events/other", auth_required=False) is not None
    assert ticket_refusal(named, "gateway", auth_required=False) is not None
    assert ticket_refusal(named, None, auth_required=False) is not None


def test_unnamed_legacy_ticket_keeps_opening_plugin_events() -> None:
    # The classic dashboard client (web/src/lib/api.ts buildWsAuthParam) mints unnamed tickets;
    # plugin event streams keep opening for them, as before the names existed:
    assert ticket_refusal({}, "plugin-events/kanban", auth_required=False) is None
    assert ticket_refusal({}, "plugin-events/anything", auth_required=False) is None
    # Own-ticket sockets stay strict for unnamed tickets:
    assert ticket_refusal({}, "host-terminal", auth_required=False) is not None
    assert ticket_refusal({}, "relay", auth_required=False) is not None


def test_core_socket_ticket_opens_no_plugin_events_socket() -> None:
    gw = {"socket": "gateway"}
    assert ticket_refusal(gw, "gateway", auth_required=False) is None
    assert ticket_refusal(gw, "plugin-events/kanban", auth_required=False) is not None


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    if failures:
        sys.exit(1)
    print("ALL TICKET TESTS PASSED")
