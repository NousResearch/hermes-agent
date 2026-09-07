"""Provisional closeout records stay durable but never become visible replies."""
import copy

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("surface", ["tui", "dashboard", "api", "cli"])
@pytest.mark.parametrize("kind", ["delegation_closeout_provisional", "delegation_waiting"])
def test_provisional_is_hidden_until_revealed_without_mutating_history(surface, kind):
    from gateway.platforms.api_server import _project_client_message
    from hermes_cli.cli_agent_setup_mixin import _collect_resume_entries
    from hermes_cli.web_routers.sessions import _project_for_display
    from tui_gateway import server

    with SessionDB() as db:
        db.create_session("closeout-display", source="tui")
        db.append_message(
            "closeout-display", "assistant", "Verified final answer",
            display_kind=kind,
            display_metadata={"hidden": True, "work_id": "work", "delivery_id": "delivery"},
        )
        history = db.get_messages_as_conversation("closeout-display")
    original = copy.deepcopy(history)

    def visible(rows):
        if surface == "tui":
            return [m["text"] for m in server._history_to_messages(rows)]
        if surface == "dashboard":
            return [m["content"] for m in _project_for_display(rows) if m.get("display_kind") != "hidden"]
        if surface == "api":
            return [m["content"] for m in map(_project_client_message, rows) if m.get("display_kind") != "hidden"]
        return [entry[1] for entry in _collect_resume_entries(rows, {}, lambda text: text)[0]]

    assert visible(history) == []
    assert history == original
    history[0].pop("display_kind")
    history[0].pop("display_metadata")
    assert "Verified final answer" in visible(history)
