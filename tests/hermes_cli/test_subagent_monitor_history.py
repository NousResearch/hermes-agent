"""The Ctrl+T list keeps finished subagents; the dock above the composer stays live-only."""
from types import SimpleNamespace

import pytest


@pytest.fixture
def store(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    db.create_session("master", "cli")
    for child, goal in (("finished-child", "Write the docs"), ("live-child", "Run the tests")):
        db.create_session(child, source="subagent", parent_session_id="master",
                          model_config={"_delegate_from": "master"})
        db.append_message(child, "user", goal)
    db.end_session("finished-child", "agent_close")
    yield db
    db.close()


@pytest.fixture
def monitor(store, monkeypatch):
    from hermes_cli.cli_subagent_monitor import SubagentMonitor
    from tools import delegate_tool_registry as registry

    monkeypatch.setattr(registry, "_active_subagents", {})
    child_agent = SimpleNamespace(session_id="live-child", steer=lambda text: True)
    registry._register_subagent(dict(subagent_id="sa-live", goal="Run the tests", owner_agent_session_id="master",
                                     started_at=1, status="running", agent=child_agent))
    cli = SimpleNamespace(agent=SimpleNamespace(session_id="master"), session_id="master", _session_db=store)
    dock = SubagentMonitor(cli)
    dock.refresh(history=True)
    return dock


def test_the_list_keeps_finished_subagents_after_live_ones(monitor):
    keys = [monitor._key(row) for row in monitor.roster]

    assert keys == ["sa-live", "finished-child"]
    finished = monitor.roster[1]
    assert finished["kind"] == "finished"
    assert finished["goal"] == "Write the docs"
    assert finished["child_session_id"] == "finished-child"


def test_the_list_heading_counts_live_and_finished_subagents(monitor):
    from agent.i18n import t

    heading = monitor.list_title()

    assert heading.startswith(t("cli.subagents.title_subagents"))
    assert t("cli.subagents.count_live", count=1) in heading
    assert t("cli.subagents.count_finished", count=1) in heading


def test_a_list_of_only_finished_subagents_is_headed_subagents(monitor):
    from agent.i18n import t

    monitor.entries = []

    assert monitor.list_title().startswith(t("cli.subagents.title_subagents"))


def test_a_live_subagent_is_listed_once_with_its_stored_session(monitor):
    live = monitor.roster[0]

    assert live["child_session_id"] == "live-child"
    assert sum(1 for row in monitor.roster if row.get("child_session_id") == "live-child") == 1


def test_the_dock_shows_only_live_subagents(monitor):
    text = monitor.dock_text(columns=100, rows=40)

    assert "Run the tests" in text
    assert "Write the docs" not in text


def test_a_finished_subagent_cannot_be_steered(monitor):
    result = monitor.control("steer", "keep going", target="finished-child")

    assert "error" in result


def test_history_is_left_out_unless_asked_for(monitor):
    monitor.refresh()

    assert [monitor._key(row) for row in monitor.roster] == ["sa-live"]


def test_the_gateway_backed_cli_store_serves_finished_subagents(store, monkeypatch):
    """The classic CLI's ``_session_db`` is the gateway session store, which only serves listed
    read methods from its read-only reader."""
    from hermes_cli.cli_subagent_monitor import SubagentMonitor
    from hermes_cli.gateway_session_store import GatewaySessionStore
    from tools import delegate_tool_registry as registry

    monkeypatch.setattr(registry, "_active_subagents", {})
    monkeypatch.setenv("HERMES_HOME", str(store.db_path.parent))
    idle = SimpleNamespace(pending=[], queued_at={}, queued_inputs={}, pending_seq=0)  # no queued prompts
    gateway_store = GatewaySessionStore(SimpleNamespace(final=False, state=lambda session_id: idle))
    cli = SimpleNamespace(agent=SimpleNamespace(session_id="master"), session_id="master", _session_db=gateway_store)
    dock = SubagentMonitor(cli)

    dock.refresh(history=True)

    assert {row["child_session_id"] for row in dock.finished} == {"finished-child", "live-child"}
    gateway_store.close_reader()


def test_no_session_store_means_live_rows_only(monkeypatch):
    from hermes_cli.cli_subagent_monitor import SubagentMonitor
    from tools import delegate_tool_registry as registry

    monkeypatch.setattr(registry, "_active_subagents", {})
    dock = SubagentMonitor(SimpleNamespace(agent=SimpleNamespace(session_id="master"), session_id="master"))

    dock.refresh(history=True)

    assert dock.roster == []
