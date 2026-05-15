from __future__ import annotations

from argparse import Namespace
import json

from hermes_cli.runtime_events import (
    append_runtime_event,
    iter_runtime_events,
    run_runtime_events,
)


def test_runtime_events_roundtrip_and_filter(tmp_path):
    append_runtime_event(
        kind="tool",
        status="fail",
        name="send_message",
        detail="delivery error",
        hermes_home=tmp_path,
    )
    append_runtime_event(
        kind="job",
        status="pass",
        name="daily-report",
        detail="ok",
        hermes_home=tmp_path,
    )

    events = iter_runtime_events(hermes_home=tmp_path, limit=10)
    assert [event["kind"] for event in events] == ["tool", "job"]

    failed_tools = iter_runtime_events(hermes_home=tmp_path, kind="tool", status="fail")
    assert len(failed_tools) == 1
    assert failed_tools[0]["name"] == "send_message"
    assert failed_tools[0]["detail"] == "delivery error"


def test_runtime_events_cli_json(monkeypatch, tmp_path, capsys):
    append_runtime_event(
        kind="tool",
        status="error",
        name="browser",
        detail="timeout",
        hermes_home=tmp_path,
    )

    monkeypatch.setattr("hermes_cli.runtime_events.get_hermes_home", lambda: tmp_path)
    run_runtime_events(Namespace(limit=5, kind=None, status=None, json=True))

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload[0]["kind"] == "tool"
    assert payload[0]["status"] == "error"
