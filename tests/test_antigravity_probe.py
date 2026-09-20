"""Behavior tests for the standalone Antigravity protocol probe."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "antigravity_probe.py"
SPEC = importlib.util.spec_from_file_location("antigravity_probe", SCRIPT)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def test_parse_ndjson_collects_events_and_non_json_diagnostics() -> None:
    parsed = probe.parse_ndjson(
        '{"event":"init","conversation_id":"c1"}\n'
        'warning: not JSON\n'
        '{"event":"result","result":{"status":"SUCCESS"}}\n'
    )

    assert [event["event"] for event in parsed.events] == ["init", "result"]
    assert parsed.diagnostics == ["warning: not JSON"]


def test_summarize_stream_exposes_observed_contract_fields() -> None:
    parsed = probe.parse_ndjson(
        '{"event":"init","conversation_id":"c1","init":'
        '{"cwd":"/tmp/probe","tools":["run_command"],'
        '"permission_mode":"request-review"}}\n'
        '{"event":"step_update","step_update":{"step_type":"tool",'
        '"state":"ERROR","tool_name":"run_command"}}\n'
        '{"event":"result","result":{"conversation_id":"c1",'
        '"status":"SUCCESS","denied_actions":[{"action":"command"}]}}\n'
    )

    summary = probe.summarize_stream(parsed)

    assert summary["conversation_ids"] == ["c1"]
    assert summary["event_types"] == ["init", "result", "step_update"]
    assert summary["step_types"] == ["tool"]
    assert summary["tool_names"] == ["run_command"]
    assert summary["permission_modes"] == ["request-review"]
    assert summary["denied_actions"] == ["command"]
