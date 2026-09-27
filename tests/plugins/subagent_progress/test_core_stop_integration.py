import json

import pytest

from test_progress import setup, report
from test_supervision import supervised
from tools.delegate_tool_results import _fire_subagent_stop_hooks


@pytest.mark.parametrize("status", ["completed", "timeout", "interrupted"])
def test_real_core_stop_emitter_retains_terminal_checkpoint(supervised, monkeypatch, status):
    plugin, ctx, parent, child, sup, clock = supervised
    assert report(plugin, completed="SOURCE_INSPECTED")["success"]
    child.session_id = "rotated-child-session"

    def emit(name, **kwargs):
        assert name == "subagent_stop"
        plugin.stop(**kwargs)

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", emit)
    entry = {"task_index": 0, "status": status, "summary": ""}
    _fire_subagent_stop_hooks([entry], {0: child}, parent)
    with plugin.db() as db:
        stored_status = db.execute("SELECT status FROM children WHERE subagent=?", (child._subagent_id,)).fetchone()[0]
        payloads = [json.loads(r[0]) for r in db.execute("SELECT payload FROM reports")]
    assert stored_status == status
    terminal = [p for p in payloads if p["kind"] == "terminal_checkpoint"]
    assert len(terminal) == 1
    assert terminal[0]["completed"] == "SOURCE_INSPECTED"
    assert terminal[0]["status"] == status
    _fire_subagent_stop_hooks([entry], {0: child}, parent)
    with plugin.db() as db:
        assert db.execute("SELECT COUNT(*) FROM reports WHERE json_extract(payload, '$.kind')='terminal_checkpoint'").fetchone()[0] == 1
