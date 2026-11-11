"""An ACP client sees what a ``skill_manage`` call in the advertised ``operations[]`` shape does.

The tool takes ``operations[]`` (the flat top-level ``action``/``name`` shape is legacy). The call goes
through the ACP tool-progress callback the agent drives (``make_tool_progress_cb``) and the real tool,
against the test's temp HERMES_HOME; the client's ``session_update`` is recorded.
"""

import asyncio
import json
import threading

import pytest

from acp_adapter.events import make_tool_progress_cb
from tools.skill_manager_tool import skill_manage

SKILL = "---\nname: deploy-notes\ndescription: Use when deploying. Steps.\n---\n# Deploy\nStep 1.\n"


class RecordingClient:
    def __init__(self):
        self.updates = []

    async def session_update(self, session_id, update):
        self.updates.append(update)


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    runner = threading.Thread(target=loop.run_forever, daemon=True)
    runner.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    runner.join(5)
    loop.close()


def _diffs(update):
    return [(c.path, c.old_text, c.new_text) for c in update.content or [] if getattr(c, "type", "") == "diff"]


def test_operations_patch_shows_the_skill_and_its_diff(loop, monkeypatch):
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")
    created = json.loads(skill_manage(action="", name="", operations=[
        {"action": "create", "name": "deploy-notes", "content": SKILL}]))
    assert created["success"], created
    client = RecordingClient()
    progress = make_tool_progress_cb(client, "s1", loop, {}, {})
    ops = [{"action": "patch", "name": "deploy-notes", "old_string": "Step 1.", "new_string": "Step 2."}]

    progress("tool.started", "skill_manage", None, {"operations": ops})
    result = skill_manage(action="", name="", operations=ops)
    progress("tool.completed", "skill_manage", None, None, result=result)

    start, done = client.updates
    assert done.status == "completed"
    seen = {
        "title": start.title,
        "proposed": _diffs(start),
        "applied": [(path.endswith("deploy-notes/SKILL.md"), "Step 1." in old, "Step 2." in new)
                    for path, old, new in _diffs(done)],
    }
    assert seen == {
        "title": "skill_manage: deploy-notes",
        "proposed": [("skills/deploy-notes/SKILL.md", "Step 1.", "Step 2.")],
        "applied": [(True, True, True)],
    }
