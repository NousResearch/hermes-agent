"""``transform_persisted_row``: an embedder may project each session-db row at flush."""

import types

from agent.session_persistence import _db_flush_row
from hermes_cli.plugins import get_plugin_manager


def test_persisted_row_hook_sees_each_row_once_with_its_index(monkeypatch):
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    seen = []

    def _project(agent, message, row, msg_idx):
        seen.append(msg_idx)
        return {**row, "content": f"[{msg_idx}] {row['content']}"}

    mgr._hooks.setdefault("transform_persisted_row", []).append(_project)
    try:
        row = _db_flush_row(types.SimpleNamespace(), {"role": "assistant", "content": "hi"}, False, 3)
    finally:
        mgr._hooks = saved
    assert seen == [3]
    assert row["content"] == "[3] hi"
