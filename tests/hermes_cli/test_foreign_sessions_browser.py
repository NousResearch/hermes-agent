"""Discovery keeps readable sessions available when a neighboring log is inaccessible."""

import json
from pathlib import Path

import pytest

from hermes_cli.foreign_sessions_browser import list_foreign_sessions


@pytest.mark.parametrize("operation", ["resolve", "stat"])
def test_discovery_skips_inaccessible_log(tmp_path, monkeypatch, operation):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    folder = tmp_path / ".codex" / "sessions"
    folder.mkdir(parents=True)
    for name in ("readable", "inaccessible"):
        (folder / f"rollout-{name}.jsonl").write_text(json.dumps({
            "type": "response_item", "payload": {"type": "message", "role": "user",
            "content": [{"type": "input_text", "text": name}]},
        }), encoding="utf-8")

    original = getattr(Path, operation)

    def access(path, *args, **kwargs):
        if path.name == "rollout-inaccessible.jsonl":
            raise PermissionError("Log access denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, operation, access)
    page = list_foreign_sessions("codex")
    assert [session["title"] for session in page["sessions"]] == ["readable"]
