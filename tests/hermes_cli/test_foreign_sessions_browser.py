"""Discovery keeps readable sessions available when a neighboring log is inaccessible."""

import json
from pathlib import Path

import pytest

import hermes_cli.foreign_sessions as foreign_sessions
import hermes_cli.foreign_sessions_browser as foreign_sessions_browser
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


def test_cowork_source_is_available_only_when_its_local_folder_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    cowork_root = tmp_path / "cowork"

    def roots(source):
        return [cowork_root] if source == "cowork" else []

    monkeypatch.setattr(foreign_sessions, "_source_roots", roots)
    monkeypatch.setattr(foreign_sessions_browser, "_source_roots", roots)
    assert "cowork" not in list_foreign_sessions()["sources"]

    transcript = (cowork_root / "account" / "workspace" / "local_session" /
                  ".claude" / "projects" / "local_shadow" / "session.jsonl")
    transcript.parent.mkdir(parents=True)
    transcript.write_text(json.dumps({
        "type": "user", "sessionId": "cowork-1", "cwd": "/workspace",
        "message": {"role": "user", "content": "Continue this Cowork task"},
    }), encoding="utf-8")
    transcript.parents[3].with_suffix(".json").write_text(json.dumps({
        "title": "Quarterly report",
        "userSelectedFolders": ["/work/Client Alpha", "/work/Shared Assets", "/work/" + "x" * 500],
    }), encoding="utf-8")

    page = list_foreign_sessions("cowork")
    assert "cowork" in page["sources"]
    assert [(row["source"], row["label"]) for row in page["sessions"]] == [
        ("cowork", "Claude Cowork")
    ]
    assert page["sessions"][0]["title"] == "Quarterly report"
    assert page["sessions"][0]["project"].startswith("Client Alpha, Shared Assets")
    assert len(page["sessions"][0]["project"]) == 180
