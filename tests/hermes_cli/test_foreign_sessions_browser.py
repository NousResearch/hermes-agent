"""Discovery keeps readable sessions available when a neighboring log is inaccessible."""

import base64
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


def test_grok_cache_discovery_normalization_and_import(tmp_path, monkeypatch):
    from hermes_state import SessionDB

    root = tmp_path / "cache"
    root.mkdir()
    monkeypatch.setattr(foreign_sessions, "_source_roots", lambda source: [root] if source == "grok" else [])
    monkeypatch.setattr(foreign_sessions_browser, "_source_roots", lambda source: [root] if source == "grok" else [])
    def blob(key):
        return root / (base64.b32encode(key.encode()).decode().rstrip("=").lower() + ".blob")

    # Unrelated slices must never be parsed, even with the same JSON envelope.
    blob("credentials").write_text("not a transcript", encoding="utf-8")
    assert "grok" not in list_foreign_sessions()["sources"]
    entry = {"kind": "send-message", "id": "reply", "message": {"type": "text", "content": "Answer"}}
    entries = [None, {"kind": "message", "id": [], "role": "user"},
               {"kind": "message", "id": "q", "role": "user", "content": "Question"},
               entry, entry,
               {"kind": "send-message", "id": "approval", "message": {"type": "local-tool-permission", "content": "omit"}},
               {"kind": "message", "id": "system", "role": "system", "content": "omit"},
               {"kind": "message", "id": "bad", "role": "user", "content": {}},
               {"kind": "message", "id": "q2", "role": "user", "content": "Next"}]
    log = blob("sand.client.slice.account.test.transcript.replicas.bot-id")
    log.write_text(json.dumps({"schemaVersion": 1, "value": {"entries": entries}}), encoding="utf-8")
    original = log.read_bytes()
    page = list_foreign_sessions("grok")
    assert "grok" in page["sources"]
    row, = page["sessions"]
    assert row["label"] == "Grok Bot"
    assert row["title"] == "Question"
    assert row["project"] is None
    roster = {"schemaVersion": 4, "value": {"rows": [{"id": "bot-id", "name": "  Test Bot  "}]}}
    blob("sand.client.slice.account.other.roster.last-roster").write_text(json.dumps(roster))
    assert list_foreign_sessions("grok")["sessions"][0]["project"] is None
    metadata = blob("sand.client.slice.account.test.roster.last-roster")
    metadata.write_text(json.dumps(roster))
    named, = list_foreign_sessions("grok")["sessions"]
    assert named["project"] == "Test Bot"
    assert named["title"] == row["title"]
    for invalid in ({"schemaVersion": 5, "value": roster["value"]},
                    {"schemaVersion": 4, "value": {"rows": [{"id": "other", "name": "Wrong Bot"}]}},
                    {"schemaVersion": 4, "value": {"rows": roster["value"]["rows"] * 2}}):
        metadata.write_text(json.dumps(invalid))
        assert list_foreign_sessions("grok")["sessions"][0]["project"] is None
    metadata.write_bytes(b" " * (1024 * 1024 + 1))
    assert list_foreign_sessions("grok")["sessions"][0]["project"] is None
    with SessionDB(tmp_path / "state.db") as db:
        preview = foreign_sessions_browser.preview_foreign_session(row["id"], db)
        assert preview["messages"] == [{"role": "user", "content": "Question"},
                                        {"role": "assistant", "content": "Answer"},
                                        {"role": "user", "content": "Next"}]
        first = foreign_sessions_browser.import_browser_session(row["id"], db, "default")
        assert foreign_sessions_browser.import_browser_session(row["id"], db, "default") == {**first, "already_imported": True}
        snapshot = foreign_sessions_browser.export_browser_session(row["id"])
        assert snapshot["origin"]["tool"] == "grok-bot"
        assert snapshot["messages"] == preview["messages"]
        assert str(log) not in json.dumps(snapshot)
        with SessionDB(tmp_path / "remote.db") as remote:
            result = foreign_sessions_browser.import_browser_snapshot(snapshot, remote, "default")
            sid = result["session_id"]
            assert [(m["role"], m["content"]) for m in remote.get_messages(sid)] == [
                (m["role"], m["content"]) for m in preview["messages"]]
    assert log.read_bytes() == original


def test_grok_cache_rejects_bad_schema_escape_and_remote_only_size_cap(tmp_path, monkeypatch):
    from hermes_state import SessionDB

    root = tmp_path / "cache"
    root.mkdir()
    monkeypatch.setattr(foreign_sessions, "_source_roots", lambda source: [root] if source == "grok" else [])
    monkeypatch.setattr(foreign_sessions_browser, "_source_roots", lambda source: [root] if source == "grok" else [])
    name = base64.b32encode(b"transcript.replicas/test").decode().rstrip("=") + ".blob"
    log = root / name
    log.write_text(json.dumps({"schemaVersion": 2, "value": {"entries": []}}), encoding="utf-8")
    assert list_foreign_sessions("grok")["unreadable"] == 1
    log.write_text(json.dumps({"schemaVersion": 1, "value": {"entries": [
        {"id": "large", "kind": "send-message", "message": {"type": "text", "content": "x" * (5 * 1024 * 1024)}}]}}), encoding="utf-8")
    row, = list_foreign_sessions("grok")["sessions"]
    with SessionDB(tmp_path / "state.db") as db:
        # The pre-existing SessionDB portability cap is independent of the source adapter.
        # Raising it proves the adapter does not add the remote snapshot cap locally.
        monkeypatch.setattr(db, "_IMPORT_MAX_SESSION_BYTES", 8 * 1024 * 1024)
        sid = foreign_sessions_browser.import_browser_session(row["id"], db, "default")["session_id"]
        assert len(db.get_messages(sid)[1]["content"]) == 5 * 1024 * 1024
    with pytest.raises(ValueError, match="cross-gateway"):
        foreign_sessions_browser.export_browser_session(row["id"])
    with log.open("wb") as stream:
        stream.truncate(32 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="32 MB"):
        foreign_sessions.parse_grok_session(log)
    outside = tmp_path / name
    log.rename(outside)
    with pytest.raises(ValueError, match="outside"):
        foreign_sessions.parse_grok_session(outside)
    assert foreign_sessions._walk("grok") == []


@pytest.mark.macos_only
def test_grok_discovery_confines_symlinks_to_verified_macos_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root, = foreign_sessions._source_roots("grok")
    root.mkdir(parents=True)
    name = base64.b32encode(b"transcript.replicas/test").decode().rstrip("=") + ".blob"
    outside = tmp_path / name
    outside.write_text("{}", encoding="utf-8")
    (root / name).symlink_to(outside)
    assert foreign_sessions._walk("grok") == []
    with pytest.raises(ValueError, match="outside"):
        foreign_sessions.parse_grok_session(root / name)
    assert foreign_sessions._source_roots("grok", platform="win32") == []
    assert foreign_sessions._source_roots("grok", platform="linux") == []
