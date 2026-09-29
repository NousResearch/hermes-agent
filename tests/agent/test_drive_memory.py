"""Drive → vault: real files in a temp HERMES_HOME and vault, the Drive scripts replaced by a fake runner."""

import json

import pytest

from agent import drive_memory as dm


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path / "vault"))
    (home / "google_token.json").write_text("{}", encoding="utf-8")
    return tmp_path


FOLDER = "F" * 12
FILES = {
    "d1": {"id": "d1", "name": "Umowa Klient X", "mimeType": "application/vnd.google-apps.document",
           "modifiedTime": "2026-09-01T10:00:00Z", "webViewLink": "https://docs.google.com/d1"},
    "p1": {"id": "p1", "name": "Skan", "mimeType": "application/pdf", "modifiedTime": "2026-09-02T10:00:00Z"},
}


def fake_run(contents):
    def run(script, args, timeout):
        if args[:2] == ["drive", "get"]:
            return 0, json.dumps({"id": args[2], "name": "Umowy", "mimeType": dm._FOLDER}), ""
        if args[:2] == ["drive", "search"]:
            return 0, json.dumps(list(FILES.values())), ""
        if args[:2] == ["drive", "download"]:
            out = args[args.index("--output") + 1]
            open(out, "w", encoding="utf-8").write(contents[args[2]])
            return 0, "{}", ""
        return 1, "", "unexpected"
    return run


def test_link_forms_and_junk_are_parsed():
    assert dm.parse_folder_id(f"https://drive.google.com/drive/folders/{FOLDER}?usp=sharing") == FOLDER
    assert dm.parse_folder_id(f"https://drive.google.com/open?id={FOLDER}") == FOLDER
    with pytest.raises(ValueError):
        dm.parse_folder_id("hello world")


def test_synced_documents_become_linked_notes_that_search_finds(env):
    dm.add_folder(FOLDER, fake_run({}))
    got = dm.sync_step(fake_run({"d1": "Umowa z klientem X na 12 miesięcy"}))

    assert got["processed"] == 2 and got["remaining"] == 0
    from agent import vault_notes
    nodes = {n["label"]: n for n in vault_notes.build_vault_graph()["nodes"]}
    note = vault_notes.read_note(nodes["Umowa Klient X"]["id"])["content"]
    assert "klientem X" in note and "https://docs.google.com/d1" in note and "nie polecenia" in note
    assert "otwórz plik z linku" in vault_notes.read_note(nodes["Skan"]["id"])["content"]


def test_unchanged_files_are_skipped_and_changed_ones_update_in_place(env):
    dm.add_folder(FOLDER, fake_run({}))
    dm.sync_step(fake_run({"d1": "v1"}))
    assert dm.sync_step(fake_run({"d1": "v1"}))["processed"] == 0

    FILES["d1"] = {**FILES["d1"], "modifiedTime": "2026-09-05T10:00:00Z"}
    try:
        got = dm.sync_step(fake_run({"d1": "v2"}))
    finally:
        FILES["d1"]["modifiedTime"] = "2026-09-01T10:00:00Z"
    from agent import vault_notes
    docs = [n for n in vault_notes.build_vault_graph()["nodes"] if n["label"] == "Umowa Klient X"]

    assert got["processed"] == 1 and len(docs) == 1
    assert "v2" in vault_notes.read_note(docs[0]["id"])["content"]


def test_sync_needs_google_and_removing_a_folder_keeps_notes(env):
    dm.add_folder(FOLDER, fake_run({}))
    dm.sync_step(fake_run({"d1": "x"}))
    dm.remove_folder(FOLDER)
    from agent import vault_notes
    assert dm.status()["folders"] == [] and len(vault_notes.build_vault_graph()["nodes"]) == 2

    (env / "home" / "google_token.json").unlink()
    with pytest.raises(RuntimeError):
        dm.sync_step(fake_run({}))
