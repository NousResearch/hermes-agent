"""The auth recovery copy must obey the original store's read policy."""
import json

import pytest


@pytest.fixture
def quarantined(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    from hermes_cli.auth import _load_auth_store

    original = home / "auth.json"
    original.write_text('{"providers":{"test":{"access_token":"QUARANTINE_CANARY_121278"', encoding="utf-8")
    assert _load_auth_store(original)["providers"] == {}
    copy = home / "auth.json.corrupt"
    assert copy.read_bytes() == original.read_bytes()
    return home, copy


def test_recovery_copy_read_refused(quarantined):
    from tools.file_tools import read_file_tool

    _, copy = quarantined
    result = json.loads(read_file_tool(str(copy), task_id="quarantine-read"))
    assert "credential store" in result.get("error", ""), result
    assert "QUARANTINE_CANARY_121278" not in json.dumps(result)


@pytest.mark.parametrize("output_mode", ["content", "files_only", "count"])
def test_directory_search_omits_recovery_copy(quarantined, output_mode):
    from tools.file_tools import search_tool

    home, copy = quarantined
    (home / "notes.txt").write_text("PUBLIC_MATCH", encoding="utf-8")
    result = json.loads(search_tool(pattern="QUARANTINE_CANARY_121278|PUBLIC_MATCH", path=str(home),
                                   output_mode=output_mode, task_id="quarantine-search"))
    assert not result.get("error"), result
    raw = json.dumps(result)
    assert "notes.txt" in raw, result
    assert "auth.json.corrupt" not in raw, result
    assert "QUARANTINE_CANARY_121278" not in raw
    assert "QUARANTINE_CANARY_121278" in copy.read_text(encoding="utf-8")


def test_direct_search_refused(quarantined):
    from tools.file_tools import search_tool

    _, copy = quarantined
    result = json.loads(search_tool(pattern=".", path=str(copy), task_id="quarantine-direct"))
    assert "credential store" in result.get("error", ""), result


@pytest.mark.parametrize("location", ["active", "root"])
def test_profile_and_root_copy_protected(tmp_path, monkeypatch, location):
    from agent import file_safety

    root = tmp_path / "hermes"
    active = root / "profiles" / "test-profile"
    active.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(active))
    monkeypatch.setattr(file_safety, "_hermes_root_path", lambda: root)
    target = (active if location == "active" else root) / "auth.json.corrupt"
    target.write_text("fixture", encoding="utf-8")
    assert file_safety.get_read_block_error(str(target))


@pytest.mark.parametrize("relative", ["auth.json.corrupt", "fixtures/auth.json.corrupt", "auth.json.corrupt-notes"])
def test_project_fixtures_remain_readable(tmp_path, monkeypatch, relative):
    from agent.file_safety import get_read_block_error

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    fixture = tmp_path / "project" / relative
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text("public fixture", encoding="utf-8")
    assert get_read_block_error(str(fixture)) is None
