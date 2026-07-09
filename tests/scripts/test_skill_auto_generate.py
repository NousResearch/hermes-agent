"""Tests for skill-auto-generate.py."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPT = Path(r"C:\Users\bbask\AppData\Local\hermes\scripts\skill-auto-generate.py")
sys.path.insert(0, str(SCRIPT.parent))
import importlib.util
spec = importlib.util.spec_from_file_location("sag", SCRIPT)
sag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sag)


# --- load_inventory ---

def test_load_inventory_returns_dict(tmp_path, monkeypatch):
    inv = tmp_path / "inv.json"
    inv.write_text('{"bash": {"path": "/usr/bin/bash"}}')
    monkeypatch.setattr(sag, "INVENTORY_PATH", inv)
    result = sag.load_inventory()
    assert "bash" in result


def test_load_inventory_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sag, "INVENTORY_PATH", tmp_path / "missing.json")
    assert sag.load_inventory() == {}


def test_load_inventory_handles_corrupt(tmp_path, monkeypatch):
    inv = tmp_path / "inv.json"
    inv.write_text("not json")
    monkeypatch.setattr(sag, "INVENTORY_PATH", inv)
    assert sag.load_inventory() == {}


# --- load_state / save_state ---

def test_load_state_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sag, "STATE_FILE", tmp_path / "state.json")
    assert sag.load_state() == {"scaffolded": {}}


def test_load_state_with_data(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    state.write_text('{"scaffolded": {"bash": {"at": "2026-07-05"}}}')
    monkeypatch.setattr(sag, "STATE_FILE", state)
    result = sag.load_state()
    assert "bash" in result["scaffolded"]


def test_save_state_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sag, "STATE_FILE", tmp_path / "state.json")
    sag.save_state({"scaffolded": {"foo": {"at": "now"}}})
    assert (tmp_path / "state.json").exists()
    loaded = json.loads((tmp_path / "state.json").read_text())
    assert "foo" in loaded["scaffolded"]


# --- list_existing_skills ---

def test_list_existing_skills(tmp_path, monkeypatch):
    monkeypatch.setattr(sag, "SKILLS_DIR", tmp_path)
    (tmp_path / "skill1").mkdir()
    (tmp_path / "skill2").mkdir()
    (tmp_path / "not_a_dir.txt").touch()
    skills = sag.list_existing_skills()
    assert "skill1" in skills
    assert "skill2" in skills
    assert "not_a_dir.txt" not in skills


def test_list_existing_skills_no_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sag, "SKILLS_DIR", tmp_path / "missing")
    assert sag.list_existing_skills() == set()


# --- detect_requirements ---

def test_detect_requirements_scoop():
    with patch.object(sag.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Name: jq\nVersion: 1.7", stderr="")
        result = sag.detect_requirements("jq")
    assert "scoop install jq" in result


def test_detect_requirements_npm():
    def fake_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        if "scoop" in cmd:
            return MagicMock(returncode=1, stdout="", stderr="not found")
        if "npm" in cmd:
            return MagicMock(returncode=0, stdout="tool@1.0.0", stderr="")
        return MagicMock(returncode=1, stdout="", stderr="")
    with patch.object(sag.subprocess, "run", side_effect=fake_run):
        result = sag.detect_requirements("tool")
    assert "npm install" in result


def test_detect_requirements_pip():
    def fake_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        if "scoop" in cmd or "npm" in cmd:
            return MagicMock(returncode=1, stdout="", stderr="not found")
        if "pip" in cmd:
            return MagicMock(returncode=0, stdout="Name: requests", stderr="")
        return MagicMock(returncode=1, stdout="", stderr="")
    with patch.object(sag.subprocess, "run", side_effect=fake_run):
        result = sag.detect_requirements("requests")
    assert "pip install requests" in result


def test_detect_requirements_unknown():
    def fake_run(*args, **kwargs):
        return MagicMock(returncode=1, stdout="", stderr="not found")
    with patch.object(sag.subprocess, "run", side_effect=fake_run):
        result = sag.detect_requirements("unknown-tool")
    assert "not auto-detected" in result


# --- generate_description ---

def test_generate_description_simple():
    desc = sag.generate_description("ripgrep")
    assert "ripgrep" in desc
    assert len(desc) <= 200


def test_generate_description_truncates_long():
    desc = sag.generate_description("a-very-long-tool-name-with-many-words")
    assert len(desc) <= 200


def test_generate_description_handles_underscores():
    desc = sag.generate_description("my_tool")
    assert "my" in desc or "tool" in desc


# --- scaffold_skill ---

def test_scaffold_skill_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sag, "SKILLS_DIR", tmp_path / "skills")
    with patch.object(sag, "detect_requirements", return_value="scoop install test-tool"):
        path = sag.scaffold_skill("test-tool")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "test-tool" in content
    assert "name: test-tool" in content
    assert "scoop install test-tool" in content


def test_scaffold_skill_handles_existing_dir(tmp_path, monkeypatch):
    """Existing skill dir should not be overwritten — but file is overwritten (acceptable)."""
    monkeypatch.setattr(sag, "SKILLS_DIR", tmp_path / "skills")
    (tmp_path / "skills" / "test-tool").mkdir(parents=True)
    (tmp_path / "skills" / "test-tool" / "SKILL.md").write_text("old")
    with patch.object(sag, "detect_requirements", return_value="scoop install test-tool"):
        path = sag.scaffold_skill("test-tool")
    # File is overwritten (acceptable — auto-regen)
    assert path.read_text(encoding="utf-8") != "old"


# --- main flow ---

def test_main_no_inventory_exits_0(tmp_path, monkeypatch):
    monkeypatch.setattr(sag, "LOG_FILE", tmp_path / "log.txt")
    monkeypatch.setattr(sag, "INVENTORY_PATH", tmp_path / "missing.json")
    r = sag.main()
    assert r == 0


def test_main_no_new_tools_exits_0(tmp_path, monkeypatch):
    monkeypatch.setattr(sag, "LOG_FILE", tmp_path / "log.txt")
    monkeypatch.setattr(sag, "INVENTORY_PATH", tmp_path / "inv.json")
    inv = tmp_path / "inv.json"
    inv.write_text('{"bash": {}}')
    monkeypatch.setattr(sag, "SKILLS_DIR", tmp_path / "skills")
    (tmp_path / "skills").mkdir(parents=True, exist_ok=True)
    (tmp_path / "skills" / "bash").mkdir()
    r = sag.main()
    assert r == 0


def test_main_with_new_tools_scaffolds(tmp_path, monkeypatch):
    monkeypatch.setattr(sag, "LOG_FILE", tmp_path / "log.txt")
    monkeypatch.setattr(sag, "INVENTORY_PATH", tmp_path / "inv.json")
    monkeypatch.setattr(sag, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(sag, "SKILLS_DIR", tmp_path / "skills")
    inv = tmp_path / "inv.json"
    # Format: {category: ["tool=path"]} (matches actual inventory format)
    inv.write_text('{"shells-build": ["newtool=C:/bin/newtool"]}')
    with patch.object(sag, "detect_requirements", return_value="scoop install newtool"):
        with patch.object(sag, "send_telegram", return_value=True) as mock_tg:
            r = sag.main()
    assert r == 0
    # Skill was created
    assert (tmp_path / "skills" / "newtool" / "SKILL.md").exists()
    # Telegram was called
    assert mock_tg.called
    # State was saved
    state = json.loads((tmp_path / "state.json").read_text())
    assert "newtool" in state["scaffolded"]


def test_main_skips_already_scaffolded(tmp_path, monkeypatch):
    monkeypatch.setattr(sag, "LOG_FILE", tmp_path / "log.txt")
    monkeypatch.setattr(sag, "INVENTORY_PATH", tmp_path / "inv.json")
    monkeypatch.setattr(sag, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(sag, "SKILLS_DIR", tmp_path / "skills")
    inv = tmp_path / "inv.json"
    inv.write_text('{"oldtool": {}}')
    state = tmp_path / "state.json"
    state.write_text('{"scaffolded": {"oldtool": {"at": "yesterday"}}}')
    with patch.object(sag, "send_telegram", return_value=True) as mock_tg:
        r = sag.main()
    assert r == 0
    # No skill created
    assert not (tmp_path / "skills" / "oldtool").exists()
    # No telegram
    assert not mock_tg.called