"""Doctor diagnoses memory entries excluded from the prompt (#121753)."""

import pytest

from hermes_cli import doctor, doctor_state
from tools import memory_tool
from tools.memory_tool_store import MemoryStore


@pytest.mark.parametrize("filename,target", [("USER.md", "user"), ("MEMORY.md", "memory")])
def test_doctor_warns_when_runtime_blocks_entry(tmp_path, monkeypatch, capsys, filename, target):
    home = tmp_path / "home"
    memories = home / "memories"
    memories.mkdir(parents=True)
    (memories / filename).write_text("A useful fact\n§\nNever write ~/.hermes/SOUL.md", encoding="utf-8")
    monkeypatch.setattr(doctor, "HERMES_HOME", home)
    monkeypatch.setattr(doctor, "_DHH", str(home))
    monkeypatch.setattr(memory_tool, "get_memory_dir", lambda: memories)

    store = MemoryStore()
    store.load_from_disk()
    assert "[BLOCKED:" in store.format_for_system_prompt(target)
    finding = doctor_state._check_directory_structure(False)
    output = capsys.readouterr().out
    assert f"{filename}: 1 blocked entries" in output
    assert "hermes_config_mod" in output
    assert any(filename in issue and "hermes_config_mod" in issue for issue in finding.issues)
    assert f"{filename} exists (" not in output


def test_doctor_reports_drift_and_configured_cap_without_modifying_memory(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    memories = home / "memories"
    memories.mkdir(parents=True)
    path = memories / "USER.md"
    raw = "A long safe memory entry\n\n"
    path.write_text(raw, encoding="utf-8")
    (home / "config.yaml").write_text("memory:\n  user_char_limit: 8\n", encoding="utf-8")
    monkeypatch.setattr(doctor, "HERMES_HOME", home)
    monkeypatch.setattr(doctor, "_DHH", str(home))

    finding = doctor_state._check_directory_structure(False)
    output = capsys.readouterr().out
    assert "USER.md does not round-trip" in output
    assert "replace/remove/batch may refuse" in output
    assert "add can still rewrite this file without a drift backup" in output
    assert "USER.md exceeds its configured char limit" in output
    assert len([issue for issue in finding.issues if "USER.md" in issue]) == 2
    assert any("add may rewrite without a backup" in issue for issue in finding.issues)
    assert path.read_text(encoding="utf-8") == raw
    assert list(memories.iterdir()) == [path]  # doctor must not create a drift backup

def test_doctor_reports_unreadable_memory_file(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    memories = home / "memories"
    memories.mkdir(parents=True)
    path = memories / "USER.md"
    raw = b"\xff\xfe"
    path.write_bytes(raw)
    monkeypatch.setattr(doctor, "HERMES_HOME", home)
    monkeypatch.setattr(doctor, "_DHH", str(home))

    finding = doctor_state._check_directory_structure(False)
    output = capsys.readouterr().out
    assert "USER.md exists but cannot be read" in output
    assert "USER.md exists (" not in output
    assert any("USER.md is unreadable" in issue for issue in finding.issues)
    assert path.read_bytes() == raw
    assert list(memories.iterdir()) == [path]
