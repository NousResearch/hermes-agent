"""Tests for `hermes memory audit` dry-run memory quality review."""

from __future__ import annotations

import json

from tools.memory_tool import ENTRY_DELIMITER


def test_collect_memory_audit_reports_policy_warnings(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)

    (tmp_path / "MEMORY.md").write_text(
        ENTRY_DELIMITER.join(
            [
                "Project uses pytest with xdist for full verification.",
                "Submitted PR #123 for this bugfix.",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "USER.md").write_text("Always answer in Russian.", encoding="utf-8")

    from hermes_cli.memory_setup import collect_memory_audit

    audit = collect_memory_audit(target="all")

    assert audit["entry_count"] == 3
    assert audit["warning_count"] == 2
    assert audit["targets"]["memory"]["entry_count"] == 2
    assert audit["targets"]["user"]["warning_count"] == 1
    assert audit["targets"]["memory"]["entries"][0]["warnings"] == []
    assert "session progress" in audit["targets"]["memory"]["entries"][1]["warnings"][0]
    assert "declarative" in audit["targets"]["user"]["entries"][0]["warnings"][0]


def test_collect_memory_audit_can_scope_to_user(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)

    (tmp_path / "MEMORY.md").write_text("Submitted PR #123", encoding="utf-8")
    (tmp_path / "USER.md").write_text("User prefers concise replies.", encoding="utf-8")

    from hermes_cli.memory_setup import collect_memory_audit

    audit = collect_memory_audit(target="user")

    assert set(audit["targets"]) == {"user"}
    assert audit["entry_count"] == 1
    assert audit["warning_count"] == 0


def test_cmd_audit_json_outputs_machine_readable_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    (tmp_path / "MEMORY.md").write_text("Submitted PR #123", encoding="utf-8")

    from hermes_cli.memory_setup import cmd_audit

    class Args:
        target = "memory"
        json = True

    cmd_audit(Args())

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["targets"]["memory"]["warning_count"] == 1
    assert captured.err == ""
