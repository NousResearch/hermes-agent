from pathlib import Path

import pytest

from hermes_cli.self_knowledge.cli import run_self_knowledge_command


def test_cli_render_prints_rendered_doc(tmp_path, capsys, monkeypatch):
    doc = tmp_path / "self.md"
    doc.write_text("<!-- AUTO-START: known -->\nold\n<!-- AUTO-END: known -->\n")
    monkeypatch.setattr(
        "hermes_cli.self_knowledge.renderer.GENERATORS",
        {"known": lambda: "new"},
    )

    code = run_self_knowledge_command(render=True, refresh=False, check=False, strict=False, doc_path=doc)

    assert code == 0
    assert "new" in capsys.readouterr().out


def test_cli_refresh_writes_doc(tmp_path, monkeypatch):
    doc = tmp_path / "self.md"
    doc.write_text("<!-- AUTO-START: known -->\nold\n<!-- AUTO-END: known -->\n")
    monkeypatch.setattr(
        "hermes_cli.self_knowledge.renderer.GENERATORS",
        {"known": lambda: "new"},
    )

    code = run_self_knowledge_command(render=False, refresh=True, check=False, strict=False, doc_path=doc)

    assert code == 0
    assert "new" in doc.read_text()


def test_cli_check_soft_returns_zero_with_findings(tmp_path, capsys):
    doc = tmp_path / "self.md"
    doc.write_text("mentions `missing/file.py`\n")

    code = run_self_knowledge_command(
        render=False,
        refresh=False,
        check=True,
        strict=False,
        doc_path=doc,
        project_root=tmp_path,
        allowlist_path=tmp_path / "missing-allowlist.txt",
    )

    assert code == 0
    assert "referenced path does not exist" in capsys.readouterr().out


def test_cli_check_strict_returns_one_with_findings(tmp_path):
    doc = tmp_path / "self.md"
    doc.write_text("mentions `missing/file.py`\n")

    code = run_self_knowledge_command(
        render=False,
        refresh=False,
        check=True,
        strict=True,
        doc_path=doc,
        project_root=tmp_path,
        allowlist_path=tmp_path / "missing-allowlist.txt",
    )

    assert code == 1
