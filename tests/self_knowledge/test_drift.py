from pathlib import Path

from hermes_cli.self_knowledge.drift import check_drift


def test_missing_file_path_reference_reports_finding(tmp_path):
    doc = tmp_path / "self.md"
    doc.write_text("Hand text mentions `missing/file.py`.\n")

    findings = check_drift(doc, project_root=tmp_path)

    assert findings
    assert findings[0].kind == "file_path"
    assert findings[0].reference == "missing/file.py"


def test_existing_file_path_reference_passes(tmp_path):
    (tmp_path / "existing").mkdir()
    (tmp_path / "existing" / "file.py").write_text("")
    doc = tmp_path / "self.md"
    doc.write_text("Hand text mentions `existing/file.py`.\n")

    assert check_drift(doc, project_root=tmp_path) == []


def test_references_inside_auto_blocks_are_ignored(tmp_path):
    doc = tmp_path / "self.md"
    doc.write_text(
        "<!-- AUTO-START: capabilities -->\n`missing/file.py`\n<!-- AUTO-END: capabilities -->\n"
    )

    assert check_drift(doc, project_root=tmp_path) == []


def test_allowlisted_reference_is_ignored(tmp_path):
    doc = tmp_path / "self.md"
    doc.write_text("Hand text mentions `missing/file.py`.\n")
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text("missing/file.py\n")

    assert check_drift(doc, project_root=tmp_path, allowlist_path=allowlist) == []


def test_symbol_reference_reports_missing_symbol(tmp_path):
    module = tmp_path / "pkg" / "mod.py"
    module.parent.mkdir()
    module.write_text("class Thing:\n    pass\n")
    doc = tmp_path / "self.md"
    doc.write_text("Hand text mentions `pkg.mod.Missing`.\n")

    findings = check_drift(doc, project_root=tmp_path)

    assert any(f.kind == "symbol" and f.reference == "pkg.mod.Missing" for f in findings)
