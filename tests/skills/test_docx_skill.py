"""Behavioral unit tests for DOCX style-reference health checks.

ZIP and XML parsing are real. Optional dependency adapters isolate this
unit from python-docx opening and lxml-specific parsing behavior; these
are not full package integration tests.
"""

import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET
import zipfile

import pytest


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PR = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/docx/scripts/docx_validate.py"
)


@pytest.fixture
def validator():
    class XMLSyntaxError(Exception):
        pass

    def fromstring(data):
        try:
            return ET.fromstring(data)
        except ET.ParseError as exc:
            raise XMLSyntaxError(str(exc)) from exc

    etree = types.ModuleType("lxml.etree")
    etree.fromstring = fromstring
    etree.XMLSyntaxError = XMLSyntaxError
    lxml = types.ModuleType("lxml")
    lxml.etree = etree
    docx = types.ModuleType("docx")
    docx.Document = Mock()
    with patch.dict(sys.modules, {"lxml": lxml, "lxml.etree": etree, "docx": docx}):
        spec = importlib.util.spec_from_file_location("docx_validator_unit", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module, docx.Document


def _content(tag=None, sid=None):
    text = "<w:r><w:t>Probe</w:t></w:r>"
    if tag == "pStyle":
        return f'<w:p><w:pPr><w:pStyle w:val="{sid}"/></w:pPr>{text}</w:p>'
    if tag == "rStyle":
        return (
            f'<w:p><w:r><w:rPr><w:rStyle w:val="{sid}"/></w:rPr>'
            "<w:t>Probe</w:t></w:r></w:p>"
        )
    if tag == "tblStyle":
        return (
            f'<w:tbl><w:tblPr><w:tblStyle w:val="{sid}"/></w:tblPr>'
            '<w:tblGrid><w:gridCol w:w="1000"/></w:tblGrid>'
            f"<w:tr><w:tc><w:p>{text}</w:p></w:tc></w:tr></w:tbl>"
        )
    return f"<w:p>{text}</w:p>"


def _package(tmp_path, *, body=(None, None), comments=None, defined=()):
    """Construct coherent relationships; missing styles are deliberate."""
    path = tmp_path / "probe.docx"
    body_xml = _content(*body)
    if comments is not None:
        body_xml += (
            '<w:p><w:commentRangeStart w:id="0"/><w:r><w:t>Anchor</w:t></w:r>'
            '<w:commentRangeEnd w:id="0"/>'
            '<w:r><w:commentReference w:id="0"/></w:r></w:p>'
        )
    overrides = "".join(
        f'<Override PartName="/word/{part}.xml" '
        f'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.{kind}+xml"/>'
        for part, kind in [("document", "document.main"), ("styles", "styles")]
        + ([("comments", "comments")] if comments is not None else [])
    )
    rels = f'<Relationship Id="rStyles" Type="{R}/styles" Target="styles.xml"/>'
    if comments is not None:
        rels += f'<Relationship Id="rComments" Type="{R}/comments" Target="comments.xml"/>'
    styles = "".join(
        f'<w:style w:type="paragraph" w:styleId="{sid}"><w:name w:val="{sid}"/></w:style>'
        for sid in defined
    )
    parts = {
        "[Content_Types].xml": (
            f'<Types xmlns="{CT}"><Default Extension="rels" '
            'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            f'<Default Extension="xml" ContentType="application/xml"/>{overrides}</Types>'
        ),
        "_rels/.rels": (
            f'<Relationships xmlns="{PR}"><Relationship Id="rDoc" '
            f'Type="{R}/officeDocument" Target="word/document.xml"/></Relationships>'
        ),
        "word/document.xml": f'<w:document xmlns:w="{W}"><w:body>{body_xml}<w:sectPr/></w:body></w:document>',
        "word/_rels/document.xml.rels": f'<Relationships xmlns="{PR}">{rels}</Relationships>',
        "word/styles.xml": f'<w:styles xmlns:w="{W}">{styles}</w:styles>',
    }
    if comments is not None:
        parts["word/comments.xml"] = comments
    with zipfile.ZipFile(path, "w") as zf:
        for name, xml in parts.items():
            zf.writestr(name, xml.encode("utf-8"))
    return path


def _comments(tag=None, sid=None):
    return (
        f'<w:comments xmlns:w="{W}"><w:comment w:id="0" w:author="Tester">'
        f"{_content(tag, sid)}</w:comment></w:comments>"
    )


def _run(validator, path, capsys):
    module, document_open = validator
    before = path.read_bytes()
    with patch.object(sys, "argv", [str(SCRIPT), str(path)]):
        rc = module.main()
    captured = capsys.readouterr()
    assert captured.err == ""
    report = json.loads(captured.out)
    document_open.assert_called_once_with(str(path))
    assert path.read_bytes() == before
    return rc, report


@pytest.mark.parametrize(
    "part,tag,defined",
    [
        pytest.param(None, None, False, id="A-no-comments"),
        pytest.param("comments", "pStyle", True, id="B-defined-comment-pStyle"),
        pytest.param("comments", "pStyle", False, id="C-missing-comment-pStyle"),
        pytest.param("comments", "rStyle", False, id="D-missing-comment-rStyle"),
        pytest.param("comments", "tblStyle", False, id="E-missing-comment-tblStyle"),
        pytest.param("body", "pStyle", False, id="F-missing-body-pStyle"),
        pytest.param("body", "rStyle", False, id="G-missing-body-rStyle"),
        pytest.param("body", "tblStyle", False, id="H-missing-body-tblStyle"),
    ],
)
def test_style_references_in_optional_comments_and_body(
    tmp_path, capsys, validator, part, tag, defined
):
    sid = "ProbeStyle"
    comments = _comments(tag, sid) if part == "comments" else None
    path = _package(
        tmp_path,
        body=(tag, sid) if part == "body" else (None, None),
        comments=comments,
        defined=(sid,) if defined else (),
    )
    rc, report = _run(validator, path, capsys)
    if part is None or defined:
        assert rc == 0
        assert report == {"ok": True, "issues": []}
    else:
        assert rc == 1
        assert report["ok"] is False
        assert any(
            issue["severity"] == "error"
            and issue["code"] == "missing-style"
            and sid in issue["detail"]
            and (part != "comments" or "comments.xml" in issue["detail"])
            for issue in report["issues"]
        )
    if part is None:
        with zipfile.ZipFile(path) as zf:
            assert "word/comments.xml" not in zf.namelist()
            assert b"comments" not in zf.read("word/_rels/document.xml.rels")
            assert b"comments" not in zf.read("[Content_Types].xml")


@pytest.mark.parametrize(
    "missing_body", [pytest.param(False, id="I-malformed-comments"),
                     pytest.param(True, id="J-malformed-comments-continue-body")]
)
def test_malformed_comments_report_and_continue(tmp_path, capsys, validator, missing_body):
    path = _package(
        tmp_path,
        body=("pStyle", "GhostBodyStyle") if missing_body else (None, None),
        comments=f'<w:comments xmlns:w="{W}"><w:comment>',
    )
    rc, report = _run(validator, path, capsys)
    assert rc == 1
    assert report["ok"] is False
    assert any(
        issue["severity"] == "error"
        and issue["code"] == "bad-comments-xml"
        and "word/comments.xml" in issue["detail"]
        for issue in report["issues"]
    )
    if missing_body:
        assert any(
            issue["severity"] == "error"
            and issue["code"] == "missing-style"
            and "GhostBodyStyle" in issue["detail"]
            for issue in report["issues"]
        )
