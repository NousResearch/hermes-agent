from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

from agent import document_parsing as dp


def test_parse_document_with_basic_backend_for_text_file(tmp_path):
    doc = tmp_path / "notes.md"
    doc.write_text("# Notes\nhello world\n", encoding="utf-8")

    parsed = dp.parse_document(doc, backend="basic")

    assert parsed.parser_backend == "basic"
    assert parsed.text == "# Notes\nhello world\n"
    assert parsed.metadata["page_count"] == 1
    assert parsed.pages[0].page_number == 1


def test_resolve_document_parser_backend_prefers_liteparse_for_supported_files(monkeypatch, tmp_path):
    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(dp, "liteparse_available", lambda: True)

    backend = dp.resolve_document_parser_backend(doc)

    assert backend == "liteparse"


def test_parse_document_uses_liteparse_python_backend(monkeypatch, tmp_path):
    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.4")

    expected = dp.ParsedDocument(
        source_path=str(doc.resolve()),
        parser_backend="liteparse-python",
        text="parsed text",
        pages=[dp.ParsedPage(page_number=1, text="parsed text")],
        metadata={"page_count": 1},
    )

    monkeypatch.setattr(dp, "liteparse_python_available", lambda: True)
    monkeypatch.setattr(dp, "_find_liteparse_cli", lambda: None)
    monkeypatch.setattr(dp, "_parse_with_liteparse_python", lambda path, config, cli_command=None: expected)

    parsed = dp.parse_document(doc, backend="liteparse")

    assert parsed == expected


def test_parse_with_liteparse_python_passes_cli_path_and_parse_options(monkeypatch, tmp_path):
    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.4")
    captured = {}

    class FakeLiteParse:
        def __init__(self, cli_path=None):
            captured["cli_path"] = cli_path

        def parse(self, file_path, **kwargs):
            captured["file_path"] = file_path
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                text="python parsed text",
                pages=[
                    SimpleNamespace(
                        pageNum=1,
                        text="python parsed text",
                        textItems=[],
                    )
                ],
            )

    monkeypatch.setitem(sys.modules, "liteparse", SimpleNamespace(LiteParse=FakeLiteParse))

    parsed = dp._parse_with_liteparse_python(
        doc.resolve(),
        {
            "ocr_enabled": False,
            "ocr_server_url": "",
            "ocr_language": "fra",
            "dpi": 300,
            "target_pages": "1-3",
            "max_pages": 12,
            "no_precise_bbox": True,
            "preserve_small_text": True,
        },
        cli_command="npx liteparse",
    )

    assert parsed.parser_backend == "liteparse-python"
    assert captured["cli_path"] == "npx liteparse"
    assert captured["file_path"] == str(doc.resolve())
    assert captured["kwargs"]["ocr_enabled"] is False
    assert captured["kwargs"]["ocr_language"] == "fra"
    assert captured["kwargs"]["max_pages"] == 12
    assert captured["kwargs"]["target_pages"] == "1-3"
    assert captured["kwargs"]["dpi"] == 300
    assert captured["kwargs"]["precise_bounding_box"] is False
    assert captured["kwargs"]["preserve_very_small_text"] is True


def test_parse_document_falls_back_to_liteparse_cli(monkeypatch, tmp_path):
    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.4")
    captured = {}

    def fake_python_backend(path, config):
        raise RuntimeError("python backend unavailable")

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "text": "cli parsed text",
                    "pages": [
                        {
                            "pageNum": 1,
                            "textItems": [
                                {
                                    "text": "cli parsed text",
                                    "bbox": [1, 2, 3, 4],
                                    "confidence": 0.9,
                                }
                            ],
                        }
                    ],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(dp, "liteparse_python_available", lambda: True)
    monkeypatch.setattr(dp, "_parse_with_liteparse_python", fake_python_backend)
    monkeypatch.setattr(dp, "_find_liteparse_cli", lambda: "npx liteparse")
    monkeypatch.setattr(dp.subprocess, "run", fake_run)

    parsed = dp.parse_document(
        doc,
        backend="liteparse",
        config={
            "parser_backend": "liteparse",
            "liteparse": {
                "ocr_enabled": False,
                "ocr_language": "fra",
                "dpi": 300,
                "target_pages": "1-2",
                "max_pages": 25,
                "no_precise_bbox": True,
                "preserve_small_text": True,
            },
        },
    )

    assert parsed.parser_backend == "liteparse-cli"
    assert parsed.text == "cli parsed text"
    assert parsed.pages[0].items[0].bbox == {"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0}
    assert captured["cmd"][:3] == ["npx", "liteparse", "parse"]
    assert "--no-ocr" in captured["cmd"]
    assert "--ocr-language" in captured["cmd"]
    assert "--target-pages" in captured["cmd"]
    assert "--no-precise-bbox" in captured["cmd"]
    assert "--preserve-small-text" in captured["cmd"]


def test_find_liteparse_cli_checks_path_local_and_npx(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    agent_dir = repo_root / "agent"
    node_bin = repo_root / "node_modules" / ".bin"
    node_bin.mkdir(parents=True)
    local_cli = node_bin / "liteparse"
    local_cli.write_text("#!/bin/sh\n", encoding="utf-8")
    local_cli.chmod(0o755)

    monkeypatch.setattr(dp, "__file__", str(agent_dir / "document_parsing.py"))
    monkeypatch.setattr(dp.shutil, "which", lambda cmd: None if cmd != "npx" else "/usr/bin/npx")

    assert dp._find_liteparse_cli() == str(local_cli)

    monkeypatch.setattr(dp.shutil, "which", lambda cmd: "/usr/local/bin/liteparse" if cmd == "liteparse" else None)
    assert dp._find_liteparse_cli() == "/usr/local/bin/liteparse"


def test_normalize_liteparse_python_result_maps_pages_and_items(tmp_path):
    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.4")
    result = SimpleNamespace(
        text="overall text",
        pages=[
            SimpleNamespace(
                pageNum=2,
                boundingBoxes=[{"x1": 20, "y1": 21, "x2": 22, "y2": 23}],
                textItems=[
                    SimpleNamespace(
                        str="line one",
                        x=10,
                        y=11,
                        width=2,
                        height=2,
                        confidence=0.75,
                    )
                ],
            )
        ],
    )

    parsed = dp._normalize_liteparse_python_result(doc.resolve(), result)

    assert parsed.parser_backend == "liteparse-python"
    assert parsed.pages[0].page_number == 2
    assert parsed.pages[0].items[0].confidence == 0.75
    assert parsed.pages[0].items[0].bbox == {"x1": 10.0, "y1": 11.0, "x2": 12.0, "y2": 13.0}
    assert parsed.pages[0].metadata["bounding_boxes"][0]["x1"] == 20.0


def test_create_document_screenshots_uses_python_backend(monkeypatch, tmp_path):
    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.4")

    class FakeShot:
        page_num = 1
        image_path = "/tmp/page_1.png"

    monkeypatch.setattr(dp, "liteparse_python_available", lambda: True)
    monkeypatch.setattr(dp, "_find_liteparse_cli", lambda: "npx liteparse")
    monkeypatch.setattr(
        dp,
        "_create_liteparse_screenshots_python",
        lambda path, config, cli_command=None: [
            dp.DocumentScreenshot(page_number=1, image_path="/tmp/page_1.png", image_format="png")
        ],
    )

    screenshots = dp.create_document_screenshots(doc, backend="liteparse")

    assert screenshots[0].page_number == 1
    assert screenshots[0].image_path == "/tmp/page_1.png"


def test_default_config_includes_documents_parser_settings():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["documents"]["parser_backend"] == "auto"
    assert DEFAULT_CONFIG["documents"]["liteparse"]["ocr_language"] == "en"
    assert DEFAULT_CONFIG["documents"]["liteparse"]["image_format"] == "png"
