import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT
    / "skills"
    / "productivity"
    / "ocr-and-documents"
    / "scripts"
    / "extract_liteparse.py"
)
SKILL_MD = REPO_ROOT / "skills" / "productivity" / "ocr-and-documents" / "SKILL.md"
DOC_MD = (
    REPO_ROOT
    / "website"
    / "docs"
    / "user-guide"
    / "skills"
    / "bundled"
    / "productivity"
    / "productivity-ocr-and-documents.md"
)


def load_script():
    spec = importlib.util.spec_from_file_location(
        "extract_liteparse_under_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_liteparse_helper_passes_markdown_mode_and_prints_text(
    monkeypatch, tmp_path, capsys
):
    module = load_script()
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    class FakeLiteParse:
        kwargs = {}
        parsed_path = None

        def __init__(self, **kwargs):
            FakeLiteParse.kwargs = kwargs

        def parse(self, path):
            FakeLiteParse.parsed_path = path
            return SimpleNamespace(text="# Heading\n\n| A | B |\n")

    fake_module = ModuleType("liteparse")
    fake_module.LiteParse = FakeLiteParse
    monkeypatch.setitem(sys.modules, "liteparse", fake_module)

    assert (
        module.main([
            str(pdf),
            "--pages",
            "1-2",
            "--max-pages",
            "1",
            "--ocr",
            "--verbose",
        ])
        == 0
    )

    assert FakeLiteParse.kwargs == {
        "ocr_enabled": True,
        "target_pages": "1-2",
        "max_pages": 1,
        "output_format": "markdown",
        "quiet": False,
    }
    assert FakeLiteParse.parsed_path == pdf
    assert capsys.readouterr().out == "# Heading\n\n| A | B |\n\n"


def test_liteparse_helper_reports_missing_dependency(monkeypatch, tmp_path, capsys):
    module = load_script()
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    real_import = builtins.__import__

    def fail_liteparse_import(name, *args, **kwargs):
        if name == "liteparse":
            raise ImportError("missing liteparse")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_liteparse_import)
    monkeypatch.delitem(sys.modules, "liteparse", raising=False)

    assert module.main([str(pdf)]) == 1
    assert "uv pip install liteparse" in capsys.readouterr().err


def test_liteparse_helper_reports_parse_failure(monkeypatch, tmp_path, capsys):
    module = load_script()
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    class FakeLiteParse:
        def __init__(self, **kwargs):
            pass

        def parse(self, path):
            raise RuntimeError("bad pdf")

    fake_module = ModuleType("liteparse")
    fake_module.LiteParse = FakeLiteParse
    monkeypatch.setitem(sys.modules, "liteparse", fake_module)

    assert module.main([str(pdf)]) == 1
    assert "liteparse failed: bad pdf" in capsys.readouterr().err


def test_ocr_skill_docs_use_realistic_liteparse_guidance():
    for text in (
        SKILL_MD.read_text(encoding="utf-8"),
        DOC_MD.read_text(encoding="utf-8"),
    ):
        assert "samples/controlled_agent_brief_table_layout.pdf" not in text
        assert "plain text despite output_format" not in text
        assert 'output_format="markdown"' in text
        assert "lightweight markdown-style text" in text
        assert "path/to/text.pdf" in text
