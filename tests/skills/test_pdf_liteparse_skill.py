import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT
    / "skills"
    / "productivity"
    / "pdf"
    / "scripts"
    / "extract_liteparse.py"
)
SKILL_MD = REPO_ROOT / "skills" / "productivity" / "pdf" / "references" / "ocr-extraction.md"
DOC_MD = (
    REPO_ROOT
    / "website"
    / "docs"
    / "user-guide"
    / "skills"
    / "bundled"
    / "productivity"
    / "productivity-pdf.md"
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
    error = capsys.readouterr().err
    assert "uv add 'liteparse==2.10.1'" in error
    assert "uv venv && uv pip install 'liteparse==2.10.1'" in error


def test_liteparse_helper_reports_constructor_failure(monkeypatch, tmp_path, capsys):
    module = load_script()
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    class FakeLiteParse:
        def __init__(self, **kwargs):
            raise RuntimeError("cannot initialize")

    fake_module = ModuleType("liteparse")
    fake_module.LiteParse = FakeLiteParse
    monkeypatch.setitem(sys.modules, "liteparse", fake_module)

    assert module.main([str(pdf)]) == 1
    assert "liteparse failed: cannot initialize" in capsys.readouterr().err


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


@pytest.mark.parametrize(
    "result",
    [SimpleNamespace(), SimpleNamespace(text=None), SimpleNamespace(text="  \n")],
)
def test_liteparse_helper_rejects_missing_or_empty_text(
    monkeypatch, tmp_path, capsys, result
):
    module = load_script()
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    class FakeLiteParse:
        def __init__(self, **kwargs):
            pass

        def parse(self, path):
            return result

    fake_module = ModuleType("liteparse")
    fake_module.LiteParse = FakeLiteParse
    monkeypatch.setitem(sys.modules, "liteparse", fake_module)

    assert module.main([str(pdf)]) == 1
    assert "liteparse returned no text" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_liteparse_helper_rejects_invalid_max_pages(value, capsys):
    module = load_script()

    with pytest.raises(SystemExit) as exc_info:
        module.parse_args(["document.pdf", "--max-pages", value])

    assert exc_info.value.code == 2
    assert "--max-pages: must be a positive integer" in capsys.readouterr().err


def test_pdf_skill_links_optional_liteparse_guidance():
    skill = REPO_ROOT / "skills" / "productivity" / "pdf" / "SKILL.md"
    for text in (skill.read_text(encoding="utf-8"), DOC_MD.read_text(encoding="utf-8")):
        assert "LiteParse" in text
        assert "scripts/extract_liteparse.py" in text
        assert "references/ocr-extraction.md" in text
    reference = SKILL_MD.read_text(encoding="utf-8")
    assert "samples/controlled_agent_brief_table_layout.pdf" not in reference
    assert 'output_format="markdown"' in reference
    assert "uv add 'liteparse==2.10.1'" in reference
    assert "reconstruction quality varies with document complexity" in reference
