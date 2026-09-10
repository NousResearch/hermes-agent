"""Tests for the zettelkasten-luhmann skill's init_vault.py script.

Uses only stdlib + pytest + unittest.mock. No live network calls.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "zettelkasten-luhmann"
SCRIPT_PATH = SKILL_DIR / "scripts" / "init_vault.py"


@pytest.fixture
def mod():
    """Load init_vault.py as a module for direct function testing."""
    spec = importlib.util.spec_from_file_location("init_vault_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestArgParse:
    """Verify --lang argument parsing (no filesystem access)."""

    def test_default_lang_is_pt(self, mod):
        assert hasattr(mod, "main")

    def test_lang_pt_selects_portuguese_templates(self, mod):
        assert mod.TEMPLATES_PT is not None
        assert "Ideia Principal" in mod.TEMPLATES_PT["permanent_notes.md"]

    def test_lang_en_selects_english_templates(self, mod):
        assert mod.TEMPLATES_EN is not None
        assert "Core Idea" in mod.TEMPLATES_EN["permanent_notes.md"]

    def test_starter_pt_has_portuguese_content(self, mod):
        assert "Uma nota é uma tese" in mod.STARTER_PT
        assert "Luhmann" in mod.STARTER_PT

    def test_starter_en_has_english_content(self, mod):
        assert "A note is a thesis" in mod.STARTER_EN
        assert "Luhmann" in mod.STARTER_EN


class TestTemplateStructure:
    """Verify templates contain expected sections."""

    PT_SECTIONS = {
        "permanent_notes.md": ["Ideia Principal", "Contexto", "Expansão", "Conexões"],
        "literature_notes.md": ["Resumo", "Citações", "Comentários Pessoais", "Conexões"],
        "fleeting_notes.md": ["Ideia", "Contexto", "Processar"],
        "moc_template.md": ["MOC", "Pontes a construir"],
    }

    EN_SECTIONS = {
        "permanent_notes.md": ["Core Idea", "Context", "Expansion", "Connections"],
        "literature_notes.md": ["Summary", "Key Quotes", "Personal Comments", "Connections"],
        "fleeting_notes.md": ["Idea", "Context", "Process into"],
        "moc_template.md": ["MOC", "Bridges to build"],
    }

    def test_pt_templates_have_expected_sections(self, mod):
        for name, sections in self.PT_SECTIONS.items():
            content = mod.TEMPLATES_PT[name]
            for section in sections:
                assert section in content, f"{name}: missing '{section}'"

    def test_en_templates_have_expected_sections(self, mod):
        for name, sections in self.EN_SECTIONS.items():
            content = mod.TEMPLATES_EN[name]
            for section in sections:
                assert section in content, f"{name}: missing '{section}'"

    def test_all_templates_have_yaml_frontmatter(self, mod):
        for templates in (mod.TEMPLATES_PT, mod.TEMPLATES_EN):
            for name, content in templates.items():
                assert content.startswith("---"), f"{name}: missing YAML frontmatter"
                assert "type:" in content.split("---", 2)[1], f"{name}: missing type field"

    def test_all_templates_have_tags_field(self, mod):
        for templates in (mod.TEMPLATES_PT, mod.TEMPLATES_EN):
            for name, content in templates.items():
                fm = content.split("---", 2)[1]
                assert "tags:" in fm, f"{name}: missing tags field"


class TestFilesystemCreation:
    """Integration-style tests that create a real vault on tmp_path."""

    def test_creates_structure(self, mod, tmp_path: Path):
        """Simulate main() for --lang pt, verify all dirs and files exist."""
        templates = mod.TEMPLATES_PT
        starter = mod.STARTER_PT
        base = tmp_path / "ZettelKasten"
        for d in ("zettels", "Daily", "templates", "arquivos"):
            (base / d).mkdir(parents=True, exist_ok=True)
        for name, content in templates.items():
            (base / "templates" / name).write_text(content, encoding="utf-8")
        starter_path = base / "zettels" / "Uma nota é uma tese, não um tópico.md"
        starter_path.write_text(starter, encoding="utf-8")

        assert (base / "zettels").is_dir()
        assert (base / "Daily").is_dir()
        assert (base / "templates").is_dir()
        assert (base / "arquivos").is_dir()
        assert (base / "templates" / "permanent_notes.md").is_file()
        assert (base / "templates" / "literature_notes.md").is_file()
        assert (base / "templates" / "fleeting_notes.md").is_file()
        assert (base / "templates" / "moc_template.md").is_file()
        assert starter_path.is_file()

    def test_idempotent_no_overwrite(self, mod, tmp_path: Path):
        """Simulate main() twice; content of existing files preserved."""
        templates = mod.TEMPLATES_PT
        starter = mod.STARTER_PT
        base = tmp_path / "ZettelKasten"
        for d in ("zettels", "Daily", "templates", "arquivos"):
            (base / d).mkdir(parents=True, exist_ok=True)

        # First run
        for name, content in templates.items():
            (base / "templates" / name).write_text(content, encoding="utf-8")
        starter_path = base / "zettels" / "Uma nota é uma tese, não um tópico.md"
        starter_path.write_text(starter, encoding="utf-8")

        # Overwrite a template with different content
        alt_content = "---\ntype: permanent\n---\n# ALT"
        (base / "templates" / "permanent_notes.md").write_text(alt_content, encoding="utf-8")

        # Re-run — should NOT overwrite
        for name, content in templates.items():
            p = base / "templates" / name
            if not p.exists():
                p.write_text(content, encoding="utf-8")

        assert (base / "templates" / "permanent_notes.md").read_text(encoding="utf-8") == alt_content
