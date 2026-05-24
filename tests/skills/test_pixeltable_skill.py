"""Tests for the Pixeltable optional skill (HARDLINE compliance)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parents[2] / 'optional-skills' / 'mlops' / 'pixeltable'
SKILL_MD = SKILL_DIR / 'SKILL.md'
SETUP_SH = SKILL_DIR / 'scripts' / 'setup.sh'
MCP_REF = SKILL_DIR / 'references' / 'mcp-integration.md'
PIPELINE_REF = SKILL_DIR / 'references' / 'multimodal-pipeline-examples.md'


@pytest.fixture(scope='module')
def skill_text() -> str:
    return SKILL_MD.read_text()


@pytest.fixture(scope='module')
def frontmatter(skill_text: str) -> str:
    parts = skill_text.split('---', 2)
    assert len(parts) >= 3, 'SKILL.md must have YAML frontmatter delimited by ---'
    return parts[1]


@pytest.fixture(scope='module')
def body(skill_text: str) -> str:
    parts = skill_text.split('---', 2)
    return parts[2]


class TestFrontmatter:
    def test_starts_with_delimiter(self, skill_text: str) -> None:
        assert skill_text.startswith('---'), 'SKILL.md must start with --- at byte 0'

    def test_name_field(self, frontmatter: str) -> None:
        m = re.search(r'^name:\s*(.+)$', frontmatter, re.MULTILINE)
        assert m, 'frontmatter must have a name field'
        name = m.group(1).strip()
        assert re.match(r'^[a-z][a-z0-9-]*$', name), f'name must be lowercase+hyphens: {name}'
        assert len(name) <= 64, f'name must be <= 64 chars: {len(name)}'

    def test_description_length(self, frontmatter: str) -> None:
        m = re.search(r'^description:\s*(.+)$', frontmatter, re.MULTILINE)
        assert m, 'frontmatter must have a description field'
        desc = m.group(1).strip()
        assert len(desc) <= 60, f'description must be <= 60 chars, got {len(desc)}: "{desc}"'

    def test_description_ends_with_period(self, frontmatter: str) -> None:
        m = re.search(r'^description:\s*(.+)$', frontmatter, re.MULTILINE)
        desc = m.group(1).strip()
        assert desc.endswith('.'), f'description must end with a period: "{desc}"'

    def test_version_field(self, frontmatter: str) -> None:
        assert re.search(r'^version:\s*\d+\.\d+\.\d+', frontmatter, re.MULTILINE), \
            'frontmatter must have a semver version field'

    def test_author_field(self, frontmatter: str) -> None:
        assert re.search(r'^author:', frontmatter, re.MULTILINE), \
            'frontmatter must have an author field'

    def test_license_field(self, frontmatter: str) -> None:
        assert re.search(r'^license:', frontmatter, re.MULTILINE), \
            'frontmatter must have a license field'

    def test_no_marketing_words(self, frontmatter: str) -> None:
        marketing = ['powerful', 'comprehensive', 'seamless', 'advanced', 'revolutionary', 'cutting-edge']
        desc_match = re.search(r'^description:\s*(.+)$', frontmatter, re.MULTILINE)
        desc = desc_match.group(1).lower()
        for word in marketing:
            assert word not in desc, f'description contains marketing word: "{word}"'

    def test_description_does_not_repeat_name(self, frontmatter: str) -> None:
        name_match = re.search(r'^name:\s*(.+)$', frontmatter, re.MULTILINE)
        desc_match = re.search(r'^description:\s*(.+)$', frontmatter, re.MULTILINE)
        name = name_match.group(1).strip()
        desc = desc_match.group(1).strip().lower()
        assert name not in desc, f'description should not repeat the skill name "{name}"'


class TestSectionStructure:
    REQUIRED_SECTIONS = [
        '## When to Use',
        '## Prerequisites',
        '## How to Run',
        '## Quick Reference',
        '## Procedure',
        '## Pitfalls',
        '## Verification',
    ]

    @pytest.mark.parametrize('section', REQUIRED_SECTIONS)
    def test_has_required_section(self, body: str, section: str) -> None:
        assert section in body, f'SKILL.md body missing required section: {section}'

    def test_has_h1_title(self, body: str) -> None:
        assert re.search(r'^# \w', body, re.MULTILINE), 'SKILL.md must have an H1 title'

    def test_line_count_reasonable(self, skill_text: str) -> None:
        lines = skill_text.strip().split('\n')
        assert 80 <= len(lines) <= 400, f'SKILL.md should be 80-400 lines, got {len(lines)}'


class TestToolReferences:
    NATIVE_TOOLS = {
        'terminal', 'web_extract', 'web_search', 'read_file', 'write_file',
        'patch', 'search_files', 'vision_analyze', 'browser_navigate',
        'delegate_task', 'image_generate', 'text_to_speech', 'cronjob',
        'memory', 'skill_view', 'todo', 'execute_code', 'native-mcp',
    }

    FORBIDDEN_TOOL_NAMES = {
        'grep', 'rg', 'cat', 'head', 'tail', 'sed', 'awk', 'find', 'ls',
        'curl', 'echo',
    }

    def test_no_forbidden_tool_references(self, body: str) -> None:
        for tool in self.FORBIDDEN_TOOL_NAMES:
            pattern = rf'`{tool}`'
            assert not re.search(pattern, body), \
                f'SKILL.md references forbidden tool `{tool}` — use the Hermes-native equivalent'


class TestSupportFiles:
    def test_setup_script_exists(self) -> None:
        assert SETUP_SH.exists(), 'scripts/setup.sh must exist'

    def test_setup_script_is_shell(self) -> None:
        text = SETUP_SH.read_text()
        assert text.startswith('#!/'), 'setup.sh must have a shebang line'
        assert 'set -' in text, 'setup.sh should use strict mode (set -e or similar)'

    def test_mcp_reference_exists(self) -> None:
        assert MCP_REF.exists(), 'references/mcp-integration.md must exist'

    def test_pipeline_reference_exists(self) -> None:
        assert PIPELINE_REF.exists(), 'references/multimodal-pipeline-examples.md must exist'

    def test_mcp_reference_has_config(self) -> None:
        text = MCP_REF.read_text()
        assert 'mcpServers:' in text, 'MCP reference must contain config.yaml entry'
        assert 'pixeltable' in text.lower(), 'MCP reference must mention pixeltable'

    def test_pipeline_reference_has_examples(self) -> None:
        text = PIPELINE_REF.read_text()
        assert 'import pixeltable' in text, 'Pipeline reference must contain runnable Python examples'
