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
APPS_REF = SKILL_DIR / 'references' / 'apps-and-cloud.md'


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

    def test_author_credits_contributor_first(self, frontmatter: str) -> None:
        m = re.search(r'^author:\s*(.+)$', frontmatter, re.MULTILINE)
        assert m, 'frontmatter must have an author field'
        author = m.group(1).strip()
        assert 'Pierre Brunelle' in author, f'author must credit Pierre Brunelle first: {author}'
        assert '@pierrebrunelle' in author, f'author must include @pierrebrunelle: {author}'
        assert author.startswith('Pierre Brunelle'), \
            f'contributor name must come first in author field: {author}'

    def test_platforms_linux_macos(self, frontmatter: str) -> None:
        m = re.search(r'^platforms:\s*\[(.+)\]\s*$', frontmatter, re.MULTILINE)
        assert m, 'frontmatter must declare platforms for Bash-dependent skill'
        platforms = {p.strip() for p in m.group(1).split(',')}
        assert platforms == {'linux', 'macos'}, \
            f'platforms must be [linux, macos], got {sorted(platforms)}'

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

    def test_h1_is_pixeltable_skill(self, body: str) -> None:
        assert re.search(r'^# Pixeltable Skill\s*$', body, re.MULTILINE), \
            "SKILL.md H1 must be '# Pixeltable Skill' per HARDLINE"

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

    def test_setup_uses_python3_m_pip(self) -> None:
        text = SETUP_SH.read_text()
        assert 'python3 -m pip' in text, 'setup.sh must install via python3 -m pip'
        assert not re.search(r'^\s*pip install\b', text, re.MULTILINE), \
            'setup.sh must not invoke bare pip (wrong interpreter risk)'

    def test_setup_emits_mcp_servers_key(self) -> None:
        text = SETUP_SH.read_text()
        assert 'mcp_servers:' in text, 'setup.sh snippet must use mcp_servers:'
        assert 'mcpServers' not in text, 'setup.sh must not emit mcpServers'

    def test_mcp_reference_exists(self) -> None:
        assert MCP_REF.exists(), 'references/mcp-integration.md must exist'

    def test_pipeline_reference_exists(self) -> None:
        assert PIPELINE_REF.exists(), 'references/multimodal-pipeline-examples.md must exist'

    def test_mcp_reference_has_config(self) -> None:
        text = MCP_REF.read_text()
        assert 'mcp_servers:' in text, 'MCP reference must contain mcp_servers: config.yaml entry'
        assert 'mcpServers' not in text, 'MCP reference must not use mcpServers'
        assert 'pixeltable' in text.lower(), 'MCP reference must mention pixeltable'
        assert 'uvx' in text and 'mcp-server-pixeltable-developer' in text, \
            'MCP reference must install the PyPI server via uvx'
        assert 'git+https://github.com/pixeltable/mcp-server-pixeltable-developer' not in text, \
            'MCP reference must not install the retired git server'
        assert 'PIXELTABLE_MCP_PROJECT_ROOT' in text, \
            'MCP reference must set PIXELTABLE_MCP_PROJECT_ROOT in env'
        assert 'PIXELTABLE_HOME' in text, 'MCP reference must set PIXELTABLE_HOME in env'

    def test_mcp_reference_uses_hermes_tool_prefix(self) -> None:
        text = MCP_REF.read_text()
        # Hermes registers mcp__<server>__<tool>. Server key pixeltable keeps its underscores.
        assert 'mcp__pixeltable__pixeltable_schema_update' in text, \
            'MCP reference must document Hermes name mcp__pixeltable__pixeltable_schema_update'
        assert 'mcp__pixeltable__pixeltable_insert_rows' in text, \
            'MCP reference must document Hermes name mcp__pixeltable__pixeltable_insert_rows'
        assert 'mcp_pixeltable_pixeltable_create_table' not in text, \
            'MCP reference must not keep the retired single-underscore create_table name'
        assert 'mcp_pixeltable_execute_python' not in text, \
            'MCP reference must not keep the retired execute_python name'
        assert 'Agent uses: pixeltable_' not in text, \
            'Examples must not call MCP tools without the Hermes mcp__ prefix'
        assert '`pixeltable_create_table`' not in text, \
            'Do not document bare MCP tool names as Hermes-callable tools'

    def test_setup_emits_pypi_mcp_install(self) -> None:
        text = SETUP_SH.read_text()
        assert 'mcp-server-pixeltable-developer' in text, \
            'setup.sh must emit the PyPI MCP server'
        assert 'git+https://github.com/pixeltable/mcp-server-pixeltable-developer' not in text, \
            'setup.sh must not emit the retired git+ uvx install'
        assert 'PIXELTABLE_MCP_PROJECT_ROOT' in text
        assert '3.11' in text, 'setup.sh must require Python 3.11+'

    def test_apps_reference_exists(self) -> None:
        assert APPS_REF.exists(), 'references/apps-and-cloud.md must exist'
        text = APPS_REF.read_text()
        assert 'pxt secret set' in text, 'apps reference must document pxt secret set'
        assert '--no-wait' in text, 'apps reference must document hosted --no-wait'
        assert 'pxt service update' in text and 'no `--no-wait`' in text, \
            'apps reference must say pxt service update has no --no-wait'

    def test_pipeline_reference_has_examples(self) -> None:
        text = PIPELINE_REF.read_text()
        assert 'import pixeltable' in text, 'Pipeline reference must contain runnable Python examples'

    def test_skill_prerequisites_use_python3_m_pip(self, skill_text: str) -> None:
        assert 'python3 -m pip install pixeltable' in skill_text, \
            'SKILL.md Prerequisites must use python3 -m pip install pixeltable'


class TestApiCorrectness:
    """Verify the skill teaches correct, non-hallucinated Pixeltable API patterns."""

    HALLUCINATED_PATTERNS = [
        (r'openai\.vision\b', 'openai.vision does not exist'),
        (r'from pixeltable\.iterators\s+import\s+FrameIterator', 'FrameIterator deprecated import'),
        (r'pxt\.Table\s*\(', 'pxt.Table() does not exist'),
        (r'pxt\.load_table\b', 'pxt.load_table does not exist'),
        (r'pxt\.connect\b', 'pxt.connect does not exist'),
        (r"if_not_exists\s*=", 'if_not_exists is not a valid kwarg; use if_exists'),
    ]

    def test_no_hallucinated_apis_in_skill(self, skill_text: str) -> None:
        code_blocks = re.findall(r'```python\n(.*?)```', skill_text, re.DOTALL)
        code_text = '\n'.join(code_blocks)
        for pattern, msg in self.HALLUCINATED_PATTERNS:
            assert not re.search(pattern, code_text), f'SKILL.md code contains hallucinated API: {msg}'

    def test_no_hallucinated_apis_in_references(self) -> None:
        text = PIPELINE_REF.read_text()
        for pattern, msg in self.HALLUCINATED_PATTERNS:
            assert not re.search(pattern, text), f'Pipeline examples contain hallucinated API: {msg}'

    def test_similarity_uses_keyword_arg(self, skill_text: str) -> None:
        code_blocks = re.findall(r'```python\n(.*?)```', skill_text, re.DOTALL)
        code_text = '\n'.join(code_blocks)
        positional = re.findall(r"\.similarity\(\s*'[^']+'\s*\)", code_text)
        assert not positional, f'SKILL.md code uses positional similarity(): {positional}'

    def test_similarity_uses_keyword_in_references(self) -> None:
        text = PIPELINE_REF.read_text()
        positional = re.findall(r"\.similarity\(\s*'[^']+'\s*\)", text)
        assert not positional, f'Pipeline examples use positional similarity(): {positional}'

    def test_idempotency_pattern_present(self, skill_text: str) -> None:
        assert "if_exists='ignore'" in skill_text, 'SKILL.md must demonstrate if_exists=ignore pattern'

    def test_collect_pattern_present(self, skill_text: str) -> None:
        assert '.collect()' in skill_text, 'SKILL.md must show .collect() to terminate queries'

    def test_columnref_query_helpers_pitfall(self, skill_text: str) -> None:
        assert 't.select(t.col)' in skill_text, \
            'SKILL.md must warn that ColumnRef query helpers were removed (use t.select)'
        assert 't.count()' in skill_text, \
            'SKILL.md must document t.count() for row counts'

    def test_image_url_is_the_column(self, skill_text: str) -> None:
        code_blocks = re.findall(r'```python\n(.*?)```', skill_text, re.DOTALL)
        code_text = '\n'.join(code_blocks)
        assert "{'url':" not in code_text, \
            "SKILL.md must pass the image column as image_url, not {'url': ...}"
        assert "'image_url': frames.frame" in code_text
        pipeline = PIPELINE_REF.read_text()
        assert "{'url':" not in pipeline
        assert "'image_url': frames.frame" in pipeline

    def test_no_retired_home_variable(self) -> None:
        for path in (SETUP_SH, MCP_REF, PIPELINE_REF):
            assert 'PXT_HOME' not in path.read_text(), f'{path.name} must not mention PXT_HOME'
        for path in (SKILL_MD, APPS_REF):
            text = path.read_text()
            assert 'PXT_HOME=' not in text, f'{path.name} must not tell the user to set PXT_HOME'
            assert 'set `PXT_HOME`' not in text
            assert 'PIXELTABLE_HOME' in text

    def test_python_311_and_upstream_guidance(self, skill_text: str) -> None:
        assert '3.11' in skill_text, 'SKILL.md must require Python 3.11+'
        assert 'references/apps-and-cloud.md' in skill_text
        apps = APPS_REF.read_text()
        assert 'Retry-After' in apps or 'retry' in skill_text.lower()
        assert 'pxt secret set' in apps
        assert 'CLIENT_OUTDATED' in apps and 'is not raised' in apps
