"""Tests for Block 6: Blueprint format + Mermaid diagrams.

 Covers:
   - blog_streams.py: ai stream has formats list with "blueprint"
   - blog_generator.py: build_blueprint_prompt generates the right structure
   - blog_assembler.py: validate_blueprint checks Mermaid + mapping table
   - blog_illustrator.py: _extract_diagram_spec extracts Mermaid code
 """
import pytest

from blog.blog_streams import STREAMS
from blog.blog_generator import build_blueprint_prompt
from blog.blog_assembler import validate_blueprint, _has_mermaid_diagram, _has_mapping_table
from blog.blog_illustrator import _extract_diagram_spec


# -- blog_streams tests -------------------------------------------------------

def test_ai_stream_has_blueprint_format():
    """AI stream has 'blueprint' in its formats list."""
    assert "blueprint" in STREAMS["ai"].get("formats", [])


def test_pm_stream_does_not_have_blueprint():
    """PM stream does not have blueprint format (essays only)."""
    assert "blueprint" not in STREAMS["pm"].get("formats", ["essay"])


# -- build_blueprint_prompt tests ---------------------------------------------

def test_blueprint_prompt_mentions_mermaid():
    """Blueprint prompt instructs the LLM to include a Mermaid diagram."""
    plan = {"topic_id": "t1", "title_hint": "kv-cache", "tags": ["ai"],
            "signals": [{"signal_id": "s1", "summary": "kv-cache architecture"}]}
    prompts = build_blueprint_prompt("ai", plan, "context", [])
    assert "mermaid" in prompts["system"].lower()
    assert "diagram" in prompts["system"].lower()


def test_blueprint_prompt_mentions_mapping_table():
    """Blueprint prompt instructs the LLM to include a mapping table."""
    plan = {"topic_id": "t1", "title_hint": "kv-cache", "tags": ["ai"],
            "signals": [{"signal_id": "s1", "summary": "kv-cache"}]}
    prompts = build_blueprint_prompt("ai", plan, "context", [])
    assert "mapping table" in prompts["system"].lower()
    assert "Concept" in prompts["system"]
    assert "Implementation" in prompts["system"]
    assert "Trade-off" in prompts["system"]


def test_blueprint_prompt_includes_retry_feedback():
    """Retry feedback is threaded into the blueprint prompt."""
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [],
            "signals": [{"signal_id": "s1", "summary": "s"}]}
    prompts = build_blueprint_prompt("ai", plan, "ctx", [],
                                     retry_feedback="missing diagram")
    assert "missing diagram" in prompts["system"]


def test_blueprint_prompt_includes_signals():
    """Signals from the plan are included in the user prompt."""
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [],
            "signals": [{"signal_id": "s1", "summary": "test signal"}]}
    prompts = build_blueprint_prompt("ai", plan, "ctx", [])
    assert "test signal" in prompts["user"]


# -- validate_blueprint tests -------------------------------------------------

def test_validate_blueprint_passes_with_mermaid_and_table():
    """Blueprint with Mermaid + table passes validation."""
    body = """# Title

```mermaid
flowchart TD
    A --> B
```

| Concept | Implementation | Trade-off |
|---------|---------------|-----------|
| Cache | Dict | Speed/RAM |
"""
    status, issues = validate_blueprint({"body_md": body})
    assert status == "ok"
    assert issues == []


def test_validate_blueprint_fails_no_mermaid():
    """Blueprint without Mermaid fails."""
    body = """# Title

| Concept | Implementation | Trade-off |
|---------|---------------|-----------|
| Cache | Dict | Speed/RAM |
"""
    status, issues = validate_blueprint({"body_md": body})
    assert status == "fail"
    assert any("Mermaid" in i for i in issues)


def test_validate_blueprint_fails_no_table():
    """Blueprint without mapping table fails."""
    body = """# Title

```mermaid
flowchart TD
    A --> B
```
"""
    status, issues = validate_blueprint({"body_md": body})
    assert status == "fail"
    assert any("mapping table" in i for i in issues)


def test_validate_blueprint_fails_comppletely_empty():
    """Empty blueprint fails both checks."""
    status, issues = validate_blueprint({"body_md": ""})
    assert status == "fail"
    assert len(issues) >= 2


def test_has_mermaid_diagram_detects_code_block():
    """_has_mermaid_diagram finds a fenced mermaid block."""
    body = "```mermaid\nflowchart TD\nA --> B\n```\n"
    assert _has_mermaid_diagram(body)


def test_has_mermaid_diagram_returns_false_for_plain_code():
    """_has_mermaid_diagram does not match non-mermaid code blocks."""
    body = "```python\nprint('hello')\n```\n"
    assert not _has_mermaid_diagram(body)


def test_has_mapping_table_detects_markdown_table():
    """_has_mapping_table finds a markdown table."""
    body = "| A | B |\n|---|---|\n| 1 | 2 |\n"
    assert _has_mapping_table(body)


def test_has_mapping_table_returns_false_for_prose():
    """_has_mapping_table returns False for prose without tables."""
    body = "This is just text without any table."
    assert not _has_mapping_table(body)


# -- _extract_diagram_spec tests ----------------------------------------------

def test_extract_diagram_spec_returns_mermaid_code():
    """_extract_diagram_spec returns the Mermaid code block content."""
    draft = {"body_md": """# Title

```mermaid
flowchart TD
    A --> B
    B --> C
```

More text.
"""}
    result = _extract_diagram_spec(draft)
    assert result is not None
    assert "flowchart TD" in result
    assert "A --> B" in result


def test_extract_diagram_spec_returns_none_when_no_mermaid():
    """_extract_diagram_spec returns None when no Mermaid block exists."""
    draft = {"body_md": "# Title\n\nNo diagram here."}
    assert _extract_diagram_spec(draft) is None


def test_extract_diagram_spec_returns_none_for_empty_body():
    """_extract_diagram_spec returns None for empty body."""
    assert _extract_diagram_spec({"body_md": ""}) is None