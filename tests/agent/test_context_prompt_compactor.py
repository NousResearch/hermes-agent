from agent.context_prompt_compactor import compact_context_prose


def test_compact_context_prose_reduces_filler_text():
    text = (
        "Please carefully make sure to use pytest for tests. "
        "It is important to note that this project simply uses Ruff. "
        "Do not forget to add type hints."
    )
    out = compact_context_prose(text)
    assert len(out) < len(text)
    assert "pytest" in out
    assert "Ruff" in out
    assert "type hints" in out


def test_compact_context_prose_preserves_url_code_and_path():
    text = (
        "Please review `pytest -q` and see https://example.com/docs carefully before editing /tmp/demo/file.py. "
        "It is important to note that this project uses tools from there."
    )
    out = compact_context_prose(text)
    assert "`pytest -q`" in out
    assert "https://example.com/docs" in out
    assert "/tmp/demo/file.py" in out


def test_compact_context_prose_skips_bullets():
    text = "- Please use pytest\n- Please use Ruff\n- Please use mypy"
    out = compact_context_prose(text)
    assert out == text


def test_compact_context_prose_skips_short_text():
    text = "Please use pytest."
    assert compact_context_prose(text) == text


def test_compact_context_prose_preserves_heading_and_compacts_body():
    text = (
        "## AGENTS.md\n\n"
        "Please carefully make sure to use pytest for tests. "
        "It is important to note that this project simply uses Ruff. "
    ) * 5
    out = compact_context_prose(text)
    assert out.startswith("## AGENTS.md\n\n")
    assert len(out) < len(text)
    assert "pytest" in out
    assert "Ruff" in out
