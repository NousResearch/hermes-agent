"""Tests for tool token estimation and curses_ui status_fn support."""


import pytest

# tiktoken is not in core/[all] deps — skip estimation tests when unavailable
_has_tiktoken = True
try:
    import tiktoken  # noqa: F401
except ImportError:
    _has_tiktoken = False

_needs_tiktoken = pytest.mark.skipif(not _has_tiktoken, reason="tiktoken not installed")


# ─── Token Estimation Tests ──────────────────────────────────────────────────






# ─── Status Function Tests ───────────────────────────────────────────────────






def test_status_fn_empty_selection():
    """Status function with no tools selected should return ~0 tokens."""
    import hermes_cli.tools_config as tc

    tc._tool_token_cache = None
    tokens = tc._estimate_tool_tokens()
    if not tokens:
        pytest.skip("tiktoken unavailable")

    from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS
    from toolsets import resolve_toolset

    ts_keys = [ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS]

    def status_fn(chosen: set) -> str:
        all_tools: set = set()
        for idx in chosen:
            all_tools.update(resolve_toolset(ts_keys[idx]))
        total = sum(tokens.get(name, 0) for name in all_tools)
        if total >= 1000:
            return f"Est. tool context: ~{total / 1000:.1f}k tokens"
        return f"Est. tool context: ~{total} tokens"

    result = status_fn(set())
    assert "~0 tokens" in result


# ─── Curses UI Status Bar Tests ──────────────────────────────────────────────


def test_curses_checklist_numbered_fallback_shows_status(monkeypatch, capsys):
    """The numbered fallback should print the status_fn output."""
    from hermes_cli.curses_ui import _numbered_fallback

    def my_status(chosen):
        return f"Selected {len(chosen)} items"

    # Simulate user pressing Enter immediately (empty input → confirm)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")

    result = _numbered_fallback(
        "Test title",
        ["Item A", "Item B", "Item C"],
        {0, 2},
        {0, 2},
        status_fn=my_status,
    )

    captured = capsys.readouterr()
    assert "Selected 2 items" in captured.out
    assert result == {0, 2}




# ─── Registry get_schema Tests ───────────────────────────────────────────────


def test_registry_get_schema_returns_schema():
    """registry.get_schema() should return a tool's schema dict."""
    from tools.registry import registry

    # Import to trigger discovery
    import model_tools  # noqa: F401

    schema = registry.get_schema("terminal")
    assert schema is not None
    assert "name" in schema
    assert schema["name"] == "terminal"
    assert "parameters" in schema


