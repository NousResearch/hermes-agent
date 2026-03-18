"""Tests for tool token estimation and curses_ui status_fn support."""

from unittest.mock import patch

import pytest


# ─── Token Estimation Tests ──────────────────────────────────────────────────


def test_estimate_tool_tokens_returns_positive_counts():
    """_estimate_tool_tokens should return a non-empty dict with positive values."""
    from hermes_cli.tools_config import _estimate_tool_tokens, _tool_token_cache

    # Clear cache to force fresh computation
    import hermes_cli.tools_config as tc
    tc._tool_token_cache = None

    tokens = _estimate_tool_tokens()

    assert isinstance(tokens, dict)
    assert len(tokens) > 0
    for name, count in tokens.items():
        assert isinstance(name, str)
        assert isinstance(count, int)
        assert count > 0, f"Tool {name} has non-positive token count: {count}"


def test_estimate_tool_tokens_is_cached():
    """Second call should return the same cached dict object."""
    import hermes_cli.tools_config as tc
    tc._tool_token_cache = None

    first = tc._estimate_tool_tokens()
    second = tc._estimate_tool_tokens()

    assert first is second


def test_estimate_tool_tokens_returns_empty_when_tiktoken_unavailable(monkeypatch):
    """Graceful degradation when tiktoken cannot be imported."""
    import hermes_cli.tools_config as tc
    tc._tool_token_cache = None

    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("mocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    result = tc._estimate_tool_tokens()

    assert result == {}

    # Reset cache for other tests
    tc._tool_token_cache = None


def test_estimate_tool_tokens_covers_known_tools():
    """Should include schemas for well-known tools like terminal, web_search."""
    import hermes_cli.tools_config as tc
    tc._tool_token_cache = None

    tokens = tc._estimate_tool_tokens()

    # These tools should always be discoverable
    for expected in ("terminal", "web_search", "read_file"):
        assert expected in tokens, f"Expected {expected!r} in token estimates"


# ─── Status Function Tests ───────────────────────────────────────────────────


def test_prompt_toolset_checklist_passes_status_fn(monkeypatch):
    """_prompt_toolset_checklist should pass a status_fn to curses_checklist."""
    import hermes_cli.tools_config as tc

    captured_kwargs = {}

    def fake_checklist(title, items, selected, *, cancel_returns=None, status_fn=None):
        captured_kwargs["status_fn"] = status_fn
        captured_kwargs["title"] = title
        return selected  # Return pre-selected unchanged

    monkeypatch.setattr("hermes_cli.curses_ui.curses_checklist", fake_checklist)

    tc._prompt_toolset_checklist("CLI", {"web", "terminal"})

    assert "status_fn" in captured_kwargs
    # If tiktoken is available, status_fn should be set
    tokens = tc._estimate_tool_tokens()
    if tokens:
        assert captured_kwargs["status_fn"] is not None


def test_status_fn_returns_formatted_token_count(monkeypatch):
    """The status_fn should return a human-readable token count string."""
    import hermes_cli.tools_config as tc
    from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS

    captured = {}

    def fake_checklist(title, items, selected, *, cancel_returns=None, status_fn=None):
        captured["status_fn"] = status_fn
        return selected

    monkeypatch.setattr("hermes_cli.curses_ui.curses_checklist", fake_checklist)

    tc._prompt_toolset_checklist("CLI", {"web", "terminal"})

    status_fn = captured.get("status_fn")
    if status_fn is None:
        pytest.skip("tiktoken unavailable; status_fn not created")

    # Find the indices for web and terminal
    idx_map = {ts_key: i for i, (ts_key, _, _) in enumerate(CONFIGURABLE_TOOLSETS)}

    # Call status_fn with web + terminal selected
    result = status_fn({idx_map["web"], idx_map["terminal"]})
    assert "tokens" in result
    assert "Est. tool context" in result


def test_status_fn_deduplicates_overlapping_tools(monkeypatch):
    """When toolsets overlap (browser includes web_search), tokens should not double-count."""
    import hermes_cli.tools_config as tc
    from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS

    captured = {}

    def fake_checklist(title, items, selected, *, cancel_returns=None, status_fn=None):
        captured["status_fn"] = status_fn
        return selected

    monkeypatch.setattr("hermes_cli.curses_ui.curses_checklist", fake_checklist)

    tc._prompt_toolset_checklist("CLI", {"web"})

    status_fn = captured.get("status_fn")
    if status_fn is None:
        pytest.skip("tiktoken unavailable; status_fn not created")

    idx_map = {ts_key: i for i, (ts_key, _, _) in enumerate(CONFIGURABLE_TOOLSETS)}

    # web alone
    web_only = status_fn({idx_map["web"]})
    # browser includes web_search, so browser + web should not double-count web_search
    browser_only = status_fn({idx_map["browser"]})
    both = status_fn({idx_map["web"], idx_map["browser"]})

    # Extract numeric token counts from strings like "~8.3k tokens" or "~350 tokens"
    import re

    def parse_tokens(s):
        m = re.search(r"~([\d.]+)k?\s+tokens", s)
        if not m:
            return 0
        val = float(m.group(1))
        if "k" in s[m.start():m.end()]:
            val *= 1000
        return val

    web_tok = parse_tokens(web_only)
    browser_tok = parse_tokens(browser_only)
    both_tok = parse_tokens(both)

    # Both together should be LESS than naive sum (due to web_search dedup)
    naive_sum = web_tok + browser_tok
    assert both_tok < naive_sum, (
        f"Expected deduplication: web({web_tok}) + browser({browser_tok}) = {naive_sum} "
        f"but combined = {both_tok}"
    )


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


def test_curses_checklist_numbered_fallback_without_status(monkeypatch, capsys):
    """The numbered fallback should work fine without status_fn."""
    from hermes_cli.curses_ui import _numbered_fallback

    monkeypatch.setattr("builtins.input", lambda _prompt="": "")

    result = _numbered_fallback(
        "Test title",
        ["Item A", "Item B"],
        {0},
        {0},
    )

    captured = capsys.readouterr()
    assert "Est. tool context" not in captured.out
    assert result == {0}


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


def test_registry_get_schema_returns_none_for_unknown():
    """registry.get_schema() should return None for unknown tools."""
    from tools.registry import registry

    assert registry.get_schema("nonexistent_tool_xyz") is None
