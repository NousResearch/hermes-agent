"""Tests for plugin context reference provider API (Issue #26193)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent.context_references import (
    BUILTIN_PREFIXES,
    ContextCompletionItem,
    ContextReferenceProvider,
    _PLUGIN_REFERENCE_PATTERN,
    _context_reference_providers,
    get_context_reference_providers,
    parse_context_references,
    register_context_reference_provider,
)


# -- helpers ---------------------------------------------------------------

class _DummyProvider(ContextReferenceProvider):
    """Minimal concrete provider for testing."""

    prefix = "test"
    description = "test provider"

    async def autocomplete(self, query: str, *, limit: int = 10) -> list[ContextCompletionItem]:
        return [ContextCompletionItem(text=f"{query}-result", meta="test")]

    async def expand(self, target: str) -> str | None:
        return f"expanded: {target}"


class _NoneExpandProvider(ContextReferenceProvider):
    """Provider whose expand() returns None (skip)."""

    prefix = "skip"
    description = "skip provider"

    async def autocomplete(self, query: str, *, limit: int = 10) -> list[ContextCompletionItem]:
        return []

    async def expand(self, target: str) -> str | None:
        return None


class _ErrorProvider(ContextReferenceProvider):
    """Provider whose expand() raises."""

    prefix = "boom"
    description = "error provider"

    async def autocomplete(self, query: str, *, limit: int = 10) -> list[ContextCompletionItem]:
        return []

    async def expand(self, target: str) -> str | None:
        raise RuntimeError("boom!")


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear plugin registry before and after each test."""
    _context_reference_providers.clear()
    yield
    _context_reference_providers.clear()


# -- registration tests ----------------------------------------------------

def test_register_valid_provider():
    p = _DummyProvider()
    register_context_reference_provider(p)
    assert "test" in get_context_reference_providers()


def test_register_rejects_builtin_prefix():
    for prefix in BUILTIN_PREFIXES:
        p = _DummyProvider()
        p.prefix = prefix
        with pytest.raises(ValueError, match="reserved"):
            register_context_reference_provider(p)


def test_register_rejects_duplicate_prefix():
    register_context_reference_provider(_DummyProvider())
    with pytest.raises(ValueError, match="already registered"):
        register_context_reference_provider(_DummyProvider())


def test_register_rejects_non_provider():
    with pytest.raises(TypeError, match="must be a ContextReferenceProvider"):
        register_context_reference_provider("not a provider")


def test_register_rejects_empty_prefix():
    p = _DummyProvider()
    p.prefix = ""
    with pytest.raises(ValueError, match="non-empty"):
        register_context_reference_provider(p)


# -- parse tests -----------------------------------------------------------

def test_parse_plugin_reference():
    register_context_reference_provider(_DummyProvider())
    refs = parse_context_references("check @test:ENG-123 and @file:README.md")
    kinds = [r.kind for r in refs]
    assert "test" in kinds
    assert "file" in kinds
    test_ref = [r for r in refs if r.kind == "test"][0]
    assert test_ref.target == "ENG-123"


def test_parse_plugin_reference_ignored_when_not_registered():
    refs = parse_context_references("check @test:ENG-123")
    assert [r.kind for r in refs] == []


def test_plugin_pattern_regex():
    m = _PLUGIN_REFERENCE_PATTERN.search("@issue:ENG-123")
    assert m is not None
    assert m.group("kind") == "issue"
    assert m.group("value") == "ENG-123"


# -- expand tests ----------------------------------------------------------

@pytest.mark.asyncio
async def test_expand_plugin_reference(tmp_path: Path):
    from agent.context_references import preprocess_context_references_async

    register_context_reference_provider(_DummyProvider())
    result = await preprocess_context_references_async(
        "check @test:ENG-123",
        cwd=tmp_path,
        context_length=10000,
    )
    assert result.expanded
    assert "expanded: ENG-123" in result.message
    assert "test:ENG-123" not in result.message or "Attached Context" in result.message


@pytest.mark.asyncio
async def test_expand_plugin_returns_none(tmp_path: Path):
    from agent.context_references import preprocess_context_references_async

    register_context_reference_provider(_NoneExpandProvider())
    result = await preprocess_context_references_async(
        "check @skip:foo",
        cwd=tmp_path,
        context_length=10000,
    )
    # expand() returned None, so the reference is parsed but no content injected
    assert not any(r.kind == "skip" and "expanded" in (result.message or "") for r in result.references)


@pytest.mark.asyncio
async def test_expand_plugin_error(tmp_path: Path):
    from agent.context_references import preprocess_context_references_async

    register_context_reference_provider(_ErrorProvider())
    result = await preprocess_context_references_async(
        "check @boom:oops",
        cwd=tmp_path,
        context_length=10000,
    )
    assert result.expanded
    assert "plugin expansion error" in result.message


# -- autocomplete tests ----------------------------------------------------

@pytest.mark.asyncio
async def test_autocomplete():
    p = _DummyProvider()
    register_context_reference_provider(p)
    items = await p.autocomplete("foo", limit=5)
    assert len(items) == 1
    assert items[0].text == "foo-result"


# -- ContextCompletionItem tests -------------------------------------------

def test_completion_item_defaults():
    item = ContextCompletionItem(text="@issue:1")
    assert item.text == "@issue:1"
    assert item.display == "@issue:1"
    assert item.meta == ""


def test_completion_item_custom():
    item = ContextCompletionItem(text="1", display="ENG-1", meta="Bug")
    assert item.display == "ENG-1"
    assert item.meta == "Bug"
