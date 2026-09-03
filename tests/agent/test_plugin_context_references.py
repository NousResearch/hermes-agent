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
    _all_context_reference_providers,
    _context_reference_providers,
    _scoped_context_reference_providers,
    get_context_reference_providers,
    parse_context_references,
    register_context_reference_provider,
    restore_context_reference_registration,
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
    _scoped_context_reference_providers.clear()
    yield
    _context_reference_providers.clear()
    _scoped_context_reference_providers.clear()


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


# -- profile-scoping tests (agent.context_references low-level API) --------
#
# register_context_reference_provider() previously wrote into one bare,
# process-global dict with no scope key at all — a multiplex/Team-Gateway
# process running one PluginManager per profile would have two profiles'
# plugins collide on the same @prefix (ValueError, silently swallowed by
# PluginContext.register_context_reference, just a log warning) whenever
# they registered the same prefix, and neither profile's registration was
# tracked in the ownership ledger, so unload/force-reload could never free
# a prefix for re-registration either.


class TestScopedRegistration:
    def test_two_scopes_can_register_the_same_prefix(self):
        provider_a = _DummyProvider()
        provider_b = _DummyProvider()
        register_context_reference_provider(provider_a, scope="profile-a")
        register_context_reference_provider(provider_b, scope="profile-b")

        assert _all_context_reference_providers(scope="profile-a")["test"] is provider_a
        assert _all_context_reference_providers(scope="profile-b")["test"] is provider_b

    def test_same_scope_duplicate_prefix_still_rejected(self):
        register_context_reference_provider(_DummyProvider(), scope="profile-a")
        with pytest.raises(ValueError, match="already registered"):
            register_context_reference_provider(_DummyProvider(), scope="profile-a")

    def test_scoped_registration_is_invisible_to_other_scopes(self):
        register_context_reference_provider(_DummyProvider(), scope="profile-a")
        assert "test" not in _all_context_reference_providers(scope="profile-b")
        assert "test" not in _all_context_reference_providers(scope=None)

    def test_scope_none_still_uses_the_legacy_global_dict(self):
        """Backward compatibility: callers that don't pass scope= are
        unaffected by the scoping change."""
        provider = _DummyProvider()
        register_context_reference_provider(provider)
        assert _context_reference_providers["test"] is provider
        assert _scoped_context_reference_providers == {}

    def test_restore_removes_the_registration(self):
        provider = _DummyProvider()
        register_context_reference_provider(provider, scope="profile-a")

        ok = restore_context_reference_registration(
            "test", provider, None, scope="profile-a"
        )

        assert ok is True
        assert "test" not in _all_context_reference_providers(scope="profile-a")
        # The now-empty per-scope bucket is dropped, not left as a stale
        # empty dict entry.
        assert "profile-a" not in _scoped_context_reference_providers

    def test_restore_is_a_noop_when_current_no_longer_installed(self):
        """A stale dispose callback (from an earlier generation) must not
        evict whatever is CURRENTLY registered under the prefix."""
        stale_provider = _DummyProvider()
        live_provider = _DummyProvider()
        register_context_reference_provider(live_provider, scope="profile-a")

        ok = restore_context_reference_registration(
            "test", stale_provider, None, scope="profile-a"
        )

        assert ok is False
        assert _all_context_reference_providers(scope="profile-a")["test"] is live_provider

    def test_unregister_then_reregister_the_same_prefix_succeeds(self):
        """The bug's second half: after a clean unload, the SAME prefix must
        be re-registerable — not permanently stuck 'already registered'."""
        first = _DummyProvider()
        register_context_reference_provider(first, scope="profile-a")
        assert restore_context_reference_registration(
            "test", first, None, scope="profile-a"
        )

        second = _DummyProvider()
        register_context_reference_provider(second, scope="profile-a")  # must not raise

        assert _all_context_reference_providers(scope="profile-a")["test"] is second


class TestPluginContextRegistrationIsScopedAndTracked:
    """Integration level: PluginContext.register_context_reference wires
    scope + ownership-ledger tracking, matching every sibling register_*
    method (register_tool, register_secret_source, ...)."""

    def _ctx(self, scope_key: str):
        from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

        manager = PluginManager(scope_key=scope_key)
        return PluginContext(
            PluginManifest(name="ctx-ref-fixture", source="user"), manager
        )

    def test_returns_a_registration_handle(self):
        ctx = self._ctx("profile-a")
        handle = ctx.register_context_reference(_DummyProvider())
        assert handle is not None

    def test_invalid_provider_returns_none_not_a_handle(self):
        ctx = self._ctx("profile-a")
        assert ctx.register_context_reference("not a provider") is None

    def test_two_profiles_register_the_same_prefix_without_colliding(self):
        ctx_a = self._ctx("profile-a")
        ctx_b = self._ctx("profile-b")
        provider_a = _DummyProvider()
        provider_b = _DummyProvider()

        handle_a = ctx_a.register_context_reference(provider_a)
        handle_b = ctx_b.register_context_reference(provider_b)

        assert handle_a is not None
        assert handle_b is not None, (
            "a second profile registering the same @prefix must not be "
            "rejected by the first profile's registration"
        )
        assert _all_context_reference_providers(scope="profile-a")["test"] is provider_a
        assert _all_context_reference_providers(scope="profile-b")["test"] is provider_b

    def test_disposing_the_handle_frees_the_prefix_for_reregistration(self):
        ctx = self._ctx("profile-a")
        handle = ctx.register_context_reference(_DummyProvider())
        assert handle is not None

        handle.dispose()

        assert "test" not in _all_context_reference_providers(scope="profile-a")
        second_handle = ctx.register_context_reference(_DummyProvider())
        assert second_handle is not None, (
            "disposing the first registration must free the prefix — a "
            "stale entry must not permanently reject re-registration"
        )
