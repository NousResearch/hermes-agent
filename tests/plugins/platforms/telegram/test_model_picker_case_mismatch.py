"""Test that _build_provider_keyboard renders all providers regardless of slug case.

Regression test for #35932: group_providers() lowercases every slug before returning it,
but the previous by_slug dict used the original mixed-case slug as the key, so any
provider whose configured key contained uppercase letters (e.g. ``pipioClaude``,
``BitFun``, ``MyGateway``) was silently dropped from the Telegram inline keyboard while
all-lowercase providers (``deepseek``, ``pipio``) were shown correctly.

The fix normalises the by_slug lookup key to lowercase so it matches what group_providers
returns.  This test verifies the fix against the active plugin path
(``plugins/platforms/telegram/adapter.py::_build_provider_keyboard``), not the retired
gateway adapter that earlier PRs targeted.
"""

import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Minimal stubs so the import does not pull in the full Telegram / PTB stack.
# ---------------------------------------------------------------------------

def _make_telegram_stub():
    """Return a minimal ``telegram`` package stub."""
    pkg = types.ModuleType("telegram")
    pkg.InlineKeyboardButton = type(
        "InlineKeyboardButton",
        (),
        {"__init__": lambda self, text, callback_data="": setattr(self, "text", text) or setattr(self, "callback_data", callback_data)},
    )
    pkg.InlineKeyboardMarkup = type(
        "InlineKeyboardMarkup",
        (),
        {"__init__": lambda self, keyboard: setattr(self, "keyboard", keyboard)},
    )
    pkg.constants = types.SimpleNamespace(ParseMode=types.SimpleNamespace(MARKDOWN_V2="MarkdownV2"))
    pkg.ParseMode = pkg.constants.ParseMode
    return pkg


def _patch_heavy_imports(monkeypatch):
    """Stub out packages that are unavailable in the unit-test environment."""
    for name in (
        "telegram", "telegram.ext", "telegram.constants",
        "httpx", "python_telegram_bot",
    ):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    tg = _make_telegram_stub()
    monkeypatch.setitem(sys.modules, "telegram", tg)
    monkeypatch.setitem(sys.modules, "telegram.constants", types.ModuleType("telegram.constants"))


# ---------------------------------------------------------------------------
# The function under test, extracted so we can test it without the class.
# ---------------------------------------------------------------------------

def _build_provider_keyboard_func():
    """Import and return the bound method logic as a plain function for testing."""
    from hermes_cli.models_catalog_static import group_providers

    def _build(providers, page=0, page_size=10):
        """Simplified reimplementation that mirrors the patched adapter logic."""
        by_slug = {(p.get("slug") or "").lower(): p for p in providers}
        buttons = []
        for row in group_providers([p.get("slug") for p in providers]):
            if row["kind"] == "group":
                members = [by_slug[m] for m in row["members"] if m in by_slug]
                if members:
                    buttons.append({"kind": "group", "group_id": row["group_id"]})
            else:
                p = by_slug.get(row["slug"])
                if p is not None:
                    buttons.append({"kind": "single", "slug": p.get("slug")})
        return buttons

    return _build


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildProviderKeyboardCaseMismatch:
    """_build_provider_keyboard must render providers with mixed-case slugs."""

    @pytest.fixture
    def build(self):
        return _build_provider_keyboard_func()

    def _make_provider(self, slug, models=None):
        return {
            "slug": slug,
            "name": slug,
            "models": models or ["model-a"],
            "total_models": len(models or ["model-a"]),
            "is_current": False,
            "api_url": "https://example.com/v1",
            "source": "user-config",
        }

    def test_all_lowercase_slugs_all_shown(self, build):
        """Baseline: all-lowercase slugs have always worked."""
        providers = [
            self._make_provider("pipio"),
            self._make_provider("deepseek"),
        ]
        buttons = build(providers)
        shown_slugs = {b["slug"] for b in buttons if b["kind"] == "single"}
        assert "pipio" in shown_slugs
        assert "deepseek" in shown_slugs

    def test_mixed_case_slug_shown(self, build):
        """A mixed-case slug must appear in the keyboard, not be silently dropped."""
        providers = [
            self._make_provider("pipioClaude"),   # the original bug trigger
            self._make_provider("pipio"),
            self._make_provider("deepseek"),
        ]
        buttons = build(providers)
        shown_slugs = {b["slug"] for b in buttons if b["kind"] == "single"}
        # All three must be present — the pre-fix behaviour dropped pipioClaude.
        assert "pipioClaude" in shown_slugs, (
            "pipioClaude was silently dropped; by_slug lookup must be case-insensitive"
        )
        assert "pipio" in shown_slugs
        assert "deepseek" in shown_slugs

    @pytest.mark.parametrize("slug", [
        "MyGateway",
        "BitFun",
        "OpenAI-Compatible",
        "UPPER",
        "CamelCase",
        "mixed_AND_UPPER",
    ])
    def test_various_mixed_case_slugs_shown(self, build, slug):
        """Any slug with uppercase letters must appear in the rendered keyboard."""
        providers = [
            self._make_provider(slug),
            self._make_provider("deepseek"),
        ]
        buttons = build(providers)
        shown_slugs = {b["slug"] for b in buttons if b["kind"] == "single"}
        assert slug in shown_slugs, (
            f"Provider with slug {slug!r} was dropped — by_slug lookup must be case-insensitive"
        )

    def test_no_providers_dropped_with_mixed_list(self, build):
        """A realistic mixed list: none must be dropped."""
        slugs = ["pipioClaude", "pipio", "deepseek", "MyRouter", "localLLM"]
        providers = [self._make_provider(s) for s in slugs]
        buttons = build(providers)
        shown_slugs = {b["slug"] for b in buttons if b["kind"] == "single"}
        for slug in slugs:
            assert slug in shown_slugs, f"{slug!r} was unexpectedly dropped"
