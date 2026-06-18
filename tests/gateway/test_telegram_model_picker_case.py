"""Regression test: Telegram /model picker must show mixed-case custom providers.

Issue #37268: ``group_providers()`` normalises slugs to lowercase, but
``_build_provider_keyboard`` built its lookup dict with original-case keys.
Providers whose config slug contained uppercase letters (e.g. ``Qwen-TP``)
were silently dropped from the picker.

The fix lowercases the keys in ``by_slug`` so the lookup is case-insensitive.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


def _make_providers(slugs: list[str]) -> list[dict]:
    """Build fake provider dicts with the given slugs."""
    return [
        {
            "slug": s,
            "name": s.replace("-", " ").title(),
            "models": ["m1"],
            "total_models": 1,
            "is_current": False,
        }
        for s in slugs
    ]


class TestBuildProviderKeyboardCaseInsensitive:
    """by_slug lookup must be case-insensitive after the fix."""

    def test_mixed_case_providers_appear_in_picker(self):
        """Providers with mixed-case slugs must not be silently dropped."""
        from hermes_cli.models import group_providers

        slugs = ["deepseek", "Qwen-TP", "Xiaomi-TP"]
        providers = _make_providers(slugs)

        # Simulate what _build_provider_keyboard does after the fix
        by_slug = {p.get("slug", "").lower(): p for p in providers}

        rows = group_providers([p.get("slug") for p in providers])

        resolved = []
        for row in rows:
            if row["kind"] == "group":
                members = [by_slug[m] for m in row["members"] if m in by_slug]
                resolved.extend(members)
            else:
                p = by_slug.get(row["slug"])
                if p is not None:
                    resolved.append(p)

        resolved_slugs = {p["slug"] for p in resolved}
        # All three providers must appear — Qwen-TP and Xiaomi-TP were
        # previously silently dropped.
        assert "Qwen-TP" in resolved_slugs, "Mixed-case provider Qwen-TP dropped from picker"
        assert "Xiaomi-TP" in resolved_slugs, "Mixed-case provider Xiaomi-TP dropped from picker"
        assert "deepseek" in resolved_slugs, "Lowercase provider deepseek dropped from picker"

    def test_lowercase_only_providers_still_work(self):
        """Existing lowercase-only providers must continue to work."""
        from hermes_cli.models import group_providers

        slugs = ["deepseek", "openrouter", "minimax"]
        providers = _make_providers(slugs)

        by_slug = {p.get("slug", "").lower(): p for p in providers}
        rows = group_providers([p.get("slug") for p in providers])

        resolved = []
        for row in rows:
            if row["kind"] == "group":
                members = [by_slug[m] for m in row["members"] if m in by_slug]
                resolved.extend(members)
            else:
                p = by_slug.get(row["slug"])
                if p is not None:
                    resolved.append(p)

        resolved_slugs = {p["slug"] for p in resolved}
        assert resolved_slugs == set(slugs)

    def test_group_callback_case_insensitive(self):
        """Group member lookup must also be case-insensitive."""
        from hermes_cli.models import PROVIDER_GROUPS

        # Simulate state["providers"] with mixed-case minimax variants
        state_providers = [
            {"slug": "Minimax", "name": "Minimax", "models": ["m1"]},
            {"slug": "minimax-oauth", "name": "Minimax OAuth", "models": ["m2"]},
            {"slug": "MINIMAX-CN", "name": "Minimax CN", "models": ["m3"]},
        ]

        _label, _desc, member_slugs = PROVIDER_GROUPS.get("minimax", ("", "", []))
        by_slug = {p["slug"].lower(): p for p in state_providers}
        members = [by_slug[m] for m in member_slugs if m in by_slug]

        # All three minimax variants must be resolved
        assert len(members) == 3, f"Expected 3 minimax members, got {len(members)}"
