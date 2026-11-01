"""Unit tests for gateway.user_tier — Phase 1 of RFC #20744."""
from __future__ import annotations

from types import SimpleNamespace

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.user_tier import (
    _DEFAULT_USER_TOOLSETS,
    filter_toolsets_for_user,
    resolve_user_tier,
)


def _source(user_id: str = "999", chat_type: str = "dm") -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        user_id=user_id,
        chat_id="c1",
        chat_type=chat_type,
    )


def _cfg_with_admin(admin_ids=None, user_toolsets=None, user_max=None):
    extra = {}
    if admin_ids is not None:
        extra["allow_admin_from"] = admin_ids
    if user_toolsets is not None:
        extra["user_toolsets"] = user_toolsets
    if user_max is not None:
        extra["user_max_iterations"] = user_max
    return {
        "platforms": {
            "discord": extra,
        },
        "gateway": {},
    }


class TestFilterToolsets:
    def test_admin_or_ungated_keeps_all(self):
        base = ["web", "memory", "terminal", "skills"]
        assert filter_toolsets_for_user(base, is_admin=True, tier_gating_enabled=True) == base
        assert filter_toolsets_for_user(base, is_admin=False, tier_gating_enabled=False) == base

    def test_non_admin_default_excludes_sensitive(self):
        base = ["web", "memory", "terminal", "skills", "browser", "cronjob"]
        out = filter_toolsets_for_user(base, is_admin=False, tier_gating_enabled=True)
        assert "web" in out
        assert "browser" in out
        assert "memory" not in out
        assert "terminal" not in out
        assert "skills" not in out
        assert "cronjob" not in out

    def test_explicit_user_toolsets_can_include_sensitive(self):
        base = ["web", "memory", "terminal"]
        out = filter_toolsets_for_user(
            base,
            is_admin=False,
            tier_gating_enabled=True,
            user_toolsets_config=["memory", "web"],
        )
        assert out == ["web", "memory"] or set(out) == {"web", "memory"}


class TestResolveUserTier:
    def test_ungated_when_no_admin_list(self):
        src = _source("999")
        d = resolve_user_tier(
            source=src,
            enabled_toolsets=["web", "memory"],
            max_iterations=40,
            user_config=_cfg_with_admin(admin_ids=None),
        )
        assert d.tier_gating_enabled is False
        assert d.is_admin is True  # ungated → treated as admin for can_run parity
        assert "memory" in d.enabled_toolsets
        assert d.max_iterations == 40

    def test_admin_keeps_full_surface(self):
        src = _source("111")
        d = resolve_user_tier(
            source=src,
            enabled_toolsets=["web", "memory", "skills", "cronjob"],
            max_iterations=40,
            user_config=_cfg_with_admin(admin_ids=["111"], user_max=5),
        )
        assert d.is_admin is True
        assert "memory" in d.enabled_toolsets
        assert d.max_iterations == 40  # no clamp for admin

    def test_non_admin_clamps_tools_and_iterations(self):
        src = _source("999")
        d = resolve_user_tier(
            source=src,
            enabled_toolsets=["web", "memory", "skills", "browser"],
            max_iterations=40,
            user_config=_cfg_with_admin(admin_ids=["111"], user_max=6),
        )
        assert d.is_admin is False
        assert d.tier_gating_enabled is True
        assert "memory" not in d.enabled_toolsets
        assert "skills" not in d.enabled_toolsets
        assert "web" in d.enabled_toolsets
        assert d.max_iterations == 6

    def test_default_user_toolsets_subset(self):
        assert "web" in _DEFAULT_USER_TOOLSETS
        assert "memory" not in _DEFAULT_USER_TOOLSETS


def _live_platform_config(admin_ids=None, user_toolsets=None, user_max=None):
    """Mimic a live PlatformConfig: an object exposing an ``.extra`` dict.

    This is the production shape the gateway hands to ``policy_for_tier`` /
    ``resolve_user_tier`` via ``platform_config=`` — distinct from the
    ``user_config`` fallback dict used by the unit tests above.
    """
    extra = {}
    if admin_ids is not None:
        extra["allow_admin_from"] = admin_ids
    if user_toolsets is not None:
        extra["user_toolsets"] = user_toolsets
    if user_max is not None:
        extra["user_max_iterations"] = user_max
    return SimpleNamespace(extra=extra)


class TestResolveUserTierLivePlatformConfig:
    """Regression tests for the live gateway branch (#67898 review).

    Previously ``policy_for_tier`` called ``policy_for_source(source,
    platform_config)`` with swapped arguments, so on the real gateway turn
    the policy resolved to a disabled/no-op state and neither the non-admin
    toolset filter nor the iteration cap applied. These tests pass a live
    ``PlatformConfig``-shaped object (``.extra``) and assert the restriction
    and clamp actually take effect.
    """

    def test_non_admin_restricted_via_live_platform_config(self):
        src = _source("999")
        d = resolve_user_tier(
            source=src,
            enabled_toolsets=["web", "memory", "skills", "browser"],
            max_iterations=40,
            platform_config=_live_platform_config(
                admin_ids=["111"], user_max=6
            ),
        )
        assert d.is_admin is False
        assert d.tier_gating_enabled is True
        assert "memory" not in d.enabled_toolsets
        assert "skills" not in d.enabled_toolsets
        assert "web" in d.enabled_toolsets
        assert d.max_iterations == 6

    def test_admin_unrestricted_via_live_platform_config(self):
        src = _source("111")
        d = resolve_user_tier(
            source=src,
            enabled_toolsets=["web", "memory", "skills", "cronjob"],
            max_iterations=40,
            platform_config=_live_platform_config(
                admin_ids=["111"], user_max=6
            ),
        )
        assert d.is_admin is True
        assert "memory" in d.enabled_toolsets
        assert d.max_iterations == 40
