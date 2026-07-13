"""Regression tests for #31996 — toolset-vs-provider consistency warnings.

Issue #31996 sub-issues 1 and 3 trace the same failure: a user wires a
backend (``image_gen.provider: minimax`` or
``auxiliary.vision.provider: openrouter``) but their explicit
``platform_toolsets`` list omits the matching toolset.  The plugin
loads, the provider resolves, the runtime client connects — but the
``image_generate`` / ``vision_analyze`` schema never reaches the LLM,
so the model never knows the tool exists.

Sub-issue 2 trace: ``auxiliary.<task>.api_key`` is silently ignored
when ``provider`` is empty/auto — the resolver falls through to the
"auto" branch which reads each backend's env var directly.

The validators below surface both classes as ``ConfigIssue`` warnings
that ``hermes doctor`` and the startup config-warning printer pick up.
"""

from __future__ import annotations

import inspect

import pytest

from hermes_cli.config import (
    ConfigIssue,
    _validate_auxiliary_api_key_has_provider,
    _validate_tool_provider_consistency,
    validate_config_structure,
)


# ---------------------------------------------------------------------------
# Tool ↔ provider consistency
# ---------------------------------------------------------------------------


class TestImageGenToolsetConsistency:
    """Detect ``image_gen.provider`` set + ``image_gen`` toolset missing."""

    def test_explicit_toolsets_missing_image_gen_with_provider_set_warns(self):
        config = {
            "image_gen": {"provider": "minimax"},
            "platform_toolsets": {"cli": ["web", "vision"]},
        }
        issues = _validate_tool_provider_consistency(config)
        assert any(
            "image_gen.provider" in i.message and "cli" in i.message
            for i in issues
        ), f"got: {[i.message for i in issues]}"

    def test_explicit_toolsets_with_image_gen_listed_does_not_warn(self):
        config = {
            "image_gen": {"provider": "minimax"},
            "platform_toolsets": {"cli": ["web", "image_gen"]},
        }
        issues = _validate_tool_provider_consistency(config)
        assert not any("image_gen.provider" in i.message for i in issues)

    def test_explicit_hermes_cli_composite_includes_image_gen(self):
        """The ``hermes-cli`` composite already wires ``image_generate``.

        Users who keep the default composite shouldn't be flagged —
        that's the out-of-the-box behaviour and it works.
        """
        config = {
            "image_gen": {"provider": "minimax"},
            "platform_toolsets": {"cli": ["hermes-cli"]},
        }
        issues = _validate_tool_provider_consistency(config)
        assert not any("image_gen.provider" in i.message for i in issues)

    def test_implicit_default_does_not_warn(self):
        """No ``platform_toolsets`` → falls back to ``hermes-cli`` defaults.

        Until the user explicitly narrows the toolset list, the runtime
        already exposes ``image_generate``.  Warning here would be
        noise on every fresh install.
        """
        config = {"image_gen": {"provider": "minimax"}}
        issues = _validate_tool_provider_consistency(config)
        assert not issues

    def test_disabled_image_gen_provider_does_not_warn(self):
        for sentinel in ("none", "off", "disabled", ""):
            config = {
                "image_gen": {"provider": sentinel},
                "platform_toolsets": {"cli": ["web"]},
            }
            issues = _validate_tool_provider_consistency(config)
            assert not any(
                "image_gen.provider" in i.message for i in issues
            ), f"sentinel {sentinel!r} unexpectedly warned"

    def test_warning_hint_offers_concrete_fix(self):
        """Hint text must show the user *exactly* what to add to their YAML."""
        config = {
            "image_gen": {"provider": "minimax"},
            "platform_toolsets": {"cli": ["web"]},
        }
        issues = _validate_tool_provider_consistency(config)
        hints = [i.hint for i in issues if "image_gen.provider" in i.message]
        assert hints
        h = hints[0]
        assert "platform_toolsets" in h
        assert "image_gen" in h
        assert "cli" in h


class TestVisionToolsetConsistency:
    """Detect ``auxiliary.vision`` configured + ``vision`` toolset missing."""

    @pytest.mark.parametrize(
        "vision_cfg",
        [
            {"provider": "openrouter"},
            {"model": "google/gemini-3-flash-preview"},
            {"base_url": "https://api.example/v1"},
            {"api_key": "sk-or-v1-…"},
        ],
    )
    def test_any_vision_field_set_with_missing_toolset_warns(self, vision_cfg):
        config = {
            "auxiliary": {"vision": vision_cfg},
            "platform_toolsets": {"cli": ["web", "image_gen"]},
        }
        issues = _validate_tool_provider_consistency(config)
        assert any(
            "auxiliary.vision" in i.message and "cli" in i.message
            for i in issues
        )

    def test_vision_toolset_listed_does_not_warn(self):
        config = {
            "auxiliary": {"vision": {"provider": "openrouter"}},
            "platform_toolsets": {"cli": ["vision"]},
        }
        issues = _validate_tool_provider_consistency(config)
        assert not any("auxiliary.vision" in i.message for i in issues)

    def test_empty_vision_section_does_not_warn(self):
        for empty in ({}, {"provider": ""}, {"provider": "  "}):
            config = {
                "auxiliary": {"vision": empty},
                "platform_toolsets": {"cli": ["web"]},
            }
            issues = _validate_tool_provider_consistency(config)
            assert not any(
                "auxiliary.vision" in i.message for i in issues
            ), f"empty {empty!r} unexpectedly warned"


class TestMultiPlatformAggregation:
    """The warning enumerates *every* platform that's missing the tool."""

    def test_multiple_bad_platforms_listed_in_single_warning(self):
        config = {
            "image_gen": {"provider": "minimax"},
            "platform_toolsets": {
                "cli": ["web"],
                "discord": ["web"],
                "telegram": ["hermes-cli"],  # OK — composite includes image_generate
            },
        }
        issues = _validate_tool_provider_consistency(config)
        msgs = [i.message for i in issues if "image_gen.provider" in i.message]
        assert msgs
        m = msgs[0]
        # Both bad platforms surfaced; the composite-using one is silent.
        assert "cli" in m
        assert "discord" in m
        assert "telegram" not in m


# ---------------------------------------------------------------------------
# auxiliary.<task>.api_key without provider warning
# ---------------------------------------------------------------------------


class TestAuxiliaryApiKeyWithoutProvider:
    """``api_key`` set without ``provider`` is a silent-drop trap (#31996.2)."""

    def test_api_key_without_provider_warns(self):
        config = {"auxiliary": {"vision": {"api_key": "sk-or-v1-…"}}}
        issues = _validate_auxiliary_api_key_has_provider(config)
        assert any(
            "auxiliary.vision.api_key" in i.message
            and "ignored" in i.message.lower()
            for i in issues
        )

    def test_api_key_with_provider_does_not_warn(self):
        config = {
            "auxiliary": {
                "vision": {
                    "api_key": "sk-or-v1-…",
                    "provider": "openrouter",
                },
            },
        }
        issues = _validate_auxiliary_api_key_has_provider(config)
        assert not issues

    def test_api_key_with_explicit_base_url_does_not_warn(self):
        """``base_url`` alone routes the key through the custom endpoint.

        ``_resolve_task_provider_model`` returns ``"custom"`` whenever
        ``base_url`` is set, and the api_key flows through that branch
        even without an explicit provider.  Don't false-positive.
        """
        config = {
            "auxiliary": {
                "vision": {
                    "api_key": "sk-…",
                    "base_url": "https://api.example.com/v1",
                },
            },
        }
        issues = _validate_auxiliary_api_key_has_provider(config)
        assert not issues

    def test_provider_auto_with_api_key_warns(self):
        """``provider: auto`` short-circuits to env-var lookup → key ignored."""
        config = {
            "auxiliary": {
                "vision": {
                    "api_key": "sk-or-v1-…",
                    "provider": "auto",
                },
            },
        }
        issues = _validate_auxiliary_api_key_has_provider(config)
        assert any("auxiliary.vision.api_key" in i.message for i in issues)

    def test_multiple_tasks_each_get_their_own_warning(self):
        config = {
            "auxiliary": {
                "vision":   {"api_key": "sk-1"},
                "compression": {"api_key": "sk-2"},
            },
        }
        issues = _validate_auxiliary_api_key_has_provider(config)
        msgs = [i.message for i in issues]
        assert any("auxiliary.vision.api_key" in m for m in msgs)
        assert any("auxiliary.compression.api_key" in m for m in msgs)

    def test_empty_or_whitespace_api_key_does_not_warn(self):
        for empty in ("", "   ", None):
            config = {"auxiliary": {"vision": {"api_key": empty}}}
            issues = _validate_auxiliary_api_key_has_provider(config)
            assert not issues, f"unexpected warning for empty key {empty!r}"


# ---------------------------------------------------------------------------
# Integration with validate_config_structure (the doctor entry point)
# ---------------------------------------------------------------------------


class TestValidateConfigStructureIntegration:
    """Both new checks are wired into the doctor's structural validator."""

    def test_validate_config_structure_surfaces_image_gen_warning(self):
        config = {
            "image_gen": {"provider": "minimax"},
            "platform_toolsets": {"cli": ["web"]},
        }
        issues = validate_config_structure(config)
        assert any(
            "image_gen.provider" in i.message
            for i in issues
        ), f"got: {[i.message for i in issues]}"

    def test_validate_config_structure_surfaces_aux_api_key_warning(self):
        config = {"auxiliary": {"vision": {"api_key": "sk-…"}}}
        issues = validate_config_structure(config)
        assert any("auxiliary.vision.api_key" in i.message for i in issues)

    def test_clean_config_does_not_introduce_new_issues(self):
        """A correctly-wired config produces no #31996-related warnings."""
        config = {
            "image_gen": {"provider": "minimax"},
            "auxiliary": {
                "vision": {"provider": "openrouter", "api_key": "sk-or-v1-…"},
            },
            "platform_toolsets": {"cli": ["hermes-cli"]},
        }
        issues = validate_config_structure(config)
        offending = [
            i for i in issues
            if "image_gen.provider" in i.message
            or "auxiliary.vision" in i.message
        ]
        assert not offending, f"unexpected: {[i.message for i in offending]}"


# ---------------------------------------------------------------------------
# Source-level guards
# ---------------------------------------------------------------------------


class TestSourceGuards:
    """Pin the integration so accidental removal is loud at code review."""

    def test_validate_config_structure_calls_both_helpers(self):
        src = inspect.getsource(validate_config_structure)
        assert "_validate_tool_provider_consistency" in src
        assert "_validate_auxiliary_api_key_has_provider" in src
        assert "31996" in src

    def test_helpers_return_typed_config_issues(self):
        config = {
            "image_gen": {"provider": "minimax"},
            "platform_toolsets": {"cli": ["web"]},
        }
        for issue in _validate_tool_provider_consistency(config):
            assert isinstance(issue, ConfigIssue)
            assert issue.severity in {"error", "warning"}
            assert issue.message
            assert issue.hint
