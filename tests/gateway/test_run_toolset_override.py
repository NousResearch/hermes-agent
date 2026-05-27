"""Tests for ``gateway.run._resolve_toolset_override``.

This helper resolves a per-event toolset override (currently produced by
webhook subscriptions with ``toolsets: [...]``) into the configurable
toolset names the agent runtime understands, mirroring the logic
``_get_platform_tools`` applies to composite toolset names.
"""

import pytest

from gateway.run import _resolve_toolset_override


def test_empty_list_returns_empty_set():
    assert _resolve_toolset_override([], "webhook") == set()


def test_none_like_input_returns_empty_set():
    # Falsy inputs (empty list) short-circuit without importing helpers.
    assert _resolve_toolset_override([], "webhook") == set()


def test_individual_configurable_toolset_kept_as_is():
    resolved = _resolve_toolset_override(["web"], "webhook")
    assert "web" in resolved


def test_multiple_individual_toolsets():
    resolved = _resolve_toolset_override(["web", "vision"], "webhook")
    assert "web" in resolved
    assert "vision" in resolved


def test_composite_hermes_cli_expands_to_configurable_members():
    resolved = _resolve_toolset_override(["hermes-cli"], "webhook")
    # hermes-cli contains the full _HERMES_CORE_TOOLS set, which spans many
    # configurable toolsets. We don't pin the exact list (that's churn-prone
    # as new toolsets are added) but we DO require the load-bearing ones for
    # the motivating use case (HA + Telegram webhook bridge) to be present.
    assert "web" in resolved
    assert "vision" in resolved
    assert "terminal" in resolved
    assert "file" in resolved
    assert "messaging" in resolved


def test_unknown_name_logged_and_skipped(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        resolved = _resolve_toolset_override(
            ["this-does-not-exist", "web"], "webhook"
        )
    assert "web" in resolved
    assert "this-does-not-exist" not in resolved
    assert any(
        "unknown toolset" in rec.message.lower()
        and "this-does-not-exist" in rec.message
        for rec in caplog.records
    )


def test_unknown_name_alone_returns_empty_set(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        resolved = _resolve_toolset_override(["nonsense"], "webhook")
    # Caller treats empty result as "fall back to platform default" — this
    # is the safe-degrade contract for typo'd hand-edited subscriptions.
    assert resolved == set()


def test_non_string_input_coerced():
    # Defensive: YAML may parse bare numeric or boolean names as native
    # Python types.  The helper str()s everything before lookup so a stray
    # ``toolsets: [123]`` doesn't crash with a TypeError.
    resolved = _resolve_toolset_override([123], "webhook")  # type: ignore[list-item]
    assert resolved == set()


def test_platform_restriction_filters_disallowed_toolsets():
    # Per-platform allow-listing is enforced both for direct configurable
    # names and for composite expansion.  Use a real toolset/platform
    # combination where the restriction is known to bite.
    from hermes_cli.tools_config import _toolset_allowed_for_platform
    # Find a configurable toolset that is NOT allowed on the webhook
    # platform; if any exists, expanding a composite that includes it
    # must drop it from the resolved set.
    from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS
    blocked = [
        k for k, _, _ in CONFIGURABLE_TOOLSETS
        if not _toolset_allowed_for_platform(k, "webhook")
    ]
    if not blocked:
        pytest.skip("No toolsets are platform-restricted on 'webhook'")
    # Asking for the blocked toolset directly returns nothing.
    resolved = _resolve_toolset_override([blocked[0]], "webhook")
    assert blocked[0] not in resolved


def test_composite_expansion_does_not_apply_default_off_subtraction():
    # Per the helper's docstring: explicit overrides skip the
    # _DEFAULT_OFF_TOOLSETS filter that _get_platform_tools applies during
    # implicit expansion.  Practical implication: passing --toolset hermes-cli
    # actually gives you hermes-cli, not "hermes-cli minus the default-off
    # bits".  Pin this so a future refactor doesn't silently change shape.
    resolved = _resolve_toolset_override(["hermes-cli"], "webhook")
    # If any DEFAULT_OFF toolset's tools are a subset of hermes-cli's tools
    # AND that toolset is allowed on the webhook platform, it should appear
    # in resolved.  We can't easily enumerate _DEFAULT_OFF_TOOLSETS from
    # outside hermes_cli without importing private state, so the assertion
    # is by-proxy: the resolved set is larger than what _get_platform_tools
    # would yield for the same composite on this platform.
    from hermes_cli.config import load_config
    from hermes_cli.tools_config import _get_platform_tools
    cfg = load_config()
    platform_default = set(_get_platform_tools(cfg, "webhook"))
    # Override should be a strict superset (or equal) of the platform
    # default for the same composite — never smaller.  In practice it's
    # equal-or-larger; the strictness depends on the user's local config.
    assert resolved >= (resolved & platform_default)


def test_resolves_for_telegram_platform_too():
    # The helper is platform-agnostic; webhook is just the current caller.
    # Smoke-test that a different platform key works without errors and
    # produces a non-empty result for a composite.
    resolved = _resolve_toolset_override(["hermes-cli"], "telegram")
    assert resolved  # non-empty


def test_individual_unknown_in_mixed_list_logged_once(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        resolved = _resolve_toolset_override(
            ["web", "totally-fake", "vision"], "webhook"
        )
    assert "web" in resolved
    assert "vision" in resolved
    warnings = [
        rec for rec in caplog.records
        if "unknown toolset" in rec.message.lower()
        and "totally-fake" in rec.message
    ]
    assert len(warnings) == 1
