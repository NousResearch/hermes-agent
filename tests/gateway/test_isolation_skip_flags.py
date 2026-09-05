"""Unit tests for the gateway isolation skip-flag resolution."""

import pytest


@pytest.mark.parametrize(
    ("platform_skip_context", "env", "expected"),
    [
        # Default: nothing set.
        (False, {}, (False, False)),
        # Per-platform latency opt-out alone skips context files, not memory.
        (True, {}, (True, False)),
        # Isolation env enables both flags, composing with the opt-out.
        (False, {"HERMES_IGNORE_RULES": "1"}, (True, True)),
        (False, {"HERMES_SAFE_MODE": "1"}, (True, True)),
        (True, {"HERMES_SAFE_MODE": "1"}, (True, True)),
        (False, {"HERMES_IGNORE_RULES": "0", "HERMES_SAFE_MODE": "0"}, (False, False)),
    ],
)
def test_resolve_gateway_isolation_skip_flags(
    monkeypatch, platform_skip_context, env, expected
):
    from gateway.run import _resolve_gateway_isolation_skip_flags

    monkeypatch.delenv("HERMES_IGNORE_RULES", raising=False)
    monkeypatch.delenv("HERMES_SAFE_MODE", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert (
        _resolve_gateway_isolation_skip_flags(platform_skip_context) == expected
    )
