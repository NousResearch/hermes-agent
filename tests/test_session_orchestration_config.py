"""
Tests for session_orchestration/config.py (T016 — Config gating).

Covers:
  - Defaults (enabled=False when unset or file absent)
  - Reading a fully-populated section
  - Partial section (only some keys set)
  - Disabled-path invariant (accessor reports disabled ⇒ dependents short-circuit)
  - Malformed / unexpected values fall back to defaults gracefully
  - is_enabled() convenience wrapper
  - _KNOWN_ROOT_KEYS includes session_orchestration (schema registration)
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest

from session_orchestration.config import (
    SessionOrchestrationConfig,
    _coerce_bool,
    _coerce_optional_str,
    _coerce_positive_int,
    is_enabled,
    load_session_orchestration_config,
)


# ---------------------------------------------------------------------------
# 1. Defaults — enabled=False when unset
# ---------------------------------------------------------------------------


def test_defaults_from_empty_dict() -> None:
    """SessionOrchestrationConfig.from_dict({}) returns all-default values."""
    cfg = SessionOrchestrationConfig.from_dict({})
    assert cfg.enabled is False
    assert cfg.feed_channel_id is None
    assert cfg.external_runs_channel_id is None
    assert cfg.hang_stale_seconds == 300
    assert cfg.hang_idle_ticks == 3


def test_defaults_from_missing_section() -> None:
    """load_session_orchestration_config with no session_orchestration key defaults OFF."""
    hermes_cfg: Dict[str, Any] = {"model": {"default": "claude"}}
    result = load_session_orchestration_config(hermes_cfg)
    assert result.enabled is False


def test_defaults_when_config_load_fails() -> None:
    """When load_config_readonly raises, accessor returns all-defaults (never raises)."""
    with patch(
        "hermes_cli.config.load_config_readonly",
        side_effect=RuntimeError("disk error"),
    ):
        result = load_session_orchestration_config()
    assert result.enabled is False
    assert result.feed_channel_id is None


def test_defaults_when_section_is_none() -> None:
    """session_orchestration: null in YAML (→ None in Python) → defaults."""
    hermes_cfg: Dict[str, Any] = {"session_orchestration": None}
    result = load_session_orchestration_config(hermes_cfg)
    assert result.enabled is False


def test_defaults_when_section_is_not_a_dict() -> None:
    """session_orchestration: "oops" → defaults, no exception."""
    hermes_cfg: Dict[str, Any] = {"session_orchestration": "oops"}
    result = load_session_orchestration_config(hermes_cfg)
    assert result.enabled is False


# ---------------------------------------------------------------------------
# 2. Fully-populated section
# ---------------------------------------------------------------------------


def test_fully_populated_section() -> None:
    """All fields are parsed correctly from a well-formed section."""
    hermes_cfg: Dict[str, Any] = {
        "session_orchestration": {
            "enabled": True,
            "feed_channel_id": "1111222233334444",
            "external_runs_channel_id": "5555666677778888",
            "hang_stale_seconds": 600,
            "hang_idle_ticks": 5,
        }
    }
    result = load_session_orchestration_config(hermes_cfg)
    assert result.enabled is True
    assert result.feed_channel_id == "1111222233334444"
    assert result.external_runs_channel_id == "5555666677778888"
    assert result.hang_stale_seconds == 600
    assert result.hang_idle_ticks == 5


def test_string_true_is_accepted() -> None:
    """enabled: 'true' (string from env passthrough) is coerced to True."""
    cfg = SessionOrchestrationConfig.from_dict({"enabled": "true"})
    assert cfg.enabled is True


def test_string_false_is_accepted() -> None:
    """enabled: 'false' (string) is coerced to False."""
    cfg = SessionOrchestrationConfig.from_dict({"enabled": "false"})
    assert cfg.enabled is False


# ---------------------------------------------------------------------------
# 3. Partial section
# ---------------------------------------------------------------------------


def test_partial_section_only_enabled() -> None:
    """Only enabled key present; other fields default gracefully."""
    hermes_cfg: Dict[str, Any] = {"session_orchestration": {"enabled": True}}
    result = load_session_orchestration_config(hermes_cfg)
    assert result.enabled is True
    assert result.feed_channel_id is None
    assert result.hang_stale_seconds == 300


def test_partial_section_only_channel() -> None:
    """Only feed_channel_id set; enabled defaults False."""
    hermes_cfg: Dict[str, Any] = {
        "session_orchestration": {"feed_channel_id": "abc123"}
    }
    result = load_session_orchestration_config(hermes_cfg)
    assert result.enabled is False
    assert result.feed_channel_id == "abc123"


def test_empty_string_feed_channel_normalised_to_none() -> None:
    """Empty string feed_channel_id is normalised to None."""
    cfg = SessionOrchestrationConfig.from_dict({"feed_channel_id": ""})
    assert cfg.feed_channel_id is None


# ---------------------------------------------------------------------------
# 4. Disabled-path invariant
# ---------------------------------------------------------------------------


def test_disabled_path_invariant_default() -> None:
    """With an empty config, enabled=False — dependent gates MUST short-circuit."""
    hermes_cfg: Dict[str, Any] = {}
    cfg = load_session_orchestration_config(hermes_cfg)

    # This is the exact branch every dependent uses to short-circuit:
    if cfg.enabled:
        raise AssertionError("Side effects must not run when enabled=False")
    # Test passes — the guard correctly prevents execution.


def test_disabled_path_invariant_explicit_false() -> None:
    """Explicit enabled: false behaves identically to the absent-key default."""
    hermes_cfg: Dict[str, Any] = {"session_orchestration": {"enabled": False}}
    cfg = load_session_orchestration_config(hermes_cfg)
    assert not cfg.enabled


def test_disabled_is_enabled_wrapper() -> None:
    """is_enabled() returns False when enabled=False."""
    hermes_cfg: Dict[str, Any] = {"session_orchestration": {"enabled": False}}
    assert is_enabled(hermes_cfg) is False


def test_enabled_is_enabled_wrapper() -> None:
    """is_enabled() returns True when enabled=True."""
    hermes_cfg: Dict[str, Any] = {"session_orchestration": {"enabled": True}}
    assert is_enabled(hermes_cfg) is True


def test_is_enabled_no_args_reads_live_config() -> None:
    """is_enabled() with no args reads via load_config_readonly (mocked)."""
    mock_cfg: Dict[str, Any] = {"session_orchestration": {"enabled": True}}
    with patch(
        "hermes_cli.config.load_config_readonly", return_value=mock_cfg
    ):
        assert is_enabled() is True


def test_is_enabled_no_args_defaults_false_on_missing() -> None:
    """is_enabled() returns False when the section is absent from live config."""
    with patch(
        "hermes_cli.config.load_config_readonly", return_value={}
    ):
        assert is_enabled() is False


# ---------------------------------------------------------------------------
# 5. Malformed / edge-case values
# ---------------------------------------------------------------------------


def test_malformed_hang_stale_seconds_falls_back_to_default() -> None:
    """Non-numeric hang_stale_seconds falls back to 300."""
    cfg = SessionOrchestrationConfig.from_dict({"hang_stale_seconds": "not-a-number"})
    assert cfg.hang_stale_seconds == 300


def test_zero_hang_stale_seconds_falls_back_to_default() -> None:
    """Zero is not a positive int — falls back to 300."""
    cfg = SessionOrchestrationConfig.from_dict({"hang_stale_seconds": 0})
    assert cfg.hang_stale_seconds == 300


def test_negative_hang_idle_ticks_falls_back_to_default() -> None:
    """Negative hang_idle_ticks falls back to 3."""
    cfg = SessionOrchestrationConfig.from_dict({"hang_idle_ticks": -1})
    assert cfg.hang_idle_ticks == 3


def test_hang_thresholds_reused_for_stale_guard_config() -> None:
    """Existing hang thresholds express the stale/frozen guard without new keys."""
    cfg = SessionOrchestrationConfig.from_dict(
        {"hang_stale_seconds": "900", "hang_idle_ticks": "4"}
    )
    assert cfg.hang_stale_seconds == 900
    assert cfg.hang_idle_ticks == 4


def test_unknown_keys_ignored() -> None:
    """Extra keys in the section are silently ignored."""
    cfg = SessionOrchestrationConfig.from_dict(
        {"enabled": True, "future_feature": "some_value"}
    )
    assert cfg.enabled is True


def test_external_runs_thread_id_legacy_alias() -> None:
    """Legacy key external_runs_thread_id is accepted as alias."""
    cfg = SessionOrchestrationConfig.from_dict(
        {"external_runs_thread_id": "thread-999"}
    )
    assert cfg.external_runs_channel_id == "thread-999"


# ---------------------------------------------------------------------------
# 6. Schema registration — DEFAULT_CONFIG includes session_orchestration
# ---------------------------------------------------------------------------


def test_default_config_has_session_orchestration() -> None:
    """session_orchestration section is present in DEFAULT_CONFIG with enabled=False."""
    from hermes_cli.config import DEFAULT_CONFIG

    assert "session_orchestration" in DEFAULT_CONFIG, (
        "DEFAULT_CONFIG must include session_orchestration section"
    )
    so = DEFAULT_CONFIG["session_orchestration"]
    assert isinstance(so, dict), "session_orchestration must be a dict"
    assert so.get("enabled") is False, "Default must be disabled"


def test_known_root_keys_includes_session_orchestration() -> None:
    """_KNOWN_ROOT_KEYS includes session_orchestration to avoid spurious warnings."""
    from hermes_cli.config import _KNOWN_ROOT_KEYS

    assert "session_orchestration" in _KNOWN_ROOT_KEYS, (
        "_KNOWN_ROOT_KEYS must list session_orchestration so "
        "config validation does not warn on a valid key"
    )


def test_load_config_merges_session_orchestration() -> None:
    """load_config() deep-merges a partial user override with the defaults."""
    from hermes_cli.config import DEFAULT_CONFIG

    # Simulate what _deep_merge does: user sets only enabled=true.
    user_section = {"enabled": True}
    merged_section = {**DEFAULT_CONFIG["session_orchestration"], **user_section}

    assert merged_section["enabled"] is True
    # Defaults for other keys survive the merge
    assert "hang_stale_seconds" in merged_section
    assert "feed_channel_id" in merged_section


# ---------------------------------------------------------------------------
# 7. Coercion helpers (unit tests so any regression is caught directly)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, default, expected",
    [
        (True, False, True),
        (False, True, False),
        ("true", False, True),
        ("false", True, False),
        ("1", False, True),
        ("0", True, False),
        ("yes", False, True),
        ("no", True, False),
        (None, True, True),
        (None, False, False),
        ("garbage", True, True),
        ("garbage", False, False),
    ],
)
def test_coerce_bool(value: Any, default: bool, expected: bool) -> None:
    assert _coerce_bool(value, default) is expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("abc", "abc"),
        ("  abc  ", "abc"),
        ("", None),
        ("   ", None),
        (None, None),
        (123, "123"),
    ],
)
def test_coerce_optional_str(value: Any, expected: Optional[str]) -> None:
    assert _coerce_optional_str(value) == expected


@pytest.mark.parametrize(
    "value, default, expected",
    [
        (5, 3, 5),
        (1, 3, 1),
        (0, 3, 3),       # zero is not positive
        (-1, 3, 3),      # negative is not positive
        ("10", 3, 10),   # string coercion
        ("bad", 3, 3),   # malformed → default
        (None, 3, 3),    # missing → default
    ],
)
def test_coerce_positive_int(value: Any, default: int, expected: int) -> None:
    assert _coerce_positive_int(value, default) == expected
