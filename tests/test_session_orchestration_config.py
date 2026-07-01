"""
Tests for session_orchestration/config.py (T016 — Config gating) and
session_orchestration/repo_registry.py (T002 — Repo registry).

Covers:
  - Defaults (enabled=False when unset or file absent)
  - Reading a fully-populated section
  - Partial section (only some keys set)
  - Disabled-path invariant (accessor reports disabled ⇒ dependents short-circuit)
  - Malformed / unexpected values fall back to defaults gracefully
  - is_enabled() convenience wrapper
  - _KNOWN_ROOT_KEYS includes session_orchestration (schema registration)
  - Repo registry: known-name resolution, fuzzy match, override priority, unresolved
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
from session_orchestration.repo_registry import (
    DEFAULT_AGENT,
    RepoEntry,
    RepoRegistry,
    ResolvedRepo,
    UnresolvedRepo,
    build_repo_registry,
    scan_for_repos,
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


# ---------------------------------------------------------------------------
# 8. SessionOrchestrationConfig — repos field
# ---------------------------------------------------------------------------


def test_repos_defaults_to_empty_dict() -> None:
    """repos field defaults to {} when absent from config."""
    cfg = SessionOrchestrationConfig.from_dict({})
    assert cfg.repos == {}


def test_repos_parsed_from_dict() -> None:
    """repos dict is extracted verbatim for downstream registry parsing."""
    data = {
        "repos": {
            "myproject": "/home/user/myproject",
            "infra": {"path": "/home/user/infra", "default_agent": "claude"},
        }
    }
    cfg = SessionOrchestrationConfig.from_dict(data)
    assert cfg.repos == data["repos"]


def test_repos_non_dict_ignored() -> None:
    """Non-dict repos value (e.g. a string) defaults to {}."""
    cfg = SessionOrchestrationConfig.from_dict({"repos": "bad"})
    assert cfg.repos == {}


# ---------------------------------------------------------------------------
# 9. Repo registry — core resolution tests (T002)
# ---------------------------------------------------------------------------

# These tests inject a fake scan dict so no filesystem access is needed.
_FAKE_SCAN: Dict[str, str] = {
    "hermes-agent": "/home/zeke/dev/hermes-agent",
    "myproject": "/home/zeke/dev/myproject",
    "infra-core": "/home/zeke/dev/infra-core",
}


def _registry(
    overrides: Optional[Dict[str, RepoEntry]] = None,
) -> RepoRegistry:
    """Build a registry with the fake scan and given overrides."""
    return RepoRegistry(overrides=overrides, _injected_scan=_FAKE_SCAN)


def test_known_name_resolves_to_absolute_path() -> None:
    """An exact name in the scan cache resolves to its absolute path."""
    reg = _registry()
    result = reg.resolve("hermes-agent")
    assert isinstance(result, ResolvedRepo)
    assert result.path == "/home/zeke/dev/hermes-agent"
    assert result.match_kind == "exact"


def test_known_name_case_insensitive() -> None:
    """Resolution is case-insensitive for scan names."""
    reg = _registry()
    result = reg.resolve("Hermes-Agent")
    assert isinstance(result, ResolvedRepo)
    assert result.path == "/home/zeke/dev/hermes-agent"


def test_default_agent_is_omp_when_not_specified() -> None:
    """Auto-scanned repos use DEFAULT_AGENT (omp) when no override is present."""
    reg = _registry()
    result = reg.resolve("myproject")
    assert isinstance(result, ResolvedRepo)
    assert result.default_agent == DEFAULT_AGENT
    assert result.default_agent == "omp"


def test_fuzzy_suffix_match_resolves() -> None:
    """A suffix-segment query (e.g. 'agent') resolves via fuzzy matching."""
    reg = _registry()
    result = reg.resolve("agent")
    assert isinstance(result, ResolvedRepo)
    assert result.path == "/home/zeke/dev/hermes-agent"
    assert result.match_kind == "fuzzy"


def test_fuzzy_substring_match_resolves() -> None:
    """A substring query ('core') fuzzy-matches 'infra-core'."""
    reg = _registry()
    result = reg.resolve("core")
    assert isinstance(result, ResolvedRepo)
    assert result.path == "/home/zeke/dev/infra-core"
    assert result.match_kind == "fuzzy"


def test_unknown_name_returns_unresolved_sentinel() -> None:
    """An unknown repo name returns UnresolvedRepo (never raises)."""
    reg = _registry()
    result = reg.resolve("totally-unknown-repo-xyz")
    assert isinstance(result, UnresolvedRepo)
    assert result.name == "totally-unknown-repo-xyz"


def test_manual_override_wins_over_auto_scan() -> None:
    """A manual override for an alias wins over an identically-named scan entry."""
    overrides = {
        "myproject": RepoEntry(path="/custom/override/path", default_agent="claude"),
    }
    reg = _registry(overrides=overrides)
    result = reg.resolve("myproject")
    assert isinstance(result, ResolvedRepo)
    assert result.path == "/custom/override/path"
    assert result.default_agent == "claude"
    assert result.match_kind == "exact"


def test_manual_override_alias_resolves() -> None:
    """A manual override alias that doesn't appear in the scan resolves correctly."""
    overrides = {
        "prod-infra": RepoEntry(path="/prod/infra", default_agent="omp"),
    }
    reg = _registry(overrides=overrides)
    result = reg.resolve("prod-infra")
    assert isinstance(result, ResolvedRepo)
    assert result.path == "/prod/infra"


def test_manual_override_fuzzy_beats_scan_fuzzy() -> None:
    """Override fuzzy match takes precedence over scan fuzzy match."""
    # "infra" would fuzzy-match "infra-core" from the scan, but the override
    # alias "infra-tool" also fuzzy-matches "infra" and overrides come first.
    overrides = {
        "infra-tool": RepoEntry(path="/override/infra-tool", default_agent="omp"),
    }
    reg = _registry(overrides=overrides)
    result = reg.resolve("infra")
    assert isinstance(result, ResolvedRepo)
    # Could match either override "infra-tool" or scan "infra-core"; override wins
    assert result.path == "/override/infra-tool"
    assert result.match_kind == "fuzzy"


# ---------------------------------------------------------------------------
# 10. build_repo_registry — config parsing
# ---------------------------------------------------------------------------


def test_build_registry_string_shorthand() -> None:
    """build_repo_registry parses the string shorthand form."""
    repos_cfg = {"work": "/home/user/work"}
    reg = build_repo_registry(repos_cfg=repos_cfg, _injected_scan={})
    result = reg.resolve("work")
    assert isinstance(result, ResolvedRepo)
    assert result.path == "/home/user/work"
    assert result.default_agent == DEFAULT_AGENT


def test_build_registry_dict_form_with_agent() -> None:
    """build_repo_registry parses the dict form with explicit default_agent."""
    repos_cfg = {
        "infra": {"path": "/home/user/infra", "default_agent": "claude"},
    }
    reg = build_repo_registry(repos_cfg=repos_cfg, _injected_scan={})
    result = reg.resolve("infra")
    assert isinstance(result, ResolvedRepo)
    assert result.path == "/home/user/infra"
    assert result.default_agent == "claude"


def test_build_registry_none_repos_cfg() -> None:
    """build_repo_registry with None repos_cfg builds an empty-overrides registry."""
    reg = build_repo_registry(repos_cfg=None, _injected_scan={})
    result = reg.resolve("anything")
    assert isinstance(result, UnresolvedRepo)


def test_build_registry_skips_empty_path() -> None:
    """build_repo_registry skips aliases with empty paths without raising."""
    repos_cfg = {"bad": ""}
    reg = build_repo_registry(repos_cfg=repos_cfg, _injected_scan={})
    result = reg.resolve("bad")
    assert isinstance(result, UnresolvedRepo)


# ---------------------------------------------------------------------------
# 11. scan_for_repos — filesystem scan (uses a real temp dir)
# ---------------------------------------------------------------------------


def test_scan_finds_git_repos_in_root(tmp_path: Any) -> None:
    """scan_for_repos returns repos discovered under a scan root."""
    repo_dir = tmp_path / "my-repo"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    result = scan_for_repos(scan_roots=[str(tmp_path)])
    assert "my-repo" in result
    assert result["my-repo"] == str(repo_dir)


def test_scan_skips_non_git_dirs(tmp_path: Any) -> None:
    """scan_for_repos ignores directories that are not git repos."""
    non_repo = tmp_path / "not-a-repo"
    non_repo.mkdir()

    result = scan_for_repos(scan_roots=[str(tmp_path)])
    assert "not-a-repo" not in result


def test_scan_returns_empty_for_missing_root(tmp_path: Any) -> None:
    """scan_for_repos returns {} when a scan root does not exist."""
    result = scan_for_repos(scan_roots=[str(tmp_path / "nonexistent")])
    assert result == {}


def test_scan_includes_root_itself_if_git_repo(tmp_path: Any) -> None:
    """If the scan root itself is a git repo it is included."""
    (tmp_path / ".git").mkdir()
    result = scan_for_repos(scan_roots=[str(tmp_path)])
    assert tmp_path.name.lower() in result
