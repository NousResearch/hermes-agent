"""PHASE 1 -- authorization decision contract tests.

Validates the inverted-default behavior mandated by the P0 corrective
design: missing/empty/malformed authorization MUST deny. Only an explicit
positive opt-in (HERMES_DISABLE_SELF_IMPROVEMENT=0) MAY allow.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.self_improvement_policy import (
    normalize_env_disabled,
    normalize_read_only_session,
    evaluate,
    ALLOW,
    DENY_ENV_DISABLED,
    DENY_READ_ONLY_SESSION,
    DENY_UNKNOWN_OPERATION,
)


def test_missing_env_disabled_denies():
    """T3: missing authorization -> DENY (PHASE 1 inversion)."""
    decision = evaluate(
        environment_disabled="",
        session_read_only="",
        operation_kind="background_review_spawn",
        origin="background_review",
    )
    assert decision.result == DENY_ENV_DISABLED, (
        f"empty env_disabled must DENY in Phase 1, got {decision.result}"
    )


def test_empty_env_disabled_denies():
    """T4: empty value -> DENY."""
    decision = evaluate(
        environment_disabled=" ",
        session_read_only="",
        operation_kind="background_review_spawn",
        origin="background_review",
    )
    assert decision.result == DENY_ENV_DISABLED


def test_malformed_env_disabled_denies():
    """T5: malformed value -> DENY."""
    decision = evaluate(
        environment_disabled="garbage",
        session_read_only="",
        operation_kind="background_review_spawn",
        origin="background_review",
    )
    assert decision.result == DENY_ENV_DISABLED


def test_explicit_disable_denies():
    """T1: HERMES_DISABLE_SELF_IMPROVEMENT=1 -> DENY."""
    decision = evaluate(
        environment_disabled="1",
        session_read_only="",
        operation_kind="background_review_spawn",
        origin="background_review",
    )
    assert decision.result == DENY_ENV_DISABLED


def test_readonly_session_denies():
    """T2: HERMES_READ_ONLY_SESSION=1 -> DENY."""
    decision = evaluate(
        environment_disabled="",
        session_read_only="1",
        operation_kind="background_review_spawn",
        origin="background_review",
    )
    assert decision.result == DENY_READ_ONLY_SESSION


def test_explicit_opt_in_allows():
    """T6: explicit positive opt-in (env=0) -> ALLOW (no stronger deny)."""
    decision = evaluate(
        environment_disabled="0",
        session_read_only="",
        operation_kind="background_review_spawn",
        origin="background_review",
    )
    assert decision.result == ALLOW, (
        f"explicit env=0 must ALLOW, got {decision.result} ({decision.reason})"
    )


def test_decision_dataclass_is_frozen():
    """T7: Decision is immutable (frozen dataclass)."""
    from agent.self_improvement_policy import Decision
    d = Decision(result=ALLOW, reason="explicit test")
    try:
        d.result = "MUTATED"
        assert False, "Decision must be frozen -- assignment should have raised"
    except Exception:
        pass


def test_normalize_env_disabled_inversion():
    """Direct assertion of the Phase 1 inversion."""
    assert normalize_env_disabled("") is True
    assert normalize_env_disabled(" ") is True
    assert normalize_env_disabled("1") is True
    assert normalize_env_disabled("true") is True
    assert normalize_env_disabled("yes") is True
    assert normalize_env_disabled("on") is True
    assert normalize_env_disabled("0") is False
    assert normalize_env_disabled("false") is False
    assert normalize_env_disabled("no") is False
    assert normalize_env_disabled("off") is False
    assert normalize_env_disabled("garbage") is True


def test_normalize_read_only_session_unchanged():
    """Phase 1 does NOT change normalize_read_only_session semantics."""
    assert normalize_read_only_session("") is False
    assert normalize_read_only_session("1") is True
    assert normalize_read_only_session("0") is False
