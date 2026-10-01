import json

import pytest

from plugins.context_engine._context_governor.protocol import (
    ContextGovernorCommandError,
    ContextGovernorProtocolError,
    parse_failure_envelope,
    require_failure_capability,
    validated_finalized_tokens,
)


def capability():
    return {
        "failure_envelope": {
            "schema": "ContextGovernorFailureV1",
            "flag": "--failure-envelope-v1",
            "stream": "stderr",
        }
    }


def test_failure_capability_is_exact():
    require_failure_capability(capability())
    bad = capability()
    bad["failure_envelope"]["stream"] = "stdout"
    with pytest.raises(ContextGovernorProtocolError):
        require_failure_capability(bad)


def test_failure_envelope_uses_code_not_display_text():
    error = parse_failure_envelope(
        json.dumps(
            {
                "schema": "ContextGovernorFailureV1",
                "operation": "compact-v2",
                "code": "cannot_meet_target",
                "details": {"target": 100, "actual": 120},
            }
        ),
        "compact-v2",
    )
    assert isinstance(error, ContextGovernorCommandError)
    assert error.code == "cannot_meet_target"
    assert error.details == {"target": 100, "actual": 120}
    assert "actual" not in str(error)


@pytest.mark.parametrize(
    "stderr",
    [
        "CannotMeetTarget: old English message",
        "{}",
        json.dumps({"schema": "ContextGovernorFailureV1", "operation": "wrong", "code": "io_failed"}),
        json.dumps({"schema": "ContextGovernorFailureV1", "operation": "compact-v2", "code": "BAD CODE"}),
        json.dumps({"schema": "ContextGovernorFailureV1", "operation": "compact-v2", "code": "io_failed", "unexpected": True}),
    ],
)
def test_malformed_or_untyped_failure_is_protocol_error(stderr):
    with pytest.raises(ContextGovernorProtocolError):
        parse_failure_envelope(stderr, "compact-v2")


@pytest.mark.parametrize("value", [None, True, -1, 1.5, "10"])
def test_final_token_field_must_be_nonnegative_integer(value):
    with pytest.raises(ContextGovernorProtocolError):
        validated_finalized_tokens({"compacted_approx_tokens": value}, 100)


def test_final_token_field_must_not_exceed_admitted_target():
    with pytest.raises(ContextGovernorProtocolError):
        validated_finalized_tokens({"compacted_approx_tokens": 101}, 100)
    assert validated_finalized_tokens({"compacted_approx_tokens": 100}, 100) == 100
