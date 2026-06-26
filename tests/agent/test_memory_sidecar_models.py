import pytest

from agent.memory_sidecar.models import (
    ContextQuery,
    Observation,
    ObservationFile,
    validate_observation_type,
    validate_privacy_status,
)


def test_validate_observation_type_rejects_unknown_value():
    with pytest.raises(ValueError, match="unknown observation_type"):
        validate_observation_type("mystery")


def test_validate_privacy_status_rejects_unknown_value():
    with pytest.raises(ValueError, match="unknown privacy_status"):
        validate_privacy_status("secret")


def test_observation_requires_mvp_type_and_confidence_range():
    with pytest.raises(ValueError, match="unknown observation_type"):
        Observation(
            session_id="s1",
            event_ts="2026-06-26T18:11:10Z",
            observation_type="mystery",
            title="Bad",
            summary="Bad",
            detail="Bad",
        )

    with pytest.raises(ValueError, match="confidence"):
        Observation(
            session_id="s1",
            event_ts="2026-06-26T18:11:10Z",
            observation_type="decision",
            title="Bad",
            summary="Bad",
            detail="Bad",
            confidence=1.5,
        )


def test_context_query_validates_limit_time_bias_and_types():
    with pytest.raises(ValueError, match="limit"):
        ContextQuery(limit=0)

    with pytest.raises(ValueError, match="time_bias"):
        ContextQuery(time_bias="fresh")

    with pytest.raises(ValueError, match="unknown observation_type"):
        ContextQuery(types=("weird",))


def test_observation_file_requires_path():
    with pytest.raises(ValueError, match="file_path"):
        ObservationFile(file_path="")
