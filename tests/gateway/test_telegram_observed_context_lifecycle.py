from types import SimpleNamespace
from unittest.mock import Mock

from gateway.run import (
    _build_gateway_agent_history,
    _build_gateway_hygiene_history,
    _select_cached_agent_history,
    _wrap_current_message_with_observed_context,
)
from gateway.session import SessionStore


OBSERVED_PROMPT = "This channel uses observed Telegram group context."
OBSERVED_HEADER = "[Observed Telegram group context"


def _provider_wire(history, current):
    messages = history + [{"role": "user", "content": current}]
    return "\n".join(
        str(message.get("api_content") or message.get("content") or "")
        for message in messages
    )


def test_hygiene_uses_canonical_replay_and_defers_when_observations_are_pending():
    first_wrapper = _wrap_current_message_with_observed_context("ASK_B", "OBS_A")
    history = [
        {"role": "user", "content": "OBS_A", "observed": True, "message_id": "1"},
        {"role": "user", "content": "ASK_B", "api_content": first_wrapper},
        {"role": "assistant", "content": "ANSWER_B"},
        {
            "role": "user",
            "content": "human draft",
            "observed": True,
            "message_id": "2",
        },
        {
            "role": "user",
            "content": "human correction",
            "observed": True,
            "message_id": "2",
        },
    ]

    replay, estimate, has_pending = _build_gateway_hygiene_history(
        history,
        channel_prompt=OBSERVED_PROMPT,
    )

    assert has_pending is True
    assert len(estimate) == len(replay) + 1
    assert estimate[-1] == {"role": "user", "content": "human correction"}
    assert "human draft" not in _provider_wire(replay, estimate[-1])


def test_three_turn_provider_history_contains_each_observation_once():
    raw_history = []
    observations = ["OBS_A", "OBS_C", "OBS_E"]
    addressed = ["ASK_B", "ASK_D", "ASK_F"]

    for index, (observed, current_text) in enumerate(
        zip(observations, addressed),
        start=1,
    ):
        raw_history.append({
            "role": "user",
            "content": observed,
            "observed": True,
            "message_id": str(index),
        })
        agent_history, observed_context = _build_gateway_agent_history(
            raw_history,
            channel_prompt=OBSERVED_PROMPT,
        )
        api_current = _wrap_current_message_with_observed_context(
            current_text,
            observed_context,
        )
        wire = _provider_wire(agent_history, api_current)

        for delivered in observations[:index]:
            assert wire.count(delivered) == 1

        raw_history.extend([
            {
                "role": "user",
                "content": current_text,
                "api_content": api_current,
            },
            {"role": "assistant", "content": f"ANSWER_{index}"},
        ])


def test_interleaved_observation_after_turn_snapshot_remains_pending():
    first_wrapper = _wrap_current_message_with_observed_context("ASK_B", "OBS_A")
    history = [
        {
            "role": "user",
            "content": "OBS_A",
            "observed": True,
            "message_id": "10",
        },
        {
            "role": "user",
            "content": "OBS_C",
            "observed": True,
            "message_id": "20",
        },
        {
            "role": "user",
            "content": "ASK_B",
            "api_content": first_wrapper,
        },
    ]

    _, observed_context = _build_gateway_agent_history(
        history,
        channel_prompt=OBSERVED_PROMPT,
    )

    assert observed_context == "OBS_C"


def test_interleaved_human_edit_preserves_only_the_new_correction():
    committed = _wrap_current_message_with_observed_context(
        "ASK", "OBS_A\nOBS_B_OLD"
    )
    history = [
        {
            "role": "user",
            "content": "OBS_A",
            "observed": True,
            "message_id": "10",
        },
        {
            "role": "user",
            "content": "OBS_B_OLD",
            "observed": True,
            "message_id": "20",
        },
        {
            "role": "user",
            "content": "OBS_B_NEW",
            "observed": True,
            "message_id": "20",
        },
        {"role": "user", "content": "ASK", "api_content": committed},
    ]

    _, observed_context = _build_gateway_agent_history(
        history,
        channel_prompt=OBSERVED_PROMPT,
    )

    assert observed_context == "OBS_B_NEW"


def test_timestamp_projection_change_does_not_redeliver_committed_observation():
    committed = _wrap_current_message_with_observed_context("ASK", "OBS_T")
    history = [
        {
            "role": "user",
            "content": "OBS_T",
            "observed": True,
            "message_id": "10",
            "timestamp": 0,
        },
        {"role": "user", "content": "ASK", "api_content": committed},
    ]

    _, observed_context = _build_gateway_agent_history(
        history,
        channel_prompt=OBSERVED_PROMPT,
        inject_timestamps=True,
    )

    assert observed_context is None


def test_edit_before_snapshot_commits_the_latest_version():
    committed = _wrap_current_message_with_observed_context(
        "ASK", "OBS_A\nOBS_B_NEW"
    )
    history = [
        {
            "role": "user",
            "content": "OBS_A",
            "observed": True,
            "message_id": "10",
        },
        {
            "role": "user",
            "content": "OBS_B_OLD",
            "observed": True,
            "message_id": "20",
        },
        {
            "role": "user",
            "content": "OBS_B_NEW",
            "observed": True,
            "message_id": "20",
        },
        {"role": "user", "content": "ASK", "api_content": committed},
    ]

    _, observed_context = _build_gateway_agent_history(
        history,
        channel_prompt=OBSERVED_PROMPT,
    )

    assert observed_context is None


def test_historical_timestamp_and_timezone_do_not_change_observation_identity():
    committed = _wrap_current_message_with_observed_context(
        "ASK", "[Thu 1970-01-01 00:00:00 UTC] OBS_T"
    )
    history = [
        {
            "role": "user",
            "content": "OBS_T",
            "observed": True,
            "message_id": "10",
            "timestamp": 0,
        },
        {"role": "user", "content": "ASK", "api_content": committed},
    ]

    _, observed_context = _build_gateway_agent_history(
        history,
        channel_prompt=OBSERVED_PROMPT,
        inject_timestamps=False,
    )

    assert observed_context is None


def test_cyclic_edit_commits_the_last_matching_historical_cut():
    committed = _wrap_current_message_with_observed_context("ASK", "OBS_A")
    history = [
        {
            "role": "user",
            "content": "OBS_A",
            "observed": True,
            "message_id": "10",
        },
        {
            "role": "user",
            "content": "OBS_B",
            "observed": True,
            "message_id": "10",
        },
        {
            "role": "user",
            "content": "OBS_A",
            "observed": True,
            "message_id": "10",
        },
        {"role": "user", "content": "ASK", "api_content": committed},
    ]

    _, observed_context = _build_gateway_agent_history(
        history,
        channel_prompt=OBSERVED_PROMPT,
    )

    assert observed_context is None


def test_multi_id_cyclic_edit_does_not_redeliver_reverted_content():
    committed = _wrap_current_message_with_observed_context(
        "ASK", "A_OLD\nB_STABLE"
    )
    history = [
        {
            "role": "user",
            "content": "A_OLD",
            "observed": True,
            "message_id": "10",
        },
        {
            "role": "user",
            "content": "B_STABLE",
            "observed": True,
            "message_id": "20",
        },
        {
            "role": "user",
            "content": "A_TEMP",
            "observed": True,
            "message_id": "10",
        },
        {
            "role": "user",
            "content": "A_OLD",
            "observed": True,
            "message_id": "10",
        },
        {"role": "user", "content": "ASK", "api_content": committed},
    ]

    _, observed_context = _build_gateway_agent_history(
        history,
        channel_prompt=OBSERVED_PROMPT,
    )

    assert observed_context is None


def test_identical_observation_retransmit_after_assignment_is_not_redelivered():
    first_wrapper = _wrap_current_message_with_observed_context("ASK_A", "OBS_X")
    history = [
        {
            "role": "user",
            "content": "OBS_X",
            "observed": True,
            "message_id": "10",
        },
        {"role": "user", "content": "ASK_A", "api_content": first_wrapper},
        {"role": "assistant", "content": "ANSWER_A"},
        {
            "role": "user",
            "content": "OBS_X",
            "observed": True,
            "message_id": "10",
        },
    ]

    _, observed_context = _build_gateway_agent_history(
        history,
        channel_prompt=OBSERVED_PROMPT,
    )

    assert observed_context is None


def test_ambiguous_synthetic_user_row_does_not_consume_real_observation():
    history = [
        {
            "role": "user",
            "content": "OBS_A",
            "observed": True,
            "message_id": "10",
        },
        {"role": "user", "content": "[internal continuation marker]"},
        {"role": "assistant", "content": "internal acknowledgement"},
    ]

    _, observed_context = _build_gateway_agent_history(
        history,
        channel_prompt=OBSERVED_PROMPT,
    )

    assert observed_context == "OBS_A"


def test_observations_are_assigned_once_across_api_content_replay():
    first_wrapper = _wrap_current_message_with_observed_context("ASK_B", "OBS_A")
    history = [
        {
            "role": "user",
            "content": "OBS_A",
            "observed": True,
            "message_id": "10",
        },
        {
            "role": "user",
            "content": "ASK_B",
            "api_content": first_wrapper,
        },
        {"role": "assistant", "content": "ANSWER_B"},
        {
            "role": "user",
            "content": "OBS_C",
            "observed": True,
            "message_id": "20",
        },
    ]

    agent_history, observed_context = _build_gateway_agent_history(
        history,
        channel_prompt=OBSERVED_PROMPT,
    )
    current = _wrap_current_message_with_observed_context("ASK_D", observed_context)
    wire = _provider_wire(agent_history, current)

    assert observed_context == "OBS_C"
    assert wire.count("OBS_A") == 1
    assert wire.count("OBS_C") == 1
    assert wire.count(OBSERVED_HEADER) == 2


def test_human_edits_keep_the_latest_content_before_assignment():
    history = [
        {
            "role": "user",
            "content": "human draft",
            "observed": True,
            "message_id": "25",
        },
        {
            "role": "user",
            "content": "human corrected",
            "observed": True,
            "message_id": "25",
        },
    ]

    _, observed_context = _build_gateway_agent_history(
        history,
        channel_prompt=OBSERVED_PROMPT,
    )

    assert observed_context == "human corrected"


def test_human_edit_after_assignment_is_delivered_as_a_correction():
    first_wrapper = _wrap_current_message_with_observed_context(
        "ASK_B", "human initial"
    )
    history = [
        {
            "role": "user",
            "content": "human initial",
            "observed": True,
            "message_id": "20",
        },
        {"role": "user", "content": "ASK_B", "api_content": first_wrapper},
        {"role": "assistant", "content": "ANSWER_B"},
        {
            "role": "user",
            "content": "human correction",
            "observed": True,
            "message_id": "20",
        },
    ]

    agent_history, observed_context = _build_gateway_agent_history(
        history,
        channel_prompt=OBSERVED_PROMPT,
    )

    assert observed_context == "human correction"
    wire = _provider_wire(agent_history, observed_context)
    assert wire.count("human initial") == 1
    assert wire.count("human correction") == 1


def test_repair_preserves_api_content_when_addressed_turns_are_adjacent():
    first_wrapper = _wrap_current_message_with_observed_context("ASK_B", "OBS_A")
    second_wrapper = _wrap_current_message_with_observed_context("ASK_D", "OBS_C")
    history = [
        {
            "role": "user",
            "content": "OBS_A",
            "observed": True,
            "message_id": "10",
        },
        {
            "role": "user",
            "content": "ASK_B",
            "api_content": first_wrapper,
            "message_id": "11",
        },
        {
            "role": "user",
            "content": "OBS_C",
            "observed": True,
            "message_id": "20",
        },
        {
            "role": "user",
            "content": "ASK_D",
            "api_content": second_wrapper,
            "message_id": "21",
        },
    ]

    agent_history, observed_context = _build_gateway_agent_history(
        history,
        channel_prompt=OBSERVED_PROMPT,
    )

    assert observed_context is None
    assert len(agent_history) == 1
    assert agent_history[0]["content"] == "ASK_B\n\nASK_D"
    assert agent_history[0]["api_content"] == f"{first_wrapper}\n\n{second_wrapper}"


def test_observed_rows_are_partitioned_before_user_alternation_repair():
    history = [
        {"role": "user", "content": "ASK_B", "message_id": "31"},
        {
            "role": "user",
            "content": "OBS_C",
            "observed": True,
            "message_id": "30",
        },
        {
            "role": "user",
            "content": "ASK_D",
            "message_id": "32",
            "api_content": _wrap_current_message_with_observed_context(
                "ASK_D", "OBS_C"
            ),
        },
        {"role": "assistant", "content": "ANSWER_D"},
        {
            "role": "user",
            "content": "OBS_E",
            "observed": True,
            "message_id": "40",
        },
    ]

    agent_history, observed_context = _build_gateway_agent_history(
        history,
        channel_prompt=OBSERVED_PROMPT,
    )

    user_messages = [m for m in agent_history if m.get("role") == "user"]
    assert len(user_messages) == 1
    assert user_messages[0]["content"] == "ASK_B\n\nASK_D"
    assert "OBS_C" not in user_messages[0]["content"]
    assert observed_context == "OBS_E"


def test_session_store_can_load_raw_transcript_without_alternation_repair():
    db = SimpleNamespace(get_messages_as_conversation=Mock(return_value=[]))
    store = object.__new__(SessionStore)
    store._db = db

    store.load_transcript("session-1", repair_alternation=False)

    db.get_messages_as_conversation.assert_called_once_with(
        "session-1",
        repair_alternation=False,
    )


def test_cached_history_is_repaired_before_length_fallback_selection():
    persisted = [{"role": "user", "content": "ASK_A\n\nASK_B"}]
    live = [
        {"role": "user", "content": "ASK_A"},
        {"role": "user", "content": "ASK_B"},
    ]

    selected = _select_cached_agent_history(persisted, live)

    assert selected is persisted
    assert live == [
        {"role": "user", "content": "ASK_A"},
        {"role": "user", "content": "ASK_B"},
    ]
