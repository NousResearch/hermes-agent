"""Native poll results follow the response and closure contracts in MSC3381."""

from __future__ import annotations

import pytest

from plugins.platforms.matrix.polls import MatrixPoll, poll_results


ROOM = "!poll:server"
CREATOR = "@creator:server"


def event(kind, payload, event_id, sender="@voter:server", ts=100, **extra):
    return {
        "type": f"org.matrix.msc3381.poll.{kind}", "event_id": event_id,
        "sender": sender, "room_id": ROOM, "origin_server_ts": ts,
        "content": {
            f"org.matrix.msc3381.poll.{kind}": payload,
            "m.relates_to": {"rel_type": "m.reference", "event_id": "$poll"},
        }, **extra,
    }


def start(disclosed=True):
    return event("start", {
        "question": {"org.matrix.msc1767.text": "Which?"},
        "kind": "org.matrix.msc3381.poll.disclosed" if disclosed else "org.matrix.msc3381.poll.undisclosed",
        "max_selections": 1,
        "answers": [{"id": "a", "org.matrix.msc1767.text": "A"}, {"id": "b", "org.matrix.msc1767.text": "B"}],
    }, "$poll", CREATOR, ts=50)



def native(raw, namespace):
    if namespace == "unstable":
        return raw
    kind = raw["type"].rsplit(".", 1)[-1]
    content = dict(raw["content"])
    payload = content.pop(f"org.matrix.msc3381.poll.{kind}")
    if namespace == "sdk_stable":
        content[f"m.poll.{kind}"] = payload
        return {**raw, "type": f"m.poll.{kind}", "content": content}
    if kind == "start":
        payload = {**payload, "kind": "m.disclosed" if payload["kind"].endswith(".disclosed") else "m.undisclosed",
                   "question": {"m.text": [{"body": "Which?"}]},
                   "answers": [{"m.id": answer["id"], "m.text": [{"body": answer["org.matrix.msc1767.text"]}]} for answer in payload["answers"]]}
        content["m.poll"] = payload
    if kind == "response":
        content["m.selections"] = payload["answers"]
    return {**raw, "type": f"m.poll.{kind}", "content": content}


@pytest.mark.parametrize("namespace", ["unstable", "sdk_stable", "stable"])
@pytest.mark.parametrize("latest,redacted,expected", [
    (["b"], False, [0, 1]),
    (["unknown"], False, [0, 0]),
    (["a", "unknown"], False, [0, 0]),
    (["a", "a"], False, [1, 0]),
    (["a", "b"], False, [1, 0]),
    ([], False, [0, 0]),
    (["b"], True, [1, 0]),
])
def test_latest_ballot_including_spoiled_ballots_replaces_previous(namespace, latest, redacted, expected):
    if namespace == "stable" and latest == ["a", "unknown"]:
        expected = [1, 0]
    poll = MatrixPoll.from_event(native(start(), namespace), ROOM)
    replacement = native(event("response", {"answers": latest}, "$second", ts=101), namespace)
    if redacted:
        replacement["unsigned"] = {"redacted_because": {"event_id": "$redaction"}}
    result = poll_results(poll, [replacement, native(event("response", {"answers": ["a"]}, "$first"), namespace)])
    assert result == {
        "poll_id": "$poll", "question": "Which?", "kind": "disclosed", "max_selections": 1,
        "closed": False, "end_event_id": None, "end_timestamp": None,
        "complete": True, "incomplete_reasons": [], "results_visible": True,
        "answers": [{"id": "a", "text": "A", "votes": expected[0]}, {"id": "b", "text": "B", "votes": expected[1]}],
        "voters": sum(expected),
    }


@pytest.mark.parametrize("disclosed,incomplete,expected_visible", [(True, False, True), (False, False, True), (False, True, False)])
def test_first_authorised_end_bounds_original_timestamps_and_incomplete_results(disclosed, incomplete, expected_visible):
    poll = MatrixPoll.from_event(start(disclosed), ROOM)
    events = [
        event("end", {}, "$unauthorized", ts=99),
        event("end", {}, "$later", CREATOR, ts=103),
        event("end", {}, "$first-end", "@moderator:server", ts=101),
        event("response", {"answers": ["b"]}, "$after", ts=102),
        event("response", {"answers": ["a"]}, "$at-end", ts=101),
    ]
    result = poll_results(poll, events, moderators={"@moderator:server"}, incomplete_reasons=["missing decryption keys"] if incomplete else [])
    assert result == {
        "poll_id": "$poll", "question": "Which?", "kind": "disclosed" if disclosed else "undisclosed", "max_selections": 1,
        "closed": True, "end_event_id": "$first-end", "end_timestamp": 101,
        "complete": not incomplete, "incomplete_reasons": ["missing decryption keys"] if incomplete else [], "results_visible": expected_visible,
        "answers": [{"id": "a", "text": "A", "votes": None if incomplete else 1}, {"id": "b", "text": "B", "votes": None if incomplete else 0}],
        "voters": None if incomplete else 1,
    }


@pytest.mark.parametrize("count,expected", [(21, [f"a{index}" for index in range(20)]), (0, None)])
def test_answers_beyond_twenty_are_truncated_and_an_empty_list_is_rejected(count, expected):
    raw = start()
    raw["content"]["org.matrix.msc3381.poll.start"]["answers"] = [
        {"id": f"a{index}", "org.matrix.msc1767.text": f"A{index}"} for index in range(count)
    ]
    if expected is None:
        with pytest.raises(ValueError):
            MatrixPoll.from_event(raw, ROOM)
        return
    assert [answer.id for answer in MatrixPoll.from_event(raw, ROOM).answers] == expected
