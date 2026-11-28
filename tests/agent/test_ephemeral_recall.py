"""``memory.ephemeral_recall`` / ``memory.ephemeral_plugin_context``: a turn's injected recall and
plugin context reach the model on THAT turn only. History rows replay without them, so the request
no longer grows by one recall block per turn; the DB sidecar still keeps the exact bytes sent, and
the prefix up to the previous turn stays byte-stable from one request to the next.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from agent.memory_context import build_memory_context_block, strip_stamped_recall
from agent.turn_context import ephemeral_history_user_content
from tests.agent import test_api_content_sidecar as sidecar
from tests.agent.test_api_content_sidecar import _chat_requests, _user_messages

wire_env = sidecar.wire_env


def _agent(make_agent, recall: str, policy: tuple[bool, bool]):
    agent = make_agent()
    manager = MagicMock()
    manager.prefetch_all.return_value = recall
    manager.describe_recall.return_value = ""
    agent._memory_manager = manager
    agent._ephemeral_injection = policy
    return agent


def _run_turns(wire_env, recalls, policy):
    """One process per turn, history reloaded from the store; returns each turn's first request."""
    make_agent, handler, db, sid = wire_env
    requests = []
    for n, recall in enumerate(recalls, 1):
        handler.captured_requests = []
        history = db.get_messages_as_conversation(sid)
        _agent(make_agent, recall, policy).run_conversation(f"question {n}", conversation_history=history, task_id=f"t{n}")
        requests.append(_chat_requests(handler)[0])
    return requests, db, sid


def _wire(messages):
    return [json.dumps(m, sort_keys=True) for m in messages]


def test_only_the_current_turn_carries_its_recall(wire_env):
    reqs, db, sid = _run_turns(wire_env, ["- alpha\n", "- beta\n", "- gamma\n"], (True, False))
    users = _user_messages(reqs[2])
    assert [u["content"] for u in users[:2]] == ["question 1\n\nPLUGIN-CTX", "question 2\n\nPLUGIN-CTX"]
    assert "<memory-context>" in users[2]["content"] and "- gamma" in users[2]["content"]
    # Nothing is lost: the store still holds the exact bytes each turn sent.
    stored = [r["api_content"] for r in db.get_messages(sid) if r["role"] == "user"]
    assert all("<memory-context>" in s for s in stored)


def test_the_prefix_before_the_previous_user_row_stays_byte_stable(wire_env):
    """Turn 3 resends turn 2's request unchanged up to turn 2's user row: the one row that drops
    its recall is the previous turn's, so the cache loses that turn's tail once, never more."""
    reqs, _db, _sid = _run_turns(wire_env, ["- alpha\n", "- beta\n", "- gamma\n"], (True, False))
    turn_2, turn_3 = reqs[1]["messages"], reqs[2]["messages"]
    idx_u2 = max(i for i, m in enumerate(turn_2) if m["role"] == "user")
    assert _wire(turn_3[:idx_u2]) == _wire(turn_2[:idx_u2])


def test_recall_is_not_deduped_against_history_that_no_longer_carries_it(wire_env):
    """With history stripped the window carries no recall, so a repeated fact is stated again."""
    reqs, _db, _sid = _run_turns(wire_env, ["- alpha\n", "- alpha\n"], (True, False))
    assert "- alpha" in _user_messages(reqs[1])[-1]["content"]


def test_plugin_context_can_be_ephemeral_too(wire_env):
    reqs, _db, _sid = _run_turns(wire_env, ["- alpha\n", "- beta\n"], (True, True))
    replayed, current = _user_messages(reqs[1])
    assert replayed["content"] == "question 1"
    assert current["content"].startswith("question 2\n\n<memory-context>") and current["content"].endswith("PLUGIN-CTX")


def test_default_policy_keeps_replaying_the_sidecar(wire_env):
    reqs, _db, _sid = _run_turns(wire_env, ["- alpha\n", "- beta\n"], (False, False))
    assert "- alpha" in _user_messages(reqs[1])[0]["content"]


def test_history_row_transform_is_pure_and_conservative():
    block = build_memory_context_block("- alpha\n")
    row = {"role": "user", "content": "hi"}
    assert ephemeral_history_user_content(row, f"hi\n\n{block}\n\nNOTE", (True, False)) == "hi\n\nNOTE"
    assert ephemeral_history_user_content(row, f"hi\n\n{block}", (True, False)) == "hi"
    # A sidecar that does not extend the row's own text (a prepended gateway note) keeps it.
    assert ephemeral_history_user_content(row, "[note]\nhi\n\nNOTE", (False, True)) == "[note]\nhi\n\nNOTE"
    assert ephemeral_history_user_content(row, None, (True, True)) is None
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    multimodal = {"role": "user", "content": [{"type": "text", "text": "what"}, image, {"type": "text", "text": block}]}
    assert ephemeral_history_user_content(multimodal, None, (True, False)) == [{"type": "text", "text": "what"}, image]
    assert strip_stamped_recall("no recall here") == "no recall here"
