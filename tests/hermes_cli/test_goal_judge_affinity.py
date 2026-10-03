"""The goal judge rides the conversation's affinity key even off the turn's thread.

Regression for #115000: the CLI runs the judge from its post-turn hook on a thread that never saw the
turn's context, so the auxiliary request carried no conversation key. OpenCode relays answered
``400 MissingSessionID``, which is now masked by a random ``oneshot-…`` key that still routes the judge
away from the conversation's warm backend. The request kwargs are built by the real
``_build_call_kwargs``; only the network send is replaced.
"""

import threading
from types import SimpleNamespace

import agent.auxiliary_client as aux
from agent.opencode_affinity import OPENCODE_SESSION_HEADER, opencode_session_headers
from agent.portal_tags import get_affinity_scope
from hermes_cli.goals import judge_goal

_PROVIDER, _BASE_URL = "opencode-go", "https://opencode.ai/zen/go/v1"


def _judge_on_fresh_thread(monkeypatch, session_id):
    sent = {}

    def fake_call_llm(*, task, messages, **_):
        sent["headers"] = aux._build_call_kwargs(_PROVIDER, "glm-5", messages, base_url=_BASE_URL, task=task)[
            "extra_headers"]
        sent["scope_during"] = get_affinity_scope()
        reply = '{"done": false, "reason": "keep going"}'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    monkeypatch.setattr(aux, "call_llm", fake_call_llm)
    worker = threading.Thread(target=lambda: sent.setdefault(
        "verdict", judge_goal("ship it", "did some work", session_id=session_id)))
    worker.start()
    worker.join(timeout=30)
    return sent


def test_judge_uses_the_main_turns_key_for_its_session(monkeypatch):
    main_turn = opencode_session_headers(_PROVIDER, _BASE_URL, "20261001_101500_abc123")
    sent = _judge_on_fresh_thread(monkeypatch, "20261001_101500_abc123")
    other = _judge_on_fresh_thread(monkeypatch, "20261001_111500_def456")

    assert sent["verdict"][0] == "continue"
    assert sent["headers"][OPENCODE_SESSION_HEADER] == main_turn[OPENCODE_SESSION_HEADER]
    assert other["headers"][OPENCODE_SESSION_HEADER] != main_turn[OPENCODE_SESSION_HEADER]


def test_judge_leaves_no_scope_behind_and_defers_to_a_host_scope(monkeypatch):
    from agent.portal_tags import reset_affinity_scope, set_affinity_scope

    _judge_on_fresh_thread(monkeypatch, "20261001_101500_abc123")
    assert get_affinity_scope() is None

    token = set_affinity_scope("kanban:t_1")
    try:
        sent = {}

        def fake_call_llm(**_):
            sent["scope"] = get_affinity_scope()
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])

        monkeypatch.setattr(aux, "call_llm", fake_call_llm)
        judge_goal("ship it", "did some work", session_id="20261001_101500_abc123")
        assert sent["scope"] == "kanban:t_1"
        assert get_affinity_scope() == "kanban:t_1"
    finally:
        reset_affinity_scope(token)
