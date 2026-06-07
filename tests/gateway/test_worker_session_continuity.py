"""A routed worker rehydrates its own transcript so routed turns keep memory.

The front sends only a ``session_id`` (it never holds the worker's history),
so ``/v1/runs`` must load the transcript from its own state.db when the front
opts in with ``continue_session`` — but stay stateless for every other client.
"""

from gateway.platforms.api_server import APIServerAdapter

_resolve = APIServerAdapter._continue_session_id


def test_loads_when_flagged_with_session_and_no_history():
    body = {"continue_session": True, "session_id": "agent:coder:tg:dm:1"}
    assert _resolve(body, [], None) == "agent:coder:tg:dm:1"


def test_off_by_default_so_existing_clients_stay_stateless():
    body = {"session_id": "agent:coder:tg:dm:1"}
    assert _resolve(body, [], None) is None


def test_explicit_history_takes_precedence():
    body = {"continue_session": True, "session_id": "s"}
    assert _resolve(body, [{"role": "user", "content": "hi"}], None) is None


def test_previous_response_id_takes_precedence():
    body = {"continue_session": True, "session_id": "s"}
    assert _resolve(body, [], "resp_42") is None


def test_no_session_id_means_nothing_to_resume():
    assert _resolve({"continue_session": True}, [], None) is None
