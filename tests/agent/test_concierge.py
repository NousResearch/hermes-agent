"""Concierge: English single-word stop/status only; rest → main."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


def test_disabled_by_default_returns_none():
    from agent.concierge import handle_concierge

    assert handle_concierge("stop") is None
    assert handle_concierge("status") is None


def test_stop_english_word():
    from agent.concierge import handle_concierge

    session = {"concierge_live_enabled": True}
    cancelled = []
    r = handle_concierge(
        "stop",
        session=session,
        cancel_callback=lambda t: cancelled.append(t),
        main_in_flight=True,
    )
    assert r is not None and r.action == "stop"
    assert cancelled == ["stop"]


def test_status_english_word():
    from agent.concierge import handle_concierge

    session = {"concierge_live_enabled": True}
    r = handle_concierge("status", session=session)
    assert r is not None and r.action == "status"


def test_korean_stop_falls_through():
    from agent.concierge import handle_concierge

    session = {"concierge_live_enabled": True}
    assert handle_concierge("멈춰", session=session) is None
    assert handle_concierge("그만", session=session) is None


def test_korean_status_falls_through():
    from agent.concierge import handle_concierge

    session = {"concierge_live_enabled": True}
    assert handle_concierge("지금 뭐 하고 있어?", session=session) is None
    assert handle_concierge("진행상황", session=session) is None


def test_free_text_falls_through():
    from agent.concierge import handle_concierge

    session = {"concierge_live_enabled": True}
    assert handle_concierge("이거 진행해", session=session) is None
    assert handle_concierge(
        "https://github.com/NousResearch/hermes-agent/pull/63603",
        session=session,
    ) is None


def test_status_inside_sentence_not_status():
    from agent.control_plane import Intent, classify

    d = classify("please check the status of the job", concierge_mode_active=True)
    assert d.intent is not Intent.STATUS
