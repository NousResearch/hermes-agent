"""A Telegram request whose turn exhausts context compression is re-run once on the fresh session the
auto-reset creates, so it is answered without the user resending it (#137565). A re-run that
exhausts the fresh session too is not re-run again. REAL gateway + adapter against the Bot API
stand-in; the fake model rejects the request as too long (on the fresh session too in the second leg).
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing

import pytest

from tests.e2e.core.platforms._drv_telegram import TelegramDriver
from tests.e2e.core.platforms._helpers import Director, GatewayUnderTest
from tests.fakes.fake_llm_provider import Error, FakeLLMServer
from tests.fakes.platforms._standin import wait_until

pytestmark = [pytest.mark.spawns_gateway_lookalike, pytest.mark.platforms("posix")]
_REQUEST = "book the dentist on friday [in:ex]"
# A server measurement, so the overflow is believed and compression runs until it gives up.
_TOO_LONG = "prompt is too long: 233153 tokens > 200000 maximum"


class _Overflowing(Director):
    """Rejects the request as too long on the exhausted session (its transcript holds the warm-up
    turn); on the fresh session only when ``fresh_too``. Decided per request, never by timing."""

    def __init__(self, fresh_too: bool) -> None:
        super().__init__()
        self.fresh_too = fresh_too

    def __call__(self, record):
        wire = json.dumps(record["body"].get("messages", []))
        if "[in:ex]" in wire and (self.fresh_too or "[in:warm]" in wire):
            return Error(400, _TOO_LONG)
        return super().__call__(record)


def _text(v) -> str:
    return (v.text or "").replace("\\", "")


@pytest.mark.parametrize("fits_after_reset", [True, False], ids=["replayed_once", "no_second_replay"])
def test_exhausted_request_is_answered_on_the_fresh_session(tmp_path, fits_after_reset):
    drv = TelegramDriver()
    drv.start()
    director = _Overflowing(fresh_too=not fits_after_reset)
    llm = FakeLLMServer(director)
    llm.start()
    gw = GatewayUnderTest(tmp_path / "rig", llm_base_url=llm.base_url, config=drv.gateway_config(),
                          env=drv.gateway_env(), ready=drv.connected)
    try:
        gw.start()
        drv.dm("warm [in:warm]")
        wait_until(lambda: drv.sends(drv.user_id), "the warm reply was sent")
        gw.wait_idle()
        drv.dm(_REQUEST)
        wait_until(lambda: any("auto-reset" in _text(v).lower() for v in drv.standin.visible(drv.user_id)),
                   "the auto-reset notice", timeout=120, on_timeout=gw.tail)
        if fits_after_reset:
            wait_until(lambda: director.turns.get("ex"), "the request re-ran", timeout=60, on_timeout=gw.tail)
        # health: allow HX013 -- absence proof: a second replay or duplicate answer would land in this window
        time.sleep(8)
        gw.wait_idle()
        with closing(sqlite3.connect(f"file:{gw.hermes_home / 'state.db'}?mode=ro", uri=True)) as db:
            sessions = db.execute("SELECT id, end_reason FROM sessions ORDER BY started_at").fetchall()
            stored = db.execute("SELECT session_id FROM messages WHERE role='user' AND content LIKE ?",
                                (f"%{_REQUEST}%",)).fetchall()
    finally:
        gw.stop()
        llm.stop()
        drv.stop()
    answers = [_text(v) for v in drv.standin.visible(drv.user_id) if "reply to ex" in _text(v)]
    fresh = sessions[-1][0]
    if fits_after_reset:
        assert answers == ["reply to ex"], answers
        assert director.turns.get("ex") == 1
        assert [sid for (sid,) in stored].count(fresh) == 1, (sessions, stored)
    else:
        # One automatic retry only: the original turn and its single re-run, then the user decides.
        assert answers == [] and len(sessions) == 3, sessions
        assert [end for _, end in sessions] == ["session_reset", "session_reset", None], sessions
