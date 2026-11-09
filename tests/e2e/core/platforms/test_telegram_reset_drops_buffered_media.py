"""A photo still in the Telegram adapter's burst buffer when ``/new`` lands on a running turn never
becomes a turn after it; ``/status`` (not a reset) leaves it alone. REAL gateway + adapter against
the Bot API stand-in (#81370)."""

from __future__ import annotations

import base64
import json
import time

import pytest

from tests.e2e.core.platforms._drv_telegram import TelegramDriver
from tests.e2e.core.platforms._helpers import Director, GatewayUnderTest
from tests.fakes.fake_llm_provider import FakeLLMServer, Text
from tests.fakes.platforms._standin import wait_until

pytestmark = [pytest.mark.spawns_gateway_lookalike, pytest.mark.platforms("posix")]
_BURST_WINDOW = 4.0  # far wider than any runner stall: the command provably lands while the photo is buffered
_CAPTION = "old photo caption"
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


@pytest.fixture
def rig(tmp_path):
    drv = TelegramDriver()
    drv.start()
    drv.standin.files["ph-1"], drv.standin.file_paths["ph-1"] = _PNG, "photos/p1.png"
    director = Director()
    llm = FakeLLMServer(director)
    llm.start()
    env = {**drv.gateway_env(), "HERMES_TELEGRAM_MEDIA_BATCH_DELAY_SECONDS": str(_BURST_WINDOW)}
    gw = GatewayUnderTest(tmp_path / "rig", llm_base_url=llm.base_url, config=drv.gateway_config(),
                          env=env, ready=drv.connected)
    try:
        gw.start()
        drv.dm("warm [in:warm]")
        wait_until(lambda: director.turns.get("warm"), "warm turn")
        wait_until(lambda: drv.sends(drv.user_id), "the warm reply was sent")
        gw.wait_idle()
        yield drv, director, gw
    finally:
        gw.stop()
        llm.stop()
        drv.stop()


def _chat(drv) -> tuple[dict, int]:
    uid = int(drv.user_id)
    return {"id": uid, "type": "private", "first_name": "Tester", "username": f"user{uid}"}, uid


def _buffer_photo(drv) -> float:
    """Send a captioned photo; returns once it sits in the burst buffer (downloaded, not flushed)."""
    chat, uid = _chat(drv)
    drv.standin._inbound(chat, uid, caption=_CAPTION, photo=[
        {"file_id": "ph-1", "file_unique_id": "u-ph-1", "width": 1, "height": 1, "file_size": len(_PNG)}])
    wait_until(lambda: drv.standin.calls_of("file_download"), "the photo reached the burst buffer")
    return time.monotonic()


def _command(drv, name: str) -> None:
    chat, uid = _chat(drv)
    drv.standin._inbound(chat, uid, text=f"/{name}", entities=[{"type": "bot_command", "offset": 0,
                                                                "length": len(name) + 1}])


def _photo_turns(director) -> list[str]:
    return [tok for tok, bs in director.requests.items() for b in bs if _CAPTION in json.dumps(b.get("messages", []))]


def _assert_photo_never_reaches_model(drv, director, buffered_at: float) -> None:
    drv.dm("fresh question [in:fresh]")
    wait_until(lambda: director.turns.get("fresh"), "fresh turn")
    # health: allow HX013 -- absence proof: a leaked photo flushes once the burst window closes
    time.sleep(max(0.0, buffered_at + _BURST_WINDOW + 2.0 - time.monotonic()))
    assert _photo_turns(director) == []


def test_new_during_a_running_turn_drops_buffered_photo(rig) -> None:
    drv, director, _gw = rig
    director.script("slow", Text("a slow answer still streaming when /new lands", chunk_chars=1, delay_per_chunk=0.3))
    drv.dm("slow [in:slow]")
    wait_until(lambda: director.turns.get("slow"), "the slow turn is running")
    buffered_at = _buffer_photo(drv)
    _command(drv, "new")
    _assert_photo_never_reaches_model(drv, director, buffered_at)


def test_status_keeps_buffered_photo(rig) -> None:
    drv, director, _gw = rig
    _buffer_photo(drv)
    _command(drv, "status")
    wait_until(lambda: _photo_turns(director), "the photo still became a turn", timeout=_BURST_WINDOW + 20.0)
