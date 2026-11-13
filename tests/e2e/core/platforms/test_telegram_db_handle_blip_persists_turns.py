"""A Telegram turn whose agent was built while the gateway's state.db handle was briefly
unavailable still reaches state.db, and so does every later turn on that cached agent.

REAL gateway + Telegram adapter against the Bot API stand-in. A probe plugin's
``pre_gateway_dispatch`` hook puts the runner's per-path handle cache into its ordinary
retry backoff for two seconds, exactly what a failed open does, so the turn's fresh agent
is built with no session DB while the handle recovers before the turn ends (#140823).
"""

from __future__ import annotations

import sqlite3
import textwrap

import pytest

from tests.e2e.core.platforms._drv_telegram import TelegramDriver
from tests.e2e.core.platforms._helpers import Director, GatewayUnderTest
from tests.fakes.fake_llm_provider import FakeLLMServer, Text
from tests.fakes.platforms._standin import wait_until

pytestmark = [pytest.mark.spawns_gateway_lookalike, pytest.mark.platforms("posix")]

_PLUGIN = textwrap.dedent('''
    import time
    from pathlib import Path


    def _blip(event=None, gateway=None, **_):
        if "[in:first]" not in (getattr(event, "text", "") or ""):
            return None
        from gateway.session_db_recovery import _Unavailable
        cache = gateway._session_db_handle_cache
        with cache.lock:
            for path in list(cache.handles):
                cache.handles.pop(path, None)
                cache._registry_backed.discard(path)
                cache._unavailable[Path(path)] = _Unavailable(failures=1, next_retry_at=time.monotonic() + 2.0)
        return None


    def register(ctx):
        ctx.register_hook("pre_gateway_dispatch", _blip)
''')


def _rows(db) -> list[tuple[str, str]]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return [(r[0], r[1]) for r in con.execute(
            "SELECT role, content FROM messages WHERE role IN ('user', 'assistant') ORDER BY id")]
    finally:
        con.close()


def test_turns_on_an_agent_built_during_a_db_handle_blip_reach_state_db(tmp_path):
    drv = TelegramDriver()
    drv.start()
    director = Director()
    # Slow enough that the handle's 2 s backoff has ended before the turn's persistence block runs.
    director.script("first", Text("first answer", chunk_chars=2, delay_per_chunk=0.3))
    llm = FakeLLMServer(director)
    llm.start()
    cfg = {**drv.gateway_config(), "plugins": {"enabled": ["dbblip"]}}
    gw = GatewayUnderTest(tmp_path / "rig", llm_base_url=llm.base_url, config=cfg,
                          env=drv.gateway_env(), ready=drv.connected)
    plugin_dir = gw.hermes_home / "plugins" / "dbblip"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: dbblip\nversion: 0.0.1\ndescription: test probe\n")
    (plugin_dir / "__init__.py").write_text(_PLUGIN)
    try:
        gw.start()
        for token in ("first", "second"):
            drv.dm(f"question {token} [in:{token}]")
            wait_until(lambda t=token: any(f"{t} answer" in v.text or f"reply to {t}" in v.text
                                           for v in drv.visible(drv.user_id)), f"{token} reply", on_timeout=gw.tail)
            gw.wait_idle()
        wait_until(lambda: len(_rows(gw.db_path)) >= 4, "both turns in state.db", timeout=15,
                   on_timeout=lambda: f"rows={_rows(gw.db_path)}\n{gw.tail()}")
        assert _rows(gw.db_path) == [
            ("user", "question first [in:first]"), ("assistant", "first answer"),
            ("user", "question second [in:second]"), ("assistant", "reply to second"),
        ]
    finally:
        gw.stop()
        llm.stop()
        drv.stop()
