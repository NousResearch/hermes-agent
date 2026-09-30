"""The gateway parses its locale catalogs before it serves, never on its event loop.

The first ``t()`` for a language parses that language's bundled YAML: hundreds of milliseconds on an
idle host and seconds under load. Gateway code calls ``t()`` on the event loop, so a cold parse there
stalls every other task, and during ``stop()`` it counts against the shutdown deadlines.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from agent import i18n, i18n_layers
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key

REPO_LOCALES = Path(__file__).resolve().parents[2] / "locales"


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


@pytest.fixture
def parses(tmp_path, monkeypatch) -> list[tuple[str, bool]]:
    """Every locale file parsed, as ``(file name, parsed on the event loop)``.

    The bundled English and German catalogs are copied to a fresh directory so that no earlier parse
    in this process can serve them."""
    bundled = tmp_path / "bundled-locales"
    bundled.mkdir()
    for lang in ("en", "de"):
        shutil.copy(REPO_LOCALES / f"{lang}.yaml", bundled / f"{lang}.yaml")
    monkeypatch.setenv("HERMES_BUNDLED_LOCALES", str(bundled))
    seen: list[tuple[str, bool]] = []
    real_parse = i18n_layers.parse_locale_file

    def recording_parse(path):
        seen.append((Path(path).name, _on_event_loop()))
        return real_parse(path)

    monkeypatch.setattr(i18n_layers, "parse_locale_file", recording_parse)
    i18n.reset_language_cache()
    yield seen
    i18n.reset_language_cache()


class _RecordingAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent: list[tuple[str, str]] = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append((chat_id, content))
        return SendResult(success=True, message_id="m")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.mark.asyncio
async def test_shutdown_notice_after_a_locale_pack_registers_does_not_parse_on_the_loop(
    tmp_path, monkeypatch, parses
):
    """A plugin that provides a locale pack can load after startup (a served profile's discovery, a
    plugin enabled at runtime). Its registration resets the language caches, and the next ``t()`` on
    the loop, here the shutdown notice, must not parse a bundled catalog again."""
    monkeypatch.setenv("HERMES_LANGUAGE", "de")
    adapter = _RecordingAdapter()
    runner = GatewayRunner(GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
        sessions_dir=tmp_path / "sessions",
    ))
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, platform_config: adapter)
    assert await runner.start()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", chat_type="dm", user_id="user-1")
    session_key = build_session_key(source)
    runner._running_agents[session_key] = object()
    runner._cache_session_source(session_key, source)
    pack = i18n_layers.register_pack("pl", i18n_layers.CORE_SURFACE, {"greeting": "Cześć"}, source="plugin:test")
    try:
        await runner._notify_active_sessions_of_shutdown()
    finally:
        i18n_layers.unregister_pack(pack)

    notice = i18n.t("gateway.shutdown.notice_shutdown", lang="de")
    assert (adapter.sent, sorted(parses)) == ([("chat-1", notice)], [("de.yaml", False), ("en.yaml", False)])
