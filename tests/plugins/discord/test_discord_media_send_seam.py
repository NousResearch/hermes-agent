"""Seam tests for DiscordMediaSendMixin (adapter.py god-file slice R3-S1).

Covers the consensus test plan: MRO identity (mixin-first), media upload
happy path via the lazy ``_read_url_image_with_redirect_guard`` seam
(patch-target regression), typing-loop lifecycle (dedup, cancel, transient
error cleanup), error paths (not-connected, unsafe URL, FileNotFoundError,
generic fallback), and the mocked-discord import path.
"""

import asyncio
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import the adapter first so the circular lazy seam resolves in gateway
# order (adapter -> mixin -> gateway.platforms.base), then the mixin.
import plugins.platforms.discord.adapter as adapter_mod  # noqa: F401
import plugins.platforms.discord.media_send_mixin as mixin_mod
from gateway.platforms.base import SendResult
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.discord.media_send_mixin import DiscordMediaSendMixin

MOVED_METHODS = [
    "send_image_file",
    "send_image",
    "send_animation",
    "send_video",
    "send_document",
    "send_typing",
    "stop_typing",
]


class _FakeBase:
    """Stands in for BasePlatformAdapter as the super() target."""

    async def send_image(self, chat_id, image_url, caption=None, reply_to=None, metadata=None):
        return ("base-send_image", chat_id)

    async def send_animation(self, chat_id, animation_url, caption=None, reply_to=None, metadata=None):
        return ("base-send_animation", chat_id)

    async def send_image_file(self, chat_id, image_path, caption=None, reply_to=None, metadata=None):
        return ("base-send_image_file", chat_id)


class _Stub(DiscordMediaSendMixin, _FakeBase):
    """Production MRO shape: mixin first, base second."""

    def __init__(self):
        self.name = "test-stub"
        self._client = None
        self._typing_tasks = {}

    def _is_forum_parent(self, channel):
        return False

    def _extract_discord_retry_after(self, exc):
        return None


# --- MRO identity -----------------------------------------------------------


@pytest.mark.parametrize("method", MOVED_METHODS)
def test_mro_identity_method_resolves_to_mixin(method):
    assert getattr(DiscordAdapter, method) is getattr(DiscordMediaSendMixin, method)


def test_mixin_is_second_in_mro():
    assert DiscordAdapter.__mro__[1] is DiscordMediaSendMixin


# --- media upload: happy path + seam ---------------------------------------


@pytest.mark.asyncio
async def test_send_image_success_flow_via_lazy_seam(monkeypatch):
    """Full send_image path; the lazy seam must honor the adapter-level patch."""
    msg = SimpleNamespace(id="12345")
    channel = SimpleNamespace(send=AsyncMock(return_value=msg))
    client = SimpleNamespace(
        get_channel=MagicMock(return_value=channel),
        fetch_channel=AsyncMock(return_value=channel),
    )
    stub = _Stub()
    stub._client = client

    fake_discord = MagicMock()
    fake_discord.File.return_value = "FAKE_FILE"
    monkeypatch.setattr(mixin_mod, "discord", fake_discord)
    monkeypatch.setattr(mixin_mod, "is_safe_url", lambda url: True)

    async def fake_guard(session, url, *, timeout, request_kwargs):
        return 200, b"fakepng", {"content-type": "image/png"}

    # Patch target lives on the ADAPTER module — the mixin's lazy
    # ``from .adapter import ...`` at the use site must pick it up.
    with patch.object(adapter_mod, "_read_url_image_with_redirect_guard", fake_guard):
        result = await stub.send_image("42", "https://example.com/x.png", caption="hi")

    assert result.success is True
    assert result.message_id == "12345"
    channel.send.assert_awaited_once()


# --- media upload: error paths ----------------------------------------------


@pytest.mark.asyncio
async def test_send_image_not_connected():
    stub = _Stub()  # _client is None
    result = await stub.send_image("42", "https://example.com/x.png")
    assert isinstance(result, SendResult)
    assert result.success is False
    assert result.error == "Not connected"


@pytest.mark.asyncio
async def test_send_image_unsafe_url_falls_back_to_super(monkeypatch):
    stub = _Stub()
    stub._client = object()  # truthy so the URL check is reached
    monkeypatch.setattr(mixin_mod, "is_safe_url", lambda url: False)
    result = await stub.send_image("42", "file:///etc/passwd")
    assert result == ("base-send_image", "42")


@pytest.mark.asyncio
async def test_send_animation_unsafe_url_falls_back_to_super(monkeypatch):
    stub = _Stub()
    stub._client = object()
    monkeypatch.setattr(mixin_mod, "is_safe_url", lambda url: False)
    result = await stub.send_animation("42", "file:///etc/passwd")
    assert result == ("base-send_animation", "42")


@pytest.mark.asyncio
async def test_send_image_file_not_found():
    stub = _Stub()
    stub._send_file_attachment = AsyncMock(side_effect=FileNotFoundError("/nope.png"))
    result = await stub.send_image_file("42", "/nope.png")
    assert result.success is False
    assert "not found" in result.error


@pytest.mark.asyncio
async def test_send_image_file_generic_error_falls_back_to_super():
    stub = _Stub()
    stub._send_file_attachment = AsyncMock(side_effect=RuntimeError("boom"))
    result = await stub.send_image_file("42", "/x.png")
    assert result == ("base-send_image_file", "42")


# --- typing loop lifecycle ---------------------------------------------------


@pytest.mark.asyncio
async def test_send_typing_not_connected_noop():
    stub = _Stub()  # _client is None
    await stub.send_typing("42")
    assert stub._typing_tasks == {}


@pytest.mark.asyncio
async def test_send_typing_dedup_and_stop_typing_cancels(monkeypatch):
    stub = _Stub()
    stub._client = SimpleNamespace(http=SimpleNamespace(request=AsyncMock()))
    fake_discord = SimpleNamespace(http=SimpleNamespace(Route=MagicMock(return_value="route")))
    monkeypatch.setattr(mixin_mod, "discord", fake_discord)

    await stub.send_typing("42")
    assert "42" in stub._typing_tasks
    first = stub._typing_tasks["42"]
    await asyncio.sleep(0)  # let the loop task start and hit the request

    # Dedup: second call must not spawn a second loop.
    await stub.send_typing("42")
    assert stub._typing_tasks["42"] is first
    assert stub._client.http.request.await_count == 1

    await stub.stop_typing("42")
    assert "42" not in stub._typing_tasks
    # The loop's inner ``except asyncio.CancelledError: return`` swallows the
    # cancellation, so the task completes normally rather than being flagged
    # cancelled — done() is the correct post-condition.
    assert first.done()


@pytest.mark.asyncio
async def test_send_typing_transient_error_cleans_up(monkeypatch):
    """Non-429 failure ends the loop and pops the task entry (finally)."""
    stub = _Stub()
    stub._client = SimpleNamespace(
        http=SimpleNamespace(request=AsyncMock(side_effect=RuntimeError("boom")))
    )
    fake_discord = SimpleNamespace(http=SimpleNamespace(Route=MagicMock(return_value="route")))
    monkeypatch.setattr(mixin_mod, "discord", fake_discord)

    await stub.send_typing("42")
    task = stub._typing_tasks["42"]
    await asyncio.wait_for(task, timeout=5)
    assert task.done()
    assert "42" not in stub._typing_tasks


@pytest.mark.asyncio
async def test_stop_typing_unknown_chat_is_noop():
    stub = _Stub()
    await stub.stop_typing("nope")  # must not raise


# --- mocked-discord import path ---------------------------------------------


def test_mixin_imports_with_mocked_discord():
    """sys.modules['discord'] mocked -> mixin must still import and bind it."""
    root = Path(__file__).resolve().parents[3]
    code = (
        "import sys\n"
        "from unittest.mock import MagicMock\n"
        "discord_mod = MagicMock()\n"
        "sys.modules['discord'] = discord_mod\n"
        "sys.modules['discord.ext'] = MagicMock()\n"
        "sys.modules['discord.ext.commands'] = MagicMock()\n"
        "import plugins.platforms.discord.media_send_mixin as mix\n"
        "assert mix.discord is discord_mod\n"
        "assert mix.DiscordMediaSendMixin is not None\n"
        "print('MOCKED-DISCORD-IMPORT-OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "MOCKED-DISCORD-IMPORT-OK" in result.stdout, result.stderr
