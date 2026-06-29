"""Regression tests for issue #54217.

WhatsApp talks to its Node.js Baileys bridge over HTTP via ``aiohttp``, but
``aiohttp`` is not a core dependency — it only arrives through another platform
extra (discord/slack/messaging). A WhatsApp-only install therefore reached the
bare ``import aiohttp`` inside ``WhatsAppAdapter.connect()`` with nothing to
import and crashed with ``ModuleNotFoundError: No module named 'aiohttp'``.

The fix registers ``platform.whatsapp`` in ``tools/lazy_deps.py`` (and a
``whatsapp`` extra in ``pyproject.toml``) and calls ``ensure("platform.whatsapp",
prompt=False)`` at the top of ``connect()`` so aiohttp auto-installs on first
connect — mirroring ``platform.slack`` / ``platform.teams``.
"""

import asyncio
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import Platform


REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_adapter():
    """Create a WhatsAppAdapter with test attributes (bypass __init__).

    Mirrors the helper in tests/gateway/test_whatsapp_connect.py.
    """
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = MagicMock()
    adapter._bridge_port = 19876
    adapter._bridge_script = "/tmp/test-bridge.js"
    adapter._session_path = Path("/tmp/test-wa-session")
    adapter._bridge_log_fh = None
    adapter._bridge_log = None
    adapter._bridge_process = None
    adapter._reply_prefix = None
    adapter._running = False
    adapter._message_handler = None
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._background_tasks = set()
    adapter._auto_tts_disabled_chats = set()
    adapter._message_queue = asyncio.Queue()
    adapter._http_session = None
    return adapter


# ---------------------------------------------------------------------------
# Wiring: registry + pyproject extra
# ---------------------------------------------------------------------------

def test_platform_whatsapp_registered_in_lazy_deps():
    from tools.lazy_deps import LAZY_DEPS

    assert "platform.whatsapp" in LAZY_DEPS, (
        "platform.whatsapp missing from LAZY_DEPS — connect() can't lazy-install aiohttp"
    )
    specs = LAZY_DEPS["platform.whatsapp"]
    assert any(s.startswith("aiohttp==") for s in specs), specs


def test_whatsapp_extra_declared_in_pyproject():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    assert "whatsapp" in extras, "missing [project.optional-dependencies] whatsapp extra"
    assert any(dep.startswith("aiohttp==") for dep in extras["whatsapp"]), extras["whatsapp"]


# ---------------------------------------------------------------------------
# connect() lazy-installs aiohttp before it is used
# ---------------------------------------------------------------------------

class TestConnectEnsuresAiohttp:

    @pytest.mark.asyncio
    async def test_connect_calls_ensure_before_using_aiohttp(self):
        """connect() must call ensure("platform.whatsapp", prompt=False) up front.

        The ensure() call sits above every other step in connect() — including
        the Node.js requirements check and, crucially, the bare ``import
        aiohttp`` further down. We stop connect() right after ensure() (Node.js
        check returns False) and assert ensure() ran with the expected args; this
        is what makes a WhatsApp-only install auto-install aiohttp instead of
        crashing (#54217).
        """
        adapter = _make_adapter()
        captured = {}

        def fake_ensure(feature, *, prompt=True):
            captured["feature"] = feature
            captured["prompt"] = prompt

        with patch("tools.lazy_deps.ensure", side_effect=fake_ensure) as mock_ensure, \
             patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=False):
            result = await adapter.connect()

        mock_ensure.assert_called_once()
        assert captured == {"feature": "platform.whatsapp", "prompt": False}
        assert result is False  # stopped at the (mocked) Node.js requirements check

    @pytest.mark.asyncio
    async def test_connect_does_not_crash_when_ensure_fails(self):
        """A failed lazy-install is swallowed — connect() returns False, not raises.

        ensure() raises FeatureUnavailable (e.g. lazy installs disabled or pip
        offline). The try/except around it must absorb that so connect() exits
        cleanly via its normal failure paths instead of propagating.
        """
        from tools.lazy_deps import FeatureUnavailable

        adapter = _make_adapter()

        with patch("tools.lazy_deps.ensure",
                   side_effect=FeatureUnavailable("platform.whatsapp", (), "disabled")) as mock_ensure, \
             patch("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", return_value=False):
            result = await adapter.connect()  # must not raise

        mock_ensure.assert_called_once()
        assert result is False
