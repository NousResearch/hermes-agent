"""
Tests for bug #12579: edit_message missing finalize parameter in platform adapters.

The base class in gateway/platforms/base.py defines:
    async def edit_message(self, chat_id: str, message_id: str, content: str, *, finalize: bool = False) -> SendResult:

All platform adapters must accept the finalize kwarg to match the base signature.
This test verifies each adapter's edit_message method:
1. Accepts finalize=True without TypeError
2. Accepts omitting finalize (backward compat)
3. Uses inspect.signature to verify the parameter exists without real API calls
"""

import inspect
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# -------------------------------------------------------------------------
# Mock telegram package (copied from conftest / test_telegram_documents.py)
# -------------------------------------------------------------------------

def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN = "Markdown"
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ParseMode.HTML = "HTML"
    mod.constants.ChatType.PRIVATE = "private"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})
    mod.error.Forbidden = type("Forbidden", (Exception,), {})
    mod.error.InvalidToken = type("InvalidToken", (Exception,), {})
    mod.error.RetryAfter = type("RetryAfter", (Exception,), {"retry_after": 1})
    mod.error.Conflict = type("Conflict", (Exception,), {})
    mod.Update.ALL_TYPES = []
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules[name] = mod
    sys.modules["telegram.error"] = mod.error


# -------------------------------------------------------------------------
# Mock discord package (copied from conftest / test_discord_send.py)
# -------------------------------------------------------------------------

def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.ui = SimpleNamespace(
        View=object,
        button=lambda *a, **k: (lambda fn: fn),
        Button=object,
    )
    discord_mod.ButtonStyle = SimpleNamespace(
        success=1, primary=2, secondary=2, danger=3,
        green=1, grey=2, blurple=2, red=3,
    )
    discord_mod.Color = SimpleNamespace(
        orange=lambda: 1, green=lambda: 2, blue=lambda: 3,
        red=lambda: 4, purple=lambda: 5,
    )
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    for name in ("discord", "discord.ext", "discord.ext.commands"):
        sys.modules[name] = discord_mod
    sys.modules["discord.ext"] = ext_mod
    sys.modules["discord.ext.commands"] = commands_mod


# -------------------------------------------------------------------------
# Mock slack package (copied from test_slack.py)
# -------------------------------------------------------------------------

def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock
    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock
    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules[name] = mod


# -------------------------------------------------------------------------
# Mock matrix-nio package (simplified from test_matrix.py)
# -------------------------------------------------------------------------

def _ensure_matrix_mock():
    if "mautrix" in sys.modules and hasattr(sys.modules["mautrix"], "__file__"):
        return
    import types
    mautrix = types.ModuleType("mautrix")
    mautrix_api = types.ModuleType("mautrix.api")

    class HTTPAPI:
        def __init__(self, base_url="", token="", **kwargs):
            self.base_url = base_url
            self.token = token
            self.session = MagicMock()
    mautrix_api.HTTPAPI = HTTPAPI
    mautrix.api = mautrix_api

    mautrix_types = types.ModuleType("mautrix.types")
    mautrix_types.EventID = str
    mautrix_types.UserID = str
    mautrix_types.RoomID = str
    mautrix.client = types.ModuleType("mautrix.client")

    class Client:
        def __init__(self, *args, **kwargs):
            pass

    mautrix.client.Client = Client
    sys.modules["mautrix"] = mautrix
    sys.modules["mautrix.api"] = mautrix_api
    sys.modules["mautrix.types"] = mautrix_types
    sys.modules["mautrix.client"] = mautrix.client


# -------------------------------------------------------------------------
# Mock whatsapp package
# -------------------------------------------------------------------------

def _ensure_whatsapp_mock():
    if "whatsapp" in sys.modules and hasattr(sys.modules["whatsapp"], "__file__"):
        return
    wa_mod = MagicMock()
    wa_mod.WhatsApp = MagicMock
    sys.modules["whatsapp"] = wa_mod


# -------------------------------------------------------------------------
# Mock mattermost package
# -------------------------------------------------------------------------

def _ensure_mattermost_mock():
    if "mattermost" in sys.modules and hasattr(sys.modules["mattermost"], "__file__"):
        return
    mm_mod = MagicMock()
    mm_mod.Client = MagicMock
    sys.modules["mattermost"] = mm_mod


# -------------------------------------------------------------------------
# Mock feishu/lark package
# -------------------------------------------------------------------------

def _ensure_feishu_mock():
    if "feishu" in sys.modules and hasattr(sys.modules["feishu"], "__file__"):
        return
    feishu_mod = MagicMock()
    lark_mod = MagicMock()
    feishu_mod.lark = lark_mod
    sys.modules["feishu"] = feishu_mod
    sys.modules["lark_oapi"] = lark_mod


# -------------------------------------------------------------------------
# Install all mocks before importing adapters
# -------------------------------------------------------------------------

_ensure_telegram_mock()
_ensure_discord_mock()
_ensure_slack_mock()
_ensure_matrix_mock()
_ensure_whatsapp_mock()
_ensure_mattermost_mock()
_ensure_feishu_mock()


# -------------------------------------------------------------------------
# Import adapters
# -------------------------------------------------------------------------

from gateway.platforms.telegram import TelegramAdapter
from gateway.platforms.discord import DiscordAdapter
from gateway.platforms.slack import SlackAdapter
from gateway.platforms.matrix import MatrixAdapter
from gateway.platforms.whatsapp import WhatsAppAdapter
from gateway.platforms.mattermost import MattermostAdapter
from gateway.platforms.feishu import FeishuAdapter


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _get_edit_message_sig(adapter_cls):
    """Return the inspect.Signature of edit_message on *adapter_cls*."""
    return inspect.signature(adapter_cls.edit_message)


def _mock_adapter(adapter_cls):
    """Return a mock 'self' instance for the adapter class."""
    return MagicMock(spec=adapter_cls)


def _check_finalize_param(sig, adapter_name):
    """Assert that *sig* has a 'finalize' keyword-only parameter with default False."""
    finalize_param = None
    for p in sig.parameters.values():
        if p.name == "finalize":
            finalize_param = p
            break
    assert finalize_param is not None, (
        f"{adapter_name}.edit_message is missing 'finalize' parameter. "
        f"Signature: {sig}"
    )
    assert finalize_param.kind == inspect.Parameter.KEYWORD_ONLY, (
        f"{adapter_name}.edit_message.finalize must be keyword-only (after *). "
        f"Got kind={finalize_param.kind}"
    )
    assert finalize_param.default is False, (
        f"{adapter_name}.edit_message.finalize default must be False, "
        f"got {finalize_param.default}"
    )


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------

@pytest.mark.parametrize("adapter_cls,adapter_name", [
    (TelegramAdapter, "TelegramAdapter"),
    (DiscordAdapter, "DiscordAdapter"),
    (SlackAdapter, "SlackAdapter"),
    (MatrixAdapter, "MatrixAdapter"),
    (WhatsAppAdapter, "WhatsAppAdapter"),
    (MattermostAdapter, "MattermostAdapter"),
    (FeishuAdapter, "FeishuAdapter"),
])
def test_edit_message_has_finalize_kwarg(adapter_cls, adapter_name):
    """Each adapter's edit_message must have 'finalize: bool = False' in its signature."""
    sig = _get_edit_message_sig(adapter_cls)
    _check_finalize_param(sig, adapter_name)


@pytest.mark.parametrize("adapter_cls,adapter_name", [
    (TelegramAdapter, "TelegramAdapter"),
    (DiscordAdapter, "DiscordAdapter"),
    (SlackAdapter, "SlackAdapter"),
    (MatrixAdapter, "MatrixAdapter"),
    (WhatsAppAdapter, "WhatsAppAdapter"),
    (MattermostAdapter, "MattermostAdapter"),
    (FeishuAdapter, "FeishuAdapter"),
])
def test_edit_message_accepts_finalize_true(adapter_cls, adapter_name):
    """Each adapter's edit_message must accept finalize=True without TypeError."""
    sig = _get_edit_message_sig(adapter_cls)
    mock_self = _mock_adapter(adapter_cls)
    try:
        sig.bind(
            mock_self,
            "chat_id",
            "message_id",
            "content",
            finalize=True,
        )
    except TypeError as e:
        pytest.fail(
            f"{adapter_name}.edit_message does not accept finalize=True: {e}"
        )


@pytest.mark.parametrize("adapter_cls,adapter_name", [
    (TelegramAdapter, "TelegramAdapter"),
    (DiscordAdapter, "DiscordAdapter"),
    (SlackAdapter, "SlackAdapter"),
    (MatrixAdapter, "MatrixAdapter"),
    (WhatsAppAdapter, "WhatsAppAdapter"),
    (MattermostAdapter, "MattermostAdapter"),
    (FeishuAdapter, "FeishuAdapter"),
])
def test_edit_message_backward_compat_no_finalize(adapter_cls, adapter_name):
    """Each adapter's edit_message must work without passing finalize (backward compat)."""
    sig = _get_edit_message_sig(adapter_cls)
    mock_self = _mock_adapter(adapter_cls)
    try:
        sig.bind(
            mock_self,
            "chat_id",
            "message_id",
            "content",
        )
    except TypeError as e:
        pytest.fail(
            f"{adapter_name}.edit_message does not accept being called without finalize: {e}"
        )
