"""Matrix boolean env flags must honour all shared truthy/falsy aliases.

Several Matrix env vars used hand-rolled truthy/falsy sets that omitted
``on`` (truthy) or ``off`` (falsy).  After routing them through
``env_var_enabled`` the full alias set is covered.  These parametrised
tests prove it.
"""

import types
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


# ── Fake mautrix (adapter imports it at module level) ─────────────────

def _make_fake_mautrix():
    mautrix = types.ModuleType("mautrix")
    mautrix_api = types.ModuleType("mautrix.api")

    class HTTPAPI:
        def __init__(self, **kw):
            self.base_url = kw.get("base_url", "")
            self.token = kw.get("token", "")
            self.session = MagicMock()
            self.session.close = AsyncMock()

    mautrix_api.HTTPAPI = HTTPAPI
    mautrix.api = mautrix_api

    mautrix_types = types.ModuleType("mautrix.types")

    class EventType:
        ROOM_MESSAGE = "m.room.message"
        REACTION = "m.reaction"
        ROOM_ENCRYPTED = "m.room.encrypted"
        ROOM_NAME = "m.room.name"

    class UserID(str):
        pass

    mautrix_types.EventType = EventType
    mautrix_types.ContentURI = str
    mautrix_types.EventID = str
    mautrix_types.RoomID = str
    mautrix_types.SyncToken = str
    mautrix_types.UserID = UserID
    mautrix_types.PaginationDirection = type(
        "PD", (), {"BACKWARD": "b", "FORWARD": "f"}
    )
    mautrix_types.PresenceState = type(
        "PS", (), {"ONLINE": "online", "OFFLINE": "offline", "UNAVAILABLE": "unavailable"}
    )
    mautrix_types.RoomCreatePreset = type(
        "RCP", (), {"PRIVATE": "private_chat", "PUBLIC": "public_chat",
                    "TRUSTED_PRIVATE": "trusted_private_chat"}
    )
    mautrix_types.TrustState = type("TS", (), {"UNVERIFIED": 0, "VERIFIED": 1})

    mautrix_client = types.ModuleType("mautrix.client")
    mautrix_client.Client = MagicMock
    mautrix.client = mautrix_client

    mautrix_crypto = types.ModuleType("mautrix.crypto")
    mautrix_crypto.OlmMachine = MagicMock
    mautrix.crypto = mautrix_crypto

    return {
        "mautrix": mautrix,
        "mautrix.api": mautrix_api,
        "mautrix.types": mautrix_types,
        "mautrix.client": mautrix_client,
        "mautrix.crypto": mautrix_crypto,
    }


def _make_adapter(monkeypatch):
    monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test")
    monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.example.org")
    with patch.dict("sys.modules", _make_fake_mautrix()):
        from plugins.platforms.matrix.adapter import MatrixAdapter
        cfg = PlatformConfig(
            enabled=True,
            token="syt_test",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@bot:example.org",
            },
        )
        return MatrixAdapter(cfg)


# ── Default-false flags: truthy aliases must enable ───────────────────

_DEFAULT_FALSE_FLAGS = [
    ("MATRIX_ALLOW_ROOM_MENTIONS", "_allow_room_mentions"),
    ("MATRIX_DM_AUTO_THREAD", "_dm_auto_thread"),
    ("MATRIX_DM_MENTION_THREADS", "_dm_mention_threads"),
    ("MATRIX_PROCESS_NOTICES", "_process_notices"),
]


@pytest.mark.parametrize("env_var,attr", _DEFAULT_FALSE_FLAGS,
                         ids=[e for e, _ in _DEFAULT_FALSE_FLAGS])
def test_default_false_flags_disabled_by_default(monkeypatch, env_var, attr):
    monkeypatch.delenv(env_var, raising=False)
    adapter = _make_adapter(monkeypatch)
    assert getattr(adapter, attr) is False


@pytest.mark.parametrize("val", ["true", "1", "yes", "on", "TRUE", "On"])
@pytest.mark.parametrize("env_var,attr", _DEFAULT_FALSE_FLAGS,
                         ids=[e for e, _ in _DEFAULT_FALSE_FLAGS])
def test_default_false_flags_truthy(monkeypatch, env_var, attr, val):
    monkeypatch.setenv(env_var, val)
    adapter = _make_adapter(monkeypatch)
    assert getattr(adapter, attr) is True


@pytest.mark.parametrize("val", ["false", "0", "no", "off", "FALSE", "Off"])
@pytest.mark.parametrize("env_var,attr", _DEFAULT_FALSE_FLAGS,
                         ids=[e for e, _ in _DEFAULT_FALSE_FLAGS])
def test_default_false_flags_falsy(monkeypatch, env_var, attr, val):
    monkeypatch.setenv(env_var, val)
    adapter = _make_adapter(monkeypatch)
    assert getattr(adapter, attr) is False


# ── Default-true flags: falsy aliases must disable ────────────────────

_DEFAULT_TRUE_FLAGS = [
    ("MATRIX_AUTO_THREAD", "_auto_thread"),
    ("MATRIX_REACTIONS", "_reactions_enabled"),
    ("MATRIX_APPROVAL_REQUIRE_SENDER", "_approval_require_sender"),
]


@pytest.mark.parametrize("env_var,attr", _DEFAULT_TRUE_FLAGS,
                         ids=[e for e, _ in _DEFAULT_TRUE_FLAGS])
def test_default_true_flags_enabled_by_default(monkeypatch, env_var, attr):
    monkeypatch.delenv(env_var, raising=False)
    adapter = _make_adapter(monkeypatch)
    assert getattr(adapter, attr) is True


@pytest.mark.parametrize("val", ["false", "0", "no", "off", "FALSE", "Off"])
@pytest.mark.parametrize("env_var,attr", _DEFAULT_TRUE_FLAGS,
                         ids=[e for e, _ in _DEFAULT_TRUE_FLAGS])
def test_default_true_flags_falsy(monkeypatch, env_var, attr, val):
    monkeypatch.setenv(env_var, val)
    adapter = _make_adapter(monkeypatch)
    assert getattr(adapter, attr) is False


@pytest.mark.parametrize("val", ["true", "1", "yes", "on", "TRUE", "On"])
@pytest.mark.parametrize("env_var,attr", _DEFAULT_TRUE_FLAGS,
                         ids=[e for e, _ in _DEFAULT_TRUE_FLAGS])
def test_default_true_flags_truthy(monkeypatch, env_var, attr, val):
    monkeypatch.setenv(env_var, val)
    adapter = _make_adapter(monkeypatch)
    assert getattr(adapter, attr) is True
