"""Regression tests for Slack interactive-caller auth under multiplex profiles.

``SlackAdapter._is_interactive_user_authorized`` gates approval-button,
slash-confirm and clarify clicks. Button clicks bypass the normal message auth
flow in ``gateway/run.py``, so this is the only gate on that path.

Two failure modes are covered here:

1. It used to resolve the gateway auth chain only by introspecting
   ``_message_handler.__self__``. On a multiplexed profile the handler is the
   closure built by ``GatewayRunner._make_profile_message_handler``, which has
   no ``__self__``, so the introspection silently yielded nothing and the
   method fell through to its env-only fallback.
2. That fallback read the process environment — the DEFAULT profile's ``.env``
   — so a secondary profile inherited another profile's allowlist and
   allow-all flags. Direction is fail-OPEN: a caller the profile never
   allowlisted could resolve its approvals.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# Minimal Slack SDK mock so SlackAdapter can be imported
# ---------------------------------------------------------------------------
def _ensure_slack_mock():
    if "slack_bolt" in sys.modules:
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    sys.modules["slack_bolt"] = slack_bolt
    sys.modules["slack_bolt.async_app"] = slack_bolt.async_app
    handler_mod = MagicMock()
    handler_mod.AsyncSocketModeHandler = MagicMock
    sys.modules["slack_bolt.adapter"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode.async_handler"] = handler_mod
    sdk_mod = MagicMock()
    sdk_mod.web = MagicMock()
    sdk_mod.web.async_client = MagicMock()
    sdk_mod.web.async_client.AsyncWebClient = MagicMock
    sys.modules["slack_sdk"] = sdk_mod
    sys.modules["slack_sdk.web"] = sdk_mod.web
    sys.modules["slack_sdk.web.async_client"] = sdk_mod.web.async_client


_ensure_slack_mock()

from agent import secret_scope  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


_AUTH_ENV_KEYS = (
    "SLACK_ALLOWED_USERS",
    "SLACK_ALLOW_ALL_USERS",
    "GATEWAY_ALLOWED_USERS",
    "GATEWAY_ALLOW_ALL_USERS",
)

OWNER = "U_OWNER_B"
STRANGER = "U_STRANGER"


@pytest.fixture
def multiplex(monkeypatch):
    """Enable/disable multiplex without leaking the module global across tests."""
    previous = secret_scope.is_multiplex_active()

    def _set(active: bool):
        secret_scope.set_multiplex_active(active)

    yield _set
    secret_scope.set_multiplex_active(previous)


@pytest.fixture
def profile_scope():
    """Install a profile secret scope, always resetting it."""
    tokens = []

    def _install(mapping):
        tokens.append(secret_scope.set_secret_scope(mapping))

    yield _install
    for token in reversed(tokens):
        secret_scope.reset_secret_scope(token)


@pytest.fixture
def clean_env(monkeypatch):
    for key in _AUTH_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def _make_adapter():
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test-token"))
    adapter._app = MagicMock()
    adapter._bot_user_id = "U_BOT"
    return adapter


def _multiplexed_handler():
    """A per-profile message handler closure — no ``__self__``, like the real one."""

    async def _handler(event):
        return None

    return _handler


def test_injected_authorization_check_is_preferred(clean_env, multiplex, profile_scope):
    """The profile-bound callback wins over process env and handler introspection."""
    clean_env.setenv("SLACK_ALLOW_ALL_USERS", "true")
    multiplex(True)
    profile_scope({"SLACK_BOT_TOKEN": "xoxb-b"})

    adapter = _make_adapter()
    adapter._message_handler = _multiplexed_handler()

    seen = []

    def check(user_id, chat_type=None, chat_id=None):
        seen.append((user_id, chat_type, chat_id))
        return user_id == OWNER

    adapter.set_authorization_check(check)

    assert adapter._is_interactive_user_authorized(STRANGER, channel_id="C1") is False
    assert adapter._is_interactive_user_authorized(OWNER, channel_id="C1") is True
    assert seen == [(STRANGER, "group", "C1"), (OWNER, "group", "C1")]


def test_dm_channel_passes_dm_chat_type(clean_env, multiplex):
    """A ``D``-prefixed channel is reported to the auth chain as a DM."""
    multiplex(False)
    adapter = _make_adapter()
    seen = []

    def check(user_id, chat_type=None, chat_id=None):
        seen.append(chat_type)
        return True

    adapter.set_authorization_check(check)

    adapter._is_interactive_user_authorized(OWNER, channel_id="D123")
    assert seen == ["dm"]


def test_multiplex_does_not_inherit_process_env_allow_all(
    clean_env, multiplex, profile_scope
):
    """A secondary profile must not inherit the default profile's allow-all flag."""
    clean_env.setenv("SLACK_ALLOW_ALL_USERS", "true")
    clean_env.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    multiplex(True)
    profile_scope({"SLACK_BOT_TOKEN": "xoxb-b", "SLACK_ALLOWED_USERS": OWNER})

    adapter = _make_adapter()
    adapter._message_handler = _multiplexed_handler()

    assert adapter._is_interactive_user_authorized(STRANGER, channel_id="C1") is False
    assert adapter._is_interactive_user_authorized(OWNER, channel_id="C1") is True


def test_multiplex_does_not_inherit_process_env_allowlist(
    clean_env, multiplex, profile_scope
):
    """A secondary profile must not inherit the default profile's allowlists."""
    clean_env.setenv("SLACK_ALLOWED_USERS", STRANGER)
    clean_env.setenv("GATEWAY_ALLOWED_USERS", STRANGER)
    multiplex(True)
    profile_scope({"SLACK_BOT_TOKEN": "xoxb-b", "SLACK_ALLOWED_USERS": OWNER})

    adapter = _make_adapter()
    adapter._message_handler = _multiplexed_handler()

    assert adapter._is_interactive_user_authorized(STRANGER, channel_id="C1") is False
    assert adapter._is_interactive_user_authorized(OWNER, channel_id="C1") is True


def test_multiplex_without_scope_fails_closed(clean_env, multiplex):
    """No scope installed under multiplex is the fail-closed signal, not an env read."""
    clean_env.setenv("SLACK_ALLOW_ALL_USERS", "true")
    clean_env.setenv("GATEWAY_ALLOWED_USERS", STRANGER)
    multiplex(True)

    adapter = _make_adapter()
    adapter._message_handler = _multiplexed_handler()

    # get_secret raises UnscopedSecretError here; it must not be downgraded
    # to an os.environ read.
    with pytest.raises(secret_scope.UnscopedSecretError):
        secret_scope.get_secret("SLACK_ALLOW_ALL_USERS")

    assert adapter._is_interactive_user_authorized(STRANGER, channel_id="C1") is False


def test_single_profile_env_fallback_unchanged(clean_env, multiplex):
    """Single-profile deployments keep reading credentials from the process env."""
    multiplex(False)
    adapter = _make_adapter()
    adapter._message_handler = _multiplexed_handler()

    assert adapter._is_interactive_user_authorized(STRANGER, channel_id="C1") is False

    clean_env.setenv("SLACK_ALLOWED_USERS", OWNER)
    assert adapter._is_interactive_user_authorized(OWNER, channel_id="C1") is True
    assert adapter._is_interactive_user_authorized(STRANGER, channel_id="C1") is False

    clean_env.delenv("SLACK_ALLOWED_USERS")
    clean_env.setenv("SLACK_ALLOW_ALL_USERS", "true")
    assert adapter._is_interactive_user_authorized(STRANGER, channel_id="C1") is True


def test_handler_introspection_still_honored_without_injected_check(
    clean_env, multiplex
):
    """Bare-adapter embedding keeps working via ``_message_handler.__self__``."""
    multiplex(False)
    adapter = _make_adapter()

    class _Runner:
        def __init__(self):
            self.seen = []

        def _is_user_authorized(self, source):
            self.seen.append(source.user_id)
            return source.user_id == OWNER

        async def handle(self, event):
            return None

    runner = _Runner()
    adapter._message_handler = runner.handle

    assert adapter._is_interactive_user_authorized(OWNER, channel_id="C1") is True
    assert adapter._is_interactive_user_authorized(STRANGER, channel_id="C1") is False
    assert runner.seen == [OWNER, STRANGER]


def test_empty_user_id_denied(multiplex):
    multiplex(False)
    adapter = _make_adapter()
    assert adapter._is_interactive_user_authorized("") is False
    assert adapter._is_interactive_user_authorized("   ") is False
