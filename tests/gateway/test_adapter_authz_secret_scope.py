"""Adapter authz gates must read GATEWAY_/PLATFORM_ allow-all + allowlists through the profile
secret scope (#77548 cluster; #72348 / #86905 precedent).

Under ``gateway.multiplex_profiles`` ``os.environ`` holds the DEFAULT profile's values. A raw
``os.getenv`` in an adapter's own gate let the default's ``GATEWAY_ALLOW_ALL_USERS=true`` open every
secondary email/QQ/WhatsApp/Matrix/Teams/Slack/LINE/DingTalk bot, and the default's allowlist decide
who may approve on a secondary bot. Two invariants per adapter: the default env never answers a
scoped gate; the scope's own opt-in/allowlist does.
"""

import contextlib
from types import SimpleNamespace

import pytest

from agent import secret_scope as ss
from gateway.config import PlatformConfig

_DEFAULT_ENV = {
    "GATEWAY_ALLOW_ALL_USERS": "true", "GATEWAY_ALLOWED_USERS": "default-admin",
    "TEAMS_ALLOW_ALL_USERS": "true", "TEAMS_ALLOWED_USERS": "default-admin",
    "MATRIX_ALLOWED_USERS": "@default-admin:example.org", "MATRIX_IGNORE_USER_PATTERNS": r"^@spam:.*",
    "WHATSAPP_ALLOWED_USERS": "+15550001111", "SLACK_ALLOW_BOTS": "all", "SLACK_API_HUMAN_USERS": "U0DEFAULT",
    "LINE_ALLOW_ALL_USERS": "true", "LINE_ALLOWED_USERS": "Udefault", "DINGTALK_ALLOWED_USERS": "default-admin",
}


@pytest.fixture
def default_env(monkeypatch):
    """Multiplex on; os.environ carries the DEFAULT profile's permissive authz."""
    for name, value in _DEFAULT_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(ss, "_MULTIPLEX_ACTIVE", True)


@contextlib.contextmanager
def _scope(secrets):
    token = ss.set_secret_scope(secrets)
    try:
        yield
    finally:
        ss.reset_secret_scope(token)


def _matrix(extra=None):
    from plugins.platforms.matrix.adapter import MatrixAdapter

    return MatrixAdapter(PlatformConfig(enabled=True, token="t", extra={"homeserver": "https://m.example",
                                                                       "user_id": "@bot:m.example", **(extra or {})}))


def _whatsapp():
    from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin

    return SimpleNamespace(_dm_allowlist_source="WHATSAPP_ALLOWED_USERS", _allow_from=set(),
                           _coerce_allow_list=WhatsAppBehaviorMixin._coerce_allow_list)


def _slack():
    from plugins.platforms.slack.adapter import SlackAdapter

    adapter = SlackAdapter.__new__(SlackAdapter)
    adapter.config = PlatformConfig(enabled=True, extra={})
    return adapter


def _dingtalk():
    from plugins.platforms.dingtalk.adapter import DingTalkAdapter

    adapter = DingTalkAdapter.__new__(DingTalkAdapter)
    adapter.config = PlatformConfig(enabled=True, extra={})
    return adapter


def _line():
    from plugins.platforms.line.adapter import LineAdapter

    return LineAdapter(PlatformConfig(enabled=True, extra={"channel_access_token": "t", "channel_secret": "s"}))


# (label, opt-in the SECONDARY scope grants, gate(scope) -> observed value, closed value, open value)
_GATES = [
    ("email.allow_all", {"GATEWAY_ALLOW_ALL_USERS": "true"},
     lambda: __import__("plugins.platforms.email.adapter", fromlist=["EmailAdapter"]).EmailAdapter._allow_all_senders(),
     False, True),
    ("email.allowlist", {"GATEWAY_ALLOWED_USERS": "bot2-admin"},
     lambda: __import__("plugins.platforms.email.adapter", fromlist=["EmailAdapter"]).EmailAdapter._allowlist_in_effect(),
     False, True),
    ("qqbot.open_dm", {"QQ_ALLOW_ALL_USERS": "true"},
     lambda: __import__("gateway.platforms.qqbot.adapter", fromlist=["QQAdapter"]).QQAdapter._open_dm_opted_in(SimpleNamespace()),
     False, True),
    ("whatsapp.open_dm", {"GATEWAY_ALLOW_ALL_USERS": "true"},
     lambda: __import__("gateway.platforms.whatsapp_common", fromlist=["WhatsAppBehaviorMixin"]).WhatsAppBehaviorMixin._open_dm_opted_in(_whatsapp()),
     False, True),
    ("whatsapp.live_allow_from", {"WHATSAPP_ALLOWED_USERS": "+15559998888"},
     lambda: __import__("gateway.platforms.whatsapp_common", fromlist=["WhatsAppBehaviorMixin"]).WhatsAppBehaviorMixin._live_dm_allow_from(_whatsapp()),
     set(), {"+15559998888"}),
    ("teams.card_action", {"TEAMS_ALLOWED_USERS": "clicker"},
     lambda: __import__("plugins.platforms.teams.adapter", fromlist=["TeamsAdapter"]).TeamsAdapter._card_action_denied(
         SimpleNamespace(aad_object_id="clicker")) is None,
     False, True),
    ("matrix.authorized_user", {"MATRIX_ALLOWED_USERS": "@bot2-admin:example.org"},
     lambda: _matrix()._is_authorized_user("@bot2-admin:example.org"),
     False, True),
    ("matrix.ignored_patterns", {"MATRIX_IGNORE_USER_PATTERNS": r"^@bot2-spam:.*"},
     lambda: [p.pattern for p in _matrix()._ignored_user_patterns],
     [], [r"^@bot2-spam:.*"]),
    ("slack.allow_bots", {"SLACK_ALLOW_BOTS": "mentions"}, lambda: _slack()._slack_allow_bots(), "none", "mentions"),
    ("slack.api_human_users", {"SLACK_API_HUMAN_USERS": "U0BOT2"},
     lambda: set(_slack()._slack_api_human_users()), set(), {"U0BOT2"}),
    ("line.allow_all", {"LINE_ALLOW_ALL_USERS": "true"}, lambda: _line().allow_all, False, True),
    ("line.allowed_users", {"LINE_ALLOWED_USERS": "Ubot2"}, lambda: _line().allowed_users, set(), {"Ubot2"}),
    ("dingtalk.allowed_users", {"DINGTALK_ALLOWED_USERS": "Bot2Admin"},
     lambda: {u.lower() for u in _dingtalk()._csv_setting("allowed_users", "DINGTALK_ALLOWED_USERS")}, set(), {"bot2admin"}),
]


@pytest.mark.parametrize("label,scope_opt_in,gate,closed,opened", _GATES, ids=[g[0] for g in _GATES])
def test_default_env_never_answers_a_scoped_gate(default_env, label, scope_opt_in, gate, closed, opened):
    """Secondary scope with NO opt-in: the default profile's permissive os.environ must not open it."""
    with _scope({}):
        assert gate() == closed


@pytest.mark.parametrize("label,scope_opt_in,gate,closed,opened", _GATES, ids=[g[0] for g in _GATES])
def test_scope_opt_in_opens_the_gate(default_env, monkeypatch, label, scope_opt_in, gate, closed, opened):
    """The secondary's own .env decides — even when the default profile's env is silent."""
    for name in _DEFAULT_ENV:
        monkeypatch.delenv(name, raising=False)
    with _scope(scope_opt_in):
        assert gate() == opened
