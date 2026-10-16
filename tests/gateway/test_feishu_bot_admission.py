"""Adapter-layer tests for Feishu bot-sender admission (``FeishuAdapter._admit``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.gateway.feishu_helpers import (
    install_dedup_state,
    make_adapter_skeleton,
    make_message,
    make_sender,
    stub_mention,
)


# --- FeishuAdapterSettings wiring ------------------------------------------


@pytest.mark.parametrize(
    "env_value, extra, expected",
    [
        (None, {}, True),
        ("false", {}, False),
        ("true", {}, True),
        ("true", {"require_mention": False}, False),
    ],
)
def test_feishu_load_settings_require_mention(monkeypatch, env_value, extra, expected):
    from plugins.platforms.feishu.adapter import FeishuAdapter

    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_test")
    if env_value is None:
        monkeypatch.delenv("FEISHU_REQUIRE_MENTION", raising=False)
    else:
        monkeypatch.setenv("FEISHU_REQUIRE_MENTION", env_value)

    settings = FeishuAdapter._load_settings(extra=extra)
    assert settings.require_mention is expected


# --- Module-level helpers --------------------------------------------------


def test_sender_identity_collects_every_non_empty_id_variant():
    from plugins.platforms.feishu.adapter import _sender_identity

    sender = SimpleNamespace(
        sender_id=SimpleNamespace(open_id="ou_x", user_id="", union_id="un_x"),
    )
    assert _sender_identity(sender) == frozenset({"ou_x", "un_x"})


@pytest.mark.parametrize("sender_type", ["bot", "app"])
def test_is_bot_sender_treats_bot_and_app_as_bot_origin(sender_type):
    from plugins.platforms.feishu.adapter import _is_bot_sender

    assert _is_bot_sender(SimpleNamespace(sender_type=sender_type)) is True


# --- _admit pipeline matrix ------------------------------------------------
#
# Covers the four-step admission pipeline (self_echo → bot_policy →
# DM bypass → group_policy + mention) as a single result-only matrix.
# Each row pins one decision in the pipeline; tests asserting call-count
# semantics live below in their own functions.


def _admit_case(
    *,
    adapter: dict | None = None,
    sender: dict | None = None,
    message: dict | None = None,
    mentions_self: bool | None = None,
    expected: str | None = None,
):
    return {
        "adapter": adapter or {},
        "sender": sender or {},
        "message": message or {},
        "mentions_self": mentions_self,
        "expected": expected,
    }


_ADMIT_CASES = [
    pytest.param(
        _admit_case(
            adapter={"bot_open_id": "ou_me", "allow_bots": "all"},
            sender={"sender_type": "bot", "open_id": "ou_me"},
            expected="self_echo",
        ),
        id="self_echo:open_id_under_all_mode",
    ),
    pytest.param(
        _admit_case(
            adapter={"bot_open_id": "", "bot_user_id": "u_me", "allow_bots": "all"},
            sender={"sender_type": "bot", "open_id": None, "user_id": "u_me"},
            expected="self_echo",
        ),
        id="self_echo:user_id_only",
    ),
    pytest.param(
        _admit_case(
            adapter={"bot_open_id": "ou_me", "allow_bots": "all"},
            sender={"sender_type": "bot", "open_id": "ou_me", "user_id": "u_me", "union_id": "un_me"},
            expected="self_echo",
        ),
        id="self_echo:mixed_ids",
    ),
    pytest.param(
        _admit_case(
            adapter={"bot_open_id": "ou_self", "bot_user_id": "u_self", "allow_bots": "all"},
            sender={"sender_type": "bot", "open_id": None, "user_id": "u_self"},
            expected="self_echo",
        ),
        id="self_echo:user_id_when_bot_user_id_set",
    ),
    pytest.param(
        _admit_case(
            adapter={"bot_open_id": "ou_self", "allow_bots": "none"},
            sender={"sender_type": "bot", "open_id": "ou_peer"},
            expected="bots_disabled",
        ),
        id="bots_disabled:mode_none",
    ),
    pytest.param(
        _admit_case(
            adapter={"bot_open_id": "ou_self", "allow_bots": ""},
            sender={"sender_type": "bot", "open_id": "ou_peer"},
            expected="bots_disabled",
        ),
        id="bots_disabled:mode_empty",
    ),
    pytest.param(
        _admit_case(
            adapter={"bot_open_id": "ou_self", "allow_bots": "loose"},
            sender={"sender_type": "bot", "open_id": "ou_peer"},
            expected="bots_disabled",
        ),
        id="bots_disabled:mode_unknown_value",
    ),
    pytest.param(
        _admit_case(
            adapter={"bot_open_id": "", "allow_bots": "none"},
            sender={"sender_type": "bot", "open_id": "ou_peer"},
            expected="bots_disabled",
        ),
        id="bots_disabled:wins_over_self_ids_unknown",
    ),
    pytest.param(
        _admit_case(
            adapter={"bot_open_id": "", "allow_bots": "all"},
            sender={"sender_type": "bot", "open_id": "ou_peer"},
            expected="self_ids_unknown",
        ),
        id="self_ids_unknown:bot_sender_no_self_ids",
    ),
    pytest.param(
        _admit_case(
            adapter={"bot_open_id": "", "allow_bots": "all"},
            sender={"sender_type": "app", "open_id": "ou_peer"},
            expected="self_ids_unknown",
        ),
        id="self_ids_unknown:app_sender_no_self_ids",
    ),
    pytest.param(
        _admit_case(
            adapter={"bot_open_id": "ou_self", "allow_bots": "all"},
            sender={"sender_type": "app", "open_id": None},
            expected="self_ids_unknown",
        ),
        id="self_ids_unknown:no_sender_ids",
    ),
    pytest.param(
        _admit_case(
            adapter={"bot_open_id": "ou_self", "allow_bots": "mentions"},
            sender={"sender_type": "bot", "open_id": "ou_peer"},
            mentions_self=False,
            expected="bot_not_mentioned",
        ),
        id="mentions_mode:not_mentioned_dm",
    ),
    pytest.param(
        _admit_case(
            adapter={"bot_open_id": "ou_self", "allow_bots": "mentions"},
            sender={"sender_type": "bot", "open_id": "ou_peer"},
            mentions_self=True,
            expected=None,
        ),
        id="mentions_mode:mentioned_dm",
    ),
    pytest.param(
        _admit_case(
            adapter={"bot_open_id": "ou_self", "allow_bots": "all"},
            sender={"sender_type": "bot", "open_id": "ou_peer"},
            mentions_self=False,
            expected=None,
        ),
        id="all_mode:not_mentioned_dm",
    ),
    pytest.param(
        _admit_case(
            adapter={"bot_open_id": "ou_self", "allow_bots": "all"},
            sender={"sender_type": "bot", "open_id": "ou_peer"},
            mentions_self=True,
            expected=None,
        ),
        id="all_mode:mentioned_dm",
    ),
    pytest.param(
        _admit_case(
            adapter={"bot_open_id": "", "allow_bots": "none"},
            sender={"sender_type": "user", "open_id": "ou_human"},
            expected=None,
        ),
        id="human:dm_admitted_regardless_of_allow_bots",
    ),
    pytest.param(
        _admit_case(
            adapter={"allow_bots": "all"},
            sender={"sender_type": "user", "open_id": "ou_human"},
            message={"message_id": "om_ok", "chat_type": "p2p"},
            expected=None,
        ),
        id="human:p2p_admitted",
    ),
    pytest.param(
        _admit_case(
            adapter={
                "bot_open_id": "ou_self",
                "require_mention": False,
                "group_policy": "open",
            },
            sender={"sender_type": "user", "open_id": "ou_human"},
            message={"chat_type": "group"},
            mentions_self=False,
            expected=None,
        ),
        id="require_mention_false:group_human_no_mention_admitted",
    ),
    pytest.param(
        _admit_case(
            adapter={
                "bot_open_id": "ou_self",
                "allow_bots": "all",
                "require_mention": False,
                "group_policy": "open",
            },
            sender={"sender_type": "bot", "open_id": "ou_peer"},
            message={"chat_type": "group"},
            mentions_self=False,
            expected=None,
        ),
        id="require_mention_false:group_bot_all_mode_admitted",
    ),
    pytest.param(
        _admit_case(
            adapter={
                "bot_open_id": "ou_self",
                "allow_bots": "mentions",
                "require_mention": False,
                "group_policy": "open",
            },
            sender={"sender_type": "bot", "open_id": "ou_peer"},
            message={"chat_type": "group"},
            mentions_self=False,
            expected="bot_not_mentioned",
        ),
        id="require_mention_false:group_bot_mentions_mode_still_gated",
    ),
]


# --- Mention call-count semantics ------------------------------------------


def test_dm_pairing_mode_forwards_unknown_sender_to_gateway_intake(monkeypatch):
    """Empty FEISHU_ALLOWED_USERS must not block pairing handshake intake."""
    monkeypatch.delenv("FEISHU_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    adapter = make_adapter_skeleton()
    adapter._allowed_group_users = frozenset()
    sender = make_sender(open_id="ou_unknown")
    message = make_message(chat_type="p2p")
    assert adapter._admit(sender, message) is None


# --- Per-group require_mention override ------------------------------------


# --- Hydration -------------------------------------------------------------


def test_hydrate_bot_identity_populates_self_ids_from_bot_v3_info(monkeypatch):
    import asyncio

    import plugins.platforms.feishu.adapter as feishu_mod
    FeishuAdapter = feishu_mod.FeishuAdapter

    class _FakeBaseRequestBuilder:
        def __init__(self):
            self._request = SimpleNamespace()

        def http_method(self, value):
            self._request.http_method = value
            return self

        def uri(self, value):
            self._request.uri = value
            return self

        def token_types(self, value):
            self._request.token_types = value
            return self

        def build(self):
            return self._request

    monkeypatch.setattr(
        feishu_mod,
        "BaseRequest",
        SimpleNamespace(builder=lambda: _FakeBaseRequestBuilder()),
        raising=False,
    )
    monkeypatch.setattr(feishu_mod, "HttpMethod", SimpleNamespace(GET="GET"), raising=False)
    monkeypatch.setattr(feishu_mod, "AccessTokenType", SimpleNamespace(TENANT="TENANT"), raising=False)

    adapter = object.__new__(FeishuAdapter)
    adapter._bot_open_id = ""
    adapter._bot_user_id = ""
    adapter._bot_name = ""
    adapter._allow_bots = "all"

    captured = {}

    def _fake_request(request):
        captured["uri"] = getattr(request, "uri", None)
        captured["http_method"] = getattr(request, "http_method", None)
        return SimpleNamespace(raw=SimpleNamespace(
            content=b'{"code":0,"bot":{"app_name":"Hermes","open_id":"ou_hydrated"}}'
        ))

    adapter._client = SimpleNamespace(request=_fake_request)

    asyncio.run(adapter._hydrate_bot_identity())

    assert captured["uri"] == "/open-apis/bot/v3/info"
    assert str(captured["http_method"]).endswith("GET")
    assert adapter._bot_open_id == "ou_hydrated"
    assert adapter._bot_name == "Hermes"
    # /bot/v3/info doesn't surface user_id, so _bot_user_id stays empty.
    assert adapter._bot_user_id == ""


def test_resolve_sender_profile_uses_open_id_for_bot_name_lookup():
    import asyncio

    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = object.__new__(FeishuAdapter)
    adapter._client = object()
    adapter._sender_name_cache = {}
    seen_ids = []

    async def _fake_fetch_bot_names(bot_ids):
        seen_ids.extend(bot_ids)
        return {"ou_peer": "Peer Bot"}

    adapter._fetch_bot_names = _fake_fetch_bot_names

    profile = asyncio.run(
        adapter._resolve_sender_profile(
            SimpleNamespace(open_id="ou_peer", user_id="u_peer", union_id="on_peer"),
            is_bot=True,
        )
    )

    assert seen_ids == ["ou_peer"]
    assert profile["user_id"] == "u_peer"
    assert profile["user_name"] == "Peer Bot"


# --- _allow_group_message matrix -------------------------------------------
#
# Bot-bypass semantics: admitted bots skip allowlist/blacklist (parallel
# human-scope filters), but channel-level locks (disabled, admin_only) and
# admin short-circuits still apply.


def _group_case(
    *,
    adapter: dict | None = None,
    admins: set | None = None,
    group_rules: dict | None = None,
    sender: dict | None = None,
    chat_id: str = "oc_1",
    is_bot: bool = False,
    expected: bool = False,
):
    return {
        "adapter": adapter or {},
        "admins": admins or set(),
        "group_rules": group_rules or {},
        "sender": sender or {},
        "chat_id": chat_id,
        "is_bot": is_bot,
        "expected": expected,
    }


def _group_rule(policy: str, **kwargs):
    from plugins.platforms.feishu.adapter import FeishuGroupRule
    return FeishuGroupRule(policy=policy, **kwargs)


_GROUP_CASES = [
    pytest.param(
        _group_case(
            sender={"sender_type": "bot", "open_id": "ou_peer"},
            is_bot=True,
            expected=True,
        ),
        id="bot:bypasses_default_allowlist",
    ),
    pytest.param(
        _group_case(
            sender={"sender_type": "user", "open_id": "ou_stranger"},
            is_bot=False,
            expected=False,
        ),
        id="human:gated_by_default_allowlist",
    ),
    pytest.param(
        _group_case(
            admins={"ou_peer"},
            sender={"sender_type": "bot", "open_id": "ou_peer"},
            is_bot=True,
            expected=True,
        ),
        id="bot:admin_short_circuit",
    ),
    pytest.param(
        _group_case(
            admins={"u_admin"},
            sender={"sender_type": "user", "open_id": None, "user_id": "u_admin"},
            is_bot=False,
            expected=True,
        ),
        id="human:admin_via_user_id",
    ),
    pytest.param(
        _group_case(
            sender={"sender_type": "bot", "open_id": "ou_peer"},
            is_bot=True,
            expected=True,
        ),
        id="bot:allowlist_skipped",
    ),
    pytest.param(
        _group_case(
            sender={"sender_type": "app", "open_id": "ou_peer"},
            is_bot=True,
            expected=True,
        ),
        id="app:allowlist_skipped",
    ),
]


# Channel-lock cases need group_rules construction; keep them in a separate
# parametrize so we can use _group_rule() (FeishuGroupRule import).
_GROUP_RULE_CASES = [
    pytest.param(
        "disabled", "bot", False,
        id="bot:disabled_policy_blocks_even_with_bypass",
    ),
    pytest.param(
        "disabled", "app", False,
        id="app:disabled_policy_blocks_even_with_bypass",
    ),
    pytest.param(
        "admin_only", "bot", False,
        id="bot:admin_only_policy_blocks_non_admin",
    ),
    pytest.param(
        "admin_only", "app", False,
        id="app:admin_only_policy_blocks_non_admin",
    ),
]


@pytest.mark.parametrize("sender_type", ["bot", "app"])
def test_allow_group_message_blacklist_is_human_scope_only(sender_type):
    # blacklist is parallel to allowlist (human-scope); admitted bots bypass
    # it. To block a specific bot, gate upstream via FEISHU_ALLOW_BOTS.
    adapter = make_adapter_skeleton()
    adapter._group_rules = {
        "oc_1": _group_rule("blacklist", blacklist={"ou_peer"})
    }
    sender = make_sender(sender_type=sender_type, open_id="ou_peer")
    assert adapter._allow_group_message(
        sender_id=sender.sender_id,
        chat_id="oc_1",
        is_bot=True,
    ) is True


# --- Realistic payload smoke -----------------------------------------------


def test_admit_accepts_realistic_bot_at_bot_group_event():
    # Locks in the real im.message.receive_v1 payload shape under mode=mentions.
    adapter = make_adapter_skeleton(bot_open_id="ou_self", allow_bots="mentions")

    mention = SimpleNamespace(
        key="@_user_1",
        id=SimpleNamespace(union_id="on_mentionUnion", user_id="", open_id="ou_self"),
        name="Hermes",
        mentioned_type="bot",
        tenant_key="tenant_ab",
    )
    message = SimpleNamespace(
        message_id="om_realistic_bot_at_bot",
        chat_id="oc_real",
        chat_type="group",
        message_type="text",
        content='{"text":"@_user_1 hello"}',
        mentions=[mention],
    )
    sender = SimpleNamespace(
        sender_type="bot",
        sender_id=SimpleNamespace(union_id="on_peerUnion", user_id="u_peer", open_id="ou_peer_bot"),
        tenant_key="tenant_ab",
    )

    assert adapter._admit(sender, message) is None


# --- Event-dispatch plumbing -----------------------------------------------


def test_handle_message_event_data_forwards_sender_when_admitted():
    import asyncio

    adapter = make_adapter_skeleton(allow_bots="all")
    install_dedup_state(adapter)
    captured = {}

    async def _fake_process_inbound_message(**kwargs):
        captured.update(kwargs)

    adapter._process_inbound_message = _fake_process_inbound_message

    sender = make_sender(sender_type="bot", open_id="ou_peer")
    data = SimpleNamespace(
        event=SimpleNamespace(
            sender=sender,
            message=make_message(message_id="om_bot_ok", chat_type="p2p"),
        )
    )

    asyncio.run(adapter._handle_message_event_data(data))
    assert captured.get("sender_id") is sender.sender_id
    assert captured.get("is_bot") is True
    assert captured.get("message_id") == "om_bot_ok"


# --- Profile-scoped admission config (#86905) -------------------------------


def test_dm_admission_config_resolves_from_profile_scope_under_multiplex(tmp_path, monkeypatch):
    """os.environ holds the DEFAULT profile's admission view; a secondary
    profile's .env must govern its own adapter — Feishu open_ids are
    app-scoped, so the default allow-list can never match the role app's
    senders, and the role profile's allow-all flag must be honored."""
    import agent.secret_scope as ss
    from plugins.platforms.feishu.adapter import FeishuAdapter

    monkeypatch.setenv("FEISHU_APP_ID", "cli_default")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_default")
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_default_view")
    monkeypatch.setenv("FEISHU_ALLOW_BOTS", "all")
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FEISHU_ALLOW_ALL_USERS", raising=False)
    (tmp_path / ".env").write_text(
        "FEISHU_APP_ID=cli_role\nFEISHU_APP_SECRET=secret_role\n"
        "FEISHU_ALLOWED_USERS=ou_role_view\n",
        encoding="utf-8",
    )

    ss.set_multiplex_active(True)
    tok = ss.set_secret_scope(ss.build_profile_secret_scope(tmp_path))
    try:
        settings = FeishuAdapter._load_settings(extra={})
    finally:
        ss.reset_secret_scope(tok)
    (tmp_path / ".env").write_text(
        "FEISHU_APP_ID=cli_role\nFEISHU_APP_SECRET=secret_role\nGATEWAY_ALLOW_ALL_USERS=true\n",
        encoding="utf-8",
    )
    tok = ss.set_secret_scope(ss.build_profile_secret_scope(tmp_path))
    try:
        allow_all = FeishuAdapter._load_settings(extra={})
    finally:
        ss.reset_secret_scope(tok)
        ss.set_multiplex_active(False)

    assert settings.app_id == "cli_role"
    assert settings.allowed_group_users == frozenset({"ou_role_view"})
    assert settings.allow_bots == "none"  # default's "all" must not leak in
    assert settings.allow_all_dm is False

    # _admit runs on the WS thread with no scope: the snapshot must carry.
    adapter = object.__new__(FeishuAdapter)
    adapter._apply_settings(settings)
    assert adapter._admit(make_sender(open_id="ou_role_view"), make_message(chat_type="p2p")) is None
    assert adapter._admit(make_sender(open_id="ou_default_view"), make_message(chat_type="p2p")) == "dm_policy_rejected"

    assert allow_all.allow_all_dm is True
    adapter = object.__new__(FeishuAdapter)
    adapter._apply_settings(allow_all)
    assert adapter._admit(make_sender(open_id="ou_anyone"), make_message(chat_type="p2p")) is None


def test_dm_admission_config_falls_back_to_os_environ_when_unscoped(monkeypatch):
    """Single-profile behavior unchanged: process env still configures DMs."""
    from plugins.platforms.feishu.adapter import FeishuAdapter

    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_test")
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_a,ou_b")

    settings = FeishuAdapter._load_settings(extra={})
    assert settings.allow_all_dm is True
    assert settings.allowed_group_users == frozenset({"ou_a", "ou_b"})
    adapter = object.__new__(FeishuAdapter)
    adapter._apply_settings(settings)
    assert adapter._admit(make_sender(open_id="ou_anyone"), make_message(chat_type="p2p")) is None
