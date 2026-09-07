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


# --- Reaction routing must clear the same admission gate -------------------


def _reaction_adapter(**kwargs):
    """Adapter whose GET-message lookup says the reacted-to message is ours."""
    adapter = make_adapter_skeleton(**kwargs)
    adapter._app_id = "cli_self_app"
    adapter._client = SimpleNamespace(
        im=SimpleNamespace(
            v1=SimpleNamespace(
                message=SimpleNamespace(
                    get=lambda _req: SimpleNamespace(
                        success=lambda: True,
                        data=SimpleNamespace(
                            items=[
                                SimpleNamespace(
                                    sender=SimpleNamespace(id="cli_self_app"),
                                    chat_id=adapter._reaction_chat_id,
                                    chat_type=adapter._reaction_chat_type,
                                )
                            ]
                        ),
                    )
                )
            )
        )
    )
    adapter._build_get_message_request = lambda _mid: object()

    async def _run_blocking(func, *args):
        return func(*args)

    adapter._run_blocking = _run_blocking

    async def _resolve_sender_profile(user_id_obj, *, is_bot=False):
        return {"user_id": getattr(user_id_obj, "open_id", None), "user_name": "U", "user_id_alt": None}

    adapter._resolve_sender_profile = _resolve_sender_profile

    async def _get_chat_info(_chat_id):
        return {"name": "Chat", "type": None}

    adapter.get_chat_info = _get_chat_info
    adapter._resolve_channel_prompt = lambda *_a, **_k: None
    adapter.build_source = lambda **kw: SimpleNamespace(**kw)
    routed: list = []

    async def _handle_message_with_guards(event):
        routed.append(event)

    adapter._handle_message_with_guards = _handle_message_with_guards
    adapter._routed = routed
    return adapter


def _reaction_data(open_id: str):
    return SimpleNamespace(
        event=SimpleNamespace(
            message_id="om_bot_msg",
            operator_type="user",
            user_id=SimpleNamespace(open_id=open_id, user_id=None, union_id=None),
            reaction_type=SimpleNamespace(emoji_type="THUMBSUP"),
        )
    )


def _run_reaction(adapter, open_id: str, *, chat_id: str, chat_type: str):
    import asyncio

    adapter._reaction_chat_id = chat_id
    adapter._reaction_chat_type = chat_type
    asyncio.run(adapter._handle_reaction_event("im.message.reaction.created_v1", _reaction_data(open_id)))
    return adapter._routed


def test_group_reaction_from_non_allowlisted_user_is_dropped(monkeypatch):
    """FEISHU_GROUP_POLICY=allowlist must gate reactions like text (#bypass)."""
    adapter = _reaction_adapter(group_policy="allowlist")
    adapter._allowed_group_users = frozenset({"ou_owner"})
    assert _run_reaction(adapter, "ou_stranger", chat_id="oc_g", chat_type="group") == []


def test_group_reaction_from_allowlisted_user_is_routed(monkeypatch):
    adapter = _reaction_adapter(group_policy="allowlist")
    adapter._allowed_group_users = frozenset({"ou_owner"})
    routed = _run_reaction(adapter, "ou_owner", chat_id="oc_g", chat_type="group")
    assert len(routed) == 1
    assert routed[0].text == "reaction:added:THUMBSUP"


def test_group_reaction_ignores_mention_requirement(monkeypatch):
    """require_mention applies to text; a reaction has no @ payload."""
    adapter = _reaction_adapter(group_policy="open", require_mention=True)
    routed = _run_reaction(adapter, "ou_anyone", chat_id="oc_g", chat_type="group")
    assert len(routed) == 1


def test_group_reaction_respects_per_chat_disabled_rule(monkeypatch):
    from plugins.platforms.feishu.adapter import FeishuGroupRule

    adapter = _reaction_adapter(group_policy="open")
    adapter._group_rules = {"oc_locked": FeishuGroupRule(policy="disabled")}
    assert _run_reaction(adapter, "ou_anyone", chat_id="oc_locked", chat_type="group") == []


def test_dm_reaction_from_non_allowlisted_user_is_dropped(monkeypatch):
    monkeypatch.delenv("FEISHU_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    adapter = _reaction_adapter()
    adapter._allowed_group_users = frozenset({"ou_owner"})
    assert _run_reaction(adapter, "ou_stranger", chat_id="oc_dm", chat_type="p2p") == []


def test_dm_reaction_from_allowlisted_user_is_routed(monkeypatch):
    monkeypatch.delenv("FEISHU_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    adapter = _reaction_adapter()
    adapter._allowed_group_users = frozenset({"ou_owner"})
    assert len(_run_reaction(adapter, "ou_owner", chat_id="oc_dm", chat_type="p2p")) == 1
