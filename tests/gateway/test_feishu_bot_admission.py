"""Adapter-layer tests for Feishu bot-sender admission (``FeishuAdapter._admit``)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

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
    "env_value, expected",
    [
        ("none", "none"),
        ("mentions", "mentions"),
        ("all", "all"),
        ("  Mentions  ", "mentions"),
    ],
)
def test_feishu_load_settings_populates_allow_bots(monkeypatch, env_value, expected):
    from gateway.platforms.feishu import FeishuAdapter

    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_test")
    monkeypatch.setenv("FEISHU_ALLOW_BOTS", env_value)

    settings = FeishuAdapter._load_settings(extra={})
    assert settings.allow_bots == expected


def test_feishu_load_settings_allow_bots_defaults_to_none(monkeypatch):
    from gateway.platforms.feishu import FeishuAdapter

    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_test")
    monkeypatch.delenv("FEISHU_ALLOW_BOTS", raising=False)

    settings = FeishuAdapter._load_settings(extra={})
    assert settings.allow_bots == "none"


def test_feishu_load_settings_ignores_extra_allow_bots(monkeypatch):
    # extra is ignored — env is single source of truth (yaml is bridged to env).
    from gateway.platforms.feishu import FeishuAdapter

    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_test")
    monkeypatch.delenv("FEISHU_ALLOW_BOTS", raising=False)

    settings = FeishuAdapter._load_settings(extra={"allow_bots": "all"})
    assert settings.allow_bots == "none"


def test_feishu_load_settings_falls_back_to_env_when_extra_missing(monkeypatch):
    from gateway.platforms.feishu import FeishuAdapter

    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_test")
    monkeypatch.setenv("FEISHU_ALLOW_BOTS", "mentions")

    settings = FeishuAdapter._load_settings(extra={})
    assert settings.allow_bots == "mentions"


def test_feishu_load_settings_warns_on_unknown_allow_bots(monkeypatch, caplog):
    import logging

    from gateway.platforms.feishu import FeishuAdapter

    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_test")
    monkeypatch.setenv("FEISHU_ALLOW_BOTS", "menton")  # typo

    with caplog.at_level(logging.WARNING, logger="gateway.platforms.feishu"):
        settings = FeishuAdapter._load_settings(extra={})

    assert settings.allow_bots == "none"
    assert any("allow_bots" in r.message and "menton" in r.message for r in caplog.records)


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
    from gateway.platforms.feishu import FeishuAdapter

    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_test")
    if env_value is None:
        monkeypatch.delenv("FEISHU_REQUIRE_MENTION", raising=False)
    else:
        monkeypatch.setenv("FEISHU_REQUIRE_MENTION", env_value)

    settings = FeishuAdapter._load_settings(extra=extra)
    assert settings.require_mention is expected


def test_feishu_load_settings_parses_per_group_require_mention(monkeypatch):
    from gateway.platforms.feishu import FeishuAdapter

    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_test")

    settings = FeishuAdapter._load_settings(extra={
        "group_rules": {
            "oc_free": {"policy": "open", "require_mention": False},
            "oc_strict": {"policy": "open", "require_mention": True},
            "oc_inherit": {"policy": "open"},
        },
    })
    assert settings.group_rules["oc_free"].require_mention is False
    assert settings.group_rules["oc_strict"].require_mention is True
    assert settings.group_rules["oc_inherit"].require_mention is None


# --- Module-level helpers --------------------------------------------------


def test_sender_identity_collects_every_non_empty_id_variant():
    from gateway.platforms.feishu import _sender_identity

    sender = SimpleNamespace(
        sender_id=SimpleNamespace(open_id="ou_x", user_id="", union_id="un_x"),
    )
    assert _sender_identity(sender) == frozenset({"ou_x", "un_x"})


def test_sender_identity_handles_missing_sender_id():
    from gateway.platforms.feishu import _sender_identity

    assert _sender_identity(SimpleNamespace()) == frozenset()


@pytest.mark.parametrize("sender_type", ["bot", "app"])
def test_is_bot_sender_treats_bot_and_app_as_bot_origin(sender_type):
    from gateway.platforms.feishu import _is_bot_sender

    assert _is_bot_sender(SimpleNamespace(sender_type=sender_type)) is True


@pytest.mark.parametrize("sender_type", ["user", "", None])
def test_is_bot_sender_rejects_non_bot_origin(sender_type):
    from gateway.platforms.feishu import _is_bot_sender

    assert _is_bot_sender(SimpleNamespace(sender_type=sender_type)) is False


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


@pytest.mark.parametrize("case", _ADMIT_CASES)
def test_admit_pipeline(case):
    adapter = make_adapter_skeleton(**case["adapter"])
    if case["mentions_self"] is not None:
        stub_mention(adapter, case["mentions_self"])
    sender = make_sender(**case["sender"])
    message = make_message(**case["message"])
    assert adapter._admit(sender, message) == case["expected"]


# --- Mention call-count semantics ------------------------------------------


def test_admit_skips_mention_check_under_all_mode():
    # Tripwire: under allow_bots=all the mention path must not be probed.
    adapter = make_adapter_skeleton(bot_open_id="ou_self", allow_bots="all")
    calls = 0

    def _tripwire(_message):
        nonlocal calls
        calls += 1
        return False

    adapter._mentions_self = _tripwire

    sender = make_sender(sender_type="bot", open_id="ou_peer")
    assert adapter._admit(sender, make_message()) is None
    assert calls == 0


def test_admit_group_mention_checked_once_per_call():
    # Stage 2 (mentions mode) and stage 4 (group require_mention) must not
    # double-evaluate _mentions_self for the same admit call.
    adapter = make_adapter_skeleton(
        bot_open_id="ou_self", allow_bots="mentions", require_mention=True,
        group_policy="open",
    )
    calls = 0

    def _counting(_message):
        nonlocal calls
        calls += 1
        return True

    adapter._mentions_self = _counting

    sender = make_sender(sender_type="bot", open_id="ou_peer")
    assert adapter._admit(sender, make_message(chat_type="group")) is None
    assert calls == 1


# --- Per-group require_mention override ------------------------------------


def test_admit_per_group_require_mention_overrides_global():
    from gateway.platforms.feishu import FeishuGroupRule

    adapter = make_adapter_skeleton(
        bot_open_id="ou_self", require_mention=True, group_policy="open",
    )
    adapter._group_rules = {
        "oc_free": FeishuGroupRule(policy="open", require_mention=False),
    }
    stub_mention(adapter, False)

    sender = make_sender(sender_type="user", open_id="ou_human")
    assert adapter._admit(sender, make_message(chat_id="oc_free", chat_type="group")) is None
    assert (
        adapter._admit(sender, make_message(chat_id="oc_other", chat_type="group"))
        == "group_policy_rejected"
    )


# --- Hydration -------------------------------------------------------------


def test_hydrate_bot_identity_populates_self_ids_from_bot_v3_info(monkeypatch):
    import asyncio

    from gateway.platforms.feishu import FeishuAdapter

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

    from gateway.platforms.feishu import FeishuAdapter

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
    from gateway.platforms.feishu import FeishuGroupRule
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


@pytest.mark.parametrize("case", _GROUP_CASES)
def test_allow_group_message_matrix(case):
    adapter = make_adapter_skeleton(**case["adapter"])
    adapter._admins = case["admins"]
    adapter._group_rules = case["group_rules"]
    sender = make_sender(**case["sender"])
    assert adapter._allow_group_message(
        sender_id=sender.sender_id,
        chat_id=case["chat_id"],
        is_bot=case["is_bot"],
    ) is case["expected"]


@pytest.mark.parametrize("policy, sender_type, expected", _GROUP_RULE_CASES)
def test_allow_group_message_channel_locks_apply_to_bots(policy, sender_type, expected):
    adapter = make_adapter_skeleton()
    adapter._group_rules = {"oc_locked": _group_rule(policy)}
    sender = make_sender(sender_type=sender_type, open_id="ou_peer")
    assert adapter._allow_group_message(
        sender_id=sender.sender_id,
        chat_id="oc_locked",
        is_bot=True,
    ) is expected


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


def test_handle_message_event_data_drops_bot_sender_by_default():
    import asyncio

    adapter = make_adapter_skeleton()
    install_dedup_state(adapter)
    processed = []

    async def _fake_process_inbound_message(**kwargs):
        processed.append(kwargs)

    adapter._process_inbound_message = _fake_process_inbound_message

    data = SimpleNamespace(
        event=SimpleNamespace(
            sender=make_sender(sender_type="bot", open_id="ou_peer"),
            message=make_message(message_id="om_bot_default", chat_type="p2p"),
        )
    )

    asyncio.run(adapter._handle_message_event_data(data))
    assert processed == []


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
