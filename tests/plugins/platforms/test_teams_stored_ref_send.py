"""Stored conversation-ref send for Teams (governance #1218).

Throwaway HTTP only — never a live Bot Framework host.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from plugins.platforms.teams.stored_ref import (
    StoredRefError,
    activity_post_url,
    classify_stored_ref,
    group_inbound_addresses_bot,
    group_inbound_should_reply,
    load_stored_refs,
    persist_inbound_ref,
    send_from_stored_ref,
)


BOT = "00000000-0000-4000-8000-0000000000aa"


def _throwaway(**overrides):
    ref = {
        "kind": "personal",
        "person": "Throwaway",
        "aad_object_id": "00000000-0000-0000-0000-000000000001",
        "tenant_id": "00000000-0000-0000-0000-000000000002",
        "bot_app_id": BOT,
        "service_url": "http://127.0.0.1:9/",
        "conversation_id": "throwaway-1218",
        "user_id": "29:throwaway-roster",
    }
    ref.update(overrides)
    return ref


def test_classify_accepts_personal_matching_bot():
    classify_stored_ref(_throwaway(), expected_bot_app_id=BOT)


def test_classify_rejects_group_without_inbound_addresser():
    with pytest.raises(StoredRefError, match="mention, reply-to-own, or unmentioned"):
        classify_stored_ref(_throwaway(kind="groupChat"), expected_bot_app_id=BOT)


def test_classify_rejects_group_with_sender_but_no_heard_via():
    with pytest.raises(StoredRefError, match="mention, reply-to-own, or unmentioned"):
        classify_stored_ref(
            _throwaway(kind="groupChat", addressed_by="29:customer-roster"),
            expected_bot_app_id=BOT,
        )


def test_classify_accepts_group_after_mention():
    classify_stored_ref(
        _throwaway(
            kind="groupChat",
            addressed_by="29:customer-roster",
            addressed_via="mention",
        ),
        expected_bot_app_id=BOT,
    )


def test_classify_accepts_group_after_reply_to_own():
    classify_stored_ref(
        _throwaway(
            kind="groupChat",
            addressed_by="29:customer-roster",
            addressed_via="reply_to_own",
        ),
        expected_bot_app_id=BOT,
    )


def test_classify_accepts_heard_unmentioned_group():
    classify_stored_ref(
        _throwaway(
            kind="groupChat",
            addressed_by="29:customer-roster",
            addressed_via="unmentioned",
        ),
        expected_bot_app_id=BOT,
    )


def test_classify_rejects_wrong_bot():
    with pytest.raises(StoredRefError, match="bot"):
        classify_stored_ref(_throwaway(), expected_bot_app_id="00000000-0000-0000-0000-000000000099")


def test_classify_rejects_reply_only_policy():
    with pytest.raises(StoredRefError, match="reply_only"):
        classify_stored_ref(
            _throwaway(outbound_policy="reply_only_until_customer_writes"),
            expected_bot_app_id=BOT,
        )


def test_activity_url_uses_service_url_and_conversation_id():
    url = activity_post_url(_throwaway())
    assert url == "http://127.0.0.1:9/v3/conversations/throwaway-1218/activities"


def test_load_stored_refs_indexes_by_conversation_id(tmp_path: Path):
    path = tmp_path / "throwaway.json"
    path.write_text(json.dumps(_throwaway()), encoding="utf-8")
    loaded = load_stored_refs(tmp_path)
    assert "throwaway-1218" in loaded
    assert loaded["throwaway-1218"]["user_id"] == "29:throwaway-roster"


def test_persist_inbound_ref_writes_personal_json(tmp_path: Path):
    dest = persist_inbound_ref(
        tmp_path,
        conversation_id="a:owner-chat",
        conversation_type="personal",
        service_url="https://smba.trafficmanager.net/teams/",
        tenant_id="00000000-0000-0000-0000-000000000002",
        bot_app_id=BOT,
        aad_object_id="00000000-0000-4000-8000-0000000000bb",
        user_id="29:owner-roster",
        person="Owner",
        filename_stem="owner",
    )
    data = json.loads(dest.read_text(encoding="utf-8"))
    assert data["kind"] == "personal"
    assert data["conversation_id"] == "a:owner-chat"
    assert data["user_id"] == "29:owner-roster"
    assert data["bot_app_id"] == BOT
    assert "outbound_policy" not in data


def test_persist_inbound_ref_does_not_unlock_reply_only(tmp_path: Path):
    locked = _throwaway(
        conversation_id="a:locked",
        outbound_policy="reply_only_until_customer_writes",
    )
    (tmp_path / "customer.json").write_text(json.dumps(locked), encoding="utf-8")
    with pytest.raises(StoredRefError, match="reply_only"):
        persist_inbound_ref(
            tmp_path,
            conversation_id="a:locked",
            conversation_type="personal",
            service_url="https://smba.trafficmanager.net/teams/",
            tenant_id=locked["tenant_id"],
            bot_app_id=BOT,
            filename_stem="unlocked",
        )
    assert not (tmp_path / "unlocked.json").exists()


def test_send_from_stored_ref_returns_activity_id_on_201():
    posted = {}

    async def poster(url, headers, body):
        posted["url"] = url
        posted["body"] = body
        posted["auth_prefix"] = str(headers.get("Authorization", ""))[:7]
        return 201, {"id": "activity-throwaway-1218"}

    async def run():
        return await send_from_stored_ref(
            _throwaway(),
            "STORED-REF-OWN-SEND",
            poster=poster,
            expected_bot_app_id=BOT,
            token="not-a-secret-for-test",
        )

    result = asyncio.run(run())
    assert result["success"] is True
    assert result["message_id"] == "activity-throwaway-1218"
    assert posted["url"].endswith("/v3/conversations/throwaway-1218/activities")
    assert posted["body"]["text"] == "STORED-REF-OWN-SEND"
    assert posted["body"]["from"]["id"] == f"28:{BOT}"
    assert posted["auth_prefix"] == "Bearer "


def test_send_from_stored_ref_http_400_is_not_success():
    async def poster(url, headers, body):
        return 400, {"error": {"message": "Invalid or unencrypted user ID"}}

    async def run():
        return await send_from_stored_ref(
            _throwaway(),
            "STORED-REF-OWN-SEND",
            poster=poster,
            expected_bot_app_id=BOT,
            token="not-a-secret-for-test",
        )

    result = asyncio.run(run())
    assert result.get("success") is not True
    assert "error" in result
    assert "400" in result["error"]


def test_send_from_stored_ref_missing_activity_id_is_not_success():
    async def poster(url, headers, body):
        return 201, {}

    async def run():
        return await send_from_stored_ref(
            _throwaway(),
            "STORED-REF-OWN-SEND",
            poster=poster,
            expected_bot_app_id=BOT,
            token="not-a-secret-for-test",
        )

    result = asyncio.run(run())
    assert result.get("success") is not True
    assert "activity id" in result["error"]


def test_send_from_stored_ref_poster_exception_is_not_success():
    async def poster(url, headers, body):
        raise TimeoutError("connector timeout")

    async def run():
        return await send_from_stored_ref(
            _throwaway(),
            "STORED-REF-OWN-SEND",
            poster=poster,
            expected_bot_app_id=BOT,
            token="not-a-secret-for-test",
        )

    result = asyncio.run(run())
    assert result.get("success") is not True
    assert "error" in result
    assert "TimeoutError" in result["error"] or "timeout" in result["error"].lower()


def test_send_from_stored_ref_status_zero_is_not_missing_activity_id():
    async def poster(url, headers, body):
        return 0, {"error": "stored-ref send: service host is not allowlisted"}

    async def run():
        return await send_from_stored_ref(
            _throwaway(),
            "STORED-REF-OWN-SEND",
            poster=poster,
            expected_bot_app_id=BOT,
            token="not-a-secret-for-test",
        )

    result = asyncio.run(run())
    assert result.get("success") is not True
    assert "activity id" not in result["error"]
    assert "failed (0)" in result["error"]


def _group_ref(**overrides):
    ref = _throwaway(
        kind="groupChat",
        conversation_id="19:group-throwaway",
        addressed_by="29:customer-roster",
        addressed_via="mention",
        last_inbound_activity_id="activity-inbound-1",
    )
    ref.update(overrides)
    return ref


def test_group_send_without_reply_is_not_a_first_post():
    async def poster(url, headers, body):
        raise AssertionError("must not POST a group first post")

    async def run():
        return await send_from_stored_ref(
            _group_ref(last_inbound_activity_id=""),
            "STORED-REF-OWN-SEND",
            poster=poster,
            expected_bot_app_id=BOT,
            token="not-a-secret-for-test",
        )

    result = asyncio.run(run())
    assert result.get("success") is not True
    assert "first post" in result["error"]


def test_group_send_replies_in_addressed_thread():
    posted = {}

    async def poster(url, headers, body):
        posted["body"] = body
        return 201, {"id": "activity-group-reply"}

    async def run():
        return await send_from_stored_ref(
            _group_ref(),
            "STORED-REF-OWN-SEND",
            poster=poster,
            expected_bot_app_id=BOT,
            token="not-a-secret-for-test",
            reply_to="activity-inbound-1",
        )

    result = asyncio.run(run())
    assert result["success"] is True
    assert result["message_id"] == "activity-group-reply"
    assert posted["body"]["replyToId"] == "activity-inbound-1"


def test_group_inbound_mention_addresses_bot():
    assert (
        group_inbound_addresses_bot(
            bot_app_id=BOT,
            entities=[{"type": "mention", "mentioned": {"id": f"28:{BOT}"}}],
        )
        == "mention"
    )


def test_group_inbound_reply_to_own_addresses_bot():
    assert (
        group_inbound_addresses_bot(
            bot_app_id=BOT,
            reply_to_id="activity-own-1",
            own_activity_ids=["activity-own-1"],
        )
        == "reply_to_own"
    )


def test_group_inbound_ambient_does_not_address_bot():
    assert (
        group_inbound_addresses_bot(
            bot_app_id=BOT,
            entities=[],
            reply_to_id="activity-someone-else",
            own_activity_ids=["activity-own-1"],
        )
        is None
    )


def test_persist_hears_unmentioned_group(tmp_path: Path):
    dest = persist_inbound_ref(
        tmp_path,
        conversation_id="19:group-throwaway",
        conversation_type="groupChat",
        service_url="https://smba.trafficmanager.net/teams/",
        tenant_id="00000000-0000-0000-0000-000000000002",
        bot_app_id=BOT,
        user_id="29:customer-roster",
        inbound_activity_id="activity-ambient",
    )
    data = json.loads(dest.read_text(encoding="utf-8"))
    assert data["addressed_via"] == "unmentioned"
    assert data["last_inbound_activity_id"] == "activity-ambient"


def test_adapter_hears_unmentioned_group(tmp_path: Path, monkeypatch):
    from plugins.platforms.teams.adapter import TeamsAdapter

    adapter = object.__new__(TeamsAdapter)
    adapter._client_id = BOT
    adapter._tenant_id = "00000000-0000-0000-0000-000000000002"
    adapter._stored_refs = {}
    adapter._own_activity_ids = {}
    monkeypatch.setattr(adapter, "_stored_ref_dir", lambda: tmp_path)

    class _Conv:
        conversation_type = "groupChat"
        id = "19:group-throwaway"
        tenant_id = "00000000-0000-0000-0000-000000000002"

    class _Activity:
        service_url = "https://smba.trafficmanager.net/teams/"
        id = "activity-ambient"
        entities = []
        reply_to_id = None

    class _From:
        aad_object_id = "00000000-0000-0000-0000-000000000001"
        id = "29:customer-roster"
        name = "Peer"

    adapter._persist_inbound_stored_ref(_Activity(), _Conv(), _From())
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["addressed_via"] == "unmentioned"
    assert data["conversation_id"] == "19:group-throwaway"
    assert adapter._stored_refs["19:group-throwaway"]["addressed_via"] == "unmentioned"


def test_mentioned_group_should_reply():
    assert group_inbound_should_reply("mention") is True


def test_replied_to_group_should_reply():
    assert group_inbound_should_reply("reply_to_own") is True


def test_unmentioned_group_default_silent():
    assert group_inbound_should_reply("unmentioned") is False
    assert group_inbound_should_reply(None) is False


def test_unmentioned_group_spoken_when_decided():
    assert group_inbound_should_reply("unmentioned", decide_speak=lambda: True) is True


def test_unmentioned_silent_does_not_post():
    async def poster(url, headers, body):
        raise AssertionError("must not POST when unmentioned default silent")

    async def run():
        return await send_from_stored_ref(
            _group_ref(addressed_via="unmentioned"),
            "STORED-REF-OWN-SEND",
            poster=poster,
            expected_bot_app_id=BOT,
            token="not-a-secret-for-test",
            reply_to="activity-inbound-1",
        )

    result = asyncio.run(run())
    assert result.get("success") is not True
    assert result.get("silent") is True
    assert "silent" in result["error"]


def test_unmentioned_spoken_posts_reply():
    posted = {}

    async def poster(url, headers, body):
        posted["body"] = body
        return 201, {"id": "activity-unmentioned-spoken"}

    async def run():
        return await send_from_stored_ref(
            _group_ref(addressed_via="unmentioned"),
            "STORED-REF-OWN-SEND",
            poster=poster,
            expected_bot_app_id=BOT,
            token="not-a-secret-for-test",
            reply_to="activity-inbound-1",
            decide_speak=lambda: True,
        )

    result = asyncio.run(run())
    assert result["success"] is True
    assert result["message_id"] == "activity-unmentioned-spoken"
    assert posted["body"]["replyToId"] == "activity-inbound-1"


def test_unmentioned_decide_speak_exception_is_silent():
    async def poster(url, headers, body):
        raise AssertionError("must not POST when decide_speak fails closed")

    def boom():
        raise RuntimeError("classifier failed")

    async def run():
        return await send_from_stored_ref(
            _group_ref(addressed_via="unmentioned"),
            "STORED-REF-OWN-SEND",
            poster=poster,
            expected_bot_app_id=BOT,
            token="not-a-secret-for-test",
            reply_to="activity-inbound-1",
            decide_speak=boom,
        )

    result = asyncio.run(run())
    assert result.get("success") is not True
    assert result.get("silent") is True


def _adapter_for_stored_send(poster, ref):
    from plugins.platforms.teams.adapter import TeamsAdapter

    adapter = object.__new__(TeamsAdapter)
    adapter._app = object()
    adapter._client_id = BOT
    adapter._own_activity_ids = {}
    adapter._stored_refs = {ref["conversation_id"]: ref}
    adapter.format_message = lambda content: content
    adapter.truncate_message = lambda content, max_length=4096, len_fn=None: [content]

    async def _token():
        return "not-a-secret-for-test"

    adapter._get_botframework_token = _token
    adapter._post_stored_activity = poster
    return adapter


def test_adapter_send_unmentioned_decide_speak_false_does_not_post():
    posted = []

    async def poster(url, headers, body):
        posted.append(body)
        raise AssertionError("must not POST when decide_speak is false")

    adapter = _adapter_for_stored_send(
        poster, _group_ref(addressed_via="unmentioned")
    )

    async def run():
        return await adapter.send(
            "19:group-throwaway",
            "STORED-REF-OWN-SEND",
            reply_to="activity-inbound-1",
            metadata={"decide_speak": False},
        )

    result = asyncio.run(run())
    assert posted == []
    assert result.success is True
    assert result.message_id is None


def test_adapter_send_unmentioned_decide_speak_true_posts_once():
    posted = []

    async def poster(url, headers, body):
        posted.append(body)
        return 201, {"id": "activity-adapter-spoken"}

    adapter = _adapter_for_stored_send(
        poster, _group_ref(addressed_via="unmentioned")
    )

    async def run():
        return await adapter.send(
            "19:group-throwaway",
            "STORED-REF-OWN-SEND",
            reply_to="activity-inbound-1",
            metadata={"decide_speak": True},
        )

    result = asyncio.run(run())
    assert len(posted) == 1
    assert posted[0]["replyToId"] == "activity-inbound-1"
    assert result.success is True
    assert result.message_id == "activity-adapter-spoken"


def test_adapter_send_unmentioned_default_and_error_are_silent():
    posted = []

    async def poster(url, headers, body):
        posted.append(body)
        raise AssertionError("must not POST on default or decision-error")

    ref = _group_ref(addressed_via="unmentioned")

    async def default_send():
        adapter = _adapter_for_stored_send(poster, ref)
        return await adapter.send(
            "19:group-throwaway",
            "STORED-REF-OWN-SEND",
            reply_to="activity-inbound-1",
        )

    def boom():
        raise RuntimeError("classifier failed")

    async def error_send():
        adapter = _adapter_for_stored_send(poster, ref)
        return await adapter.send(
            "19:group-throwaway",
            "STORED-REF-OWN-SEND",
            reply_to="activity-inbound-1",
            metadata={"decide_speak": boom},
        )

    default = asyncio.run(default_send())
    error = asyncio.run(error_send())
    assert posted == []
    assert default.success is True and default.message_id is None
    assert error.success is True and error.message_id is None


def test_two_senders_keep_own_refs_latest_unmentioned_not_older_mention(
    tmp_path: Path,
):
    conv = "19:group-throwaway"
    common = dict(
        conversation_id=conv,
        conversation_type="groupChat",
        service_url="https://smba.trafficmanager.net/teams/",
        tenant_id="00000000-0000-0000-0000-000000000002",
        bot_app_id=BOT,
    )
    persist_inbound_ref(
        tmp_path,
        **common,
        user_id="29:zulu-roster",
        person="Zulu",
        filename_stem="Zulu",
        inbound_activity_id="old-mentioned",
        addressed_via="mention",
    )
    persist_inbound_ref(
        tmp_path,
        **common,
        user_id="29:alpha-roster",
        person="Alpha",
        filename_stem="Alpha",
        inbound_activity_id="latest-unmentioned",
        addressed_via="unmentioned",
    )
    by_person = {
        json.loads(path.read_text(encoding="utf-8"))["person"]: json.loads(
            path.read_text(encoding="utf-8")
        )
        for path in tmp_path.glob("*.json")
    }
    zulu = by_person["Zulu"]
    alpha = by_person["Alpha"]
    assert zulu["addressed_via"] == "mention"
    assert zulu["last_inbound_activity_id"] == "old-mentioned"
    assert zulu["addressed_by"] == "29:zulu-roster"
    assert alpha["addressed_via"] == "unmentioned"
    assert alpha["last_inbound_activity_id"] == "latest-unmentioned"
    assert alpha["addressed_by"] == "29:alpha-roster"
    loaded = load_stored_refs(tmp_path)[conv]
    assert loaded["addressed_via"] == "unmentioned"
    assert loaded["last_inbound_activity_id"] == "latest-unmentioned"
    assert loaded["addressed_by"] == "29:alpha-roster"


def _persist_same_person(tmp_path: Path, *, personal_first: bool) -> None:
    person = "Pat"
    user_id = "29:same-roster"
    common = dict(
        service_url="https://smba.trafficmanager.net/teams/",
        tenant_id="00000000-0000-0000-0000-000000000002",
        bot_app_id=BOT,
        user_id=user_id,
        person=person,
    )
    personal = dict(
        conversation_id="19:personal-throwaway",
        conversation_type="personal",
        **common,
    )
    group = dict(
        conversation_id="19:group-throwaway",
        conversation_type="groupChat",
        inbound_activity_id="group-line",
        addressed_via="unmentioned",
        **common,
    )
    first, second = (personal, group) if personal_first else (group, personal)
    persist_inbound_ref(tmp_path, **first)
    persist_inbound_ref(tmp_path, **second)


@pytest.mark.parametrize("personal_first", [True, False])
def test_same_person_personal_and_group_keep_separate_refs(
    tmp_path: Path, personal_first: bool
):
    _persist_same_person(tmp_path, personal_first=personal_first)
    loaded = load_stored_refs(tmp_path)
    personal = loaded["19:personal-throwaway"]
    group = loaded["19:group-throwaway"]
    assert personal["kind"] == "personal"
    assert personal["person"] == "Pat"
    assert personal["user_id"] == "29:same-roster"
    assert group["kind"] == "groupChat"
    assert group["person"] == "Pat"
    assert group["conversation_id"] == "19:group-throwaway"
    assert personal["conversation_id"] == "19:personal-throwaway"
    names = {path.name for path in tmp_path.glob("*.json")}
    assert len(names) == 2
