"""Wire contracts from Tencent's openclaw-weixin 2.4.9 channel."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from urllib.parse import parse_qs, urlparse

import pytest

from gateway.config import PlatformConfig
from gateway.platforms import weixin


class Response:
    ok = True
    status = 200

    def __init__(self, body):
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def text(self):
        return json.dumps(self.body)


class QrSession:
    def __init__(self, statuses):
        self.statuses = iter(statuses)
        self.posts = []
        self.gets = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return Response({"qrcode": "qr/token?", "qrcode_img_content": "https://weixin.qq.com/qr"})

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        if "get_bot_qrcode" in url:
            return Response({"qrcode": "qr/token?", "qrcode_img_content": "https://weixin.qq.com/qr"})
        return Response(next(self.statuses))


CONFIRMED = {
    "status": "confirmed", "ilink_bot_id": "bot@im.bot", "bot_token": "new-token",
    "baseurl": "https://ilinkai.weixin.qq.com", "ilink_user_id": "user@im.wechat",
}


@pytest.mark.asyncio
async def test_qr_login_posts_only_this_profiles_recent_account_tokens(tmp_path, monkeypatch):
    other_profile = tmp_path / "other-profile"
    weixin.save_weixin_account(str(other_profile), account_id="other-bot", token="other-token", base_url=weixin.ILINK_BASE_URL)
    weixin.save_weixin_account(str(tmp_path), account_id="old-bot", token="old-token", base_url=weixin.ILINK_BASE_URL)
    account_dir = tmp_path / "weixin" / "accounts"
    (account_dir / "old-bot.context-tokens.json").write_text('{"token":"peer-context"}', encoding="utf-8")
    session = QrSession([CONFIRMED])
    monkeypatch.setattr(weixin, "_new_session", lambda: session)
    monkeypatch.setattr(weixin, "_print_qr", Mock())

    credentials = await weixin.qr_login(str(tmp_path), bot_type="3&unexpected=1")

    assert credentials["token"] == "new-token"
    qr_url, request = session.posts[0]
    assert parse_qs(urlparse(qr_url).query) == {"bot_type": ["3&unexpected=1"]}
    assert json.loads(request["data"])["local_token_list"] == ["old-token"]
    assert "Authorization" not in request["headers"]
    assert json.loads(request["data"])["base_info"]["bot_agent"] == "Hermes"
    assert parse_qs(urlparse(session.gets[0][0]).query) == {"qrcode": ["qr/token?"]}
    assert weixin.load_weixin_account(str(tmp_path), "bot@im.bot")["token"] == "new-token"


@pytest.mark.asyncio
@pytest.mark.parametrize("refresh_status", ["expired", "verify_code_blocked"])
async def test_verification_retry_and_qr_refresh_reset_redirect_and_code(tmp_path, monkeypatch, refresh_status):
    session = QrSession([
        {"status": "scaned_but_redirect", "redirect_host": "ilinkai2.weixin.qq.com"},
        {"status": "need_verifycode"},
        {"status": "need_verifycode"},
        {"status": refresh_status},
        CONFIRMED,
    ])
    monkeypatch.setattr(weixin, "_new_session", lambda: session)
    monkeypatch.setattr(weixin, "_print_qr", Mock())
    monkeypatch.setattr(weixin.asyncio, "sleep", AsyncMock())
    code_reader = AsyncMock(side_effect=["123&456", "654321"])

    credentials = await weixin.qr_login(
        str(tmp_path), verify_code_reader=code_reader, bot_agent="Hermes/9.9 (Desktop) 中文 Invalid", route_tag=42,
    )

    assert credentials["account_id"] == "bot@im.bot"
    assert code_reader.await_count == 2
    assert len(session.posts) == 2
    assert all(json.loads(request["data"])["base_info"]["bot_agent"] == "Hermes/9.9 (Desktop)" for _, request in session.posts)
    assert all(request["headers"]["SKRouteTag"] == "42" for _, request in [*session.posts, *session.gets])
    assert urlparse(session.gets[1][0]).hostname == "ilinkai2.weixin.qq.com"
    assert parse_qs(urlparse(session.gets[2][0]).query)["verify_code"] == ["123&456"]
    assert parse_qs(urlparse(session.gets[3][0]).query)["verify_code"] == ["654321"]
    assert urlparse(session.gets[4][0]).hostname == "ilinkai.weixin.qq.com"
    assert "verify_code" not in parse_qs(urlparse(session.gets[4][0]).query)


@pytest.mark.asyncio
async def test_rebinding_removes_only_same_profile_same_user_stale_accounts(tmp_path, monkeypatch):
    foreign_home = tmp_path / "other-profile"
    for home, account_id, user_id in [
        (tmp_path, "old-bot", CONFIRMED["ilink_user_id"]),
        (tmp_path, "other-bot", "other-user"),
        (foreign_home, "foreign-bot", CONFIRMED["ilink_user_id"]),
    ]:
        weixin.save_weixin_account(str(home), account_id=account_id, token=f"{account_id}-token", base_url=weixin.ILINK_BASE_URL, user_id=user_id)
    old_dir = tmp_path / "weixin" / "accounts"
    for suffix in (".sync.json", ".context-tokens.json"):
        (old_dir / f"old-bot{suffix}").write_text('{"peer":"stale"}', encoding="utf-8")
    source = tmp_path / "attachment.txt"
    source.write_text("original attachment", encoding="utf-8")
    stale_store = weixin.WeixinQuoteStore(str(tmp_path), "old-bot")
    other_store = weixin.WeixinQuoteStore(str(tmp_path), "other-bot")
    stale_store.put("peer", "1", "stale quote", str(source))
    other_store.put("peer", "1", "other quote", str(source))
    stale_media = stale_store.find("peer", "1")["media_path"]
    session = QrSession([CONFIRMED])
    monkeypatch.setattr(weixin, "_new_session", lambda: session)
    monkeypatch.setattr(weixin, "_print_qr", Mock())

    credentials = await weixin.qr_login(str(tmp_path))

    assert credentials["account_id"] == CONFIRMED["ilink_bot_id"]
    assert {account["account_id"] for account in weixin.list_weixin_accounts(str(tmp_path))} == {"other-bot", CONFIRMED["ilink_bot_id"]}
    assert not any(old_dir.glob("old-bot*.json"))
    assert stale_store.find("peer", "1") is None
    assert not Path(stale_media).exists()
    assert other_store.find("peer", "1")["body"] == "other quote"
    assert source.read_text(encoding="utf-8") == "original attachment"
    assert weixin.load_weixin_account(str(foreign_home), "foreign-bot")["token"] == "foreign-bot-token"


@pytest.mark.asyncio
async def test_incomplete_rebind_preserves_existing_account(tmp_path, monkeypatch):
    weixin.save_weixin_account(str(tmp_path), account_id="old-bot", token="old-token", base_url=weixin.ILINK_BASE_URL,
                              user_id=CONFIRMED["ilink_user_id"])
    session = QrSession([{**CONFIRMED, "bot_token": ""}])
    monkeypatch.setattr(weixin, "_new_session", lambda: session)
    monkeypatch.setattr(weixin, "_print_qr", Mock())

    assert await weixin.qr_login(str(tmp_path)) is None
    assert weixin.load_weixin_account(str(tmp_path), "old-bot")["token"] == "old-token"


@pytest.mark.asyncio
async def test_already_bound_qr_login_restores_credentials_without_rewriting(tmp_path, monkeypatch):
    weixin.save_weixin_account(str(tmp_path), account_id="saved-bot", token="saved-token", base_url=weixin.ILINK_BASE_URL)
    path = tmp_path / "weixin" / "accounts" / "saved-bot.json"
    original = path.read_bytes()
    session = QrSession([{"status": "binded_redirect"}])
    monkeypatch.setattr(weixin, "_new_session", lambda: session)
    monkeypatch.setattr(weixin, "_print_qr", Mock())

    credentials = await weixin.qr_login(str(tmp_path))

    assert credentials["account_id"] == "saved-bot"
    assert credentials["token"] == "saved-token"
    assert path.read_bytes() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("account_ids, server_id, expected", [
    ([], "", None),
    (["first-bot", "second-bot"], "", None),
    (["first-bot", "second-bot"], "second-bot", "second-bot"),
    (["first-bot"], "unknown-bot", None),
])
async def test_already_bound_login_requires_unambiguous_local_identity(tmp_path, monkeypatch, account_ids, server_id, expected):
    for account_id in account_ids:
        weixin.save_weixin_account(str(tmp_path), account_id=account_id, token=f"token-{account_id}", base_url=weixin.ILINK_BASE_URL)
    session = QrSession([{"status": "binded_redirect", "ilink_bot_id": server_id}])
    monkeypatch.setattr(weixin, "_new_session", lambda: session)
    monkeypatch.setattr(weixin, "_print_qr", Mock())

    credentials = await weixin.qr_login(str(tmp_path))

    assert (credentials["account_id"] if credentials else None) == expected


@pytest.mark.asyncio
async def test_qr_login_stops_after_three_codes(tmp_path, monkeypatch):
    session = QrSession([{"status": "expired"}] * 3)
    monkeypatch.setattr(weixin, "_new_session", lambda: session)
    monkeypatch.setattr(weixin, "_print_qr", Mock())
    monkeypatch.setattr(weixin.asyncio, "sleep", AsyncMock())

    assert await weixin.qr_login(str(tmp_path)) is None
    assert len(session.posts) == 3


@pytest.mark.asyncio
async def test_verify_prompt_eof_ends_login(tmp_path, monkeypatch):
    session = QrSession([{"status": "need_verifycode"}])
    monkeypatch.setattr(weixin, "_new_session", lambda: session)
    monkeypatch.setattr(weixin, "_print_qr", Mock())
    monkeypatch.setattr(weixin, "_read_verify_code", AsyncMock(side_effect=EOFError))

    assert await weixin.qr_login(str(tmp_path)) is None


@pytest.mark.asyncio
async def test_terminal_verify_prompt_preserves_leading_zeroes(tmp_path, monkeypatch):
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    session = QrSession([{"status": "need_verifycode"}, CONFIRMED])
    monkeypatch.setattr(weixin, "_new_session", lambda: session)
    monkeypatch.setattr(weixin, "_print_qr", Mock())
    with create_pipe_input() as terminal_input:
        monkeypatch.setattr("prompt_toolkit.PromptSession", lambda: PromptSession(input=terminal_input, output=DummyOutput()))
        terminal_input.send_text("012345\n")

        credentials = await weixin.qr_login(str(tmp_path))

    assert credentials["account_id"] == "bot@im.bot"
    assert parse_qs(urlparse(session.gets[1][0]).query)["verify_code"] == ["012345"]


@pytest.mark.asyncio
async def test_lifecycle_notifies_ilink_before_polling_and_closing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = weixin.WeixinAdapter(PlatformConfig(enabled=True, token="test-token", extra={
        "account_id": "test-bot", "bot_agent": "Hermes/1.0", "route_tag": "17",
    }))
    poll_session, send_session = Mock(), Mock()
    for session in (poll_session, send_session):
        session.closed = False
        session.close = AsyncMock()
    monkeypatch.setattr(weixin, "_new_session", Mock(side_effect=[poll_session, send_session]))
    monkeypatch.setattr(weixin, "check_weixin_requirements", lambda: True)
    monkeypatch.setattr(adapter, "_acquire_platform_lock", Mock(return_value=True))
    monkeypatch.setattr(adapter, "_release_platform_lock", Mock())
    monkeypatch.setattr(adapter, "_wire_plugin_handlers", Mock())
    seen = []

    async def api_post(session, **kwargs):
        assert not session.closed
        assert adapter._poll_task is None
        assert kwargs["bot_agent"] == "Hermes/1.0"
        assert kwargs["route_tag"] == "17"
        seen.append(kwargs["endpoint"])
        if kwargs["endpoint"].endswith("notifystart"):
            raise RuntimeError("temporary network error")
        return {"ret": 0}

    monkeypatch.setattr(weixin, "_api_post", api_post)
    monkeypatch.setattr(adapter, "_poll_loop", AsyncMock())

    assert await adapter.connect()
    await adapter.disconnect()

    assert seen == ["ilink/bot/msg/notifystart", "ilink/bot/msg/notifystop"]
    send_session.close.assert_awaited_once()
    poll_session.close.assert_awaited_once()
