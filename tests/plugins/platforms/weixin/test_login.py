"""Exercise Weixin binding through the HTTP transport boundary."""

import argparse
import base64
import json
import time
from urllib.parse import parse_qs, urlparse
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import web

from hermes_cli.config import load_config, save_config
from gateway.config import PlatformConfig
from plugins.platforms.weixin import adapter
from plugins.platforms.weixin import login


TICKET = "a" * 32
CONFIRMED = {
    "status": "confirmed", "ilink_bot_id": "test-bot", "bot_token": "new-token",
    "baseurl": "https://ilinkai.weixin.qq.com", "ilink_user_id": "test-user",
}


class Response:
    def __init__(self, payload):
        self.payload = payload
        self.ok = True
        self.status = 200

    async def __aenter__(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self

    async def __aexit__(self, *args):
        return None

    async def text(self):
        return json.dumps(self.payload)


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def request(self, method, url, **kwargs):
        self.calls.append((method.upper(), url, kwargs))
        return Response(next(self.responses))

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)


def qr(value="qr-token"):
    return {"qrcode": value, "qrcode_img_content": f"liteapp://bind/{value}"}


def ticket(value=TICKET, **overrides):
    return {"bot_spam_ticket": value, "auth_url": "liteapp://realname", "expire_time": int(time.time()) + 600, **overrides}


@pytest.fixture
def login_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(adapter.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(adapter, "_print_qr", Mock())

    def prepare(responses, answers=()):
        session = Session(responses)
        monkeypatch.setattr(adapter, "_new_session", lambda: session)
        inputs = iter(answers)
        def read_input(_):
            try:
                return next(inputs)
            except StopIteration:
                raise EOFError from None

        prompt = Mock(side_effect=read_input)
        monkeypatch.setattr("builtins.input", prompt)
        return session, prompt

    return tmp_path, prepare


def posts(session):
    return [json.loads(kwargs["data"]) for method, _, kwargs in session.calls if method == "POST"]


def status_queries(session):
    return [parse_qs(urlparse(url).query) for _, url, _ in session.calls if "get_qrcode_status" in url]


@pytest.mark.asyncio
async def test_default_binding_posts_local_tokens_without_optional_checks(login_env):
    home, prepare = login_env
    adapter.save_weixin_account(str(home), account_id="previous", token="old-token", base_url="https://example.com")
    session, prompt = prepare([qr(), CONFIRMED])

    result = await adapter.qr_login(str(home))

    assert result["token"] == "new-token"
    assert posts(session) == [{"local_token_list": ["old-token"]}]
    assert all("botrealnameverifycgi" not in url for _, url, _ in session.calls)
    prompt.assert_not_called()
    assert adapter.load_weixin_account(str(home), "test-bot")["token"] == "new-token"


@pytest.mark.asyncio
@pytest.mark.parametrize("request_pin", [False, True])
async def test_pin_retry_then_clear_after_scan(login_env, request_pin):
    home, prepare = login_env
    session, prompt = prepare([
        qr("token&encoded"), {"status": "confirmed_but_need_verifycode", **{k: v for k, v in CONFIRMED.items() if k != "status"}},
        {"status": "need_verifycode"}, {"status": "scaned"}, CONFIRMED,
    ], ["123456", "234567"])

    result = await adapter.qr_login(str(home), verify_pin=request_pin)

    assert result["account_id"] == "test-bot"
    assert prompt.call_count == 2
    assert "不匹配" in prompt.call_args_list[1].args[0]
    queries = status_queries(session)
    assert [q.get("verify_code") for q in queries] == [None, ["123456"], ["234567"], None]
    assert all(q["qrcode"] == ["token&encoded"] for q in queries)
    if request_pin:
        assert json.loads(base64.b64decode(posts(session)[0]["bot_spam_business_data"])) == {"verify_pin": 1}
    else:
        assert "bot_spam_business_data" not in posts(session)[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["", " y ", "Y"])
async def test_self_owner_skips_realname(login_env, answer):
    home, prepare = login_env
    session, prompt = prepare([qr(), CONFIRMED], ["invalid", answer])

    assert await adapter.qr_login(str(home), prompt_bind_owner=True)

    assert prompt.call_count == 2
    assert posts(session) == [{"local_token_list": []}]
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_other_owner_authorizes_before_binding_and_keeps_pin_independent(login_env):
    home, prepare = login_env
    session, _ = prepare([{"data": ticket()}, {"data": {"auth_status": "0"}}, {"auth_status": "1"}, qr(), CONFIRMED], ["N"])

    assert await adapter.qr_login(str(home), prompt_bind_owner=True, verify_pin=True)

    assert [method for method, _, _ in session.calls] == ["GET", "GET", "GET", "POST", "GET"]
    first_query = parse_qs(urlparse(session.calls[0][1]).query)
    assert first_query["step"] == ["get_auth_url"]
    assert first_query["plugin_version"] == [adapter.CHANNEL_VERSION]
    assert adapter.CHANNEL_VERSION.startswith("3.0.0")
    assert session.calls[0][2]["headers"] == {"Accept": "application/json"}
    body = posts(session)[0]
    assert body["bot_spam_ticket"] == TICKET
    assert "bot_spam_business_data" in body
    assert "base_info" not in body
    assert "Authorization" not in session.calls[3][2]["headers"]
    assert adapter._print_qr.call_args_list[0].args[1] == "liteapp://realname"
    saved = (home / "weixin/accounts/test-bot.json").read_text()
    assert TICKET not in saved and "verify_pin" not in saved


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [ticket(bot_spam_ticket="bad"), ticket(expire_time=0), {"ret": 1, **ticket()}, {"auth_url": "liteapp://realname"}])
async def test_invalid_realname_ticket_stops_without_binding(login_env, payload):
    home, prepare = login_env
    session, _ = prepare([payload], ["n"])

    assert await adapter.qr_login(str(home), prompt_bind_owner=True) is None

    assert not posts(session)
    assert not (home / "weixin/accounts/test-bot.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_status", [2, 9, None])
async def test_denied_or_unknown_authorization_does_not_bind(login_env, auth_status):
    home, prepare = login_env
    session, _ = prepare([ticket(), {"auth_status": auth_status}], ["n"])

    assert await adapter.qr_login(str(home), prompt_bind_owner=True) is None

    assert not posts(session)


@pytest.mark.asyncio
async def test_invalid_ticket_can_be_replaced_only_once(login_env):
    home, prepare = login_env
    session, _ = prepare([ticket(), {"auth_status": 3}, ticket("b" * 32), {"auth_status": 3}], ["n"])

    assert await adapter.qr_login(str(home), prompt_bind_owner=True) is None

    assert not posts(session)
    assert len(session.calls) == 4
    assert adapter._print_qr.call_count == 2


@pytest.mark.asyncio
async def test_replaced_ticket_and_transient_query_failure_recover(login_env):
    home, prepare = login_env
    session, _ = prepare([ticket(), OSError("network failure"), {"auth_status": 3}, ticket("b" * 32), {"auth_status": 1}, qr(), CONFIRMED], ["n"])

    assert await adapter.qr_login(str(home), prompt_bind_owner=True)

    assert posts(session)[0]["bot_spam_ticket"] == "b" * 32
    assert "bot_spam_business_data" not in posts(session)[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("refresh_status", ["expired", "verify_code_blocked"])
async def test_refresh_preserves_options_resets_pin_and_poll_host(login_env, refresh_status):
    home, prepare = login_env
    session, _ = prepare([
        ticket(), {"auth_status": 1}, qr("first"), {"status": "scaned_but_redirect", "redirect_host": "other.weixin.qq.com"},
        {"status": "need_verifycode"}, {"status": refresh_status}, qr("second"), CONFIRMED,
    ], ["n", "123456"])

    assert await adapter.qr_login(str(home), prompt_bind_owner=True, verify_pin=True)

    assert posts(session)[0] == posts(session)[1]
    queries = status_queries(session)
    assert queries[-2]["verify_code"] == ["123456"]
    assert "verify_code" not in queries[-1]
    assert queries[-1]["qrcode"] == ["second"]
    assert urlparse(session.calls[-1][1]).hostname == "ilinkai.weixin.qq.com"


@pytest.mark.asyncio
async def test_refresh_limit_is_three_total_qr_codes(login_env):
    home, prepare = login_env
    session, _ = prepare([qr(), {"status": "expired"}, qr(), {"status": "verify_code_blocked"}, qr(), {"status": "expired"}])

    assert await adapter.qr_login(str(home)) is None

    assert len(posts(session)) == 3


@pytest.mark.asyncio
async def test_ticket_near_expiry_prevents_qr_refresh(login_env):
    home, prepare = login_env
    session, _ = prepare([ticket(expire_time=int(time.time()) + 45), {"auth_status": 1}, qr(), {"status": "expired"}], ["n"])

    assert await adapter.qr_login(str(home), prompt_bind_owner=True) is None

    assert len(posts(session)) == 1


@pytest.mark.asyncio
async def test_already_bound_does_not_replace_credentials(login_env):
    home, prepare = login_env
    adapter.save_weixin_account(str(home), account_id="test-bot", token="old-token", base_url="https://example.com")
    prepare([qr(), {"status": "binded_redirect", **{k: v for k, v in CONFIRMED.items() if k != "status"}}])

    assert await adapter.qr_login(str(home)) is None

    assert adapter.load_weixin_account(str(home), "test-bot")["token"] == "old-token"


@pytest.mark.asyncio
async def test_request_failure_does_not_log_secret_payloads(login_env, caplog):
    home, prepare = login_env
    prepare([OSError(f"secret-token {TICKET} verify_code=123456")])

    assert await adapter.qr_login(str(home)) is None

    assert TICKET not in caplog.text and "123456" not in caplog.text and "secret-token" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("configured, overrides, expected_pin, expected_prompt", [
    ({"verify_pin": True, "prompt_bind_owner": True}, {}, True, True),
    ({"verify_pin": "false", "prompt_bind_owner": "true"}, {}, False, False),
    ({"verify_pin": True, "prompt_bind_owner": True}, {"verify_pin": False, "prompt_bind_owner": False}, False, False),
])
async def test_real_config_and_explicit_overrides(login_env, configured, overrides, expected_pin, expected_prompt):
    home, prepare = login_env
    config = load_config()
    config["platforms"] = {"weixin": {"extra": configured}}
    save_config(config)
    session, prompt = prepare([qr(), CONFIRMED], ["y"])

    assert await adapter.qr_login(str(home), **overrides)

    assert ("bot_spam_business_data" in posts(session)[0]) == expected_pin
    assert bool(prompt.call_count) == expected_prompt


@pytest.mark.asyncio
async def test_test_endpoint_and_proxy_only_affect_realname_requests(login_env):
    home, prepare = login_env
    config = load_config()
    config["platforms"] = {"weixin": {"extra": {"route_tag": "10086"}}}
    save_config(config)
    session, _ = prepare([ticket(), {"auth_status": 1}, qr(), CONFIRMED], ["n"])

    assert await adapter.qr_login(
        str(home), prompt_bind_owner=True, acctweb_base_url="http://127.0.0.1:8080/ignored?unused=1",
        acctweb_proxy="http://127.0.0.1:8888/ignored",
    )

    for index, (_, url, kwargs) in enumerate(session.calls):
        parsed = urlparse(url)
        assert "_cradle_hostname" not in parse_qs(parsed.query)
        assert "SKRouteTag" not in kwargs["headers"]
        if index < 2:
            assert parsed.netloc == "127.0.0.1:8080"
            assert parsed.path == "/security/acct/botrealnameverifycgi"
            assert kwargs["proxy"] == "http://127.0.0.1:8888"
            assert kwargs["headers"] == {"Accept": "application/json"}
        else:
            assert parsed.hostname == "ilinkai.weixin.qq.com"
            assert "proxy" not in kwargs


def test_cli_rejects_internal_route_flag():
    from plugins.platforms.weixin.cli import register_cli

    parser = argparse.ArgumentParser()
    register_cli(parser)

    with pytest.raises(SystemExit) as error:
        parser.parse_args(["login", "--route-tag", "10086"])

    assert error.value.code == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [
    {"acctweb_base_url": "file:///bad"}, {"acctweb_proxy": "http://localhost:invalid"}, {"acctweb_base_url": "http://[invalid"},
])
async def test_invalid_test_endpoint_stops_before_http(login_env, override):
    home, prepare = login_env
    session, _ = prepare([])

    assert await adapter.qr_login(str(home), **override) is None

    assert session.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("expiry", [float(int(time.time()) + 600), str(int(time.time()) + 600)])
async def test_realname_accepts_numeric_and_digit_string_timestamps(login_env, expiry):
    home, prepare = login_env
    session, _ = prepare([ticket(expire_time=expiry), {"auth_status": 1.0}, qr(), CONFIRMED], ["n"])

    assert await adapter.qr_login(str(home), prompt_bind_owner=True)

    assert posts(session)[0]["bot_spam_ticket"] == TICKET


@pytest.mark.asyncio
async def test_pin_and_incomplete_confirmation_never_save_credentials(login_env):
    home, prepare = login_env
    prepare([qr(), {**CONFIRMED, "status": "confirmed_but_need_verifycode"}], [])

    assert await adapter.qr_login(str(home)) is None

    assert not (home / "weixin/accounts/test-bot.json").exists()
    prepare([qr(), {"status": "confirmed", "ilink_bot_id": "test-bot"}])
    assert await adapter.qr_login(str(home)) is None
    assert not (home / "weixin/accounts/test-bot.json").exists()


@pytest.mark.asyncio
async def test_invalid_pin_reprompts_without_sending_it(login_env):
    home, prepare = login_env
    session, prompt = prepare([qr(), {"status": "need_verifycode"}, CONFIRMED], ["letters", "", " 123456 "])

    assert await adapter.qr_login(str(home))

    assert prompt.call_count == 3
    assert status_queries(session)[-1]["verify_code"] == ["123456"]


@pytest.mark.asyncio
async def test_timeout_stops_polling_and_never_saves_credentials(login_env, monkeypatch):
    home, prepare = login_env
    session, _ = prepare([qr(), {"status": "wait"}])
    clock = Mock()
    clock.time.return_value = time.time()
    clock.monotonic.return_value = 0

    async def advance(_):
        clock.monotonic.return_value += 600

    monkeypatch.setattr(login, "time", clock)
    monkeypatch.setattr(login.asyncio, "sleep", advance)

    assert await adapter.qr_login(str(home)) is None

    assert len(status_queries(session)) == 1
    assert not (home / "weixin/accounts/test-bot.json").exists()


def test_registered_cli_runs_binding_and_persists_env(login_env, monkeypatch):
    home, prepare = login_env
    session, _ = prepare([ticket(), {"auth_status": 1}, qr(), {"status": "need_verifycode"}, CONFIRMED], ["n", "123456"])
    from hermes_cli import gateway as gw
    from hermes_cli import gateway_setup_wizard as wizard

    monkeypatch.setattr(wizard, "confirm_reconfigure", lambda *args: True)
    monkeypatch.setattr(wizard, "offer_home_channel", Mock(side_effect=AssertionError("No channel prompt after binding")))
    monkeypatch.setattr(wizard, "prompt_csv", Mock(side_effect=AssertionError("No allowlist prompt after binding")))
    monkeypatch.setattr(gw, "prompt_yes_no", lambda *args: True)
    monkeypatch.setattr(gw, "prompt_choice", Mock(side_effect=AssertionError("No access-policy menu after binding")))
    ctx = Mock()
    adapter.register(ctx)
    command = ctx.register_cli_command.call_args.kwargs
    parser = argparse.ArgumentParser()
    command["setup_fn"](parser)

    assert command["handler_fn"](parser.parse_args(["login", "--prompt-bind-owner", "--verify-pin"])) == 0

    assert posts(session)[0]["bot_spam_ticket"] == TICKET
    assert "bot_spam_business_data" in posts(session)[0]
    env = (home / ".env").read_text()
    assert "WEIXIN_ACCOUNT_ID=test-bot" in env and "WEIXIN_TOKEN=new-token" in env
    assert "123456" not in env and TICKET not in env
    assert "WEIXIN_DM_POLICY=allowlist" in env
    assert "WEIXIN_ALLOWED_USERS=test-user" in env
    assert "WEIXIN_ALLOW_ALL_USERS=false" in env
    assert "WEIXIN_GROUP_POLICY=disabled" in env
    assert "WEIXIN_HOME_CHANNEL=test-user" in env
    assert "WEIXIN_HOME_CHANNEL_NAME=Weixin" in env
    bound_adapter = adapter.WeixinAdapter(PlatformConfig(enabled=True, token="new-token", extra={"account_id": "test-bot"}))
    assert bound_adapter._is_dm_allowed("test-user")
    assert not bound_adapter._is_dm_allowed("another-user")
    assert bound_adapter._group_policy == "disabled"


def test_setup_without_binding_user_preserves_existing_env(login_env, monkeypatch):
    home, prepare = login_env
    prepare([qr(), {**CONFIRMED, "ilink_user_id": ""}])
    from hermes_cli import gateway as gw
    from hermes_cli import gateway_setup_wizard as wizard

    env_path = home / ".env"
    old_env = "WEIXIN_ACCOUNT_ID=existing-bot\nWEIXIN_TOKEN=existing-token\nWEIXIN_ALLOWED_USERS=existing-user\n"
    env_path.write_text(old_env)
    monkeypatch.setattr(wizard, "confirm_reconfigure", lambda *args: True)
    monkeypatch.setattr(gw, "prompt_yes_no", lambda *args: True)
    monkeypatch.setattr(gw, "prompt_choice", Mock(side_effect=AssertionError("No access-policy menu after binding")))

    assert adapter.interactive_setup() is False

    assert env_path.read_text() == old_env


@pytest.mark.asyncio
async def test_real_http_body_headers_and_pin_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(adapter, "_print_qr", Mock())
    answers = iter(["n", "123456"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    calls = []

    async def respond(request):
        calls.append((request.method, dict(request.query), dict(request.headers), await request.text()))
        if request.path.endswith("botrealnameverifycgi"):
            return web.json_response(ticket() if request.query["step"] == "get_auth_url" else {"auth_status": 1})
        if request.method == "POST":
            return web.json_response(qr("token&encoded"))
        return web.json_response(CONFIRMED if request.query.get("verify_code") == "123456" else {"status": "need_verifycode"})

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", respond)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    origin = f"http://127.0.0.1:{runner.addresses[0][1]}"
    monkeypatch.setattr(login, "ILINK_BASE_URL", origin)
    try:
        result = await adapter.qr_login(
            str(tmp_path), prompt_bind_owner=True, verify_pin=True, acctweb_base_url=origin,
        )
    finally:
        await runner.cleanup()

    assert result["token"] == "new-token"
    assert calls[0][1]["plugin_version"] == adapter.CHANNEL_VERSION
    assert "AuthorizationType" not in calls[0][2]
    assert calls[2][0] == "POST"
    assert calls[2][2]["iLink-App-ClientVersion"] == str(3 << 16)
    assert json.loads(calls[2][3]) == {
        "local_token_list": [], "bot_spam_ticket": TICKET, "bot_spam_business_data": "eyJ2ZXJpZnlfcGluIjoxfQ==",
    }
    assert calls[-1][1]["qrcode"] == "token&encoded"
    assert calls[-1][1]["verify_code"] == "123456"
    assert all("_cradle_hostname" not in call[1] and "SKRouteTag" not in call[2] for call in calls)
