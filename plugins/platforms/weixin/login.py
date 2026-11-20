"""Interactive iLink binding, optional owner authorization, and phone PIN checks."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode, urlsplit

import aiohttp

from .protocol import CHANNEL_VERSION, ILINK_APP_CLIENT_VERSION, ILINK_APP_ID, ILINK_BASE_URL

logger = logging.getLogger(__name__)
ACCTWEB_ORIGIN = "https://weixin110.qq.com"
ACCTWEB_PATH = "/security/acct/botrealnameverifycgi"


class LoginError(Exception):
    """A safe, user-facing failure message without server payloads or credentials."""


class RequestError(LoginError):
    """A transient HTTP or JSON transport failure."""


def _origin(value: str) -> str:
    try:
        parsed = urlsplit(value.strip())
        parsed.port
    except ValueError:
        raise LoginError("测试服务地址或端口无效。") from None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise LoginError("测试服务地址必须是有效的 HTTP 或 HTTPS 地址。")
    return f"{parsed.scheme}://{parsed.netloc}"


@dataclass(frozen=True)
class LoginOptions:
    prompt_bind_owner: bool = False
    verify_pin: bool = False
    acctweb_base_url: str = ACCTWEB_ORIGIN
    acctweb_proxy: str = ""


def load_login_options(**overrides) -> LoginOptions:
    """Read the normal platform extras; explicit CLI booleans override configuration."""
    from hermes_cli.config import load_config_readonly

    config = load_config_readonly()
    platforms = config.get("platforms", {})
    platform = platforms.get("weixin", {}) if isinstance(platforms, dict) else {}
    extra = platform.get("extra", {}) if isinstance(platform, dict) else {}
    extra = extra if isinstance(extra, dict) else {}
    values = {key: overrides.get(key) if overrides.get(key) is not None else extra.get(key) for key in LoginOptions.__dataclass_fields__}
    return LoginOptions(
        prompt_bind_owner=values["prompt_bind_owner"] is True,
        verify_pin=values["verify_pin"] is True,
        acctweb_base_url=_origin(str(values["acctweb_base_url"] or ACCTWEB_ORIGIN)),
        acctweb_proxy=_origin(str(values["acctweb_proxy"])) if values["acctweb_proxy"] else "",
    )


def local_tokens(hermes_home: str) -> list[str]:
    """Use the ten most recently saved account tokens; ignore auxiliary state files."""
    directory = Path(hermes_home) / "weixin" / "accounts"
    paths = sorted(directory.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    tokens: list[str] = []
    for path in paths:
        if path.name.endswith((".context-tokens.json", ".sync.json")):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        token = data.get("token") if isinstance(data, dict) else None
        if isinstance(token, str) and token.strip():
            tokens.append(token.strip())
        if len(tokens) == 10:
            break
    return tokens


def _payload(response: dict) -> dict:
    for key in ("ret", "errcode"):
        if response.get(key) not in (None, 0, "0"):
            raise LoginError("实名授权服务返回失败，连接已停止。")
    data = response.get("data", response)
    if not isinstance(data, dict):
        raise LoginError("实名授权服务返回无效数据，连接已停止。")
    return data


def _number(value) -> float:
    if isinstance(value, bool):
        raise LoginError("实名授权服务返回无效数据，连接已停止。")
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        value = float(value)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise LoginError("实名授权服务返回无效数据，连接已停止。")
    return value


@dataclass(frozen=True)
class Ticket:
    value: str
    auth_url: str
    expires_at: float

    @classmethod
    def parse(cls, response: dict) -> Ticket:
        data = _payload(response)
        value, auth_url = data.get("bot_spam_ticket"), data.get("auth_url")
        expires_at = _number(data.get("expire_time"))
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{32}", value):
            raise LoginError("无法获取绑定凭证，连接已停止。")
        if not isinstance(auth_url, str) or not auth_url.strip() or expires_at <= time.time():
            raise LoginError("无法获取绑定凭证，连接已停止。")
        return cls(value, auth_url, expires_at)


class LoginFlow:
    def __init__(self, session, *, options: LoginOptions, tokens: list[str], headers: Callable, render_qr: Callable,
                 bot_type: str = "3", timeout_seconds: int = 480):
        self.session = session
        self.options = options
        self.tokens = tokens
        self.headers = headers
        self.render_qr = render_qr
        self.bot_type = bot_type
        self.timeout_seconds = timeout_seconds
        self.deadline = time.monotonic() + timeout_seconds
        self.ticket: Ticket | None = None
        self.pending_code = ""
        self.poll_base = ILINK_BASE_URL
        self.scanned_printed = False

    @staticmethod
    def _url(origin: str, path: str, **params) -> str:
        return f"{origin}{path}?{urlencode(params)}"

    async def _request(self, method: str, url: str, *, headers: dict, timeout: int, **kwargs) -> dict:
        async def perform():
            async with self.session.request(method, url, headers=headers, **kwargs) as response:
                if not response.ok:
                    raise RequestError("微信绑定服务请求失败，请稍后重试。")
                payload = json.loads(await response.text())
                if not isinstance(payload, dict):
                    raise LoginError("微信绑定服务返回无效数据，连接已停止。")
                return payload

        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise LoginError("微信绑定超时，请重新开始连接。")
        try:
            return await asyncio.wait_for(perform(), timeout=min(timeout, remaining))
        except (aiohttp.ClientError, OSError, TimeoutError, ValueError):
            raise RequestError("微信绑定服务请求失败，请稍后重试。") from None

    async def _acctweb(self, step: str, **params) -> dict:
        kwargs = {"proxy": self.options.acctweb_proxy} if self.options.acctweb_proxy else {}
        return await self._request(
            "GET", self._url(self.options.acctweb_base_url, ACCTWEB_PATH, step=step, **params),
            headers={"Accept": "application/json"}, timeout=10, **kwargs,
        )

    async def _new_ticket(self) -> Ticket:
        ticket = Ticket.parse(await self._acctweb("get_auth_url", plugin_version=CHANNEL_VERSION))
        self.deadline = min(self.deadline, time.monotonic() + ticket.expires_at - time.time())
        print("\n请用你本人的手机微信扫描以下二维码，完成实名信息授权以连接此 Hermes 到他人的微信 ClawBot：")
        self.render_qr("", ticket.auth_url, report_render_error=True)
        return ticket

    async def _authorize(self) -> Ticket:
        ticket = await self._new_ticket()
        refreshed = False
        while time.monotonic() < self.deadline:
            try:
                result = await self._acctweb("query_auth_result", bot_spam_ticket=ticket.value)
            except RequestError:
                logger.warning("weixin: real-name authorization query failed; retrying")
                await asyncio.sleep(1)
                continue
            status = _number(_payload(result).get("auth_status"))
            if status == 1:
                return ticket
            if status == 2:
                raise LoginError("实名授权未通过，无法连接。")
            if status == 3:
                if refreshed:
                    raise LoginError("绑定凭证无效，请重新开始连接。")
                refreshed = True
                ticket = await self._new_ticket()
                continue
            if status != 0:
                raise LoginError("实名授权状态无效，连接已停止。")
            await asyncio.sleep(1)
        raise LoginError("实名授权超时，请重新开始连接。")

    def _other_owner(self) -> bool:
        if not self.options.prompt_bind_owner:
            return False
        while True:
            answer = input("连接此 Hermes 到你本人的微信 ClawBot？直接回车默认为 Y [Y/n] ").strip().lower()
            if answer in {"", "y", "n"}:
                return answer == "n"
            print("请输入 Y 或 N。")

    async def _new_qr(self) -> str:
        payload = {"local_token_list": self.tokens}
        if self.ticket:
            payload["bot_spam_ticket"] = self.ticket.value
        if self.options.verify_pin:
            payload["bot_spam_business_data"] = base64.b64encode(b'{"verify_pin":1}').decode("ascii")
        body = json.dumps(payload, separators=(",", ":"))
        headers = self.headers(None, body)
        response = await self._request(
            "POST", self._url(ILINK_BASE_URL, "/ilink/bot/get_bot_qrcode", bot_type=self.bot_type),
            headers=headers, timeout=35, data=body,
        )
        qrcode, url = response.get("qrcode"), response.get("qrcode_img_content")
        if not isinstance(qrcode, str) or not qrcode or not isinstance(url, str) or not url:
            raise LoginError("微信二维码返回不完整，连接已停止。")
        self.pending_code = ""
        self.poll_base = ILINK_BASE_URL
        self.scanned_printed = False
        print("\n让连接此 Hermes 的用户用手机微信扫描以下二维码：")
        self.render_qr(qrcode, url, report_render_error=True)
        return qrcode

    async def _status(self, qrcode: str) -> dict:
        params = {"qrcode": qrcode}
        if self.pending_code:
            params["verify_code"] = self.pending_code
        headers = {"iLink-App-Id": ILINK_APP_ID, "iLink-App-ClientVersion": str(ILINK_APP_CLIENT_VERSION)}
        return await self._request(
            "GET", self._url(self.poll_base, "/ilink/bot/get_qrcode_status", **params), headers=headers, timeout=35,
        )

    def _read_pin(self) -> None:
        prompt = "你输入的数字不匹配，请重新输入：" if self.pending_code else "输入手机微信显示的数字，以继续连接："
        while True:
            code = input(prompt).strip()
            if re.fullmatch(r"[0-9]+", code):
                self.pending_code = code
                return
            prompt = "请输入手机微信显示的纯数字 PIN："

    def _credentials(self, response: dict) -> dict[str, str]:
        credentials = {
            "account_id": response.get("ilink_bot_id"), "token": response.get("bot_token"),
            "base_url": response.get("baseurl") or ILINK_BASE_URL, "user_id": response.get("ilink_user_id") or "",
        }
        if not all(isinstance(value, str) for value in credentials.values()) or not credentials["account_id"] or not credentials["token"]:
            raise LoginError("微信确认成功但返回的账号凭据不完整。")
        return credentials

    async def _poll(self, qrcode: str) -> dict[str, str] | None:
        qr_count = 1
        while time.monotonic() < self.deadline:
            try:
                response = await self._status(qrcode)
            except RequestError:
                logger.warning("weixin: QR status request failed; retrying")
                await asyncio.sleep(1)
                continue
            status = response.get("status")
            if not isinstance(status, str):
                raise LoginError("微信二维码状态无效，连接已停止。")
            if "need_verifycode" in status:
                self._read_pin()
                continue
            if status == "confirmed":
                return self._credentials(response)
            if status == "binded_redirect":
                print("\n已连接过此 Hermes，无需重复连接。")
                return None
            if status in {"expired", "verify_code_blocked"}:
                qr_count += 1
                if qr_count > 3:
                    raise LoginError("二维码多次失效或 PIN 多次错误，连接已停止，请稍后再试。")
                if self.ticket and self.ticket.expires_at - time.time() <= 60:
                    raise LoginError("绑定凭证即将过期，请重新开始连接。")
                print(f"\n二维码已过期或 PIN 多次错误，正在刷新... ({qr_count}/3)")
                qrcode = await self._new_qr()
            else:
                self._progress(status, response)
            await asyncio.sleep(1)
        raise LoginError("微信登录超时，请重新开始连接。")

    def _progress(self, status: str, response: dict) -> None:
        if status == "scaned":
            self.pending_code = ""
            if not self.scanned_printed:
                print("\n已扫码，正在验证，请在手机微信确认。")
                self.scanned_printed = True
        elif status == "scaned_but_redirect" and response.get("redirect_host"):
            host = response["redirect_host"]
            if not isinstance(host, str) or not re.fullmatch(r"[a-zA-Z0-9.-]+(?::[0-9]+)?", host):
                raise LoginError("微信二维码跳转地址无效，连接已停止。")
            self.poll_base = _origin(f"https://{host}")

    async def run(self) -> dict[str, str] | None:
        try:
            other_owner = self._other_owner()
            self.deadline = time.monotonic() + self.timeout_seconds
            if other_owner:
                self.ticket = await self._authorize()
            self.deadline = time.monotonic() + self.timeout_seconds
            return await self._poll(await self._new_qr())
        except LoginError as exc:
            logger.error("weixin: %s", exc)
            print(f"\n{exc}")
            return None
        except (EOFError, KeyboardInterrupt):
            print("\n微信连接已取消。")
            return None
