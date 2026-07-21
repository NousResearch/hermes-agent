"""Telegram launch verification and opaque Mini App sessions."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

SESSION_COOKIE = "__Host-hermes-mini"
SESSION_TTL = 30 * 60
SESSION_IDLE_TTL = 5 * 60
MAX_INIT_DATA_BYTES = 16 * 1024
PRIVATE_SESSION_RATE_LIMIT = 120


@dataclass
class MiniAppAuth:
    """In-memory, fail-closed authentication boundary for one Mini App process."""

    bot_token: str
    allowed_users_raw: str
    public_url: str
    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    init_uses: dict[str, float] = field(default_factory=dict)
    exchange_events: dict[str, list[float]] = field(default_factory=dict)

    @property
    def allowed_users(self) -> set[str]:
        return {
            value
            for value in re.split(r"[,\s]+", self.allowed_users_raw)
            if value.isdigit()
        }

    @property
    def public_origin(self) -> str:
        parsed = urllib.parse.urlsplit(self.public_url.strip())
        if parsed.scheme != "https" or not parsed.netloc:
            return ""
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            return ""
        if parsed.path not in {"", "/"}:
            return ""
        return f"{parsed.scheme}://{parsed.netloc}"

    @staticmethod
    def _key(raw: str) -> str:
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _user_agent_hash(user_agent: str) -> str:
        return hashlib.sha256(user_agent.encode()).hexdigest()

    def prune(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        for key, record in list(self.sessions.items()):
            if (
                now >= record["expires"]
                or now - record["last_seen"] >= SESSION_IDLE_TTL
            ):
                self.sessions.pop(key, None)
        for fingerprint, expires in list(self.init_uses.items()):
            if now >= expires:
                self.init_uses.pop(fingerprint, None)
        for uid, events in list(self.exchange_events.items()):
            fresh = [stamp for stamp in events if now - stamp < 60]
            if fresh:
                self.exchange_events[uid] = fresh
            else:
                self.exchange_events.pop(uid, None)

    def verify_init_data(self, init_data: str, *, max_age: int = 300) -> dict[str, Any]:
        if not self.bot_token:
            raise HTTPException(500, "TELEGRAM_BOT_TOKEN is missing")
        if not init_data:
            raise HTTPException(401, "Missing Telegram initData")

        pairs = urllib.parse.parse_qsl(init_data, keep_blank_values=True)
        keys = [key for key, _ in pairs]
        if len(keys) != len(set(keys)):
            raise HTTPException(401, "Duplicate initData keys")
        parsed = dict(pairs)
        received_hash = parsed.pop("hash", "")
        if not received_hash:
            raise HTTPException(401, "Missing initData hash")

        data_check = "\n".join(
            f"{key}={value}" for key, value in sorted(parsed.items())
        )
        secret = hmac.new(
            b"WebAppData", self.bot_token.encode(), hashlib.sha256
        ).digest()
        calculated = hmac.new(secret, data_check.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(calculated, received_hash):
            raise HTTPException(401, "Invalid Telegram initData")

        try:
            auth_date = int(parsed.get("auth_date", "0") or 0)
        except ValueError as exc:
            raise HTTPException(401, "Invalid initData auth_date") from exc
        now = time.time()
        if not auth_date:
            raise HTTPException(401, "Missing initData auth_date")
        if now - auth_date > max_age:
            raise HTTPException(401, "Expired Telegram initData")
        if auth_date - now > 300:
            raise HTTPException(401, "Future Telegram initData")
        if not self.allowed_users:
            raise HTTPException(
                503, "Hermes Mini App is not configured with allowed users"
            )

        try:
            user = json.loads(parsed.get("user", "{}") or "{}")
        except json.JSONDecodeError as exc:
            raise HTTPException(401, "Invalid initData user") from exc
        uid = str(user.get("id", "")) if isinstance(user, dict) else ""
        if not uid or uid not in self.allowed_users:
            raise HTTPException(403, "Not allowed")
        return {"user": user, "query": parsed}

    def record(
        self, raw: str, user_agent: str, *, touch: bool = True
    ) -> dict[str, Any] | None:
        if not raw:
            return None
        now = time.time()
        self.prune(now)
        record = self.sessions.get(self._key(raw))
        if not record or record.get("uid") not in self.allowed_users:
            return None
        if not hmac.compare_digest(
            record.get("ua_hash", ""), self._user_agent_hash(user_agent)
        ):
            return None
        if touch:
            record["last_seen"] = now
        return record

    def exchange(self, request: Request) -> JSONResponse:
        init_data = request.headers.get("x-telegram-init-data", "")
        auth = self.verify_init_data(init_data, max_age=300)
        uid = str(auth["user"]["id"])
        now = time.time()
        self.prune(now)

        fingerprint = self._key(init_data)
        if fingerprint in self.init_uses:
            raise HTTPException(
                401,
                "Telegram launch data was already used; reopen Hermes from the bot menu",
            )
        events = self.exchange_events.setdefault(uid, [])
        if len(events) >= 6:
            raise HTTPException(429, "Too many Mini App session requests")
        events.append(now)

        for key, record in list(self.sessions.items()):
            if record.get("uid") == uid:
                self.sessions.pop(key, None)

        raw = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(32)
        self.sessions[self._key(raw)] = {
            "uid": uid,
            "user": auth["user"],
            "csrf_token": csrf_token,
            "ua_hash": self._user_agent_hash(request.headers.get("user-agent", "")),
            "created": now,
            "last_seen": now,
            "expires": now + SESSION_TTL,
            "request_events": [],
        }
        self.init_uses[fingerprint] = now + 300
        response = JSONResponse({
            "ok": True,
            "csrf_token": csrf_token,
            "expires_in": SESSION_TTL,
        })
        response.set_cookie(
            SESSION_COOKIE,
            raw,
            max_age=SESSION_TTL,
            secure=True,
            httponly=True,
            samesite="strict",
            path="/",
        )
        return response

    def logout(self, request: Request) -> JSONResponse:
        raw = request.cookies.get(SESSION_COOKIE, "")
        record = self.record(raw, request.headers.get("user-agent", ""), touch=False)
        if not record:
            raise HTTPException(401, "An authenticated Mini App session is required")
        csrf_token = request.headers.get("x-csrf-token", "")
        if not csrf_token or not hmac.compare_digest(csrf_token, record["csrf_token"]):
            raise HTTPException(403, "Invalid CSRF token")
        if not self.public_origin:
            raise HTTPException(503, "Hermes Mini App public URL is not configured")
        if request.headers.get("origin", "") != self.public_origin:
            raise HTTPException(403, "Cross-origin request rejected")
        if request.headers.get("sec-fetch-site", "") != "same-origin":
            raise HTTPException(403, "Cross-site request rejected")

        self.sessions.pop(self._key(raw), None)
        response = JSONResponse({"ok": True})
        response.delete_cookie(
            SESSION_COOKIE,
            path="/",
            secure=True,
            httponly=True,
            samesite="strict",
        )
        return response
