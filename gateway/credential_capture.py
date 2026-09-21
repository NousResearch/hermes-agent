"""Deterministic, model-before-boundary capture of explicitly authorized logins."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Optional

from agent.vault_store import scrub_secret_from_text

_INTENTS = (
    "保存账号密码",
    "请保存账号密码",
    "授权保存账号密码",
    "保存登录凭据",
    "请保存登录凭据",
    "保存网站凭据",
    "请保存网站凭据",
    "/credential save",
    "save login credential",
    "save website credential",
)
_FIELD = re.compile(r"^\s*([^:：]{1,24})\s*[:：]\s*(.*?)\s*$")
_ALIASES = {
    "origin": {"网站", "网址", "站点", "url", "origin", "site", "website"},
    "identifier": {"账号", "帐号", "用户名", "用户", "邮箱", "手机号", "identifier", "username", "email", "phone", "account"},
    "password": {"密码", "口令", "password", "passcode"},
    "instruction": {"继续", "后续", "指令", "然后", "continue", "instruction", "next"},
}


@dataclass
class CredentialCapture:
    origin: str = ""
    identifier: str = ""
    identifier_type: str = "username"
    password: str = ""
    instruction: str = ""
    error: str = ""

    def clear(self) -> None:
        self.identifier = ""
        self.password = ""


@dataclass(frozen=True)
class CredentialCaptureResult:
    handled: bool
    reply: Optional[str] = None


def _canonical_field(name: str) -> Optional[str]:
    lowered = name.strip().lower()
    return next((key for key, aliases in _ALIASES.items() if lowered in aliases), None)


def _identifier_type(field_name: str, identifier: str) -> str:
    lowered = field_name.strip().lower()
    if lowered in {"邮箱", "email"} or "@" in identifier:
        return "email"
    if lowered in {"手机号", "phone"} or identifier.lstrip("+").isdigit():
        return "phone"
    return "username"


def _has_explicit_intent(text: str) -> bool:
    first = next((line.strip().lower() for line in text.splitlines() if line.strip()), "")
    return any(first == intent or first.startswith(intent + " ") for intent in _INTENTS)


def _scrub_credentials(text: str, capture: CredentialCapture) -> str:
    scrubbed = str(text or "")
    for value in sorted({capture.identifier, capture.password} - {""}, key=len, reverse=True):
        scrubbed = scrubbed.replace(value, "[REDACTED]")
    return scrub_secret_from_text(
        scrubbed, {"identifier": capture.identifier, "password": capture.password}
    )


def _scrub_payload(value, capture: CredentialCapture, memo=None):
    """Copy a normalized platform payload while removing the captured credential values."""
    if isinstance(value, str):
        return _scrub_credentials(value, capture)
    if isinstance(value, bytes):
        scrubbed = value
        for secret in (capture.identifier, capture.password):
            if secret:
                scrubbed = scrubbed.replace(secret.encode("utf-8"), b"[REDACTED]")
        return scrubbed
    if value is None or isinstance(value, (bool, int, float)):
        return value
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, dict):
        scrubbed = {}
        memo[id(value)] = scrubbed
        for key, item in value.items():
            scrubbed[_scrub_payload(key, capture, memo)] = _scrub_payload(item, capture, memo)
        return scrubbed
    if isinstance(value, list):
        scrubbed = []
        memo[id(value)] = scrubbed
        scrubbed.extend(_scrub_payload(item, capture, memo) for item in value)
        return scrubbed
    if isinstance(value, tuple):
        items = tuple(_scrub_payload(item, capture, memo) for item in value)
        return type(value)(*items) if hasattr(value, "_fields") else items
    if isinstance(value, (set, frozenset)):
        return type(value)(_scrub_payload(item, capture, memo) for item in value)
    try:
        scrubbed = copy.copy(value)
        attributes = vars(value)
    except (TypeError, copy.Error):
        return None
    memo[id(value)] = scrubbed
    for name, item in attributes.items():
        try:
            setattr(scrubbed, name, _scrub_payload(item, capture, memo))
        except (AttributeError, TypeError):
            return None
    return scrubbed


def prepare_credential_capture(event) -> bool:
    """Replace an explicit save message before hooks, logs, persistence, or the model can see it."""
    if getattr(event, "_credential_capture", None) is not None or getattr(event, "internal", False):
        return getattr(event, "_credential_capture", None) is not None
    text = str(getattr(event, "text", "") or "")
    if not _has_explicit_intent(text):
        return False

    capture = CredentialCapture()
    first_field_name = ""
    for index, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if not stripped or (index == 0 and any(stripped.lower().startswith(intent) for intent in _INTENTS)):
            continue
        match = _FIELD.match(line)
        if not match:
            continue
        raw_name, value = match.groups()
        field = _canonical_field(raw_name)
        if field is None:
            continue
        setattr(capture, field, value.strip())
        if field == "identifier":
            first_field_name = raw_name
    capture.identifier_type = _identifier_type(first_field_name, capture.identifier)
    missing = [name for name, value in (("网站", capture.origin), ("账号", capture.identifier), ("密码", capture.password)) if not value]
    if missing:
        capture.error = f"缺少字段：{'、'.join(missing)}"

    event._credential_capture = capture
    event.text = "[凭据保存请求已在模型前拦截，等待写入凭据库]"
    event.raw_message = _scrub_payload(getattr(event, "raw_message", None), capture)
    event.metadata = _scrub_payload(getattr(event, "metadata", {}), capture)
    for field in ("reply_to_text", "channel_context", "channel_prompt"):
        value = getattr(event, field, None)
        if isinstance(value, str):
            setattr(event, field, _scrub_credentials(value, capture))
    if capture.instruction:
        event.text += f"\n继续执行：{_scrub_credentials(capture.instruction, capture)}"
    return True


def discard_credential_capture(event) -> None:
    capture = getattr(event, "_credential_capture", None)
    if capture is not None:
        capture.clear()
    event._credential_capture = None


def save_authorized_credential_capture(event, broker=None) -> CredentialCaptureResult:
    """Persist a previously sanitized capture. Call only after gateway authorization succeeds."""
    capture = getattr(event, "_credential_capture", None)
    if capture is None:
        return CredentialCaptureResult(False)
    if capture.error:
        reply = (
            f"凭据没有保存：{capture.error}。请使用：\n"
            "保存账号密码\n网站: https://example.com\n账号: your-account\n密码: your-password\n"
            "继续: 可选的后续任务"
        )
        discard_credential_capture(event)
        return CredentialCaptureResult(True, reply)

    if broker is None:
        from agent.credential_broker import get_credential_broker
        broker = get_credential_broker()
    try:
        saved = broker.save_login(
            label=capture.origin,
            origin=capture.origin,
            identifier_type=capture.identifier_type,
            identifier=capture.identifier,
            password=capture.password,
        )
        origin = saved.meta.origin or capture.origin
        instruction = _scrub_credentials(capture.instruction, capture)
        event.text = f"[登录凭据已保存到 {saved.meta.id}，绑定 {origin}]"
        if instruction:
            event.text += f"\n继续执行：{instruction}"
            return CredentialCaptureResult(True)
        action = "更新" if saved.action == "updated" else "保存"
        return CredentialCaptureResult(True, f"✅ 登录凭据已{action}到 {saved.meta.id}，绑定 {origin}。")
    except Exception as exc:
        error = _scrub_credentials(str(exc), capture)[:240]
        return CredentialCaptureResult(True, f"凭据保存失败：{error}")
    finally:
        discard_credential_capture(event)
