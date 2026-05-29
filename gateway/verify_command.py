import os
import re
from typing import Any, Optional, TypedDict

import aiohttp


class VerifyCommand(TypedDict):
    code: str


_VERIFY_RE = re.compile(r"^\s*/?verify\s+(\d{6})\s*$", re.IGNORECASE)


def parse_verify_command(text: str) -> Optional[VerifyCommand]:
    m = _VERIFY_RE.match(text or "")
    return {"code": m.group(1)} if m else None


async def redeem_verify_code(*, platform: str, code: str, user_id: str) -> dict[str, Any]:
    base = os.getenv("HERMES_SAAS_URL", "").rstrip("/")
    token = os.getenv("HERMES_INTERNAL_TOKEN", "")
    if not base or not token:
        return {"ok": False, "reason": "network_error"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base}/api/integrations/pairing-code/redeem",
                json={"platform": platform, "code": code, "user_id": user_id},
                headers={"X-Hermes-Token": token},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return {"ok": False, "reason": "network_error"}
                return await resp.json()
    except Exception:
        return {"ok": False, "reason": "network_error"}


def verify_ack_text(result: dict[str, Any]) -> str:
    if result.get("ok") is True:
        return "✅ 已將你加為 owner，現在可以開始對話"
    reason = result.get("reason")
    return {
        "already_owner": "ℹ️ 你已經是 owner",
        "rate_limited": "❌ 嘗試太多次，請稍後再試",
        "invalid_or_expired": "❌ 驗證碼無效或已過期，請聯絡管理員重新產生",
    }.get(reason, "❌ 驗證失敗，請稍後再試")
