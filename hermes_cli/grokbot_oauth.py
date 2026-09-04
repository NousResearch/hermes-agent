"""Grok Bot PKCE for `hermes auth add grokbot`.

Cursor sand loginDeepControl poller. Tokens stay in ~/.grokbot/session.json
(mode 0600). The credential pool stores a listable copy so `hermes auth list`
shows grokbot as oauth, not a dummy yaml api_key.
"""

from __future__ import annotations

from typing import Any

from agent.grokbot import login as grokbot_login


def existing_session() -> dict[str, Any] | None:
    try:
        sess = grokbot_login._load()
    except Exception:
        return None
    if not isinstance(sess, dict):
        return None
    if not sess.get("accessToken") or not sess.get("refreshToken"):
        return None
    return sess


def run_pkce_login(*, open_browser: bool = True, timeout: float = 300.0) -> dict[str, Any]:
    return grokbot_login.login_session(open_browser=open_browser, timeout=timeout)


def session_label(sess: dict[str, Any]) -> str:
    auth_id = str(sess.get("authId") or "")
    email = str(sess.get("email") or "")
    if email:
        return email
    if auth_id:
        return auth_id.split("|", 1)[0] + "|…"
    return "grokbot-pkce"
