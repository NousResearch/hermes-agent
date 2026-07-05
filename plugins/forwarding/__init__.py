"""forwarding plugin — declarative cross-channel message relay.

Rules live in ``$HERMES_HOME/forwarding-rules.json``::

    {"rules": [
        {"from": {"platform": "wecom",    "chat": "<source jid>"},
         "to":   {"platform": "whatsapp", "chat": "<target jid>"},
         "forward_only": false, "media": true, "prefix": ""}
    ]}

On every inbound user message (``pre_gateway_dispatch``), each matching rule
relays the text — and the first image when ``media: true`` — to the target
chat via the target platform's connected adapter. ``forward_only: true`` also
drops the message from the agent (pure relay, no reply). Relay sends are
scheduled on the running loop and best-effort: a failure is logged and never
blocks the originating message.

The rules file is hot-reloaded on mtime change, so edits take effect without a
gateway restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_RULES_CACHE: Optional[Dict[str, Any]] = None
_RULES_MTIME: float = -1.0


def _rules_path() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home) / "forwarding-rules.json"


def _load_rules() -> List[Dict[str, Any]]:
    """Return the rule list, hot-reloading the file on mtime change."""
    global _RULES_CACHE, _RULES_MTIME
    path = _rules_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _RULES_CACHE, _RULES_MTIME = {"rules": []}, -1.0
        return []
    if _RULES_CACHE is None or mtime != _RULES_MTIME:
        try:
            _RULES_CACHE = json.loads(path.read_text(encoding="utf-8"))
            _RULES_MTIME = mtime
        except Exception as exc:  # noqa: BLE001
            logger.warning("forwarding: could not parse %s: %s", path, exc)
            _RULES_CACHE = {"rules": []}
    rules = (_RULES_CACHE or {}).get("rules") or []
    return [r for r in rules if isinstance(r, dict)]


def _platform_value(platform: Any) -> str:
    return str(getattr(platform, "value", platform) or "").strip().lower()


def _rule_matches(rule: Dict[str, Any], src_platform: str, src_chat: str) -> bool:
    frm = rule.get("from") or {}
    return (
        str(frm.get("platform", "")).strip().lower() == src_platform
        and str(frm.get("chat", "")).strip() == src_chat
    )


async def _relay(gateway: Any, rule: Dict[str, Any], text: str, media_urls: List[str]) -> None:
    """Send one relayed message to the rule's target chat (best-effort)."""
    from gateway.config import Platform

    to = rule.get("to") or {}
    try:
        target_platform = Platform(str(to.get("platform", "")).strip().lower())
    except ValueError:
        logger.warning("forwarding: unknown target platform %r", to.get("platform"))
        return
    adapter = (getattr(gateway, "adapters", {}) or {}).get(target_platform)
    if adapter is None:
        logger.warning("forwarding: target platform %s not connected", target_platform)
        return
    target_chat = str(to.get("chat", "")).strip()
    if not target_chat:
        return

    prefix = str(rule.get("prefix") or "")
    body = f"{prefix}{text}" if text else prefix
    try:
        if body.strip():
            await adapter.send(target_chat, body)
        if rule.get("media") and media_urls:
            # Relay the first image best-effort; other media types are a
            # follow-up (the common recruitment/announce case is text+image).
            try:
                await adapter.send_image(target_chat, media_urls[0], caption=None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("forwarding: media relay failed: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "forwarding: relay to %s/%s failed: %s", target_platform, target_chat, exc
        )


def _on_inbound(event: Any = None, gateway: Any = None, **_: Any) -> Optional[Dict[str, Any]]:
    """pre_gateway_dispatch hook: relay matching inbound messages."""
    if event is None or gateway is None:
        return None
    source = getattr(event, "source", None)
    src_platform = _platform_value(getattr(source, "platform", None))
    src_chat = str(getattr(source, "chat_id", "") or "").strip()
    if not src_platform or not src_chat:
        return None

    matched = [r for r in _load_rules() if _rule_matches(r, src_platform, src_chat)]
    if not matched:
        return None

    text = getattr(event, "text", "") or ""
    media_urls = [u for u in (getattr(event, "media_urls", None) or []) if isinstance(u, str)]
    for rule in matched:
        try:
            asyncio.get_running_loop().create_task(_relay(gateway, rule, text, media_urls))
        except RuntimeError:
            logger.warning("forwarding: no running loop; relay skipped")

    if any(r.get("forward_only") for r in matched):
        return {"action": "skip", "reason": "forwarded"}
    return None


def register(ctx: Any) -> None:
    ctx.register_hook("pre_gateway_dispatch", _on_inbound)
