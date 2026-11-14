"""Gateway intake guardrail: automatic hold for block-tier messages
(Phase 2, owner decision 14 Nov 2026).

Every inbound non-command message is screened through the local RNR
Judgement Engine (``/v1/screen``) BEFORE a session turn starts:

- ``allow``      -> proceeds (outcome ``accepted`` reported)
- ``review``     -> proceeds, logged for the owner dashboard
                    (outcome ``escalated``)
- ``block``      -> HELD: no agent turn runs. Optional hold reply to
                    the sender, one owner alert per sender per
                    debounce window (outcome ``escalated``)
- ``chain_down`` -> proceeds (the engine already applied its
                    fail_mode rule; hard-three held inside the
                    engine), alert informs the owner (outcome
                    ``escalated``)
- unreachable    -> proceeds + alert. We cannot hold what we cannot
                    judge, and the bridge allowlist still gates WHO
                    may talk at all; a dead engine must not mute the
                    household (documented, deliberate fail-open).

The judge advises, it never replaces owner gates: holds are audited
(alerts + logs + outcomes), and only review-tier, chain-down holds
and escalations reach the owner.

Configuration is environment-driven (the gateway's existing override
pattern, cf. GATEWAY_MULTIPLEX_PROFILES / RNR_JUDGE_SITE) so the
core change is inert until the operator opts in:

- HERMES_GATEWAY_GUARDRAIL            "" (off) | "1"/"true" (on)
- HERMES_GATEWAY_GUARDRAIL_ENDPOINT   default http://127.0.0.1:8201/v1/screen
- HERMES_GATEWAY_GUARDRAIL_TIMEOUT_S  default 4.0
- HERMES_GATEWAY_GUARDRAIL_CALLER     default hermes-gateway
- HERMES_GATEWAY_GUARDRAIL_HOLD_REPLY default "" (silent hold)
- HERMES_GATEWAY_GUARDRAIL_OWNER_CHAT default "" (no alert)
- HERMES_GATEWAY_GUARDRAIL_DEBOUNCE_S default 600

All HTTP work is off-loop (asyncio.to_thread) with a short timeout
so a slow engine never delays intake. The screen text is truncated
to 4000 chars. Errors never raise into the caller: worst case the
message proceeds and the miss is logged + alerted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger("hermes.gateway.guardrail")

_LOG_PATH = Path(
    os.environ.get("HERMES_GATEWAY_GUARDRAIL_LOG",
                   os.path.expanduser("~/.hermes/logs/gateway-guardrail.log")))
_STATE_PATH = Path(os.path.expanduser(
    "~/.hermes/logs/gateway-guardrail-alerts.json"))
MAX_TEXT_CHARS = 4000
_ALERT_TEXT_CAP = 160


def _flag(name: str) -> str:
    return os.environ.get(name, "").strip()


def guardrail_enabled() -> bool:
    return _flag("HERMES_GATEWAY_GUARDRAIL").lower() in ("1", "true", "yes", "on")


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    headers = dict(headers, **{"Content-Type": "application/json"})
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _log(record: dict) -> None:
    record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    except OSError:
        logger.warning("guardrail: log write failed", exc_info=True)


def _debounced(sender: str, window_s: float) -> bool:
    now = time.time()
    state: dict = {}
    try:
        state = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            state = {}
    except (OSError, ValueError):
        pass
    if now - float(state.get(sender, 0)) < window_s:
        return False
    state[sender] = now
    state = dict(sorted(state.items())[-200:])
    try:
        _STATE_PATH.write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        pass
    return True


async def _send(adapter, chat_id: str, text: str) -> None:
    try:
        await adapter.send(chat_id, text)
    except Exception:
        logger.warning("guardrail: send failed", exc_info=True)


def _report_outcome(result: dict, outcome: str, note: str,
                    question_id: str | None) -> None:
    request_id = result.get("request_id")
    if not request_id:
        return
    endpoint = _flag("HERMES_GATEWAY_GUARDRAIL_ENDPOINT")
    outcome_url = endpoint.rsplit("/", 1)[0] + "/outcome"
    try:
        _post(outcome_url, {
            "request_id": request_id,
            "question_id": question_id,
            "outcome": outcome,
            "note": note[:200],
        }, {}, 3.0)
    except Exception:
        logger.debug("guardrail: outcome report failed", exc_info=True)


async def screen(adapter, event) -> bool:
    """Screen one inbound event; True = HELD (caller must not run a
    turn for it). Never raises."""
    if not guardrail_enabled():
        return False
    # Gateway control traffic and slash commands are trusted syntax;
    # screening them could wedge remote admin of a misbehaving bot.
    if getattr(event, "allow_gateway_control", False) or event.get_command():
        return False
    text = (getattr(event, "text", "") or "").strip()
    if not text:
        return False
    sender = str(getattr(event, "user_id", None)
                 or getattr(event.source, "user_id", None) or "unknown")
    platform = str(getattr(event.source, "platform", "?"))
    chat_id = str(getattr(event.source, "chat_id", "") or "")
    endpoint = _flag("HERMES_GATEWAY_GUARDRAIL_ENDPOINT") or \
        "http://127.0.0.1:8201/v1/screen"
    timeout = float(_flag("HERMES_GATEWAY_GUARDRAIL_TIMEOUT_S") or 4.0)
    caller = _flag("HERMES_GATEWAY_GUARDRAIL_CALLER") or "hermes-gateway"
    base = {"sender": sender, "platform": platform, "preview": text[:120]}

    def call() -> dict:
        return _post(endpoint, {"text": text[:MAX_TEXT_CHARS]},
                     {"X-Judge-Caller": caller}, timeout)

    try:
        result = await asyncio.to_thread(call)
    except Exception as err:
        _log(dict(base, tier="engine_unreachable", error=str(err)[:160]))
        await _maybe_alert(adapter, sender, platform,
                           "screen unavailable (engine unreachable) - "
                           "message allowed")
        return False

    tier = result.get("tier")
    reasons = result.get("reasons") or []
    _log(dict(base, tier=tier, reasons=reasons,
              served_by=(result.get("served_by") or {}).get("provider"),
              scores=result.get("scores")))
    question_id = ("q_" + reasons[0]) if reasons else None
    if tier == "block":
        await _maybe_alert(adapter, sender, platform,
                           "%s-tier (%s) - held automatically"
                           % (tier, ", ".join(reasons)[:_ALERT_TEXT_CAP]))
        reply = _flag("HERMES_GATEWAY_GUARDRAIL_HOLD_REPLY")
        if reply and chat_id:
            await _send(adapter, chat_id, reply)
        _report_outcome(result, "escalated",
                        "block-tier held automatically (Phase 2)", question_id)
        return True
    if tier in ("review", "chain_down"):
        if tier == "chain_down":
            await _maybe_alert(adapter, sender, platform,
                               "chain-down - engine held or failed per its "
                               "own rule; message allowed through Phase 2")
        _report_outcome(result, "escalated",
                        "%s-tier (Phase 2 advisory pass)" % tier, question_id)
        return False
    _report_outcome(result, "accepted", "allow-tier (Phase 2)", question_id)
    return False


async def _maybe_alert(adapter, sender: str, platform: str,
                       detail: str) -> None:
    owner_chat = _flag("HERMES_GATEWAY_GUARDRAIL_OWNER_CHAT")
    if not owner_chat:
        return
    window = float(_flag("HERMES_GATEWAY_GUARDRAIL_DEBOUNCE_S") or 600)
    if not _debounced(sender, window):
        return
    await _send(adapter, owner_chat,
                "[guardrail] %s on %s: %s." % (sender, platform, detail))
