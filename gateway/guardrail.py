"""Gateway intake guardrail: automatic hold for block-tier messages
(Phase 2, owner decision 14 Nov 2026; review-fix revision r2).

Every inbound message that is NOT a known gateway command is screened
through the local RNR Judgement Engine (``/v1/screen``) BEFORE a
session turn starts:

- ``allow``      -> proceeds (outcome ``accepted`` reported)
- ``review``     -> proceeds, logged for the owner dashboard
                    (outcome ``escalated``)
- ``block``      -> HELD: no agent turn runs. Optional hold reply to
                    the sender, one owner alert per sender per
                    debounce window (outcome ``escalated``)
- ``chain_down`` -> proceeds (the engine already applied its
                    fail_mode rule; hard-three held inside the
                    engine), alert informs the owner
- unreachable / malformed response -> proceeds + alert. We cannot
  hold what we cannot judge, and the bridge allowlist still gates
  WHO may talk at all; a dead or lying engine must not mute the
  household (documented, deliberate fail-open).

Command handling (review B2): ONLY known gateway commands skip the
gate (remote admin must never be wedged by a judge). Unknown
command-shaped text ('/zzz payload') is admitted downstream as plain
text, so it IS screened like any other message. The
``allow_gateway_control`` field is intentionally NOT consulted: it
defaults True on real traffic (it means "may resolve commands", not
"this is a control event"), so gating on it would skip every real
message (review B1).

No-raise guarantee (review B3): the public entry point wraps every
step; any error degrades to "proceed" with a log line. All engine
response parsing is type-defensive; all numeric env values fall back
to defaults; corrupted debounce state is tolerated.

Event-loop safety (review B4): ALL HTTP work - screen and outcome -
runs off the event loop; the screen call additionally has an overall
deadline (per-socket timeout + 1s) so a drip-feeding peer cannot
stretch it, and outcome reports are fire-and-forget tasks that can
never block or delay an intake decision.

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
- HERMES_GATEWAY_GUARDRAIL_LOG        default ~/.hermes/logs/gateway-guardrail.log

Known limitation (review I4, informational): self-initiated traffic
(routing wake ticks, synthetic self-messages) is not universally
identifiable at this layer; a declined synthetic tick could be held.
Observed as a rare self-notice nuisance, not a security issue.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger("hermes.gateway.guardrail")

def _log_path() -> Path:
    # Resolved per call, not at import: the log override must honour env set
    # (or patched by tests) after this module is imported; production env is
    # fixed at process start, so behaviour there is unchanged.
    return Path(os.environ.get(
        "HERMES_GATEWAY_GUARDRAIL_LOG",
        os.path.expanduser("~/.hermes/logs/gateway-guardrail.log")))

_STATE_PATH = Path(os.path.expanduser(
    "~/.hermes/logs/gateway-guardrail-alerts.json"))
MAX_TEXT_CHARS = 4000
_ALERT_TEXT_CAP = 160
_DEFAULT_ENDPOINT = "http://127.0.0.1:8201/v1/screen"

# Fire-and-forget outcome tasks (kept referenced so CPython cannot GC
# them mid-run; discarded on completion).
_OUTCOME_TASKS: set = set()


def _flag(name: str) -> str:
    return os.environ.get(name, "").strip()


def guardrail_enabled() -> bool:
    return _flag("HERMES_GATEWAY_GUARDRAIL").lower() in ("1", "true", "yes", "on")


def _float_env(name: str, default: float) -> float:
    try:
        value = float(_flag(name))
        if not math.isfinite(value) or value <= 0:
            return default
        return value
    except (TypeError, ValueError):
        return default


def _effective_endpoint() -> str:
    return _flag("HERMES_GATEWAY_GUARDRAIL_ENDPOINT") or _DEFAULT_ENDPOINT


def _outcome_endpoint() -> str:
    endpoint = _effective_endpoint()
    if "/v1/screen" in endpoint:
        return endpoint.replace("/v1/screen", "/v1/outcome")
    return endpoint.rsplit("/", 1)[0] + "/outcome"


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    headers = dict(headers, **{"Content-Type": "application/json"})
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _log(record: dict) -> None:
    record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    except OSError:
        logger.warning("guardrail: log write failed", exc_info=True)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _debounced(alert_key: str, window_s: float) -> bool:
    now = time.time()
    state: dict = {}
    try:
        loaded = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            state = loaded
    except (OSError, ValueError):
        pass
    if now - _safe_float(state.get(alert_key), 0.0) < window_s:
        return False
    state[alert_key] = now
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


def _report_outcome_sync(result: dict, outcome: str, note: str,
                         question_id) -> None:
    request_id = result.get("request_id")
    if not request_id:
        return
    _post(_outcome_endpoint(), {
        "request_id": request_id,
        "question_id": question_id,
        "outcome": outcome,
        "note": note[:200],
    }, {}, 3.0)


def _spawn_outcome_report(result: dict, outcome: str, note: str,
                          question_id) -> None:
    """Fire-and-forget, off-loop, deadline-capped outcome report.

    Never awaited by the intake path (review B4): a slow or dead
    outcome endpoint cannot delay or block a gate decision.
    """
    try:
        task = asyncio.get_running_loop().create_task(asyncio.wait_for(
            asyncio.to_thread(_report_outcome_sync, result, outcome,
                              note, question_id),
            timeout=4.0))
    except RuntimeError:
        return
    _OUTCOME_TASKS.add(task)

    def _done(finished):
        _OUTCOME_TASKS.discard(finished)
        if not finished.cancelled() and finished.exception() is not None:
            logger.debug("guardrail: outcome report failed",
                         exc_info=finished.exception())

    task.add_done_callback(_done)


async def drain_outcome_tasks(timeout: float = 5.0) -> None:
    """Best-effort wait for pending outcome reports (tests, clean
    shutdown). Never raises: a stuck report is logged by the task
    callback and forgotten."""
    if _OUTCOME_TASKS:
        try:
            await asyncio.wait_for(
                asyncio.gather(*list(_OUTCOME_TASKS),
                               return_exceptions=True),
                timeout=timeout)
        except TimeoutError:
            for stuck in list(_OUTCOME_TASKS):
                stuck.cancel()


def _known_gateway_command(event) -> bool:
    try:
        command = event.get_command()
    except Exception:
        return False
    if not command:
        return False
    try:
        from hermes_cli.commands import is_gateway_known_command
        return bool(is_gateway_known_command(command))
    except Exception:
        logger.warning("guardrail: command classification failed; "
                       "screening as plain text", exc_info=True)
        return False


async def screen(adapter, event) -> bool:
    """Screen one inbound event; True = HELD (caller must not run a
    turn for it). NEVER raises (review B3): every failure degrades to
    'proceed', logged."""
    try:
        return await _screen_inner(adapter, event)
    except Exception:
        logger.exception("guardrail: unexpected error; proceeding")
        return False


async def _screen_inner(adapter, event) -> bool:
    if not guardrail_enabled():
        return False
    # Only KNOWN gateway commands skip the gate (remote admin safety).
    # Unknown '/zzz payload' text is plain text downstream - screened.
    if _known_gateway_command(event):
        return False
    text = (getattr(event, "text", "") or "").strip()
    if not text:
        return False
    sender = str(getattr(event, "user_id", None)
                 or getattr(event.source, "user_id", None) or "unknown")
    platform = str(getattr(event.source, "platform", "?"))
    chat_id = str(getattr(event.source, "chat_id", "") or "")
    endpoint = _effective_endpoint()
    timeout = _float_env("HERMES_GATEWAY_GUARDRAIL_TIMEOUT_S", 4.0)
    caller = _flag("HERMES_GATEWAY_GUARDRAIL_CALLER") or "hermes-gateway"
    base = {"sender": sender, "platform": platform, "preview": text[:120]}

    def call() -> dict:
        return _post(endpoint, {"text": text[:MAX_TEXT_CHARS]},
                     {"X-Judge-Caller": caller}, timeout)

    try:
        # Overall deadline: socket timeout + 1s guards drip-fed bodies.
        result = await asyncio.wait_for(asyncio.to_thread(call),
                                        timeout=timeout + 1.0)
        if not isinstance(result, dict):
            raise ValueError(f"non-dict engine response: {type(result)!r}")
    except Exception as err:
        _log(dict(base, tier="engine_unreachable_or_malformed",
                  error=str(err)[:160]))
        await _maybe_alert(adapter, sender, platform,
                           "screen unavailable (engine unreachable or "
                           "malformed response) - message allowed")
        return False

    tier = result.get("tier")
    tier = tier if isinstance(tier, str) else None
    raw_reasons = result.get("reasons")
    if not isinstance(raw_reasons, list):
        raw_reasons = []
    reasons = [str(r) for r in raw_reasons
               if isinstance(r, str) and r.strip()][:8]
    served_by = result.get("served_by")
    served_by = served_by.get("provider") if isinstance(served_by, dict) else None
    scores = result.get("scores") if isinstance(result.get("scores"), dict) else {}
    _log(dict(base, tier=tier, reasons=reasons,
              served_by=served_by, scores=scores))
    question_id = ("q_" + reasons[0]) if reasons else None
    if tier == "block":
        await _maybe_alert(adapter, sender, platform,
                           f"{tier}-tier ({', '.join(reasons)[:_ALERT_TEXT_CAP]})"
                           " - held automatically")
        reply = _flag("HERMES_GATEWAY_GUARDRAIL_HOLD_REPLY")
        if reply and chat_id:
            await _send(adapter, chat_id, reply)
        _spawn_outcome_report(result, "escalated",
                              "block-tier held automatically (Phase 2)",
                              question_id)
        return True
    if tier in ("review", "chain_down"):
        if tier == "chain_down":
            await _maybe_alert(adapter, sender, platform,
                               "chain-down - engine held or failed per its "
                               "own rule; message allowed through Phase 2")
        _spawn_outcome_report(result, "escalated",
                              f"{tier}-tier (Phase 2 advisory pass)",
                              question_id)
        return False
    _spawn_outcome_report(result, "accepted", "allow-tier (Phase 2)",
                          question_id)
    return False


async def _maybe_alert(adapter, sender: str, platform: str,
                       detail: str) -> None:
    owner_chat = _flag("HERMES_GATEWAY_GUARDRAIL_OWNER_CHAT")
    if not owner_chat:
        return
    window = _float_env("HERMES_GATEWAY_GUARDRAIL_DEBOUNCE_S", 600.0)
    if not _debounced(f"{platform}:{sender}", window):
        return
    await _send(adapter, owner_chat,
                f"[guardrail] {sender} on {platform}: {detail}.")
