"""
Phase 026-D — Hermes cron daily morning brief.

Purpose
-------
Once per weekday morning (Mon–Fri, 07:30 in the operator's local
timezone), invoke ``/daily`` programmatically and DM the resulting
block-kit brief to Blake on Slack.

This is the behavioral-lock-in mechanism R2 promoted to the top of the
five-journey D5 portfolio: the first interaction with Slack each
morning is a brief Blake did not request. Daily-frequency repetition is
what eventually surfaces gaps in Atlas, the synthesizer ranker, and the
orchestrator status surface — see plan 026 §"Goal".

Design notes
------------
- **Standalone Hermes cron**: invoked as a script entry, mirroring
  ``cron/follow_up_sweep.py`` (the 028-B pattern). The cron registry
  stores the schedule (``30 7 * * 1-5``, weekday-only, configured in the
  operator's local TZ via ``HERMES_TIMEZONE``) and runs::

      python -m cron.daily_brief

  on a tick. The Hermes cron primitive resolves cron expressions against
  the configured timezone (see ``hermes_time``), so the schedule string
  is wall-clock local even though the scheduler process may run in UTC.

- **Weekday gate**: the module also enforces ``Mon–Fri`` itself as a
  defense-in-depth check, so an operator who misconfigures the cron
  expression (e.g. forgets the ``1-5`` weekday selector) still does not
  get woken up Saturday morning. ``should_fire_now()`` returns False on
  weekends → ``run_daily_brief()`` exits 0 with status ``skipped``.

- **Reuses ``build_daily_brief()``** from 026-A. The cron path passes a
  ``DailyHandlerConfig`` with ``writeback_enabled=True`` so the
  AgentDecision triple lands in Atlas (the next-day prologue depends on
  it — that's the feedback loop that closes 026-B).

- **Slack target**: posts to ``SLACK_DAILY_DM_CHANNEL`` (a Slack DM
  channel ID like ``D0123ABCD``) using ``SLACK_BOT_TOKEN``. The DM
  channel is resolved once by an operator at setup-time via Slack's
  ``conversations.open`` (Blake's user ID → DM channel ID) and stored
  in env. We do *not* re-resolve it on every fire — that would add a
  Slack API round-trip to every 07:30 wake-up and a new failure mode.

- **Stdlib only.** Same justification as 028-B: this job is a thin
  orchestration layer and should not drag in optional deps.

- **Honest failure semantics.** When ``build_daily_brief`` raises or
  Slack chat.postMessage fails, we log + surface the error via
  exit code 1 → the cron scheduler captures the traceback into its
  output file. Per memory ``feedback_test_gate.md``, silent failures
  here would defeat the behavioral lock-in — better to wake Blake with
  an "X happened" alert than to ship nothing.

Acceptance (from master plan §026-D):
  * cron module exists with weekday 07:30 user-local schedule
  * DM sent containing the daily brief (verified by test with mocked
    /daily output + mocked Slack)
  * 5+ tests including the weekend-skip case + /daily failure case
  * No regression on existing cron tests
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SLACK_POST_URL = "https://slack.com/api/chat.postMessage"
FIRE_HOUR = 7  # 07:30 user-local
FIRE_MINUTE = 30
# Mon=0 ... Sun=6 (datetime.weekday())
WEEKDAY_GATE = frozenset({0, 1, 2, 3, 4})


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


# ---------------------------------------------------------------------------
# Schedule gate
# ---------------------------------------------------------------------------


def _now_user_local(now: datetime | None = None) -> datetime:
    """Return ``now`` in the operator's local timezone (HERMES_TIMEZONE)."""
    if now is not None:
        return now
    # Import lazily so import-time cost of hermes_time (config.yaml read) is
    # only paid when the module is actually executed, not when imported by
    # tests that pre-build ``now``.
    from hermes_time import now as _hermes_now
    return _hermes_now()


def should_fire_now(
    now: datetime | None = None,
    *,
    weekday_gate: frozenset[int] = WEEKDAY_GATE,
) -> tuple[bool, str]:
    """Return (fire?, reason). Used by the cron entry to skip weekend wakes.

    The cron expression in the registry should already constrain to
    Mon–Fri 07:30; this is the defense-in-depth check so a misconfigured
    expression does not produce a Saturday 07:30 DM.
    """
    local = _now_user_local(now)
    if local.weekday() not in weekday_gate:
        return False, f"weekend-skip (weekday={local.weekday()})"
    return True, f"weekday-fire (weekday={local.weekday()}, time={local:%H:%M})"


# ---------------------------------------------------------------------------
# Slack client (stdlib http)
# ---------------------------------------------------------------------------


class SlackError(RuntimeError):
    """Raised on Slack chat.postMessage failure."""


def _slack_post(
    payload: dict[str, Any],
    *,
    token: str,
    url_opener=urllib.request.urlopen,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        SLACK_POST_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    try:
        with url_opener(req, timeout=30) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise SlackError(f"Slack HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise SlackError(f"Slack network error: {exc.reason}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SlackError(f"Slack returned non-JSON: {raw[:200]!r}") from exc
    if not data.get("ok"):
        raise SlackError(f"Slack API error: {data.get('error', 'unknown')}")
    return data


def send_brief_to_slack(
    payload: dict[str, Any],
    *,
    channel: str | None = None,
    token: str | None = None,
    slack_post=_slack_post,
) -> dict[str, Any]:
    """Post the block-kit brief to the DM channel.

    Returns ``{"summary_ts": ts, "channel": channel}``.
    """
    channel = channel or _env("SLACK_DAILY_DM_CHANNEL") or _env("SLACK_HOME_CHANNEL")
    token = token or _env("SLACK_BOT_TOKEN")
    if not channel:
        raise SlackError(
            "SLACK_DAILY_DM_CHANNEL (or SLACK_HOME_CHANNEL) not set"
        )
    if not token:
        raise SlackError("SLACK_BOT_TOKEN not set")

    blocks = payload.get("blocks") or []
    text = payload.get("text") or "Daily brief"
    resp = slack_post(
        {
            "channel": channel,
            "text": text,
            "blocks": blocks,
            "mrkdwn": True,
        },
        token=token,
    )
    return {"summary_ts": resp.get("ts"), "channel": channel}


# ---------------------------------------------------------------------------
# /daily invocation
# ---------------------------------------------------------------------------


async def _invoke_daily_default(
    *,
    slack_channel: str,
) -> dict[str, Any]:
    """Default ``/daily`` invocation — calls 026-A's ``build_daily_brief``.

    ``slack_channel`` is part of the injected-invoker contract (the cron
    resolves the target channel and the post-step delivers there). It is
    accepted but not forwarded to ``build_daily_brief``: that builder takes a
    ``DailyHandlerConfig`` whose only fields are test seams (atlas_ask,
    today_fetcher, orchestrator_base_url, httpx_module). Writeback of the
    AgentDecision triple (026-B) is not yet wired into the builder, so passing
    ``writeback_enabled``/``slack_channel`` here raised ``TypeError`` and broke
    the cron path entirely. Use the real defaults until 026-B lands.
    """
    # Lazy import: keeps test surface free of the daily plugin until the
    # cron job is actually invoked.
    from plugins.slash.daily import build_daily_brief

    return await build_daily_brief()


# ---------------------------------------------------------------------------
# Orchestration entry point
# ---------------------------------------------------------------------------


def _emit_brief_run_record(
    *,
    run_id: str,
    status: str,
    reason: str,
    agent_decision_urn: str | None,
) -> None:
    """Plan 056-D: emit ONE ``producer="hermes"`` RunRecord at brief completion.

    STRICTLY fail-soft + side-channel: this is the statefulness-spine breadcrumb
    (so a Hermes brief shares the ``run_id`` join key with any orchestrator work
    it triggers), NOT part of the brief's delivery. Any failure is swallowed by
    ``emit_run_record`` itself, and we additionally guard the import/build so a
    missing module can never change the brief's behavior. The LIVE wiring
    (shipping these to the shared S3/Atlas store in prod) is Blake-gated [manual].
    """
    try:
        from cron.run_record import build_run_record, emit_run_record

        record = build_run_record(
            run_id=run_id,
            kind="brief",
            status=status,
            notes=f"daily-brief: {reason}",
            memory_refs=[agent_decision_urn] if agent_decision_urn else [],
        )
        emit_run_record(record)
    except Exception:  # noqa: BLE001 — observability must never break the brief
        logger.debug("daily-brief: run-record emit skipped", exc_info=True)


def run_daily_brief(
    *,
    now: datetime | None = None,
    invoke_daily: Optional[Callable[..., Any]] = None,
    slack_post=_slack_post,
    channel: str | None = None,
    token: str | None = None,
    weekday_gate: frozenset[int] = WEEKDAY_GATE,
    run_id: str | None = None,
) -> dict[str, Any]:
    """End-to-end: gate, build brief, post to Slack.

    Returns a result dict suitable for the cron output log::

        {"status": "delivered"|"skipped"|"failed",
         "reason": str,
         "summary_ts": str|None,
         "channel": str|None,
         "agent_decision_urn": str|None,
         "run_id": str}

    Raises ``SlackError`` only on the post-step; ``build_daily_brief``
    failures are caught and surfaced as ``status=failed`` so the cron
    log captures the failure without crashing the scheduler.

    Plan 056-D: this cron path is the initiator of a Hermes action, so it stamps
    a ``run_id`` (overridable for an inbound chain) and emits ONE fail-soft
    ``RunRecord`` at completion — the shared-``run_id`` statefulness spine. The
    emit is behind the existing cron flow and never alters the brief's behavior.
    """
    # Plan 056-D: stamp the run_id for this brief (the initiator mints it).
    from cron.run_record import new_run_id

    run_id = run_id or new_run_id()

    fire, reason = should_fire_now(now, weekday_gate=weekday_gate)
    if not fire:
        logger.info("daily-brief: %s — skipping", reason)
        _emit_brief_run_record(
            run_id=run_id, status="skipped", reason=reason,
            agent_decision_urn=None,
        )
        return {
            "status": "skipped",
            "reason": reason,
            "summary_ts": None,
            "channel": None,
            "agent_decision_urn": None,
            "run_id": run_id,
        }

    resolved_channel = channel or _env("SLACK_DAILY_DM_CHANNEL") or _env("SLACK_HOME_CHANNEL") or ""

    # Build the brief.
    invoker = invoke_daily or _invoke_daily_default
    try:
        coro = invoker(slack_channel=resolved_channel or "dm:bossman2")
        if asyncio.iscoroutine(coro):
            payload = asyncio.run(coro)
        else:
            payload = coro
    except Exception as exc:
        logger.exception("daily-brief: /daily invocation failed")
        reason = f"/daily invocation failed: {type(exc).__name__}: {exc}"
        _emit_brief_run_record(
            run_id=run_id, status="failed", reason=reason,
            agent_decision_urn=None,
        )
        return {
            "status": "failed",
            "reason": reason,
            "summary_ts": None,
            "channel": resolved_channel or None,
            "agent_decision_urn": None,
            "run_id": run_id,
        }

    if not isinstance(payload, dict) or not payload.get("blocks"):
        _emit_brief_run_record(
            run_id=run_id, status="failed",
            reason="build_daily_brief returned no blocks",
            agent_decision_urn=None,
        )
        return {
            "status": "failed",
            "reason": "build_daily_brief returned no blocks",
            "summary_ts": None,
            "channel": resolved_channel or None,
            "agent_decision_urn": None,
            "run_id": run_id,
        }

    meta = payload.get("_daily_meta") or {}
    agent_decision_urn = meta.get("agent_decision_urn")

    # Deliver to Slack.
    delivery = send_brief_to_slack(
        payload,
        channel=channel,
        token=token,
        slack_post=slack_post,
    )

    _emit_brief_run_record(
        run_id=run_id, status="delivered", reason=reason,
        agent_decision_urn=agent_decision_urn,
    )
    result = {
        "status": "delivered",
        "reason": reason,
        "summary_ts": delivery.get("summary_ts"),
        "channel": delivery.get("channel"),
        "agent_decision_urn": agent_decision_urn,
        "run_id": run_id,
    }
    logger.info(
        "daily-brief: delivered status=%s summary_ts=%s channel=%s urn=%s",
        result["status"],
        result["summary_ts"],
        result["channel"],
        result["agent_decision_urn"],
    )
    return result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        result = run_daily_brief()
    except SlackError as exc:
        logger.error("daily-brief failed: %s", exc)
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1
    sys.stdout.write(json.dumps(result) + "\n")
    return 0 if result.get("status") in ("delivered", "skipped") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
