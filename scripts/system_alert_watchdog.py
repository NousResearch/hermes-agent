#!/usr/bin/env python3
"""Dry-run / read-only System Alert watchdog.

A deliberately *separate* alerting path from the Kanban checkpoint engine
(``~/.hermes/scripts/kanban_checkpoint.py``). The checkpoint watchdog reasons
about board/pipeline state and is relayed verbatim; this module reasons about
**system health** (disk, load, heartbeat freshness, …) and produces classified
"System Alert" notifications. Keeping the two paths separate means a noisy board
never drowns out a real system alert, and a system probe failure never pollutes
the checkpoint proposal stream.

Hard rules (mirroring the checkpoint engine's safety posture):
  - Dry-run is the default and performs ZERO outbound delivery. A simulated
    delivery produces the exact Telegram-ready text it *would* have sent and a
    sanitized evidence record, but never contacts Telegram (or anything else).
  - Real delivery is opt-in via env (mode ``active`` AND
    ``HERMES_SYSTEM_ALERT_ACTIVE_ENABLED``) and is **critical-only**: only a
    ``critical`` alert is ever transmitted, through the approved local Hermes
    pathway (``tools.send_message_tool._send_to_platform`` + gateway config,
    reusing cron's Telegram home-channel resolution — see
    :func:`_default_telegram_transport`). Warning/info are NEVER sent over this
    transport — they remain local-only/simulated even in working hours. The
    transport is dependency-injected so tests pass a fake sender and assert no
    real send ever happens.
  - Probes are read-only and safe: ``shutil.disk_usage`` / ``os.getloadavg`` /
    ``os.stat`` only. No sudo, no service control, no network, no mutation.
  - Evidence is logged locally and sanitized: secrets are redacted and only a
    bounded, redacted detail string is ever persisted — never raw probe payloads
    or Kernohan-confidential material.

Alert severities:
  - ``critical`` — page-worthy; delivered even outside Jeslyn's working hours.
  - ``warning``  — routine; delivered only during working hours, otherwise
    deferred to the next in-hours run.
  - ``info``     — local-only; recorded to the evidence log, never sent to
    Telegram.

Delivery is additionally gated by a per-key cooldown so a stuck condition does
not re-page every run (dedupe/cooldown).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Callable, Iterable, Sequence
from zoneinfo import ZoneInfo

# Resolve the Hermes home dir via the profile-aware single source of truth so
# this watchdog writes its evidence/cooldown state under the *active* profile's
# tree (honouring HERMES_HOME / context overrides) rather than blindly assuming
# ~/.hermes. Falls back to the env/default if hermes_constants isn't importable
# (e.g. the script run standalone outside the repo). Same import shim as
# ``scripts/profile-tui.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from hermes_constants import get_hermes_home
except ImportError:
    def get_hermes_home() -> Path:  # type: ignore[misc]
        val = (os.environ.get("HERMES_HOME") or "").strip()
        return Path(val) if val else Path.home() / ".hermes"

HERMES_HOME = get_hermes_home()
# Default local, sanitized evidence log. JSON-lines; one decision record per
# line. Lives under the active Hermes home so it never lands inside the repo tree.
DEFAULT_EVIDENCE_LOG = HERMES_HOME / "system_alerts" / "evidence.log"

# Jeslyn's working hours, in New Zealand local time (Mon–Fri 07:00–15:30 NZ).
# Reused from the checkpoint engine's convention so the two watchdogs agree on
# what "off-hours" means.
NZ_TZ = ZoneInfo("Pacific/Auckland")
WORK_START = dtime(7, 0)
WORK_END = dtime(15, 30)

# --- Severity model --------------------------------------------------------
CRITICAL = "critical"
WARNING = "warning"
INFO = "info"
SEVERITIES = (CRITICAL, WARNING, INFO)
# Display emoji per severity for the Telegram-ready message header.
SEVERITY_EMOJI = {CRITICAL: "🔴", WARNING: "🟠", INFO: "🔵"}

# --- Delivery modes --------------------------------------------------------
MODE_DRY_RUN = "dry-run"
MODE_ACTIVE = "active"
MODE_ENV = "HERMES_SYSTEM_ALERT_MODE"
ACTIVE_ENABLED_ENV = "HERMES_SYSTEM_ALERT_ACTIVE_ENABLED"

# --- Delivery decisions ----------------------------------------------------
# What the watchdog decided to do with an alert. Every alert produces exactly
# one decision and one evidence record, whether or not it is delivered.
DELIVER = "deliver"          # critical — would send to Telegram (simulated in dry-run)
LOCAL_ONLY = "local-only"    # warning/info — recorded locally, never sent
DEFERRED = "deferred"        # reserved: held until working hours. Unused in the
                             # critical-only scope (warnings are local-only
                             # regardless of hours); kept for back-compat.
SUPPRESSED = "suppressed"    # within cooldown / duplicate this run

# Per-severity cooldown: once an alert key is delivered, an identical key is
# suppressed until this many seconds have elapsed. Critical re-pages sooner than
# a routine warning. Info is local-only so its cooldown is irrelevant.
COOLDOWN_SECONDS = {
    CRITICAL: 15 * 60,
    WARNING: 60 * 60,
    INFO: 0,
}

# Bound how many alerts a single run will surface so a probe storm cannot flood
# the channel.
MAX_ALERTS_PER_RUN = 20

# Default probe thresholds. Percentages are of-capacity used; load is the
# 1-minute average per CPU. Conservative defaults; callers/tests override.
DISK_WARN_PCT = 85.0
DISK_CRIT_PCT = 95.0
LOADAVG_WARN = 4.0
LOADAVG_CRIT = 8.0
# A heartbeat/sentinel file older than this is stale (warning) / very stale
# (critical).
HEARTBEAT_WARN_AGE = 30 * 60
HEARTBEAT_CRIT_AGE = 2 * 60 * 60

# Patterns used to redact obvious secrets from alert detail and evidence.
# Kept in lockstep with the checkpoint engine's ``_SECRET_PATTERNS`` so neither
# watchdog can leak a credential into a relayed message or a local log.
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password|passwd|pwd)\s*[:=]\s*\S+"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[abprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
)


def redact(text: str | None, *, max_len: int = 200) -> str:
    """Strip obvious secrets and bound length. Mirrors the checkpoint engine.

    Newlines are collapsed so a multi-line probe payload cannot smuggle an
    un-redacted continuation past a line-oriented scan, and the result is
    truncated so an oversized blob cannot bloat the evidence log.
    """
    if not text:
        return ""
    out = str(text)
    for pat in _SECRET_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    out = out.replace("\n", " ").strip()
    if len(out) > max_len:
        out = out[: max_len - 1] + "…"
    return out


def is_jeslyn_working_hours(now: datetime | int | float | None = None) -> bool:
    """True when ``now`` falls within Jeslyn's NZ working hours.

    Mon–Fri, WORK_START–WORK_END inclusive, evaluated in Pacific/Auckland.
    ``now`` may be a ``datetime``, an epoch timestamp, or ``None`` for now. A
    naive datetime is read as NZ local; an aware datetime is converted to NZ.
    Semantics intentionally match the checkpoint engine's function of the same
    name so the two watchdogs never disagree about off-hours.
    """
    if now is None:
        local = datetime.now(NZ_TZ)
    elif isinstance(now, datetime):
        local = now.replace(tzinfo=NZ_TZ) if now.tzinfo is None else now.astimezone(NZ_TZ)
    else:
        local = datetime.fromtimestamp(float(now), NZ_TZ)
    if local.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    return WORK_START <= local.time() <= WORK_END


# ---------------------------------------------------------------------------
# Alert model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Alert:
    """A single classified system alert.

    ``key`` is the dedupe/cooldown identity: two alerts about the same
    underlying condition (e.g. "disk:/ critical") must share a key so repeats
    collapse and cooldown applies. ``detail`` is free text and is ALWAYS routed
    through :func:`redact` before it reaches a message or the evidence log.
    """
    severity: str
    key: str
    title: str
    detail: str = ""
    source: str = "system"
    created_at: int = 0

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}")


def make_alert(
    severity: str,
    key: str,
    title: str,
    detail: str = "",
    *,
    source: str = "system",
    now: int | None = None,
) -> Alert:
    """Construct an :class:`Alert`, stamping ``created_at`` if not supplied."""
    return Alert(
        severity=severity,
        key=key,
        title=title,
        detail=detail,
        source=source,
        created_at=int(now) if now is not None else int(time.time()),
    )


def classify(value: float, *, warn: float, crit: float) -> str:
    """Map a numeric reading to a severity given warn/crit thresholds.

    At-or-above ``crit`` is :data:`CRITICAL`, at-or-above ``warn`` is
    :data:`WARNING`, otherwise :data:`INFO`. ``crit`` must be >= ``warn``.
    """
    if crit < warn:
        raise ValueError("crit threshold must be >= warn threshold")
    if value >= crit:
        return CRITICAL
    if value >= warn:
        return WARNING
    return INFO


# ---------------------------------------------------------------------------
# Telegram formatting (dry-run/simulation only — never actually sent here)
# ---------------------------------------------------------------------------


def format_telegram(alert: Alert) -> str:
    """Render one alert as the plain-text Telegram message it WOULD send.

    Deliberately plain text (no MarkdownV2 escaping games): the gateway relay is
    responsible for platform escaping. The detail is redacted here so a secret
    can never reach the rendered message even if a probe captured one.
    """
    emoji = SEVERITY_EMOJI.get(alert.severity, "")
    header = f"{emoji} System Alert [{alert.severity.upper()}] — {alert.title}".strip()
    lines = [header, f"source: {alert.source}", f"key: {alert.key}"]
    detail = redact(alert.detail)
    if detail:
        lines.append(detail)
    return "\n".join(lines)


def format_telegram_batch(alerts: Sequence[Alert]) -> str:
    """Render a set of deliverable alerts as one combined Telegram message."""
    return "\n\n".join(format_telegram(a) for a in alerts)


# ---------------------------------------------------------------------------
# Dedupe / cooldown / delivery decisions
# ---------------------------------------------------------------------------


@dataclass
class CooldownState:
    """Last-delivered epoch second per alert key.

    Persisted as sanitized JSON ({key: ts}); carries no secret/detail material,
    only opaque alert keys and timestamps. Loaded/saved by the caller so tests
    can drive it in-memory.
    """
    last_delivered: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "CooldownState":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            ld = {str(k): int(v) for k, v in dict(raw.get("last_delivered", {})).items()}
            return cls(last_delivered=ld)
        except Exception:
            return cls()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"last_delivered": self.last_delivered}, sort_keys=True),
            encoding="utf-8",
        )

    def in_cooldown(self, alert: Alert, now: int) -> bool:
        cd = COOLDOWN_SECONDS.get(alert.severity, 0)
        if cd <= 0:
            return False
        last = self.last_delivered.get(alert.key)
        return last is not None and (now - last) < cd

    def record_delivery(self, alert: Alert, now: int) -> None:
        self.last_delivered[alert.key] = int(now)


@dataclass(frozen=True)
class Decision:
    """The watchdog's routing decision for one alert."""
    alert: Alert
    action: str   # DELIVER | LOCAL_ONLY | DEFERRED | SUPPRESSED
    reason: str


def dedupe(alerts: Iterable[Alert]) -> list[Alert]:
    """Collapse duplicate keys within a single run, keeping the first seen.

    When two alerts share a key, the more severe one wins so a flapping
    condition that emits both a warning and a critical in one sweep surfaces as
    critical rather than whichever happened to be appended first.
    """
    order: list[str] = []
    chosen: dict[str, Alert] = {}
    rank = {CRITICAL: 3, WARNING: 2, INFO: 1}
    for a in alerts:
        if a.key not in chosen:
            order.append(a.key)
            chosen[a.key] = a
        elif rank[a.severity] > rank[chosen[a.key].severity]:
            chosen[a.key] = a
    return [chosen[k] for k in order]


def decide(alert: Alert, *, now: int, state: CooldownState, work_hours: bool) -> Decision:
    """Decide what to do with one (already-deduped) alert.

    Critical-only Telegram activation scope. Only ``critical`` is ever routed to
    Telegram; ``warning`` and ``info`` are local-only (recorded to the evidence
    log, never rendered into a Telegram message nor placed in ``deliveries``):

      - info        → local-only (recorded, never sent).
      - warning     → local-only (recorded, never sent) — warning delivery is
                      intentionally NOT part of this transport, so it stays
                      local-only regardless of working hours.
      - critical within cooldown → suppressed.
      - critical    → deliver (dry-run simulates; active+enabled sends).

    The critical path is intentionally NOT gated on ``work_hours`` — a critical
    system alert pages even outside Jeslyn's working hours; ``work_hours`` no
    longer changes any routing decision in this scope (warnings never deliver,
    criticals always do) and is retained only for signature/back-compat.
    """
    if alert.severity in (INFO, WARNING):
        return Decision(alert, LOCAL_ONLY, f"{alert.severity} severity is local-only (not sent)")
    # critical from here down.
    if state.in_cooldown(alert, now):
        cd = COOLDOWN_SECONDS.get(alert.severity, 0)
        return Decision(alert, SUPPRESSED, f"within {cd}s cooldown for key")
    if not work_hours:
        return Decision(alert, DELIVER, "critical: delivered despite off-hours")
    return Decision(alert, DELIVER, "critical in working hours")


# ---------------------------------------------------------------------------
# Delivery (simulated by default; critical-only real Telegram transport)
# ---------------------------------------------------------------------------

# A transport is any callable that takes the rendered (already-redacted)
# Telegram message and attempts a single send, returning a result dict with at
# minimum ``{"sent": bool}`` and optionally ``{"transport": str, "error": str}``.
# It MUST NOT raise for ordinary send failures (return ``error`` instead) and
# MUST NOT echo the message body or any secret into ``error``. Dependency
# injection: :func:`deliver`/:func:`run` accept a ``transport`` so tests pass a
# fake sender and assert exactly when (and when not) it is invoked.
Transport = Callable[[str], dict]


def _active_delivery_enabled() -> bool:
    return os.environ.get(ACTIVE_ENABLED_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _default_telegram_transport(message: str) -> dict:
    """Send ``message`` to Jeslyn's Telegram home channel via the approved
    local Hermes pathway. Never raises; returns a sanitized result dict.

    Single source of truth for credentials/targets — nothing is duplicated:
      * ``gateway.config.load_gateway_config`` supplies the Telegram bot config,
      * ``cron.scheduler._get_home_target_chat_id`` / ``_get_home_target_thread_id``
        resolve the configured home channel (the same target cron's
        ``deliver="telegram"`` uses),
      * ``tools.send_message_tool._send_to_platform`` performs the send.
    This mirrors the standalone send path in ``cron.scheduler._deliver_result``.

    On any failure a ``{"sent": False, ...}`` record is returned with a redacted,
    body-free ``error`` so a transport failure can be logged as evidence without
    leaking the message or a secret.
    """
    try:
        from gateway.config import load_gateway_config, Platform
        from tools.send_message_tool import _send_to_platform
        from cron.scheduler import (
            _get_home_target_chat_id,
            _get_home_target_thread_id,
        )
    except Exception as exc:  # gateway/cron not importable in this context
        return {"sent": False, "transport": "telegram", "error": f"import failed: {type(exc).__name__}"}

    try:
        config = load_gateway_config()
    except Exception as exc:
        return {"sent": False, "transport": "telegram", "error": f"config load failed: {type(exc).__name__}"}

    try:
        platform = Platform("telegram")
    except Exception as exc:
        return {"sent": False, "transport": "telegram", "error": f"unknown platform: {type(exc).__name__}"}

    pconfig = config.platforms.get(platform)
    if not pconfig or not getattr(pconfig, "enabled", False):
        return {"sent": False, "transport": "telegram", "error": "telegram not configured/enabled"}

    chat_id = _get_home_target_chat_id("telegram")
    if not chat_id:
        return {"sent": False, "transport": "telegram", "error": "no telegram home channel configured"}
    thread_id = _get_home_target_thread_id("telegram")

    coro = _send_to_platform(platform, pconfig, chat_id, message, thread_id=thread_id)
    try:
        result = asyncio.run(coro)
    except RuntimeError:
        # A loop is already running in this thread; asyncio.run never awaited the
        # coro — close it to avoid a "never awaited" warning, then send from a
        # fresh worker thread that has no running loop (mirrors cron's fallback).
        coro.close()
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                asyncio.run,
                _send_to_platform(platform, pconfig, chat_id, message, thread_id=thread_id),
            )
            try:
                result = future.result(timeout=30)
            except Exception as exc:
                return {"sent": False, "transport": "telegram", "error": f"send failed: {type(exc).__name__}"}
    except Exception as exc:
        return {"sent": False, "transport": "telegram", "error": f"send failed: {type(exc).__name__}"}

    if result and result.get("error"):
        return {"sent": False, "transport": "telegram", "error": redact(str(result["error"]))}
    return {"sent": True, "transport": "telegram"}


def deliver(decision: Decision, *, mode: str, transport: Transport | None = None) -> dict:
    """Produce a delivery record for a DELIVER decision.

    Critical-only real delivery. A real Telegram send happens **only** when all
    of the following hold: ``mode == MODE_ACTIVE``, the
    :data:`ACTIVE_ENABLED_ENV` gate is enabled, AND the alert severity is
    :data:`CRITICAL`. In that case the (injected or default) ``transport`` is
    invoked exactly once and its outcome is reflected in the record.

    Every other case — dry-run (the default), the active gate disabled, or a
    non-critical (``warning``/``info``) alert even during working hours — is a
    pure simulation: ``sent=False``, ``simulated=True``, the transport is NOT
    called, and the record carries the exact ``message`` that *would* have been
    sent. This is what keeps warning/info local-only over this transport.

    The returned record exposes ``sent`` / ``simulated`` / ``transport`` and, on
    a real-send failure, a redacted body-free ``error`` — fields the caller
    folds into the sanitized evidence log.
    """
    message = format_telegram(decision.alert)
    eligible = (
        mode == MODE_ACTIVE
        and _active_delivery_enabled()
        and decision.alert.severity == CRITICAL
    )
    if not eligible:
        return {"sent": False, "simulated": True, "transport": "telegram", "message": message}

    sender = transport if transport is not None else _default_telegram_transport
    try:
        res = sender(message) or {}
    except Exception as exc:
        # A transport that raises is recorded as a failed (not simulated) send,
        # with a redacted, body-free error.
        return {
            "sent": False,
            "simulated": False,
            "transport": "telegram",
            "error": f"transport raised: {type(exc).__name__}",
            "message": message,
        }
    record = {
        "sent": bool(res.get("sent")),
        "simulated": False,
        "transport": res.get("transport", "telegram"),
        "message": message,
    }
    err = res.get("error")
    if err:
        record["error"] = redact(str(err))
    return record


# ---------------------------------------------------------------------------
# Sanitized local evidence logging
# ---------------------------------------------------------------------------


def evidence_record(decision: Decision, delivery: dict | None) -> dict:
    """Build the sanitized evidence dict for one decision.

    Only opaque/redacted fields are persisted: severity, key, redacted title and
    detail, source, the routing action/reason, and the simulated/sent flags.
    Raw probe payloads and any secret-bearing text are never written.
    """
    a = decision.alert
    rec = {
        "ts": a.created_at,
        "severity": a.severity,
        "key": a.key,
        "title": redact(a.title, max_len=120),
        "detail": redact(a.detail),
        "source": a.source,
        "action": decision.action,
        "reason": decision.reason,
    }
    if delivery is not None:
        rec["sent"] = bool(delivery.get("sent"))
        rec["simulated"] = bool(delivery.get("simulated"))
        rec["transport"] = delivery.get("transport")
        # Transport failures are recorded for evidence, but only the redacted,
        # body-free error string — never the message itself (``delivery["message"]``
        # is deliberately NOT copied into the persisted record).
        err = delivery.get("error")
        if err:
            rec["error"] = redact(str(err))
    return rec


def append_evidence(records: Sequence[dict], path: Path) -> None:
    """Append sanitized JSON-line evidence records to the local log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    decisions: list[Decision]
    deliveries: dict[str, dict]   # alert.key -> delivery record (DELIVER only)
    evidence: list[dict]

    @property
    def delivered(self) -> list[Decision]:
        return [d for d in self.decisions if d.action == DELIVER]

    @property
    def telegram_message(self) -> str:
        return format_telegram_batch([d.alert for d in self.delivered])


def run(
    alerts: Iterable[Alert],
    *,
    now: int | None = None,
    mode: str = MODE_DRY_RUN,
    state: CooldownState | None = None,
    work_hours: bool | None = None,
    transport: Transport | None = None,
) -> RunResult:
    """Route a set of raw alerts through dedupe → decide → deliver → evidence.

    Pure with respect to I/O: it never writes the evidence log or the cooldown
    state file (the caller persists those). A simulated delivery DOES update the
    in-memory cooldown ``state`` so a repeat of the same key within the cooldown
    window is suppressed on the next call — this is what makes dedupe/cooldown
    testable without real sends.

    ``transport`` is the injected sender passed through to :func:`deliver`; it is
    only ever invoked for an eligible critical send (active mode + enabled gate),
    so in dry-run or for warning/info alerts it is never called.
    """
    now = int(now) if now is not None else int(time.time())
    state = state if state is not None else CooldownState()
    if work_hours is None:
        work_hours = is_jeslyn_working_hours(now)

    deduped = dedupe(alerts)[:MAX_ALERTS_PER_RUN]
    decisions: list[Decision] = []
    deliveries: dict[str, dict] = {}
    evidence: list[dict] = []
    for alert in deduped:
        decision = decide(alert, now=now, state=state, work_hours=work_hours)
        delivery = None
        if decision.action == DELIVER:
            delivery = deliver(decision, mode=mode, transport=transport)
            deliveries[alert.key] = delivery
            # Apply cooldown for a simulated or a successfully-sent delivery so a
            # repeat of the same key within the window is suppressed. A *failed*
            # real send is deliberately NOT recorded, so a critical that could not
            # be transmitted is retried on the next run rather than silenced.
            if delivery.get("simulated") or delivery.get("sent"):
                state.record_delivery(alert, now)
        decisions.append(decision)
        evidence.append(evidence_record(decision, delivery))
    return RunResult(decisions=decisions, deliveries=deliveries, evidence=evidence)


# ---------------------------------------------------------------------------
# Safe, read-only probes (used by the CLI; injectable for tests)
# ---------------------------------------------------------------------------

Probe = Callable[[int], list[Alert]]


def probe_disk(path: str = "/", *, now: int | None = None,
               warn: float = DISK_WARN_PCT, crit: float = DISK_CRIT_PCT) -> list[Alert]:
    """Read-only disk-capacity probe via ``shutil.disk_usage`` (stat only)."""
    now = int(now) if now is not None else int(time.time())
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return [make_alert(WARNING, f"disk:{path}", "disk probe failed",
                           f"{type(exc).__name__}", source="disk", now=now)]
    pct = (usage.used / usage.total * 100.0) if usage.total else 0.0
    sev = classify(pct, warn=warn, crit=crit)
    if sev == INFO:
        return []
    return [make_alert(sev, f"disk:{path}", f"disk usage high on {path}",
                       f"{pct:.1f}% used", source="disk", now=now)]


def probe_loadavg(*, now: int | None = None,
                  warn: float = LOADAVG_WARN, crit: float = LOADAVG_CRIT) -> list[Alert]:
    """Read-only 1-minute load-average probe via ``os.getloadavg``."""
    now = int(now) if now is not None else int(time.time())
    try:
        one_min = os.getloadavg()[0]
    except (OSError, AttributeError):
        return []
    sev = classify(one_min, warn=warn, crit=crit)
    if sev == INFO:
        return []
    return [make_alert(sev, "loadavg:1m", "load average high",
                       f"1m load {one_min:.2f}", source="load", now=now)]


def probe_heartbeat(path: str | Path, *, now: int | None = None,
                    warn_age: int = HEARTBEAT_WARN_AGE,
                    crit_age: int = HEARTBEAT_CRIT_AGE) -> list[Alert]:
    """Read-only heartbeat-freshness probe via ``os.stat`` (mtime only)."""
    now = int(now) if now is not None else int(time.time())
    p = Path(path)
    try:
        age = now - int(p.stat().st_mtime)
    except OSError:
        return [make_alert(CRITICAL, f"heartbeat:{p.name}", "heartbeat missing",
                           f"no heartbeat file at {redact(str(p))}",
                           source="heartbeat", now=now)]
    if age >= crit_age:
        return [make_alert(CRITICAL, f"heartbeat:{p.name}", "heartbeat very stale",
                           f"{age}s old", source="heartbeat", now=now)]
    if age >= warn_age:
        return [make_alert(WARNING, f"heartbeat:{p.name}", "heartbeat stale",
                           f"{age}s old", source="heartbeat", now=now)]
    return []


def gather(probes: Sequence[Probe], *, now: int | None = None) -> list[Alert]:
    """Run a set of probes and flatten their alerts. Probe errors are isolated
    so one failing probe cannot abort the sweep."""
    now = int(now) if now is not None else int(time.time())
    out: list[Alert] = []
    for probe in probes:
        try:
            out.extend(probe(now))
        except Exception:  # pragma: no cover - defensive; probes own their errors
            continue
    return out


def default_probes() -> list[Probe]:
    return [lambda now: probe_disk("/", now=now), lambda now: probe_loadavg(now=now)]


# ---------------------------------------------------------------------------
# Active cron wrapper generation
# ---------------------------------------------------------------------------

# Prefix for the wrapper's private temp directory. The full mktemp template is
# ``<TMPDIR>/<prefix>.XXXXXX`` — the trailing run of X's is REQUIRED: macOS/BSD
# ``mktemp`` only substitutes a run of X's that is the *last* component of the
# template, so a template like ``.../system-alert-active.XXXXXX.out`` (X's
# followed by ``.out``) is rejected on BSD and leaves an empty/garbage path. We
# therefore make a temp *directory* with trailing X's (portable on GNU and BSD)
# and place fixed-name capture files inside it, never a ``.XXXXXX.out`` file
# template. See the regression coverage in test_system_alert_watchdog.py.
ACTIVE_WRAPPER_TMP_PREFIX = "system-alert-active"
# A clearly non-alert marker for wrapper-level (process) failures, kept distinct
# from the "System Alert [CRITICAL]" wording of a real alert so a wrapper hiccup
# is never mistaken for a paging-worthy system condition.
WRAPPER_ERROR_MARKER = "[watchdog-error]"


def render_active_wrapper(
    script_path: str | Path,
    *,
    python: str = "python3",
    mode: str = MODE_ACTIVE,
    tmp_prefix: str = ACTIVE_WRAPPER_TMP_PREFIX,
) -> str:
    """Render the shell wrapper that runs this watchdog under the active cron job.

    The wrapper is deliberately thin and read-only-safe:

      * It runs ``<python> <script_path> --mode <mode>`` with stdout/stderr
        redirected into a *private temp directory* created with
        ``mktemp -d "<TMPDIR>/<prefix>.XXXXXX"``. The template's last component
        is a bare run of X's, which is the only shape both GNU and macOS/BSD
        ``mktemp`` accept — the previous ``.XXXXXX.out`` template failed on BSD.
      * In ``active`` mode the watchdog process is run with
        :data:`ACTIVE_ENABLED_ENV` ``=1`` so the critical-only Telegram delivery
        gate is actually open under the cron job. The env assignment is scoped to
        just the watchdog invocation (not exported wrapper-wide), and the Python
        :func:`deliver` still enforces the critical-only contract — warning/info
        are never sent regardless of this gate. In ``dry-run`` mode the gate is
        intentionally left unset so the wrapper only ever simulates.
      * On a healthy / no-alert run it emits NOTHING. The watchdog's own chatty
        status stdout is captured into the temp dir and discarded, so cron relays
        an empty body and Jeslyn sees no message. Any genuine *critical* alert is
        delivered by the watchdog itself through the approved Telegram transport,
        not by this wrapper — preserving the critical-only delivery contract.
      * Only a real failure of the watchdog *process* (nonzero exit) produces
        output, and that output is a single terse :data:`WRAPPER_ERROR_MARKER`
        health line — never "System Alert"/"CRITICAL" wording, and never the
        captured stderr body (which could carry incidental sensitive text). The
        detail stays in the local, sanitized evidence log.
      * The temp dir is always removed via an EXIT trap. ``set -e`` is
        intentionally NOT used so the wrapper can inspect the watchdog's exit
        code rather than aborting on it.

    Paths are shell-quoted so spaces/odd characters in the python or script path
    cannot break or inject into the generated script.
    """
    if mode not in (MODE_DRY_RUN, MODE_ACTIVE):
        raise ValueError(f"unknown mode {mode!r}")
    py_q = shlex.quote(str(python))
    script_q = shlex.quote(str(script_path))
    prefix_q = shlex.quote(str(tmp_prefix))
    # Open the active-delivery env gate for the watchdog process ONLY in active
    # mode, scoped to the single invocation. The env name is a fixed constant
    # (no shell metacharacters) so no quoting is required. Left empty in dry-run
    # so that wrapper never opens the gate and only ever simulates.
    gate_assign = f"{ACTIVE_ENABLED_ENV}=1 " if mode == MODE_ACTIVE else ""
    return f"""#!/usr/bin/env bash
# AUTO-GENERATED by system_alert_watchdog.py --emit-active-wrapper. Do not edit
# by hand; regenerate from the watchdog module so the temp-file handling and the
# critical-only silence contract stay in lockstep with the Python source.
#
# Active System Alert watchdog wrapper. Runs the read-only watchdog in
# {mode} mode, capturing its output to a private temp dir so a healthy/no-alert
# run relays NOTHING through cron. Real critical alerts are delivered by the
# watchdog's own Telegram transport, never by this wrapper.
set -u

PYTHON="${{HERMES_PYTHON:-{py_q}}}"
SCRIPT={script_q}

# Portable temp capture: BSD/macOS mktemp only substitutes a trailing run of X's
# that is the LAST path component, so we create a temp *directory* (valid on GNU
# and BSD) and put fixed-name capture files inside it — never a per-file template
# whose X-run is followed by a dot-suffix, which is not trailing and so fails on
# macOS/BSD mktemp.
workdir="$(mktemp -d "${{TMPDIR:-/tmp}}/{tmp_prefix}.XXXXXX")" || {{
    echo "{WRAPPER_ERROR_MARKER} could not create temp dir for system_alert_watchdog" >&2
    exit 1
}}
trap 'rm -rf "$workdir"' EXIT
out="$workdir/run.out"
err="$workdir/run.err"

# Active delivery gate: the watchdog's critical-only Telegram send is gated on
# BOTH --mode active AND {ACTIVE_ENABLED_ENV}. We open the env gate here (scoped
# to just this invocation) so a genuine critical alert is actually delivered
# under the active cron job; the Python deliver() still enforces critical-only,
# so warning/info are never sent regardless of this gate. In dry-run this prefix
# is empty, so the wrapper only ever simulates.
{gate_assign}"$PYTHON" "$SCRIPT" --mode {mode} >"$out" 2>"$err"
rc=$?

if [ "$rc" -ne 0 ]; then
    # The watchdog PROCESS itself failed — this is NOT a system alert. Surface a
    # single terse, non-alert health line (no real-alert header or severity
    # wording, and no captured stderr body, which could carry incidental
    # sensitive text) so ops can investigate via the local sanitized evidence log.
    echo "{WRAPPER_ERROR_MARKER} system_alert_watchdog {mode} run failed (rc=$rc); see local evidence log"
    exit 0
fi

# Healthy / no-alert: stay silent. The watchdog's status stdout was captured and
# is discarded with the temp dir; cron relays an empty body. Done.
exit 0
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run System Alert watchdog")
    parser.add_argument("--mode", default=os.environ.get(MODE_ENV, MODE_DRY_RUN),
                        choices=(MODE_DRY_RUN, MODE_ACTIVE),
                        help="dry-run (default) simulates; active does critical-only "
                             "real delivery when HERMES_SYSTEM_ALERT_ACTIVE_ENABLED is set")
    parser.add_argument("--evidence-log", default=str(DEFAULT_EVIDENCE_LOG))
    parser.add_argument("--state-file",
                        default=str(HERMES_HOME / "system_alerts" / "cooldown.json"))
    parser.add_argument(
        "--emit-active-wrapper",
        action="store_true",
        help="print the portable active-mode cron wrapper shell script to stdout "
             "(for (re)generating the deployed wrapper) and exit; runs no probes",
    )
    args = parser.parse_args(argv)

    if args.emit_active_wrapper:
        # Emit-only: generate the wrapper for THIS script with the running
        # interpreter and exit without probing or touching any state/evidence.
        sys.stdout.write(
            render_active_wrapper(Path(__file__).resolve(), python=sys.executable)
        )
        return 0

    now = int(time.time())
    state_path = Path(args.state_file)
    state = CooldownState.load(state_path)
    alerts = gather(default_probes(), now=now)
    result = run(alerts, now=now, mode=args.mode, state=state)

    append_evidence(result.evidence, Path(args.evidence_log))
    state.save(state_path)

    print(f"# System Alert watchdog ({args.mode}) — {len(result.decisions)} alert(s)")
    if result.delivered:
        print("\n## Would deliver (simulated):\n")
        print(result.telegram_message)
    for d in result.decisions:
        if d.action != DELIVER:
            print(f"- [{d.action}] {d.alert.severity} {d.alert.key}: {d.reason}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
