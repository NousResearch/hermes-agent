"""Cron-echo suppression: a cron delivery is a machine ping, not a request.

A cron job delivers to a group topic as the owning profile's bot. The receiving
gateway sees that post as an ordinary inbound message from another bot, so the
group agent answers it — every time, even when the job's own state has not
changed. The reply adds a message and zero information, and the load it adds is
what trips the interrupt-recursion guard on a busy topic.

This module owns two decisions:

* **Is this inbound message a cron envelope?** (a machine ping, not a request)
* **Is the ping worth answering?** ("reply only when something CHANGED")

The envelope is the one the delivery layer writes (``cron/scheduler_delivery.py``):
``Cronjob Response: <job name>`` / ``(job_id: <id>)`` / a dashed divider. The
content under the divider is the job's actual output, which is where a genuine
escalation can hide.

**The default is silence.** A cron delivery to a group topic is addressed to
nobody: it is the owning profile's bot reporting its own status. The receiving
group agent answering it adds a message and zero information, and the load that
adds is what trips the interrupt-recursion guard on a busy topic. So an envelope
is dropped at ingress — no turn, no reply, no agent work.

The changed-state decision is the second half: when an operator asks for
change-driven replies (``gateway.cron_echo.reply_on_change_only: true``), the
suppressor answers a ping only when the job's payload fingerprint CHANGED. The
fingerprint ignores volatile noise (failure counters, timestamps) so the same
outage re-reported with a new counter is the SAME state and stays silent, while
a different error class, host, or identity is a different state and is answered.
The fingerprint lives in ``<hermes_home>/state/cron-echo/<job-key>.json`` so it
survives restarts and is readable by an operator asking "why did the bot answer
that?".

Escalations never go silent: a cron payload that carries an escalation marker
(``I need help with **title**``) is always admitted, and a human message that
merely *replies to* a cron post is not an envelope at all — it is ordinary
human text and is admitted by the normal path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

logger = logging.getLogger("gateway.run")

__all__ = [
    "CronEchoSettings",
    "CronEchoVerdict",
    "is_cron_envelope",
    "parse_cron_envelope",
    "load_settings",
    "settings_from_config",
    "CronEchoSuppressor",
]

# The header the delivery layer writes. Mirrors ``cron/scheduler_delivery.py``;
# keep the two in step (the delivery side is the definition, this side the reader).
CRON_ENVELOPE_HEADER = "Cronjob Response:"

# ``(job_id: <id>)`` — the delivery layer's second line.
_JOB_ID_RE = re.compile(r"^\(job_id:\s*(?P<job_id>[0-9a-fA-F-]{4,})\)\s*$", re.MULTILINE)

# The dashed divider between the envelope header and the payload.
_DIVIDER_RE = re.compile(r"^-{3,}\s*$", re.MULTILINE)

# The delivery layer's footer ("To stop or manage this job, send me a new message").
_FOOTER_RE = re.compile(r"To stop or manage this job, send me a new message", re.IGNORECASE)

# A blocked-card escalation. The text is the agent's, not the cron's, so a
# message carrying it is a real request and must never be silenced.
_ESCALATION_MARKERS = (
    "I need help with",
    "NEEDS HUMAN",
    "NEEDS ATTENTION",
    "🚨 ESCALATION",
)

# A cron payload may itself carry an escalation (a job that reports a blocked
# card). Those must reach the operator even though the envelope is a machine ping.
_ESCALATION_PAYLOAD_RE = re.compile(
    r"(?:I need help with|🚨\s*ESCALATION|NEEDS\s+HUMAN(?:\s+ATTENTION)?\b)",
    re.IGNORECASE,
)

# A job key safe to use as a filename.
_UNSAFE_KEY_RE = re.compile(r"[^A-Za-z0-9._-]+")

_TRUTHY = frozenset({"true", "1", "yes", "on"})
_FALSY = frozenset({"false", "0", "no", "off"})


@dataclass(frozen=True)
class CronEchoSettings:
    """Settings for the cron-echo suppressor. Read from ``gateway.cron_echo``."""

    enabled: bool = True
    # When True (the default), a cron envelope is dropped outright: a delivery to a
    # group topic is never addressed to the receiving agent, so answering it adds a
    # message and no information. Escalations are exempt (see below).
    #
    # When False, the suppressor instead keeps the envelope and answers it only when
    # the job's payload fingerprint CHANGED since the last answer — the "reply only
    # when something changed" rule. Off by default because the first delivery of a
    # steady-state job is not news, and a cron ping is not a request.
    reply_on_change_only: bool = False
    # How long a fingerprint stays authoritative, in seconds. 0 = forever.
    state_ttl_seconds: float = 0.0
    # Window in which an unchanged payload is treated as already answered. 0 = forever.
    dedupe_seconds: float = 0.0


@dataclass(frozen=True)
class CronEchoVerdict:
    """What the suppressor decided about one inbound message."""

    #: True when the message is a cron delivery envelope.
    is_envelope: bool = False
    #: True when the envelope carries a genuine escalation and must be admitted.
    is_escalation: bool = False
    #: True when the message should be dropped (no agent turn, no reply).
    suppress: bool = False
    #: The job name from the envelope, when there is one.
    job_name: str = ""
    #: The job id from the envelope, when there is one.
    job_id: str = ""
    #: Why the message was suppressed — for the log line and the operator.
    reason: str = ""

    @property
    def job_key(self) -> str:
        """A stable key for the state file: the job id when known, else the name."""
        raw = self.job_id or self.job_name
        return _UNSAFE_KEY_RE.sub("-", raw).strip("-") or "unknown"


def _as_bool(raw: Any, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower() if raw is not None else ""
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    return default


def _as_non_negative(raw: Any, default: float) -> float:
    if isinstance(raw, bool):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def settings_from_config(cfg: Any) -> CronEchoSettings:
    """Read ``gateway.cron_echo`` from a loaded config dict. Unusable values keep the default.

    A config dict that cannot be read at all (the YAML loader is unavailable, the
    block is the wrong shape) keeps the defaults rather than raising: the
    suppressor must never be the reason ingress fails.
    """
    try:
        from hermes_cli.config import cfg_get

        block = cfg_get(cfg, "gateway", "cron_echo", default=None)
    except Exception:
        logger.debug("cron-echo: config block unreadable, keeping defaults", exc_info=True)
        return CronEchoSettings()
    if not isinstance(block, dict):
        return CronEchoSettings()
    defaults = CronEchoSettings()
    return CronEchoSettings(
        enabled=_as_bool(block.get("enabled"), defaults.enabled),
        reply_on_change_only=_as_bool(block.get("reply_on_change_only"), defaults.reply_on_change_only),
        state_ttl_seconds=_as_non_negative(block.get("state_ttl_seconds"), defaults.state_ttl_seconds),
        dedupe_seconds=_as_non_negative(block.get("dedupe_seconds"), defaults.dedupe_seconds),
    )


def load_settings() -> CronEchoSettings:
    """Settings from the live config.yaml. Defaults when the config cannot be read."""
    try:
        from hermes_cli.config import load_config_readonly

        return settings_from_config(load_config_readonly())
    except Exception:
        logger.debug("cron-echo: config could not be loaded, keeping defaults", exc_info=True)
        return CronEchoSettings()


def parse_cron_envelope(text: Any) -> Optional[CronEchoVerdict]:
    """Parse a cron delivery envelope out of ``text``; None when it is not one.

    The envelope is only recognised at the START of the message. A human message
    that quotes a cron post ("can you kill the VPN alert?") is not an envelope
    and must keep the normal path.
    """
    if not isinstance(text, str):
        return None
    stripped = text.lstrip()
    if not stripped.startswith(CRON_ENVELOPE_HEADER):
        return None

    header_line, _, remainder = stripped.partition("\n")
    # "Cronjob Response: <job name>" — the name may be empty, but the header must be alone on its line.
    if "\n" in header_line and not header_line.startswith(CRON_ENVELOPE_HEADER):
        return None
    job_name = header_line[len(CRON_ENVELOPE_HEADER):].strip()

    job_id = ""
    id_match = _JOB_ID_RE.search(remainder)
    if id_match:
        job_id = id_match.group("job_id")

    # The payload is what sits under the divider. Without a divider the whole
    # remainder is the payload.
    payload = remainder
    divider = _DIVIDER_RE.search(remainder)
    if divider:
        payload = remainder[divider.end():]
    # The delivery layer's footer is not payload.
    footer = _FOOTER_RE.search(payload)
    if footer:
        payload = payload[: footer.start()]

    escalation = bool(_ESCALATION_PAYLOAD_RE.search(payload))
    return CronEchoVerdict(is_envelope=True, is_escalation=escalation, job_name=job_name, job_id=job_id)


def is_cron_envelope(text: Any) -> bool:
    """True when ``text`` is a cron delivery envelope (machine ping, not a request)."""
    return parse_cron_envelope(text) is not None


def _extract_payload(text: Any) -> str:
    """The job's own output: what sits under the envelope's divider."""
    if not isinstance(text, str):
        return ""
    stripped = text.lstrip()
    remainder = stripped.partition("\n")[2]
    divider = _DIVIDER_RE.search(remainder)
    payload = remainder[divider.end():] if divider else remainder
    footer = _FOOTER_RE.search(payload)
    if footer:
        payload = payload[: footer.start()]
    return payload


# Volatile noise: the parts of a status payload that change every tick without the
# situation changing. A watchdog page that says "consecutive failures: 12" becomes
# "consecutive failures: 18" on the next tick — the same outage, a different number.
# Keying the fingerprint on the raw text would then read every page as a change and
# the hub would answer every one of them, which is the behaviour this module exists
# to stop. These patterns are normalized away so the fingerprint tracks the
# SITUATION (which host, which error class, which identity) rather than the counters.
_VOLATILE_PATTERNS = (
    # ISO-8601 timestamps (2026-10-24T03:26:10Z) and bare dates.
    re.compile(r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}(?:\.\d+)?z?", re.IGNORECASE),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    # Clock times (03:26:10, 3:26 pm).
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b", re.IGNORECASE),
    # Durations that grow with the outage ("for 15 minutes", "3h12m").
    re.compile(r"\b\d+\s*(?:h|m|s|min|mins|minutes|seconds|hours)\b", re.IGNORECASE),
    # "consecutive failures: 12" / "failures=18" / "streak=3" — counters, not identity.
    re.compile(
        r"\b(consecutive[\s_-]+(?:failures?|errors?)|failures?|errors?|streak|attempts?|"
        r"retries|count|total|uptime|age)\b\s*[:=]?\s*\d+",
        re.IGNORECASE,
    ),
    # "12 consecutive failures" (counter before the noun).
    re.compile(r"\b\d+\s+(consecutive[\s_-]+(?:failures?|errors?)|failures?|errors?)\b", re.IGNORECASE),
)


def normalize_payload(payload: str) -> str:
    """Strip volatile counters and timestamps so a fingerprint tracks the situation."""
    normalized = " ".join(str(payload).split()).lower()
    for pattern in _VOLATILE_PATTERNS:
        normalized = pattern.sub(" ", normalized)
    # Collapse the whitespace the substitutions leave behind.
    return " ".join(normalized.split())


def payload_fingerprint(text: Any) -> str:
    """A stable fingerprint of the cron payload, used to decide "something changed".

    The fingerprint covers the payload with volatile noise removed (see
    ``normalize_payload``), so the same outage re-reported with a new failure counter
    is the SAME state and stays silent, while a genuinely different failure class,
    host, or identity is a different state and is answered.
    """
    verdict = parse_cron_envelope(text)
    payload = _extract_payload(text) if verdict is not None else (text if isinstance(text, str) else "")
    normalized = normalize_payload(payload)
    return hashlib.sha256(normalized.encode("utf-8", "replace")).hexdigest()[:32]


class CronEchoSuppressor:
    """Decides whether a cron delivery deserves an agent turn, and remembers the last answer.

    State lives in ``<hermes_home>/state/cron-echo/<job-key>.json`` so the decision
    survives a gateway restart: without it, every restart would re-answer the
    last ping of every job.
    """

    def __init__(
        self,
        settings_provider=load_settings,
        state_dir: Optional[Any] = None,
        clock=time.time,
    ) -> None:
        self._settings_provider = settings_provider
        self._state_dir = state_dir
        self._clock = clock
        self._lock = threading.Lock()

    # -- state file ---------------------------------------------------------

    def _resolve_state_dir(self) -> Optional[Any]:
        if self._state_dir is not None:
            return self._state_dir
        try:
            from hermes_constants import get_hermes_home

            return get_hermes_home() / "state" / "cron-echo"
        except Exception:
            logger.debug("cron-echo: state dir unavailable, fingerprints will not persist", exc_info=True)
            return None

    def _state_path(self, job_key: str) -> Optional[Any]:
        state_dir = self._resolve_state_dir()
        if state_dir is None:
            return None
        try:
            state_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.debug("cron-echo: state dir %s not writable: %s", state_dir, exc)
            return None
        return state_dir / f"{job_key}.json"

    def _read_state(self, job_key: str) -> dict:
        path = self._state_path(job_key)
        if path is None:
            return {}
        try:
            raw = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            logger.debug("cron-echo: could not read state for %s: %s", job_key, exc)
            return {}
        try:
            data = json.loads(raw)
        except (ValueError, TypeError) as exc:
            logger.debug("cron-echo: state for %s is not valid JSON: %s", job_key, exc)
            return {}
        return data if isinstance(data, dict) else {}

    def _write_state(self, job_key: str, data: dict) -> None:
        path = self._state_path(job_key)
        if path is None:
            return
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(path)
        except (OSError, TypeError, ValueError) as exc:
            logger.debug("cron-echo: could not persist state for %s: %s", job_key, exc)

    # -- decision -----------------------------------------------------------

    def verdict(self, text: Any) -> CronEchoVerdict:
        """Decide what to do with an inbound message body."""
        parsed = parse_cron_envelope(text)
        if parsed is None:
            return CronEchoVerdict()

        settings = self._settings_provider()
        if not settings.enabled:
            return parsed

        if parsed.is_escalation:
            # A payload carrying an escalation is a real request wearing an
            # envelope. Answer it, and remember the fingerprint so a repeat of
            # the SAME escalation does not re-answer.
            return parsed

        if not settings.reply_on_change_only:
            # The default: a cron delivery to a group topic is a machine ping
            # addressed to nobody. Drop it before a turn exists — no reply, no
            # turn, no agent work. This is what stops the hub from acknowledging
            # another profile's status ping 14-18 times a day.
            return CronEchoVerdict(
                is_envelope=True,
                is_escalation=False,
                suppress=True,
                job_name=parsed.job_name,
                job_id=parsed.job_id,
                reason="cron envelope (machine ping, not a request)",
            )

        fingerprint = payload_fingerprint(text)
        with self._lock:
            state = self._read_state(parsed.job_key)
            now = self._clock()
            last_fingerprint = str(state.get("fingerprint") or "")
            last_answered = float(state.get("answered_at") or 0.0)

            if last_fingerprint == fingerprint:
                age = now - last_answered if last_answered else float("inf")
                if settings.dedupe_seconds <= 0 or age < settings.dedupe_seconds:
                    return CronEchoVerdict(
                        is_envelope=True,
                        suppress=True,
                        job_name=parsed.job_name,
                        job_id=parsed.job_id,
                        reason="cron payload unchanged since the last answer",
                    )

            # Changed: record the new fingerprint and answer. The first delivery of
            # a job is ALSO a change (no prior fingerprint) — with
            # ``reply_on_change_only`` on, that is deliberate: the operator asked
            # for change-driven replies, and an unseen job is a new state.
            record = {
                "fingerprint": fingerprint,
                "answered_at": now,
                "job_name": parsed.job_name,
                "job_id": parsed.job_id,
            }
            if settings.state_ttl_seconds > 0:
                record["expires_at"] = now + settings.state_ttl_seconds
            self._write_state(parsed.job_key, record)

        return CronEchoVerdict(
            is_envelope=True,
            is_escalation=False,
            suppress=False,
            job_name=parsed.job_name,
            job_id=parsed.job_id,
            reason="cron payload changed",
        )

    def suppress(self, text: Any) -> Tuple[bool, str]:
        """``(suppress, reason)`` for one inbound message body."""
        verdict = self.verdict(text)
        return verdict.suppress, verdict.reason
