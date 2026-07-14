"""Rule-based severity classification for infra alerts.

Rule-based, not an LLM judge — avoids the small-model over-triggering risk
flagged in bead androux-8dg0. Defaults to "batched" for anything not
explicitly listed as urgent: failing toward not-interrupting is the safer
default given the goal is reducing false-alarm interrupts, not missing
real ones (real service-down/security alerts are explicitly listed below).
"""
from typing import Literal

Severity = Literal["urgent", "batched"]

# Alert types that always interrupt in real time, regardless of source.
_URGENT_ALERT_TYPES = {
    "ServiceDown",
    "HighAttackVolume",
    "SecurityIncident",
    "DataLossRisk",
}


def classify_severity(*, source: str, alert_type: str) -> Severity:
    """Classify an alert's urgency.

    Args:
        source: the host/service the alert originated from (e.g. an IP:port,
            a service name, or a watcher name like "prometheus"/"opnsense").
        alert_type: the alert's type/name as reported by the source
            (e.g. "ServiceDown", "HighSystemLoad", "DiskSpaceLow").

    Returns:
        "urgent" if this should interrupt in real time (Telegram/ntfy
        immediately); "batched" if it should fold into the daily debrief.
    """
    if alert_type in _URGENT_ALERT_TYPES:
        return "urgent"
    return "batched"
