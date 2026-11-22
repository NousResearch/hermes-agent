"""Fingerprint and suppress repeated Grafana Slack alert notifications."""

from __future__ import annotations

import html
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlparse


_ALERT_BLOCK_RE = re.compile(
    r"\*(?P<status>Firing|Resolved)\*:\s*(?P<title>.*?)\s*\n\s*\n\s*"
    r"\*Summary:\*\s*(?P<summary>.*?)\s*\n(?P<body>.*?)(?=\n\s*\*(?:Firing|Resolved)\*:\s|\Z)",
    re.DOTALL,
)
_ALERT_URL_RE = re.compile(r"<(https?://[^>|]+)\|Alert>")


@dataclass(frozen=True)
class AlertOccurrence:
    status: str
    fingerprint: str
    summary: str
    message_ts: str
    channel_id: str
    current_permalink: str


@dataclass(frozen=True)
class DuplicateAlert:
    fingerprint: str
    summary: str
    original_permalink: str
    current_permalink: str


def slack_permalink(channel_id: str, message_ts: str) -> str:
    """Build a workspace-independent Slack message URL."""
    return f"https://slack.com/archives/{channel_id}/p{str(message_ts).replace('.', '')}"


def _alert_text(event: dict[str, Any]) -> str:
    parts = [str(event.get("text") or "")]
    for attachment in event.get("attachments") or []:
        if isinstance(attachment, dict):
            parts.append(str(attachment.get("text") or ""))
    return "\n".join(part for part in parts if part)


def _alert_url(block: str) -> str:
    decoded = html.unescape(block)
    match = re.search(r"<(https?://[^>|]*matcher=[^>|]+)", decoded)
    if match:
        return match.group(1)
    match = _ALERT_URL_RE.search(decoded)
    return match.group(1) if match else ""


def _fingerprint(alert_url: str) -> str:
    parsed = urlparse(alert_url)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    labels: dict[str, str] = {}
    for key, value in query:
        if key != "matcher":
            continue
        value = unquote(value)
        label, separator, label_value = value.partition("=")
        if separator and label and label != "CreatedBy":
            labels[label] = label_value
    rule_uid = labels.pop("__alert_rule_uid__", "")
    if not rule_uid:
        return ""
    identity = "|".join(f"{key}={labels[key]}" for key in sorted(labels))
    return f"{rule_uid}|{identity}" if identity else rule_uid


def parse_alerts(event: dict[str, Any], *, channel_id: str, message_ts: str) -> list[AlertOccurrence]:
    """Extract Grafana alert occurrences from one Slack message."""
    text = html.unescape(_alert_text(event))
    current_permalink = slack_permalink(channel_id, message_ts)
    occurrences: list[AlertOccurrence] = []
    for match in _ALERT_BLOCK_RE.finditer(text):
        body = match.group("body")
        fingerprint = _fingerprint(_alert_url(body))
        if not fingerprint:
            continue
        occurrences.append(AlertOccurrence(
            status=match.group("status").lower(),
            fingerprint=fingerprint,
            summary=match.group("summary").strip(),
            message_ts=message_ts,
            channel_id=channel_id,
            current_permalink=current_permalink,
        ))
    return occurrences


class AlertFingerprintStore:
    """Small persistent store for active alert fingerprints.

    The gateway normally runs as one process, but writes are still serialized and atomic so a
    restart or concurrent Slack callback cannot leave a partially written JSON file.
    """

    def __init__(self, path: str | Path, *, ttl_seconds: float = 30 * 24 * 3600) -> None:
        self.path = Path(path)
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = self._load()

    def observe(self, event: dict[str, Any], *, channel_id: str, message_ts: str) -> list[DuplicateAlert]:
        occurrences = parse_alerts(event, channel_id=channel_id, message_ts=message_ts)
        if not occurrences:
            return []
        now = time.time()
        duplicates: list[DuplicateAlert] = []
        changed = False
        with self._lock:
            changed |= self._expire(now)
            resolved_fingerprints = {
                occurrence.fingerprint for occurrence in occurrences if occurrence.status == "resolved"
            }
            for fingerprint in resolved_fingerprints:
                if self._entries.pop(fingerprint, None) is not None:
                    changed = True
            for occurrence in occurrences:
                if occurrence.status == "resolved" or occurrence.fingerprint in resolved_fingerprints:
                    continue
                original = self._entries.get(occurrence.fingerprint)
                if original is None:
                    self._entries[occurrence.fingerprint] = {
                        "summary": occurrence.summary,
                        "permalink": occurrence.current_permalink,
                        "channel_id": occurrence.channel_id,
                        "message_ts": occurrence.message_ts,
                        "updated_at": now,
                    }
                    changed = True
                    continue
                original["updated_at"] = now
                duplicates.append(DuplicateAlert(
                    fingerprint=occurrence.fingerprint,
                    summary=str(original.get("summary") or occurrence.summary),
                    original_permalink=str(original.get("permalink") or occurrence.current_permalink),
                    current_permalink=occurrence.current_permalink,
                ))
                changed = True
            if changed:
                self._save()
        return duplicates

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}

    def _expire(self, now: float) -> bool:
        expired = [
            key for key, value in self._entries.items()
            if now - float(value.get("updated_at", 0)) > self.ttl_seconds
        ]
        for key in expired:
            self._entries.pop(key, None)
        return bool(expired)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix="slack-alerts-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self._entries, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def default_store_path() -> Path:
    root = Path(os.environ.get("HERMES_HOME") or os.environ.get("HOME") or ".")
    return root / "state" / "slack-alert-fingerprints.json"
