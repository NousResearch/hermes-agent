"""Operational alerts — kill switch, losses, gate spikes, failed txs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional

from hermes_trader.config import TraderConfig, TRADER_HOME_SUBDIR
from hermes_trader.memory.episodes import EpisodeStore
from hermes_trader.risk.gate import is_kill_switch_active


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def default_alerts_path() -> Path:
    return _hermes_home() / TRADER_HOME_SUBDIR / "alerts.jsonl"


@dataclass(frozen=True)
class Alert:
    kind: str
    message: str
    severity: str = "warning"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "message": self.message, "severity": self.severity}


class AlertStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or default_alerts_path()

    def append(self, alert: Alert) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            **alert.to_dict(),
        }
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False))
            handle.write("\n")

    def emit(self, alerts: List[Alert]) -> None:
        for alert in alerts:
            self.append(alert)


def _parse_ts(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def evaluate_alerts(
    config: TraderConfig,
    *,
    episode_store: Optional[EpisodeStore] = None,
    cycles_log_path: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> List[Alert]:
    """Detect alert conditions from recent trader activity."""
    current = now or datetime.now(timezone.utc)
    alerts: list[Alert] = []
    store = episode_store or EpisodeStore()

    if is_kill_switch_active():
        alerts.append(
            Alert(
                kind="kill_switch",
                message="HERMES_TRADER_KILL_SWITCH is active — all write tools halted",
                severity="critical",
            )
        )

    closed = store.list_closed_episodes(limit=20)
    losses = 0
    for ep in closed:
        if (ep.pnl_usd or 0) < 0:
            losses += 1
        else:
            break
    if losses >= config.consecutive_loss_alert_count:
        alerts.append(
            Alert(
                kind="consecutive_losses",
                message=f"{losses} consecutive losing episodes detected",
                severity="warning",
            )
        )

    failed = _recent_failed_executions(store)
    if failed:
        alerts.append(
            Alert(
                kind="failed_tx",
                message=f"{failed} recent execution failure(s) in episode ledger",
                severity="warning",
            )
        )

    rejects = _gate_rejects_in_last_hour(cycles_log_path, current)
    if rejects >= config.gate_reject_spike_threshold:
        alerts.append(
            Alert(
                kind="gate_block_spike",
                message=(
                    f"{rejects} gate rejections in the last hour "
                    f"(threshold {config.gate_reject_spike_threshold})"
                ),
                severity="warning",
            )
        )

    return alerts


def _recent_failed_executions(store: EpisodeStore) -> int:
    count = 0
    for ep in store.list_episodes(limit=50):
        execution = ep.execution or {}
        if execution.get("status") in {"error", "failed"}:
            count += 1
    return count


def _gate_rejects_in_last_hour(
    cycles_log_path: Optional[Path],
    now: datetime,
) -> int:
    if cycles_log_path is None:
        from hermes_trader.loop.audit import default_cycle_log_path

        cycles_log_path = default_cycle_log_path()
    if not cycles_log_path.is_file():
        return 0
    cutoff = now - timedelta(hours=1)
    rejects = 0
    for line in cycles_log_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = _parse_ts(str(row.get("timestamp", "")))
        if ts and ts < cutoff:
            continue
        if row.get("approved") is False or row.get("reason_code"):
            rejects += 1
    return rejects