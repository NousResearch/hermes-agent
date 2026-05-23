from dataclasses import dataclass
from datetime import datetime, timezone

from agent.resource_policy import build_quota_resource_status


@dataclass(frozen=True)
class _UsageWindow:
    label: str
    used_percent: float | None = None
    reset_at: object = None


@dataclass(frozen=True)
class _UsageSnapshot:
    provider: str = "openai-codex"
    source: str = "usage_api"
    fetched_at: datetime = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)
    plan: str = "Pro"
    windows: tuple[_UsageWindow, ...] = ()
    unavailable_reason: str | None = None


def _snapshot(*windows: _UsageWindow) -> _UsageSnapshot:
    return _UsageSnapshot(windows=tuple(windows))


def test_build_quota_resource_status_uses_most_constrained_remaining_window():
    snapshot = _snapshot(
        _UsageWindow(label="Session", used_percent=15.0),
        _UsageWindow(label="Weekly", used_percent=92.0),
    )

    status = build_quota_resource_status(snapshot)

    assert status.provider == "openai-codex"
    assert status.ok is True
    assert status.status == "warning"
    assert status.remaining_percent == 8.0
    assert status.windows["session"].remaining_percent == 85.0
    assert status.windows["weekly"].remaining_percent == 8.0


def test_build_quota_resource_status_marks_zero_as_exhausted():
    snapshot = _snapshot(_UsageWindow(label="Weekly", used_percent=100.0))

    status = build_quota_resource_status(snapshot)

    assert status.ok is True
    assert status.status == "exhausted"
    assert status.remaining_percent == 0.0


def test_build_quota_resource_status_marks_missing_snapshot_as_degraded():
    status = build_quota_resource_status(None)

    assert status.ok is False
    assert status.status == "degraded"
    assert status.remaining_percent is None
    assert status.error == "quota unavailable"
