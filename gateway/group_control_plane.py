"""Shared internal models for oral group control and runtime status."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _normalize_text_list(values: list[Any] | tuple[Any, ...] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


@dataclass(frozen=True)
class NormalizedGroupControlRequest:
    """Platform-agnostic group-control mutation/request."""

    action: str
    target: str
    delivery_target: str | None = None
    daily_report_enabled: bool | None = None
    daily_report_target: str | None = None
    manual_report_target: str | None = None

    def to_tool_args(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "action": self.action,
            "target": self.target,
        }
        if self.delivery_target:
            payload["delivery_target"] = self.delivery_target
        if self.daily_report_enabled is not None:
            payload["daily_report_enabled"] = self.daily_report_enabled
        if self.daily_report_target:
            payload["daily_report_target"] = self.daily_report_target
        if self.manual_report_target:
            payload["manual_report_target"] = self.manual_report_target
        return payload


@dataclass(frozen=True)
class GroupRuntimeSnapshot:
    """Normalized runtime view for one group-like target."""

    platform_label: str
    target_label: str
    effective_mode: str
    can_reply_in_group: bool
    archive_enabled: bool
    daily_report_enabled: bool
    daily_targets: tuple[str, ...]
    manual_targets: tuple[str, ...]
    worker_names: tuple[str, ...]

    def to_status_details(self) -> dict[str, object]:
        return {
            "platform_label": self.platform_label,
            "target_label": self.target_label,
            "effective_mode": self.effective_mode,
            "can_reply_in_group": self.can_reply_in_group,
            "archive_enabled": self.archive_enabled,
            "daily_report_enabled": self.daily_report_enabled,
            "daily_targets": list(self.daily_targets),
            "manual_targets": list(self.manual_targets),
            "worker_names": list(self.worker_names),
        }


def build_group_runtime_snapshot(
    *,
    platform_label: str,
    target_label: str,
    effective_mode: str,
    archive_enabled: bool,
    daily_report_enabled: bool,
    daily_targets: list[Any] | tuple[Any, ...] | None,
    manual_targets: list[Any] | tuple[Any, ...] | None,
    worker_names: list[Any] | tuple[Any, ...] | None,
) -> GroupRuntimeSnapshot:
    normalized_mode = str(effective_mode or "default").strip() or "default"
    return GroupRuntimeSnapshot(
        platform_label=str(platform_label or "").strip(),
        target_label=str(target_label or "").strip(),
        effective_mode=normalized_mode,
        can_reply_in_group=normalized_mode not in {"collect_only", "disabled"},
        archive_enabled=bool(archive_enabled),
        daily_report_enabled=bool(daily_report_enabled),
        daily_targets=tuple(_normalize_text_list(daily_targets)),
        manual_targets=tuple(_normalize_text_list(manual_targets)),
        worker_names=tuple(_normalize_text_list(worker_names)),
    )
