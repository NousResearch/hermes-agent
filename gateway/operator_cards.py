"""Platform-agnostic operator-card contract and plaintext fallback.

Operator cards carry bounded, human-readable control-surface metadata between
the gateway and platform adapters.  This module intentionally contains no
Discord (or other platform) dependency; adapters decide whether to render the
validated contract richly or use :func:`render_operator_card_text`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit


CARD_TYPES = frozenset(
    {
        "approval",
        "task_run",
        "digest",
        "deal",
        "capture",
        "ops_alert",
        "thread_header",
    }
)
SEVERITIES = frozenset({"done", "info", "needs_review", "blocked", "critical"})
ACTION_STYLES = frozenset({"primary", "secondary", "success", "danger", "link"})

_TOP_LEVEL_FIELDS = frozenset(
    {
        "kind",
        "version",
        "card_type",
        "title",
        "severity",
        "summary",
        "fields",
        "actions",
        "links",
        "state_ref",
    }
)
_ACTION_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_STATE_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")

_SEVERITY_DISPLAY = {
    "done": ("✅", "Done"),
    "info": ("🔵", "Info"),
    "needs_review": ("🟡", "Needs review"),
    "blocked": ("🔴", "Blocked"),
    "critical": ("🚨", "Critical"),
}


class OperatorCardValidationError(ValueError):
    """Raised when untrusted operator-card data violates the contract."""


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OperatorCardValidationError(f"{field_name} must be an object")
    return value


def _bounded_string(
    source: Mapping[str, Any],
    field_name: str,
    *,
    max_length: int,
) -> str:
    if field_name not in source:
        raise OperatorCardValidationError(f"missing required field: {field_name}")
    value = source[field_name]
    if not isinstance(value, str) or not value.strip():
        raise OperatorCardValidationError(f"{field_name} must be a non-empty string")
    value = value.strip()
    if len(value) > max_length:
        raise OperatorCardValidationError(
            f"{field_name} exceeds the {max_length}-character limit"
        )
    return value


def _object_list(source: Mapping[str, Any], field_name: str, *, max_items: int) -> list[Mapping[str, Any]]:
    value = source.get(field_name, [])
    if not isinstance(value, list):
        raise OperatorCardValidationError(f"{field_name} must be a list")
    if len(value) > max_items:
        raise OperatorCardValidationError(f"{field_name} exceeds the {max_items}-item limit")
    return [_mapping(item, f"{field_name}[{index}]") for index, item in enumerate(value)]


def _reject_unknown_fields(source: Mapping[str, Any], allowed: set[str] | frozenset[str], context: str) -> None:
    unknown = set(source) - set(allowed)
    if unknown:
        rendered = ", ".join(sorted(str(name) for name in unknown))
        raise OperatorCardValidationError(f"{context} has unknown fields: {rendered}")


@dataclass(frozen=True, slots=True)
class OperatorCardField:
    label: str
    value: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OperatorCardField":
        _reject_unknown_fields(payload, {"label", "value"}, "field")
        return cls(
            label=_bounded_string(payload, "label", max_length=80),
            value=_bounded_string(payload, "value", max_length=1024),
        )

    def to_mapping(self) -> dict[str, str]:
        return {"label": self.label, "value": self.value}


@dataclass(frozen=True, slots=True)
class OperatorCardAction:
    id: str
    label: str
    style: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OperatorCardAction":
        _reject_unknown_fields(payload, {"id", "label", "style"}, "action")
        action_id = _bounded_string(payload, "id", max_length=64)
        if not _ACTION_ID_RE.fullmatch(action_id):
            raise OperatorCardValidationError("action id must use lowercase letters, digits, hyphens, or underscores")
        style = _bounded_string(payload, "style", max_length=16)
        if style not in ACTION_STYLES:
            raise OperatorCardValidationError(f"unsupported action style: {style}")
        return cls(
            id=action_id,
            label=_bounded_string(payload, "label", max_length=80),
            style=style,
        )

    def to_mapping(self) -> dict[str, str]:
        return {"id": self.id, "label": self.label, "style": self.style}


@dataclass(frozen=True, slots=True)
class OperatorCardLink:
    label: str
    url: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OperatorCardLink":
        _reject_unknown_fields(payload, {"label", "url"}, "link")
        url = _bounded_string(payload, "url", max_length=2048)
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise OperatorCardValidationError("link url must be an http(s) URL without credentials")
        return cls(
            label=_bounded_string(payload, "label", max_length=80),
            url=url,
        )

    def to_mapping(self) -> dict[str, str]:
        return {"label": self.label, "url": self.url}


@dataclass(frozen=True, slots=True)
class OperatorCard:
    card_type: str
    title: str
    severity: str
    summary: str
    fields: tuple[OperatorCardField, ...]
    actions: tuple[OperatorCardAction, ...]
    links: tuple[OperatorCardLink, ...]
    state_ref: str
    kind: str = "operator_card"
    version: int = 1

    @classmethod
    def from_mapping(cls, raw_payload: Mapping[str, Any]) -> "OperatorCard":
        payload = _mapping(raw_payload, "operator_card")
        _reject_unknown_fields(payload, _TOP_LEVEL_FIELDS, "operator_card")

        if "kind" not in payload:
            raise OperatorCardValidationError("missing required field: kind")
        if payload["kind"] != "operator_card":
            raise OperatorCardValidationError("kind must be operator_card")
        if "version" not in payload:
            raise OperatorCardValidationError("missing required field: version")
        if isinstance(payload["version"], bool) or payload["version"] != 1:
            raise OperatorCardValidationError("version must be 1")

        card_type = _bounded_string(payload, "card_type", max_length=32)
        if card_type not in CARD_TYPES:
            raise OperatorCardValidationError(f"unsupported card_type: {card_type}")
        severity = _bounded_string(payload, "severity", max_length=32)
        if severity not in SEVERITIES:
            raise OperatorCardValidationError(f"unsupported severity: {severity}")

        fields = tuple(
            OperatorCardField.from_mapping(item)
            for item in _object_list(payload, "fields", max_items=12)
        )
        actions = tuple(
            OperatorCardAction.from_mapping(item)
            for item in _object_list(payload, "actions", max_items=5)
        )
        action_ids = [action.id for action in actions]
        if len(action_ids) != len(set(action_ids)):
            raise OperatorCardValidationError("action ids must be unique")
        links = tuple(
            OperatorCardLink.from_mapping(item)
            for item in _object_list(payload, "links", max_items=5)
        )

        state_ref = _bounded_string(payload, "state_ref", max_length=200)
        if not _STATE_REF_RE.fullmatch(state_ref):
            raise OperatorCardValidationError("state_ref must be an opaque identifier")

        return cls(
            card_type=card_type,
            title=_bounded_string(payload, "title", max_length=120),
            severity=severity,
            summary=_bounded_string(payload, "summary", max_length=500),
            fields=fields,
            actions=actions,
            links=links,
            state_ref=state_ref,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "version": self.version,
            "card_type": self.card_type,
            "title": self.title,
            "severity": self.severity,
            "summary": self.summary,
            "fields": [field.to_mapping() for field in self.fields],
            "actions": [action.to_mapping() for action in self.actions],
            "links": [link.to_mapping() for link in self.links],
            "state_ref": self.state_ref,
        }


def render_operator_card_text(card: OperatorCard, *, max_length: int = 2000) -> str:
    """Render a compact Markdown fallback shared by non-rich platforms."""
    emoji, severity_label = _SEVERITY_DISPLAY[card.severity]
    lines = [f"{emoji} **{severity_label} — {card.title}**", card.summary]
    lines.extend(f"{field.label}: {field.value}" for field in card.fields)
    if card.actions:
        lines.append("Actions: " + " · ".join(action.label for action in card.actions))
    if card.links:
        links = " · ".join(f"[{link.label}]({link.url})" for link in card.links)
        lines.append(f"Links: {links}")

    rendered = "\n".join(lines)
    if len(rendered) <= max_length:
        return rendered
    if max_length <= 0:
        return ""
    if max_length == 1:
        return "…"
    return rendered[: max_length - 1].rstrip() + "…"
