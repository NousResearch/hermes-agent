"""Typed webhook route configuration with legacy-shape compatibility."""

from __future__ import annotations

import re
from typing import Any, Literal, Mapping
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

_ROUTE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class WebhookRouteConfig(BaseModel):
    """The canonical, profile-scoped webhook route contract."""

    # Unknown fields are retained so legacy/provider metadata survives a
    # canonical read/write migration.
    model_config = ConfigDict(extra="allow")

    name: str
    enabled: bool = True
    description: str = ""
    profile: str = "default"
    events: list[str] = Field(default_factory=list)
    signature_mode: Literal[
        "github", "gitlab", "svix", "generic_v2", "generic_v1"
    ] = "generic_v2"
    secret_ref: str
    prompt: str = ""
    skills: list[str] = Field(default_factory=list)
    script: str | None = None
    filters: list[dict] = Field(default_factory=list)
    model: str | None = None
    session_mode: Literal["event", "thread", "keyed"] = "event"
    session_key_template: str | None = None
    approval_mode: Literal["deny", "delivery_target"] = "deny"
    response_mode: Literal["accepted", "wait", "callback"] = "accepted"
    callback: dict | None = None
    deliveries: list[dict] = Field(default_factory=list)
    deliver_only: bool = False
    completion_script: str | None = None
    _legacy_secret_present: bool = PrivateAttr(default=True)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not _ROUTE_NAME_RE.fullmatch(value):
            raise ValueError(
                "route name must use lowercase letters, digits, hyphens, and underscores"
            )
        return value

    @field_validator("secret_ref")
    @classmethod
    def validate_secret_ref(cls, value: str) -> str:
        # An empty value is retained for legacy routes that intentionally
        # inherit the adapter's global secret at runtime.
        if not isinstance(value, str):
            raise ValueError("secret_ref must be a string")
        return value

    @field_validator("profile")
    @classmethod
    def validate_profile(cls, value: str) -> str:
        value = value.strip()
        if not value or not _ROUTE_NAME_RE.fullmatch(value):
            raise ValueError("profile must be a valid profile name")
        return value

    @model_validator(mode="after")
    def validate_policy_combinations(self) -> "WebhookRouteConfig":
        if self.deliver_only and not self.deliveries:
            raise ValueError("deliver_only routes require at least one delivery target")
        if self.deliver_only and self.response_mode != "accepted":
            raise ValueError("deliver_only routes must use response_mode='accepted'")
        if self.response_mode == "callback":
            if not self.callback or not isinstance(self.callback.get("url"), str):
                raise ValueError("callback response_mode requires callback.url")
            parsed = urlparse(self.callback["url"])
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("callback.url must be an absolute http(s) URL")
        elif self.callback is not None:
            raise ValueError("callback is only valid with response_mode='callback'")
        if self.session_mode == "keyed" and not self.session_key_template:
            raise ValueError("keyed session_mode requires session_key_template")
        if self.session_mode != "keyed" and self.session_key_template is not None:
            raise ValueError(
                "session_key_template is only valid with session_mode='keyed'"
            )
        if self.approval_mode == "delivery_target" and not self.deliveries:
            raise ValueError(
                "approval_mode='delivery_target' requires a delivery target"
            )
        return self


def _legacy_delivery(route: Mapping[str, Any]) -> list[dict]:
    """Translate the legacy singular delivery pair to canonical deliveries."""
    target = route.get("deliver")
    extra = route.get("deliver_extra")
    if target is None and not extra:
        return []
    delivery: dict[str, Any] = {}
    if target is not None:
        delivery["target"] = target
    if isinstance(extra, Mapping):
        delivery.update(extra)
    elif extra is not None:
        delivery["extra"] = extra
    return [delivery]


def from_legacy_route(
    name: str, route: Mapping[str, Any], *, profile: str = "default"
) -> WebhookRouteConfig:
    """Load an old CLI/config route without discarding legacy information.

    ``secret``, ``deliver``, and ``deliver_extra`` are represented by the
    canonical ``secret_ref`` and ``deliveries`` fields. Other legacy keys not
    in the canonical contract are retained as Pydantic extras, which keeps
    metadata such as ``created_at`` lossless during a read/write migration.
    """
    raw = dict(route)
    canonical = {
        "name": raw.pop("name", name),
        "profile": raw.pop("profile", profile),
        "secret_ref": raw.pop("secret_ref", raw.pop("secret", "")),
        "deliveries": raw.pop("deliveries", _legacy_delivery(route)),
    }
    for field in (
        "enabled", "description", "events", "signature_mode", "prompt", "skills",
        "script", "filters", "model", "session_mode", "session_key_template",
        "approval_mode", "response_mode", "callback", "deliver_only",
        "completion_script",
    ):
        if field in raw:
            canonical[field] = raw.pop(field)
    # Preserve legacy data that has no canonical field (created_at, future CLI
    # fields, and provider-specific metadata) as model extras.
    canonical.update(raw)
    result = WebhookRouteConfig.model_validate(canonical)
    result._legacy_secret_present = "secret" in route or "secret_ref" in route
    return result


def to_legacy_route(route: WebhookRouteConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Project a canonical route to the exact dict shape used by old callers."""
    model = route if isinstance(route, WebhookRouteConfig) else WebhookRouteConfig.model_validate(route)
    data = model.model_dump(exclude={"name", "profile", "secret_ref", "deliveries"})
    if model.secret_ref or model._legacy_secret_present:
        data["secret"] = model.secret_ref
    data["profile"] = model.profile
    if model.deliveries:
        first = dict(model.deliveries[0])
        data["deliver"] = first.pop("target", "log")
        data["deliver_extra"] = first
    else:
        data.setdefault("deliver", "log")
        data.setdefault("deliver_extra", {})
    # Canonical-only fields are harmless to new consumers but old webhook
    # code should continue to see its familiar singular delivery keys.
    return data


# Friendly aliases for callers that prefer loader/serializer terminology.
load_legacy_route = from_legacy_route
serialize_legacy_route = to_legacy_route

__all__ = [
    "WebhookRouteConfig",
    "from_legacy_route",
    "load_legacy_route",
    "to_legacy_route",
    "serialize_legacy_route",
]
