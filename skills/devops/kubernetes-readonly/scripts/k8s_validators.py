"""Shared regex validators for kubernetes-readonly skill."""

from __future__ import annotations

import re

_RESOURCE = re.compile(r"^[a-z0-9][a-z0-9_.-]*(/[a-z0-9][a-z0-9_.-]*)?$", re.IGNORECASE)
_DNS_LABEL = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
_OBJECT_NAME = re.compile(r"^[a-z0-9]([-a-z0-9_.]*[a-z0-9])?$", re.IGNORECASE)


def check_resource(v: str) -> str:
    v = v.strip()
    if not _RESOURCE.fullmatch(v):
        raise ValueError("invalid resource expression")
    return v


def check_ns(v: str | None) -> str | None:
    if v is None:
        return None
    v = v.strip()
    if len(v) > 63 or not _DNS_LABEL.fullmatch(v):
        raise ValueError("invalid namespace")
    return v


def check_name(v: str | None) -> str | None:
    if v is None:
        return None
    v = v.strip()
    if len(v) > 253 or not _OBJECT_NAME.fullmatch(v):
        raise ValueError("invalid object name")
    return v
