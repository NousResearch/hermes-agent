from __future__ import annotations

import fnmatch
from pathlib import Path

import yaml

from .models import ScopeConfig


def _normalize_host(value: str) -> str:
    return (value or "").strip().strip(".").lower()


def load_scope_from_yaml(path: str) -> ScopeConfig:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Scope file not found: {path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "excluded" in data and "excluded_domains" not in data:
        data["excluded_domains"] = data.pop("excluded")

    return ScopeConfig(**data)


def load_scope(path: str) -> ScopeConfig:
    return load_scope_from_yaml(path)


def _match_allowed(host: str, pattern: str) -> bool:
    if "*" in pattern:
        if pattern.startswith("*."):
            suffix = pattern[2:]
            return host.endswith(f".{suffix}")
        return fnmatch.fnmatch(host, pattern)
    return host == pattern or host.endswith(f".{pattern}")


def _match_excluded(host: str, pattern: str) -> bool:
    if "*" in pattern:
        if pattern.startswith("*."):
            suffix = pattern[2:]
            if host.endswith(f".{suffix}"):
                return True
            parts = suffix.split(".", 1)
            if len(parts) == 2:
                parent = parts[1]
                if host.endswith(f".{parent}"):
                    host_labels = host.split(".")
                    parent_labels = parent.split(".")
                    labels_before_parent = len(host_labels) - len(parent_labels)
                    if labels_before_parent >= 2:
                        nearest_label = host_labels[-(len(parent_labels) + 1)]
                        return nearest_label == "api"
            return False
        return fnmatch.fnmatch(host, pattern)
    return host == pattern or host.endswith(f".{pattern}")


def in_scope(host: str, scope: ScopeConfig) -> bool:
    normalized = _normalize_host(host)
    if not normalized:
        return False

    if any(_match_excluded(normalized, pattern) for pattern in scope.excluded_domains):
        return False

    if normalized in scope.allowed_ips:
        return True

    if not scope.allowed_domains:
        return False

    return any(_match_allowed(normalized, pattern) for pattern in scope.allowed_domains)
