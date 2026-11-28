from __future__ import annotations

import re
import shlex
from typing import Iterable, List, Optional

from .models import ScopeConfig
from .scope import in_scope, load_scope_from_yaml


HOST_RE = re.compile(r"^([a-z0-9][a-z0-9\-]*\.)+[a-z0-9\-]{2,}$", re.IGNORECASE)
IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")


class ViolationError(RuntimeError):
    pass


class Guardian:
    def __init__(self, scope: ScopeConfig | str):
        self.scope_config = load_scope_from_yaml(scope) if isinstance(scope, str) else scope
        self.allowed_patterns = set(self.scope_config.allowed_domains)
        self.excluded_patterns = set(self.scope_config.excluded_domains)

    def is_allowed(self, target: str) -> bool:
        return in_scope(target, self.scope_config)

    def _is_excluded(self, target: str) -> bool:
        normalized = (target or "").strip().lower().strip(".")
        for pattern in self.excluded_patterns:
            if "*" in pattern:
                suffix = pattern.replace("*.", "")
                if normalized.endswith(f".{suffix}"):
                    return True
                parts = suffix.split(".", 1)
                if len(parts) == 2 and normalized.endswith(f".{parts[1]}"):
                    host_labels = normalized.split(".")
                    parent_labels = parts[1].split(".")
                    labels_before_parent = len(host_labels) - len(parent_labels)
                    if labels_before_parent >= 2:
                        nearest_label = host_labels[-(len(parent_labels) + 1)]
                        if nearest_label == "api":
                            return True
            elif normalized == pattern or normalized.endswith(f".{pattern}"):
                return True
        return False

    def _extract_targets(self, command: str) -> List[str]:
        tokens = shlex.split(command or "")
        targets: list[str] = []
        expect_value_for = {"-d", "--domain", "-u", "--url", "-h", "--host", "--target"}

        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in expect_value_for and i + 1 < len(tokens):
                targets.append(tokens[i + 1])
                i += 2
                continue

            if token.startswith("-"):
                i += 1
                continue

            cleaned = token.strip().strip(",")
            if HOST_RE.match(cleaned) or IP_RE.match(cleaned):
                targets.append(cleaned)
            i += 1

        seen = set()
        unique_targets = []
        for target in targets:
            normalized = target.lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_targets.append(target)
        return unique_targets

    def check_command(self, command: str, context: Optional[dict] = None) -> bool:
        context = context or {}
        ctx_target = context.get("target")
        context_targets: list[str] = []

        if isinstance(ctx_target, str) and ctx_target.strip():
            context_targets.append(ctx_target.strip())
        elif isinstance(ctx_target, Iterable):
            context_targets.extend(str(t).strip() for t in ctx_target if str(t).strip())

        targets = context_targets + self._extract_targets(command)
        if not targets:
            raise ViolationError("Aucune cible détectée dans la commande (hors scope par défaut)")

        for target in targets:
            if self._is_excluded(target):
                raise ViolationError(f"Cible exclue par le scope: {target} (exclu)")
            if not self.is_allowed(target):
                raise ViolationError(f"Cible hors scope: {target} (hors scope)")

        return True
