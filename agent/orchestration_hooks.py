"""Lifecycle hooks for Hermes orchestration runs.

This is a tiny in-process hook runner used by the orchestration runtime.  It is
separate from the plugin hook bus on purpose: workflow YAML and tests need a
small deterministic surface, while production call sites can bridge results into
plugins or shell hooks later.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, DefaultDict
from collections import defaultdict

logger = logging.getLogger(__name__)
Hook = Callable[[str, dict[str, Any]], Any]


@dataclass
class HookResult:
    event: str
    ok: bool
    output: Any = None
    error: str = ""


class HookRunner:
    def __init__(self) -> None:
        self._hooks: DefaultDict[str, list[Hook]] = defaultdict(list)

    def register(self, event: str, hook: Hook) -> None:
        self._hooks[event].append(hook)

    def run(self, event: str, payload: dict[str, Any] | None = None) -> list[HookResult]:
        payload = dict(payload or {})
        results: list[HookResult] = []
        for hook in list(self._hooks.get(event, [])) + list(self._hooks.get("*", [])):
            try:
                results.append(HookResult(event=event, ok=True, output=hook(event, payload)))
            except Exception as exc:  # hooks are observational by default
                logger.warning("orchestration hook %s failed: %s", event, exc)
                results.append(HookResult(event=event, ok=False, error=str(exc)))
        return results


GLOBAL_HOOKS = HookRunner()


def run_lifecycle_hooks(event: str, payload: dict[str, Any] | None = None) -> list[HookResult]:
    return GLOBAL_HOOKS.run(event, payload)
