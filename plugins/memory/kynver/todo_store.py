"""Kynver-owned Hermes todo store — plan progress rows + in_progress focus (not running)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from tools.todo_tool import TodoStore

from .agentos_bridge import KynverAgentOSClient, KynverAgentOSError
from .operating_config import OperatingLinkage, load_operating_linkage
from .plan_progress import (
    inspect_todo_write,
    project_todo_write,
    reconcile_todos_from_kynver,
)
from .pre_transition import PreTransitionError

logger = logging.getLogger(__name__)


class KynverTodoStore:
    """TodoStore that projects to Kynver plan progress; falls back to local on failure."""

    def __init__(
        self,
        client: KynverAgentOSClient,
        *,
        linkage: Optional[OperatingLinkage] = None,
        allow_fallback: bool = True,
        degraded: bool = False,
    ):
        self._client = client
        self._linkage = linkage or load_operating_linkage()
        self._allow_fallback = allow_fallback
        self._degraded = degraded
        self._local = TodoStore()

    @property
    def degraded(self) -> bool:
        return self._degraded

    def _mark_degraded(self, reason: str) -> None:
        self._degraded = True
        logger.warning("Kynver todo store degraded to local fallback: %s", reason)

    def read(self) -> List[Dict[str, str]]:
        if self._degraded or not self._linkage.plan_id:
            return self._local.read()
        try:
            return reconcile_todos_from_kynver(
                self._client,
                self._linkage,
                self._local.read(),
            )
        except KynverAgentOSError as exc:
            if not self._allow_fallback:
                raise
            self._mark_degraded(str(exc))
            return self._local.read()
        except Exception as exc:
            if not self._allow_fallback:
                raise KynverAgentOSError(str(exc)) from exc
            self._mark_degraded(str(exc))
            return self._local.read()

    def write(self, todos: List[Dict[str, Any]], merge: bool = False) -> List[Dict[str, str]]:
        local_items = self._local.write(todos, merge=merge)

        if self._degraded or not self._linkage.plan_id:
            return local_items

        try:
            blocked = inspect_todo_write(
                self._client,
                self._linkage,
                list(todos),
                merge=merge,
            )
            if blocked:
                raise PreTransitionError(blocked)

            project_todo_write(
                self._client,
                self._linkage,
                list(todos),
                merge=merge,
            )
            return self.read()
        except PreTransitionError:
            raise
        except KynverAgentOSError as exc:
            if not self._allow_fallback:
                raise
            self._mark_degraded(str(exc))
            return self._local.read()
        except Exception as exc:
            if not self._allow_fallback:
                raise KynverAgentOSError(str(exc)) from exc
            self._mark_degraded(str(exc))
            return self._local.read()

    def has_items(self) -> bool:
        return bool(self.read())

    def format_for_injection(self) -> Optional[str]:
        base = TodoStore()
        base._items = self.read()  # noqa: SLF001 — reuse formatter
        text = base.format_for_injection()
        if self._degraded and text:
            return text + "\n[Kynver todo: degraded — using local fallback cache]"
        if self._degraded:
            return "[Kynver todo: degraded — local fallback; AgentOS plan progress unavailable]"
        return text
