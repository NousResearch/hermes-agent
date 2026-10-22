"""OpenAI Codex/GPT subscription usage for the classic CLI status bar.

The Codex backend exposes account rate-limit windows through the same usage
endpoint used by ``/usage``.  This module keeps that network work off the UI
thread, polls every five minutes, and asks the CLI to repaint when data changes.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional


def is_gpt_route(provider: str, model: str) -> bool:
    """Return true only for Hermes' subscription-backed Codex/GPT route."""
    p = str(provider or "").strip().lower()
    m = str(model or "").strip().lower()
    return p == "openai-codex" and ("gpt" in m or "codex" in m)


def _countdown(reset_at) -> str:
    if not reset_at:
        return ""
    remaining = reset_at.timestamp() - time.time()
    if remaining <= 0:
        return "reset"
    hours, rem = divmod(int(remaining), 3600)
    minutes = rem // 60
    if hours >= 48:
        return f"{hours // 24}d{hours % 24:02d}h"
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"


def build_segment(snapshot, width: int) -> list[tuple[str, str]]:
    """Build status-bar fragments, using the same used-percent convention as GLM.

    The Codex backend reports whichever limit windows apply to the plan (e.g.
    ProLite carries only a weekly ``primary_window`` and ``secondary_window``
    is null), so labels come from the payload's ``label`` field rather than a
    fixed positional mapping.
    """
    if snapshot is None or width < 52 or not getattr(snapshot, "available", False):
        return []
    plan = getattr(snapshot, "plan", None)
    seg: list[tuple[str, str]] = [("dim", " │ "), ("strong", "GPT")]
    if plan and width >= 76:
        seg.append(("dim", f" {plan}"))
    window_labels = {"session": "5h", "weekly": "wk", "week": "wk"}
    windows = list(getattr(snapshot, "windows", ()) or ())
    for index, window in enumerate(windows[:2]):
        raw_label = str(getattr(window, "label", "") or "").strip().lower()
        label = window_labels.get(raw_label, "5h" if index == 0 else "wk")
        used = getattr(window, "used_percent", None)
        if index:
            seg.append(("dim", " · "))
        seg.append(("dim", f" {label} "))
        if used is None:
            seg.append(("dim", "--"))
        else:
            used_i = max(0, min(100, round(float(used))))
            kind = "crit" if used_i >= 95 else "warn" if used_i >= 80 else "good"
            seg.append((kind, f"{used_i}%"))
        reset = _countdown(getattr(window, "reset_at", None))
        if reset and width >= 76:
            seg.append(("dim", f" {reset}"))
    return seg


class UsagePoller:
    """One daemon poller whose context follows the active CLI model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._context: Optional[tuple[str, str, str, str]] = None
        self._snapshot = None
        self._refresh: Optional[Callable[[], None]] = None

    def start(
        self,
        *,
        provider: str,
        model: str,
        base_url: str = "",
        api_key: str = "",
        refresh: Optional[Callable[[], None]] = None,
    ) -> None:
        context = (provider or "", model or "", base_url or "", api_key or "")
        with self._lock:
            self._refresh = refresh or self._refresh
            if context != self._context:
                self._context = context
                self._snapshot = None
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="gpt-usage-poller", daemon=True
                )
                self._thread.start()

    def snapshot(self):
        with self._lock:
            return self._snapshot

    def clear(self) -> None:
        """Forget the previous route when the active model leaves GPT."""
        with self._lock:
            self._context = None
            self._snapshot = None

    def _run(self) -> None:
        while True:
            with self._lock:
                context = self._context
            if context and is_gpt_route(context[0], context[1]):
                try:
                    from agent.account_usage import fetch_account_usage
                    snapshot = fetch_account_usage(
                        context[0], base_url=context[2], api_key=context[3]
                    )
                except Exception:
                    snapshot = None
                with self._lock:
                    if context == self._context:
                        self._snapshot = snapshot
            for _ in range(5):
                time.sleep(60)
                callback = self._refresh
                if callback:
                    try:
                        callback()
                    except Exception:
                        pass


_poller = UsagePoller()


def get_poller() -> UsagePoller:
    return _poller
