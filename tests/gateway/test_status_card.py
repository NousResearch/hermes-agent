"""Focused tests for the compact gateway ``/status`` card."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from agent.i18n import t
from gateway import status_card
from gateway.status_card import StatusCardSnapshot


def _english(key: str, **kwargs) -> str:
    return t(key, lang="en", **kwargs)


def _snapshot(**overrides) -> StatusCardSnapshot:
    values = {
        "version": "0.15.1",
        "commit": "c6501c0",
        "gateway_uptime_seconds": 2 * 3600 + 14 * 60,
        "system_uptime_seconds": 1 * 86400 + 5 * 3600,
        "model": "gpt-5.5-1",
        "provider": "azure-foundry",
        "fallbacks": (
            "deepseek/deepseek-v4-pro",
            "deepseek/deepseek-v4-flash",
        ),
        "input_tokens": 64_000,
        "output_tokens": 3_000,
        "cache_read_tokens": 14_000,
        "cache_write_tokens": 0,
        "cost_usd": 0.12,
        "cost_status": "estimated",
        "context_tokens": 64_000,
        "context_limit": 1_000_000,
        "compactions": 3,
        "session_id": "20260602_xxx",
        "task_state": "Running",
        "active_tasks": 1,
        "queue_mode": "steer",
        "queue_depth": 0,
    }
    values.update(overrides)
    return StatusCardSnapshot(**values)


def test_format_hermes_status_card_matches_compact_messaging_shape():
    text = status_card.format_hermes_status_card(
        _snapshot(),
        translate=_english,
    )

    assert text.splitlines() == [
        "🪽 Hermes 0.15.1 (c6501c0)",
        "⏱️ **Uptime:** gateway 2h 14m · system 1d 5h",
        "🧠 **Model:** gpt-5.5-1 · **Provider:** azure-foundry",
        "🔄 **Fallbacks:** deepseek/deepseek-v4-pro, deepseek/deepseek-v4-flash",
        "🧮 **Tokens:** 64k in / 3k out · 💵 **Cost:** ~$0.12",
        "🗄️ **Cache:** 18% hit · 14k read, 0 write",
        "📚 **Context:** 64k/1.0m (6%) · 🧹 **Compactions:** 3",
        "🧵 **Session:** `20260602_xxx`",
        "📌 **Tasks:** Running · 1 active",
        "🪢 **Queue:** steer (depth 0)",
    ]
    assert "Execution" not in text
    assert "Runtime" not in text
    assert "Platforms" not in text


def test_format_hermes_status_card_omits_unknown_optional_metrics():
    text = status_card.format_hermes_status_card(
        _snapshot(
            commit=None,
            gateway_uptime_seconds=None,
            system_uptime_seconds=None,
            fallbacks=(),
            cost_usd=None,
            cost_status=None,
            context_tokens=None,
            context_limit=None,
            compactions=None,
        ),
        translate=_english,
    )

    assert "🪽 Hermes 0.15.1" in text
    assert "gateway unknown · system unknown" in text
    assert "🔄 **Fallbacks:** none" in text
    assert "🧮 **Tokens:** 64k in / 3k out" in text
    assert "Cost" not in text
    assert "Context" not in text
    assert "Compactions" not in text


def test_collect_uptime_uses_windows_safe_psutil_path_without_ps(monkeypatch):
    """The Windows path must use psutil and never invoke a ``ps`` command."""

    class _Process:
        def __init__(self, pid: int):
            assert pid == 4242

        @staticmethod
        def create_time() -> float:
            return 900.0

    fake_psutil = SimpleNamespace(
        Process=_Process,
        boot_time=lambda: 100.0,
    )
    subprocess_calls = []

    def _forbid_subprocess(*args, **kwargs):
        subprocess_calls.append((args, kwargs))
        raise AssertionError("uptime collection must not shell out")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(status_card.subprocess, "run", _forbid_subprocess)

    assert status_card.collect_uptime_seconds(pid=4242, now=1_000.0) == (100, 900)
    assert subprocess_calls == []


def test_collect_uptime_degrades_each_metric_independently():
    class _DeniedProcess:
        def __init__(self, _pid: int):
            raise PermissionError("process metadata denied")

    fake_psutil = SimpleNamespace(
        Process=_DeniedProcess,
        boot_time=lambda: 100.0,
    )

    assert status_card.collect_uptime_seconds(
        pid=4242,
        now=1_000.0,
        psutil_module=fake_psutil,
    ) == (None, 900)


def test_get_revision_passes_windows_safe_creation_flags(monkeypatch, tmp_path):
    captured = {}

    def _run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="abcdef12\n")

    monkeypatch.setattr(status_card.subprocess, "run", _run)
    monkeypatch.setattr(
        "hermes_cli._subprocess_compat.windows_hide_flags",
        lambda: 0x08000000,
    )

    assert status_card.get_hermes_revision(tmp_path) == "abcdef12"
    assert captured["argv"] == ["git", "rev-parse", "--short=8", "HEAD"]
    assert captured["kwargs"]["creationflags"] == 0x08000000
