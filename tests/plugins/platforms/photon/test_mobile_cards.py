"""Mobile supervisor-card formatting for Photon/iMessage progress."""
from __future__ import annotations

import pytest

from gateway.config import PlatformConfig
from gateway.run import (
    _format_adapter_tool_progress,
    _should_drain_progress_for_no_edit_adapter,
)
from gateway.stream_events import ToolCallChunk
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch, *, mobile_cards=True) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(
        enabled=True,
        token="",
        extra={"mobile_cards": mobile_cards},
    )
    return PhotonAdapter(cfg)


def test_tool_progress_renders_compact_supervisor_card(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)

    rendered = adapter.format_tool_event(
        ToolCallChunk(
            tool_name="mcp_google_multi_gmail_search",
            preview="checking all NYU open items and inbox state",
            index=5,
        ),
        preview_max_len=80,
    )

    assert rendered is not None
    assert "**Hermes status**" in rendered
    assert "Working — iteration 6" in rendered
    assert "Tool: `mcp_google_multi_gmail_search`" in rendered
    assert "Current: checking all NYU open items" in rendered
    assert "Boundary: status only" in rendered


def test_tool_progress_preview_is_single_line_and_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)

    rendered = adapter.format_tool_event(
        ToolCallChunk(
            tool_name="terminal",
            preview="first line\nsecond line with extra detail",
            index=0,
        ),
        preview_max_len=18,
    )

    assert rendered is not None
    assert "first line second…" in rendered
    assert "\nsecond line" not in rendered


def test_active_gateway_progress_path_uses_photon_mobile_card_formatter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)

    rendered = _format_adapter_tool_progress(
        adapter,
        tool_name="mcp_google_multi_gmail_search",
        preview="checking all NYU open items and inbox state",
        args={"query": "NYU"},
        mode="all",
        preview_max_len=80,
        index=5,
    )

    assert rendered is not None
    assert "**Hermes status**" in rendered
    assert "Working — iteration 6" in rendered
    assert "Tool: `mcp_google_multi_gmail_search`" in rendered
    assert "Current: checking all NYU open items" in rendered
    assert "Boundary: status only" in rendered


def test_no_edit_progress_guard_allows_photon_cards_without_opening_all_no_edit_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)

    assert _should_drain_progress_for_no_edit_adapter(adapter) is False


def test_mobile_cards_can_fall_back_to_base_tool_chrome(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch, mobile_cards=False)

    rendered = adapter.format_tool_event(
        ToolCallChunk(tool_name="terminal", preview="date", index=0),
        preview_max_len=40,
    )

    assert rendered is not None
    assert "terminal" in rendered
    assert "Hermes status" not in rendered
    assert "Boundary: status only" not in rendered
    assert _should_drain_progress_for_no_edit_adapter(adapter) is True


def test_mobile_cards_can_be_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    monkeypatch.setenv("PHOTON_MOBILE_CARDS", "false")
    adapter = PhotonAdapter(PlatformConfig(enabled=True, token="", extra={}))

    rendered = adapter.format_tool_event(
        ToolCallChunk(tool_name="terminal", preview="date", index=0),
        preview_max_len=40,
    )

    assert rendered is not None
    assert "terminal" in rendered
    assert "Hermes status" not in rendered
    assert "Boundary: status only" not in rendered
    assert _should_drain_progress_for_no_edit_adapter(adapter) is True
