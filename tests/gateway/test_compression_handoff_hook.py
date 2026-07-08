from __future__ import annotations

from datetime import datetime, timezone

from gateway.compression_handoff import (
    maybe_write_compression_handoff,
    render_compression_handoff_artifact,
    should_write_compression_handoff,
)
from hermes_cli.config import DEFAULT_CONFIG


def _ctx(count: int = 2) -> dict:
    return {
        "platform": "telegram",
        "session_id": "session/with unsafe chars",
        "old_session_id": "old-session",
        "in_place": False,
        "compression_count": count,
    }


def _cfg(**handoff_overrides) -> dict:
    cfg = {
        "compression": {
            "handoff": {
                "enabled": True,
                "min_compression_count": 2,
                "notify": True,
            }
        }
    }
    cfg["compression"]["handoff"].update(handoff_overrides)
    return cfg


def test_default_config_disables_compression_handoff(tmp_path):
    assert DEFAULT_CONFIG["compression"]["handoff"]["enabled"] is False
    assert not should_write_compression_handoff(
        "session:compress",
        _ctx(99),
        DEFAULT_CONFIG,
    )
    assert maybe_write_compression_handoff(
        "session:compress",
        _ctx(99),
        DEFAULT_CONFIG,
        hermes_home=tmp_path,
    ) is None
    assert not list(tmp_path.rglob("*.md"))


def test_enabled_handoff_writes_metadata_artifact(tmp_path):
    generated_at = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)

    result = maybe_write_compression_handoff(
        "session:compress",
        _ctx(2),
        _cfg(),
        hermes_home=tmp_path,
        generated_at=generated_at,
    )

    assert result is not None
    assert result.notify is True
    assert result.path.exists()
    assert result.path.parent == tmp_path / "handoffs" / "compression"
    assert "session-with-unsafe-chars" in result.path.name
    text = result.path.read_text(encoding="utf-8")
    assert "Mutation status: GENERATED_LOCAL_ARTIFACT_ONLY" in text
    assert "Compression count: 2" in text
    assert "not** a full transcript audit" in text
    assert "handoff artifact:" in result.notice


def test_handoff_respects_threshold_and_event_type(tmp_path):
    assert maybe_write_compression_handoff(
        "session:compress",
        _ctx(1),
        _cfg(min_compression_count=2),
        hermes_home=tmp_path,
    ) is None
    assert maybe_write_compression_handoff(
        "agent:end",
        _ctx(9),
        _cfg(min_compression_count=2),
        hermes_home=tmp_path,
    ) is None
    assert not list(tmp_path.rglob("*.md"))


def test_handoff_respects_relative_output_dir_and_notify_false(tmp_path):
    result = maybe_write_compression_handoff(
        "session:compress",
        _ctx(3),
        _cfg(output_dir="custom-handoffs", notify=False),
        hermes_home=tmp_path,
        generated_at=datetime(2026, 7, 8, 12, 1, tzinfo=timezone.utc),
    )

    assert result is not None
    assert result.notify is False
    assert result.path.parent == tmp_path / "custom-handoffs"


def test_render_handles_in_place_boundary():
    text = render_compression_handoff_artifact(
        {
            "platform": "cli",
            "session_id": "same-session",
            "old_session_id": "",
            "in_place": True,
            "compression_count": 4,
        },
        generated_at=datetime(2026, 7, 8, 12, 2, tzinfo=timezone.utc),
    )

    assert "Boundary mode: in-place compaction" in text
    assert "Previous session id: `same session / unavailable`" in text
