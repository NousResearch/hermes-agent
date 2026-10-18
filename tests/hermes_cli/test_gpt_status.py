from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from hermes_cli.gpt_status import build_segment, is_gpt_route


def test_gpt_route_is_strictly_subscription_backed():
    assert is_gpt_route("openai-codex", "gpt-5.5")
    assert is_gpt_route("openai-codex", "gpt-5.3-codex")
    assert not is_gpt_route("deepseek", "gpt-5.5")
    assert not is_gpt_route("openai-codex", "o3-mini")
    assert not is_gpt_route("zai", "glm-4.5")


def test_build_segment_renders_windows_and_reset_countdowns():
    now = datetime.now(timezone.utc)
    snapshot = SimpleNamespace(
        available=True,
        plan="Plus",
        windows=(
            SimpleNamespace(label="Session", used_percent=21, reset_at=now + timedelta(hours=2, minutes=5)),
            SimpleNamespace(label="Weekly", used_percent=4, reset_at=now + timedelta(days=6)),
        ),
    )
    text = "".join(piece for _kind, piece in build_segment(snapshot, 120))
    assert "GPT Plus" in text
    assert "5h 21%" in text
    assert "wk 4%" in text
    assert "2h" in text


def test_build_segment_handles_weekly_only_plan():
    """ProLite-style plans return only a weekly primary window (secondary null)."""
    now = datetime.now(timezone.utc)
    snapshot = SimpleNamespace(
        available=True,
        plan="Prolite",
        windows=(
            SimpleNamespace(label="Weekly", used_percent=76, reset_at=now + timedelta(days=1, hours=5)),
        ),
    )
    text = "".join(piece for _kind, piece in build_segment(snapshot, 120))
    assert "GPT Prolite" in text
    assert "wk 76%" in text
    assert "5h" not in text.split("Prolite", 1)[1]


def test_build_segment_hides_unavailable_or_narrow():
    assert build_segment(None, 120) == []
    assert build_segment(SimpleNamespace(available=False), 120) == []
    assert build_segment(SimpleNamespace(available=True), 51) == []
