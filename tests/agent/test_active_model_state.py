"""Tests for the fallback-aware "which model is actually answering" snapshot.

The dashboard badge is built from the configured model, so a runtime fallback is
invisible unless the agent records the route it flipped to. These tests pin that
bookkeeping: the record/read round-trip, the write-elision rules (one turn in a
fallback storm must not rewrite the file, but a long fallback must not expire
either), the staleness horizon, and that nothing here can ever raise into a turn.

Real imports against a temp ``HERMES_HOME`` — the same rubric the other
resolution-chain tests (fallback, credential pool) follow.
"""

import json
import os
import time
from pathlib import Path

import pytest

from agent.active_model_state import (
    DEFAULT_MAX_AGE_SECONDS,
    active_model_path,
    clear_active_model,
    read_active_model,
    record_active_model,
    record_agent_model,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A profile home, exported so the ambient resolution path is exercised too."""
    path = tmp_path / "hermes"
    path.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(path))
    return path


class _Agent:
    """Stand-in for the attributes the record helper reads off a real agent."""

    def __init__(self, model="gpt-5", provider="openai", fallback=False):
        self.model = model
        self.provider = provider
        self._provider_fallback_active = fallback


def test_records_the_route_that_is_serving_turns(home):
    record_active_model("claude-sonnet-4", "anthropic", fallback=True)

    snapshot = read_active_model()
    assert snapshot["model"] == "claude-sonnet-4"
    assert snapshot["provider"] == "anthropic"
    assert snapshot["fallback"] is True
    assert snapshot["at"] == pytest.approx(time.time(), abs=30)


def test_record_agent_model_reads_the_agent_state(home):
    record_agent_model(_Agent("backup-model", "custom", fallback=True))

    snapshot = read_active_model()
    assert (snapshot["model"], snapshot["provider"], snapshot["fallback"]) == (
        "backup-model", "custom", True,
    )


def test_primary_and_fallback_routes_are_distinguished(home):
    record_agent_model(_Agent("primary", "p", fallback=False))
    assert read_active_model()["fallback"] is False

    record_agent_model(_Agent("backup", "p", fallback=True))
    snapshot = read_active_model()
    assert (snapshot["model"], snapshot["fallback"]) == ("backup", True)


def test_snapshot_lives_in_the_profile_home(home):
    record_active_model("m", "p")
    assert active_model_path() == home / "active_model.json"
    assert json.loads((home / "active_model.json").read_text(encoding="utf-8"))["model"] == "m"


def test_same_route_is_not_rewritten_every_turn(home):
    """A fallback that keeps re-activating must not rewrite the file per event."""
    now = 1_000_000.0
    record_active_model("backup", "p", fallback=True, now=now)
    path = home / "active_model.json"
    before = path.stat().st_mtime_ns

    record_active_model("backup", "p", fallback=True, now=now + 60)

    assert path.stat().st_mtime_ns == before
    assert read_active_model(now=now + 60)["at"] == now


def test_long_fallback_stays_fresh(home):
    """A fallback outliving half the horizon refreshes, else the badge would drop it."""
    now = 2_000_000.0
    record_active_model("backup", "p", fallback=True, now=now)

    later = now + DEFAULT_MAX_AGE_SECONDS / 2
    record_active_model("backup", "p", fallback=True, now=later)

    assert read_active_model(now=later + 10)["at"] == later
    assert read_active_model(now=later + 10)["fallback"] is True


def test_stale_snapshot_reads_as_unknown(home):
    record_active_model("backup", "p", fallback=True, now=1_000.0)

    assert read_active_model(now=1_000.0 + DEFAULT_MAX_AGE_SECONDS + 1) == {}
    # Explicit horizon, same rule.
    assert read_active_model(max_age_seconds=5, now=1_006.0) == {}


def test_missing_or_corrupt_state_is_unknown_not_an_error(home):
    assert read_active_model() == {}

    (home / "active_model.json").write_text("{not json", encoding="utf-8")
    assert read_active_model() == {}

    (home / "active_model.json").write_text('["a list"]', encoding="utf-8")
    assert read_active_model() == {}

    (home / "active_model.json").write_text('{"at": 99999999999.0}', encoding="utf-8")
    assert read_active_model() == {}


def test_empty_model_is_not_recorded(home):
    record_active_model("", "p")
    record_active_model(None, "p")
    record_active_model("   ", "p")

    assert not (home / "active_model.json").exists()
    assert read_active_model() == {}


def test_clear_removes_the_snapshot(home):
    record_active_model("m", "p")
    clear_active_model()

    assert not (home / "active_model.json").exists()
    clear_active_model()  # idempotent


def test_unwritable_home_never_raises(tmp_path):
    """Bookkeeping is best-effort: a broken home must not fail the turn."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")

    record_active_model("m", "p", home=blocker)  # no exception
    assert read_active_model(home=blocker) == {}


def test_route_helper_surfaces_the_fallback(tmp_path, monkeypatch):
    """`/api/model/info` adds the active route only when a snapshot is known."""
    from hermes_cli.web_routers.models import active_model_fields

    home = tmp_path / "reachable"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert active_model_fields() == {}

    record_active_model("backup", "custom", fallback=True)
    assert active_model_fields() == {
        "active_model": "backup", "active_model_provider": "custom", "fallback_active": True,
    }

    record_active_model("primary", "custom", fallback=False)
    assert active_model_fields() == {
        "active_model": "primary", "active_model_provider": "custom", "fallback_active": False,
    }


def test_snapshot_is_not_world_readable(home):
    record_active_model("m", "p")
    mode = (home / "active_model.json").stat().st_mode & 0o777
    if os.name != "nt":
        assert mode == 0o600


def test_path_object_home_is_accepted(tmp_path):
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    record_active_model("m", "p", home=Path(explicit))

    assert read_active_model(home=str(explicit))["model"] == "m"
