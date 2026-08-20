"""Hermetic tests for the Robin optional skill.

Stdlib + pytest only; NO live network and NO package installation. Each test runs
against an isolated temporary Robin state directory and drives the bundled CLI
entry points directly.

    scripts/run_tests.sh tests/skills/test_robin_skill.py -q
    python3 -m pytest tests/skills/test_robin_skill.py -q
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import pytest

# Resolve the skill's runtime package across layouts: hermes-agent (tests/skills/ ->
# optional-skills/productivity/robin/scripts) and the standalone Robin dev repo (src/).
_HERE = Path(__file__).resolve()
_CANDIDATES = [
    _HERE.parents[2] / "optional-skills" / "productivity" / "robin" / "scripts",
    _HERE.parents[2] / "src",
]
SCRIPTS = next((c for c in _CANDIDATES if (c / "robin" / "cli.py").exists()), _CANDIDATES[0])
sys.path.insert(0, str(SCRIPTS))

from robin.cli import (  # noqa: E402
    add_main,
    doctor_main,
    entries_main,
    review_main,
    search_main,
    topics_main,
)
from robin.config import media_path, topics_path  # noqa: E402


@contextlib.contextmanager
def robin_state(tmp_path):
    """Yield a fresh, initialized Robin state directory."""
    state = tmp_path / "data" / "robin"
    (state / "topics").mkdir(parents=True)
    (state / "media").mkdir(parents=True)
    config = {
        "topics_dir": "topics",
        "media_dir": "media",
        "min_items_before_review": 1,
        "review_cooldown_days": 60,
    }
    (state / "robin-config.json").write_text(json.dumps(config), encoding="utf-8")
    (state / "robin-review-index.json").write_text(json.dumps({"items": {}}), encoding="utf-8")
    yield state


def _add(state, capsys, **flags):
    argv = ["--state-dir", str(state), "--json"]
    for key, value in flags.items():
        argv += [f"--{key.replace('_', '-')}", value]
    add_main(argv)
    return json.loads(capsys.readouterr().out)


def test_add_and_search_roundtrip(tmp_path, capsys):
    with robin_state(tmp_path) as state:
        added = _add(state, capsys, topic="AI", content="Useful note", description="Short label", tags="ai,notes")
        assert added["entry_type"] == "text"

        search_main(["--state-dir", str(state), "Useful", "--json"])
        result = json.loads(capsys.readouterr().out)
        assert result["count"] == 1
        assert result["entries"][0]["id"] == added["id"]

        # Query is positional; tag and topic filters also work.
        search_main(["--state-dir", str(state), "--tags", "ai,notes", "--json"])
        assert json.loads(capsys.readouterr().out)["count"] == 1


def test_add_source_entry(tmp_path, capsys):
    with robin_state(tmp_path) as state:
        added = _add(
            state, capsys,
            topic="Reading", content="Key takeaway", description="Article title",
            source="https://example.com",
        )
        assert added["topic"].lower() == "reading"
        assert added["entry_type"] == "text"


def test_review_surfaces_and_rate(tmp_path, capsys):
    with robin_state(tmp_path) as state:
        added = _add(state, capsys, topic="AI", content="Recall me", description="A note to resurface")

        review_main(["--state-dir", str(state), "--json"])
        surfaced = json.loads(capsys.readouterr().out)
        assert surfaced["status"] == "ok"
        assert surfaced["id"] == added["id"]

        review_main(["--state-dir", str(state), "--rate", added["id"], "5", "--json"])
        rated = json.loads(capsys.readouterr().out)
        assert rated["rating"] == 5


def test_entries_move_and_delete(tmp_path, capsys):
    with robin_state(tmp_path) as state:
        added = _add(state, capsys, topic="AI", content="Movable", description="An entry to move then delete")

        entries_main(["--state-dir", str(state), "--move", added["id"], "--topic", "New Topic", "--json"])
        capsys.readouterr()
        search_main(["--state-dir", str(state), "--topic", "New Topic", "--json"])
        assert json.loads(capsys.readouterr().out)["count"] == 1

        entries_main(["--state-dir", str(state), "--delete", added["id"], "--json"])
        capsys.readouterr()
        search_main(["--state-dir", str(state), "Movable", "--json"])
        assert json.loads(capsys.readouterr().out)["count"] == 0


def test_topics_lists_saved_topic(tmp_path, capsys):
    with robin_state(tmp_path) as state:
        _add(state, capsys, topic="AI", content="Note", description="A short description here")
        topics_main(["--state-dir", str(state), "--json"])
        out = json.loads(capsys.readouterr().out)
        assert "ai" in json.dumps(out).lower()


def test_doctor_reports_healthy(tmp_path, capsys):
    with robin_state(tmp_path) as state:
        _add(state, capsys, topic="AI", content="Note", description="A short description here")
        with pytest.raises(SystemExit) as exc:
            doctor_main(["--state-dir", str(state), "--json"])
        assert exc.value.code in (0, None)


@pytest.mark.parametrize("field, fn", [("topics_dir", topics_path), ("media_dir", media_path)])
@pytest.mark.parametrize("bad_value", ["/etc", "../escape", "topics/../../escape"])
def test_content_paths_cannot_escape_state_dir(tmp_path, field, fn, bad_value):
    with robin_state(tmp_path) as state:
        with pytest.raises(SystemExit, match="must be a relative path inside the state directory"):
            fn({field: bad_value}, str(state))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
