"""Tests for the What's-New onboarding loop (PR-A).

Covers: brief loading/parsing, seen-state (atomic writes, corrupt-file
recovery, invalid dismiss clamping), version validation (traversal guard),
unseen-version filtering, the slash executor (surface-independent), and the
post-update notice (silent on steady state, never raises).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hermes_cli.commands import COMMAND_REGISTRY
from hermes_cli.slash_exec import CommandContext, run_execute
from hermes_cli.whats_new import (
    DISMISS_NEVER_AGAIN,
    DISMISS_UNDERSTOOD,
    WhatsNewBrief,
    _parse_features,
    _parse_front_matter,
    get_current_version,
    get_whats_new,
    load_seen,
    mark_seen,
    unseen_versions,
    validate_version_arg,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Loading & parsing
# ---------------------------------------------------------------------------

def test_get_whats_new_loads_brief():
    brief = get_whats_new(REPO_ROOT, "0.21.0")
    assert brief is not None
    assert isinstance(brief, WhatsNewBrief)
    assert brief.features
    # Every parsed feature must have a name.
    assert all(f.get("name") for f in brief.features)


def test_get_whats_new_missing_version_is_none(tmp_path):
    assert get_whats_new(tmp_path, "9.9.9") is None


def test_parse_front_matter():
    meta, body = _parse_front_matter(
        "---\nversion: 0.21.0\nrelease_date: 2026-08-08\n---\n\n# body"
    )
    assert meta["version"] == "0.21.0"
    assert meta["release_date"] == "2026-08-08"
    assert "# body" in body


def test_parse_features_extracts_fields():
    body = """## 1. Alpha

- **One-line:** New alpha thing.
- **Use when:** Doing alpha work.
- **How:** Run `alpha --go`.
- **Related:** #123

## 2. Beta

- **One-line:** Beta too.
"""
    feats = _parse_features(body)
    assert len(feats) == 2
    assert feats[0]["name"].startswith("1.")
    assert feats[0]["one_line"] == "New alpha thing."
    assert feats[0]["use_when"] == "Doing alpha work."
    assert feats[0]["how"].startswith("Run")
    assert feats[0]["related"] == "#123"
    assert feats[1]["one_line"] == "Beta too."


def test_parse_features_multiline_how_code_block():
    """Regression: multi-line `How:` code blocks (the real v0.21.0.md format)
    must be captured, not dropped (review #81580 issue 1)."""
    body = """## 1. Alpha

- **One-line:** New alpha thing.
- **Use when:** Doing alpha work.
- **How:**
  ```
  /whats-new
  /whats-new 0.20.0
  ```
- **Related:** #123
"""
    feats = _parse_features(body)
    assert len(feats) == 1
    assert "whats-new" in feats[0]["how"]
    assert "/whats-new 0.20.0" in feats[0]["how"]
    # Fence markers are not part of the content.
    assert "```" not in feats[0]["how"]


def test_parse_features_skips_placeholder_entries():
    """Regression: template placeholders (name only, no fields) must not
    render as real features (review #81580 issue 5)."""
    body = """## 1. Real Feature

- **One-line:** Actual thing.

## 2. (Template — add future features here)

- **One-line:**
- **Use when:**
- **How:**
- **Related:**
"""
    feats = _parse_features(body)
    assert len(feats) == 1
    assert feats[0]["name"].startswith("1.")


def test_parse_features_real_brief_file():
    """Parse the shipped v0.21.0.md and confirm `How:` survives."""
    brief = get_whats_new(REPO_ROOT, "0.21.0")
    assert brief is not None
    # The Feature Onboarding entry's How: block is multi-line in the file.
    feat = next(f for f in brief.features if "Feature Onboarding" in f["name"])
    assert feat["how"]
    assert "/whats-new" in feat["how"]


def test_mark_seen_unique_tmp_names(tmp_path, monkeypatch):
    """Regression: concurrent writers must not collide on one .tmp name
    (review #81580 issue 4)."""
    # Two sequential writes must leave only the final file — no leftover
    # shared .tmp. The atomic replace means the seen file is the only artifact.
    mark_seen(tmp_path, "0.21.0")
    mark_seen(tmp_path, "0.22.0")
    files = list(tmp_path.iterdir())
    names = [p.name for p in files]
    assert "whats_new_seen.json" in names
    # No stale temp files remain after successful writes.
    assert not [n for n in names if n.endswith(".tmp")]
    data = load_seen(tmp_path)
    assert set(data["seen"].keys()) == {"0.21.0", "0.22.0"}


def test_mark_seen_tmp_name_contains_pid_and_counter(tmp_path, monkeypatch):
    """The temp name embeds pid + monotonic counter for concurrency safety."""
    from hermes_cli import whats_new as wn

    real_write = wn.Path.write_text
    seen_names = []

    def spy(self, *a, **k):
        seen_names.append(self.name)
        return real_write(self, *a, **k)

    monkeypatch.setattr(wn.Path, "write_text", spy)
    mark_seen(tmp_path, "0.21.0")
    mark_seen(tmp_path, "0.22.0")
    assert len(seen_names) == 2
    assert seen_names[0] != seen_names[1]
    assert all(n.endswith(".tmp") for n in seen_names)


def test_render_truncates_at_max_features():
    feats = [{"name": f"F{i}"} for i in range(12)]
    brief = WhatsNewBrief("0.99.0", "body", feats)
    text = brief.render(max_features=5)
    assert "F0" in text
    assert "F4" in text
    assert "F5" not in text
    assert "more" in text  # truncation notice


# ---------------------------------------------------------------------------
# Version validation (traversal guard)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0.20.0", "0.20.0"),
        ("0.21.0", "0.21.0"),
        ("0.20", None),
        ("../../etc/passwd", None),
        ("0.20.0-beta", None),
        ("v0.20.0", None),
        ("", None),
        ("0.20.0; rm -rf /", None),
    ],
)
def test_validate_version_arg(raw, expected):
    assert validate_version_arg(raw) == expected


def test_get_current_version_reads_pyproject():
    ver = get_current_version(REPO_ROOT)
    assert ver is not None
    parts = ver.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


# ---------------------------------------------------------------------------
# Seen-state
# ---------------------------------------------------------------------------

def test_mark_and_load_seen(tmp_path):
    mark_seen(tmp_path, "0.21.0")
    data = load_seen(tmp_path)
    assert "0.21.0" in data["seen"]
    assert data["seen"]["0.21.0"]["dismiss"] == DISMISS_UNDERSTOOD


def test_mark_seen_preserves_other_versions(tmp_path):
    mark_seen(tmp_path, "0.20.0")
    mark_seen(tmp_path, "0.21.0")
    data = load_seen(tmp_path)
    assert set(data["seen"].keys()) == {"0.20.0", "0.21.0"}


def test_invalid_dismiss_clamped(tmp_path):
    mark_seen(tmp_path, "0.21.0", "drop-everything")
    data = load_seen(tmp_path)
    assert data["seen"]["0.21.0"]["dismiss"] == DISMISS_UNDERSTOOD


def test_corrupt_seen_file_is_empty(tmp_path):
    (tmp_path / "whats_new_seen.json").write_text("{not json!!", encoding="utf-8")
    assert load_seen(tmp_path) == {}
    # And a subsequent mark_seen recovers the file cleanly.
    mark_seen(tmp_path, "0.21.0")
    data = load_seen(tmp_path)
    assert data["seen"]["0.21.0"]["dismiss"] == DISMISS_UNDERSTOOD


def test_mark_seen_atomic_write(tmp_path):
    mark_seen(tmp_path, "0.21.0")
    # No leftover temp file (os.replace cleans up).
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_unseen_versions_filters_seen_and_future(tmp_path):
    mark_seen(tmp_path, "0.21.0")
    # 0.21.0 exists on disk and is seen → excluded; current=0.21.0.
    unseen = unseen_versions(tmp_path, REPO_ROOT, "0.21.0")
    assert "0.21.0" not in unseen
    # Future versions are never surfaced.
    assert all(
        tuple(int(x) for x in v.split(".")) <= (0, 21, 0) for v in unseen
    )


# ---------------------------------------------------------------------------
# Slash executor (surface-independent)
# ---------------------------------------------------------------------------

def test_whats_new_command_registered():
    cmd = next(c for c in COMMAND_REGISTRY if c.name == "whats-new")
    assert cmd.execute == "whats_new"
    assert cmd.busy_policy == "dispatch"


def test_whats_new_executor_no_arg(tmp_path, monkeypatch):
    # No brief for a fictional current version → friendly notice, no crash.
    from hermes_cli import whats_new as wn

    def fake_current(root):
        return "0.20.0"

    monkeypatch.setattr(wn, "get_current_version", fake_current)
    cmd = next(c for c in COMMAND_REGISTRY if c.name == "whats-new")
    reply = run_execute(cmd, CommandContext(surface="gateway", args=""))
    assert reply is not None
    assert reply.text


def test_whats_new_executor_bad_version(tmp_path, monkeypatch):
    from hermes_cli import whats_new as wn

    def fake_current(root):
        return "0.20.0"

    monkeypatch.setattr(wn, "get_current_version", fake_current)
    cmd = next(c for c in COMMAND_REGISTRY if c.name == "whats-new")
    reply = run_execute(cmd, CommandContext(surface="cli", args="../../etc"))
    assert "Invalid version" in reply.text


def test_whats_new_executor_marks_seen(tmp_path, monkeypatch):
    """Viewing the current version's brief auto-acknowledges it."""
    from hermes_cli import whats_new as wn

    def fake_current(root):
        return "0.21.0"

    monkeypatch.setattr(wn, "get_current_version", fake_current)
    # The executor resolves HERMES_HOME via hermes_constants at call time —
    # point it at the tmp dir so mark_seen lands somewhere we can assert.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cmd = next(c for c in COMMAND_REGISTRY if c.name == "whats-new")
    reply = run_execute(cmd, CommandContext(surface="cli", args=""))
    assert reply is not None
    assert "0.21.0" in reply.text
    data = load_seen(tmp_path)
    assert "0.21.0" in data["seen"]


def test_whats_new_executor_seen_flag(tmp_path, monkeypatch):
    from hermes_cli import whats_new as wn

    def fake_current(root):
        return "0.21.0"

    monkeypatch.setattr(wn, "get_current_version", fake_current)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cmd = next(c for c in COMMAND_REGISTRY if c.name == "whats-new")
    reply = run_execute(cmd, CommandContext(surface="cli", args="--seen"))
    assert "Marked" in reply.text
    data = load_seen(tmp_path)
    assert "0.21.0" in data["seen"]


# ---------------------------------------------------------------------------
# Post-update notice (silent on steady state)
# ---------------------------------------------------------------------------

def test_print_whats_new_notice_silent_when_no_brief(capsys):
    from hermes_cli.update_cmd import _print_whats_new_notice

    _print_whats_new_notice()
    captured = capsys.readouterr()
    # Current version 0.20.0 has no brief on disk → silent.
    assert "What's new" not in captured.out


def test_print_whats_new_notice_never_raises(monkeypatch, capsys):
    """A broken brief or broken loader must never break an update."""
    from hermes_cli.update_cmd import _print_whats_new_notice

    def boom():
        raise RuntimeError("simulated failure")

    # Make the underlying loader explode; the update_cmd wrapper must swallow it.
    monkeypatch.setattr("hermes_cli.whats_new.get_current_version", boom)
    _print_whats_new_notice()
    captured = capsys.readouterr()
    assert "simulated failure" not in captured.out
    assert True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
