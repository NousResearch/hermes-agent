"""Tests for `hermes curator usage` subcommand.

Verifies that:
- The command renders a header + one row per skill.
- Rows carry use/view/patch/activity counts and a provenance label.
- --sort re-orders the table correctly (numeric and timestamp variants).
- --provenance filters to a single origin.
- --limit truncates the output.
- Empty skills directory / filtered-to-nothing prints a clean "no skills" line.
- An unrecognised --sort value returns exit-code 2.
"""

from __future__ import annotations

import io
from argparse import Namespace
from contextlib import redirect_stdout

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_usage(monkeypatch, report_rows, *, sort="activity_count", provenance=None, limit=None):
    """Monkeypatch usage_report() and run _cmd_usage, returning (rc, output)."""
    import tools.skill_usage as skill_usage
    import hermes_cli.curator as curator_cli

    monkeypatch.setattr(skill_usage, "usage_report", lambda: list(report_rows))

    args = Namespace(sort=sort, provenance=provenance, limit=limit)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = curator_cli._cmd_usage(args)
    return rc, buf.getvalue()


def _make_row(name, *, use=0, view=0, patch=0, last=None, prov="agent"):
    return {
        "name": name,
        "use_count": use,
        "view_count": view,
        "patch_count": patch,
        "activity_count": use + view + patch,
        "last_activity_at": last,
        "provenance": prov,
        "_persisted": True,
    }


# ---------------------------------------------------------------------------
# Basic output structure
# ---------------------------------------------------------------------------

def test_usage_header_and_row_present(monkeypatch):
    rows = [_make_row("my-skill", use=5, view=3, patch=1, last="2026-04-01T00:00:00+00:00")]
    rc, out = _run_usage(monkeypatch, rows)

    assert rc == 0
    assert "NAME" in out
    assert "USE" in out
    assert "VIEW" in out
    assert "PATCH" in out
    assert "ACTIVITY" in out
    assert "LAST_SEEN" in out
    assert "PROVENANCE" in out
    assert "my-skill" in out
    assert "agent" in out


def test_usage_counts_appear_correctly(monkeypatch):
    rows = [_make_row("alpha", use=10, view=5, patch=2, last="2026-05-01T00:00:00+00:00")]
    rc, out = _run_usage(monkeypatch, rows)

    assert rc == 0
    # use=10, view=5, patch=2 → activity=17
    assert "   10" in out   # USE column right-aligned in 5 chars
    assert "    5" in out   # VIEW column
    assert "    2" in out   # PATCH column
    assert "      17" in out  # ACTIVITY column right-aligned in 8 chars


def test_usage_never_seen_shows_never(monkeypatch):
    """A skill with no activity should print 'never' in the LAST_SEEN column."""
    rows = [_make_row("idle-skill")]
    rc, out = _run_usage(monkeypatch, rows)

    assert rc == 0
    assert "never" in out


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def test_sort_by_activity_count_default(monkeypatch):
    rows = [
        _make_row("low",  use=1),
        _make_row("high", use=50),
        _make_row("mid",  use=10),
    ]
    rc, out = _run_usage(monkeypatch, rows)  # default sort = activity_count

    assert rc == 0
    lines = [l for l in out.splitlines() if l and not l.startswith("NAME") and not l.startswith("-")]
    assert lines[0].startswith("high")
    assert lines[1].startswith("mid")
    assert lines[2].startswith("low")


def test_sort_by_use_count(monkeypatch):
    rows = [
        _make_row("a", use=1, view=100),
        _make_row("b", use=99, view=1),
    ]
    rc, out = _run_usage(monkeypatch, rows, sort="use_count")

    assert rc == 0
    lines = [l for l in out.splitlines() if l and not l.startswith("NAME") and not l.startswith("-")]
    assert lines[0].startswith("b")   # b has higher use_count
    assert lines[1].startswith("a")


def test_sort_by_view_count(monkeypatch):
    rows = [
        _make_row("viewed", use=0, view=99),
        _make_row("used",   use=99, view=1),
    ]
    rc, out = _run_usage(monkeypatch, rows, sort="view_count")

    assert rc == 0
    lines = [l for l in out.splitlines() if l and not l.startswith("NAME") and not l.startswith("-")]
    assert lines[0].startswith("viewed")


def test_sort_by_last_activity_at(monkeypatch):
    rows = [
        _make_row("old",   last="2025-01-01T00:00:00+00:00"),
        _make_row("recent",last="2026-06-01T00:00:00+00:00"),
        _make_row("never"),  # last=None → sorts to bottom
    ]
    rc, out = _run_usage(monkeypatch, rows, sort="last_activity_at")

    assert rc == 0
    lines = [l for l in out.splitlines() if l and not l.startswith("NAME") and not l.startswith("-")]
    assert lines[0].startswith("recent")
    assert lines[1].startswith("old")
    assert lines[2].startswith("never")


# ---------------------------------------------------------------------------
# Provenance filtering
# ---------------------------------------------------------------------------

def test_provenance_filter_agent_only(monkeypatch):
    rows = [
        _make_row("agent-skill",   prov="agent"),
        _make_row("bundled-skill", prov="bundled"),
        _make_row("hub-skill",     prov="hub"),
    ]
    rc, out = _run_usage(monkeypatch, rows, provenance="agent")

    assert rc == 0
    assert "agent-skill" in out
    assert "bundled-skill" not in out
    assert "hub-skill" not in out


def test_provenance_filter_bundled(monkeypatch):
    rows = [
        _make_row("plan",  prov="bundled"),
        _make_row("custom", prov="agent"),
    ]
    rc, out = _run_usage(monkeypatch, rows, provenance="bundled")

    assert rc == 0
    assert "plan" in out
    assert "custom" not in out


def test_provenance_filter_no_match_prints_empty(monkeypatch):
    rows = [_make_row("skill", prov="agent")]
    rc, out = _run_usage(monkeypatch, rows, provenance="hub")

    assert rc == 0
    assert "no skills found" in out
    assert "provenance=hub" in out


# ---------------------------------------------------------------------------
# Limit
# ---------------------------------------------------------------------------

def test_limit_truncates_output(monkeypatch):
    rows = [_make_row(f"skill-{i}", use=i) for i in range(10, 0, -1)]
    rc, out = _run_usage(monkeypatch, rows, limit=3)

    assert rc == 0
    data_lines = [l for l in out.splitlines() if l and not l.startswith("NAME") and not l.startswith("-")]
    assert len(data_lines) == 3
    # Highest use_count (activity_count) should be first
    assert data_lines[0].startswith("skill-10")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_usage_report_prints_no_skills(monkeypatch):
    rc, out = _run_usage(monkeypatch, [])
    assert rc == 0
    assert "no skills found" in out


def test_invalid_sort_returns_exit_code_2(monkeypatch):
    """An invalid sort key that bypasses argparse should exit with code 2."""
    import tools.skill_usage as skill_usage
    import hermes_cli.curator as curator_cli

    monkeypatch.setattr(skill_usage, "usage_report", lambda: [_make_row("x")])

    args = Namespace(sort="nonexistent_field", provenance=None, limit=None)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = curator_cli._cmd_usage(args)

    assert rc == 2


def test_multiple_provenances_all_shown_by_default(monkeypatch):
    rows = [
        _make_row("a", prov="agent"),
        _make_row("b", prov="bundled"),
        _make_row("h", prov="hub"),
    ]
    rc, out = _run_usage(monkeypatch, rows)

    assert rc == 0
    assert "agent" in out
    assert "bundled" in out
    assert "hub" in out


def test_usage_single_skill_no_crash(monkeypatch):
    """Smoke test: one skill with all-zero counters should render cleanly."""
    rows = [_make_row("only-skill")]
    rc, out = _run_usage(monkeypatch, rows)
    assert rc == 0
    assert "only-skill" in out


def test_usage_cli_main_invocation(tmp_path, monkeypatch):
    """End-to-end: run via cli_main(['usage', '--limit', '2'])."""
    import importlib
    import hermes_constants
    import tools.skill_usage as skill_usage

    home = tmp_path / ".hermes"
    skills = home / "skills"
    skills.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(tmp_path.__class__, "home", classmethod(lambda cls: tmp_path))
    importlib.reload(hermes_constants)
    importlib.reload(skill_usage)

    # Write two skills to disk so usage_report() finds them
    for name in ("skill-a", "skill-b"):
        d = skills / name
        d.mkdir()
        (d / "SKILL.md").write_text(
            "---\nname: {name}\ndescription: test\nversion: 1.0.0\n---\n".format(name=name)
        )

    from hermes_cli.curator import cli_main
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main(["usage", "--limit", "2"])

    assert rc == 0
    out = buf.getvalue()
    assert "NAME" in out
