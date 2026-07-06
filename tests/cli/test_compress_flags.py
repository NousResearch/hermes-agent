"""Tests for /compress --preview/--dry-run/--aggressive flags and the
/compact alias (PR #3243 salvage).

Covers the pure helpers in ``hermes_cli.partial_compress`` plus alias
resolution in the command registry. The CLI and gateway surfaces both
route through these helpers, so the flag semantics are pinned here once.
"""

from hermes_cli.commands import COMMANDS, resolve_command
from hermes_cli.partial_compress import (
    DEFAULT_KEEP_LAST,
    extract_compress_flags,
    parse_partial_compress_args,
    summarize_compress_preview,
)


def _history(n_pairs: int) -> list[dict[str, str]]:
    h: list[dict[str, str]] = []
    for i in range(n_pairs):
        h.append({"role": "user", "content": f"u{i}"})
        h.append({"role": "assistant", "content": f"a{i}"})
    return h


# ── /compact alias resolution ─────────────────────────────────────────


def test_compact_resolves_to_compress():
    cmd = resolve_command("compact")
    assert cmd is not None
    assert cmd.name == "compress"
    assert "compact" in cmd.aliases


def test_compact_alias_with_slash():
    cmd = resolve_command("/compact")
    assert cmd is not None and cmd.name == "compress"


def test_compact_listed_in_flat_commands():
    assert "/compact" in COMMANDS
    assert "alias for /compress" in COMMANDS["/compact"]


def test_compress_args_hint_documents_preview():
    cmd = resolve_command("compress")
    assert cmd is not None
    assert "--preview" in (cmd.args_hint or "")


# ── extract_compress_flags ────────────────────────────────────────────


def test_no_flags_passthrough():
    rest, preview, aggressive = extract_compress_flags("here 3")
    assert rest == "here 3"
    assert preview is False
    assert aggressive is False


def test_preview_flag_stripped():
    rest, preview, aggressive = extract_compress_flags("--preview")
    assert rest == ""
    assert preview is True
    assert aggressive is False


def test_dry_run_is_preview():
    for form in ("--dry-run", "--dryrun", "--DRY-RUN"):
        _, preview, _ = extract_compress_flags(form)
        assert preview is True, form


def test_aggressive_flag_detected():
    rest, preview, aggressive = extract_compress_flags("--aggressive")
    assert rest == ""
    assert preview is False
    assert aggressive is True


def test_flags_coexist_with_here_form():
    rest, preview, aggressive = extract_compress_flags("--preview here 4")
    assert rest == "here 4"
    assert preview is True
    partial, keep, focus = parse_partial_compress_args(rest)
    assert partial is True and keep == 4 and focus is None


def test_flags_coexist_with_focus_topic():
    rest, preview, _ = extract_compress_flags("database schema --dry-run")
    assert rest == "database schema"
    assert preview is True
    partial, _, focus = parse_partial_compress_args(rest)
    assert partial is False and focus == "database schema"


def test_aggressive_dry_run_combo():
    rest, preview, aggressive = extract_compress_flags("--aggressive --dry-run")
    assert rest == ""
    assert preview is True and aggressive is True


def test_empty_args():
    rest, preview, aggressive = extract_compress_flags("")
    assert rest == "" and preview is False and aggressive is False


# ── summarize_compress_preview ────────────────────────────────────────


def test_preview_full_compress_counts():
    hist = _history(5)
    report = summarize_compress_preview(hist, False, DEFAULT_KEEP_LAST, None, 1234)
    assert report["head_count"] == 10
    assert report["tail_count"] == 0
    assert report["total"] == 10
    assert report["partial"] is False
    joined = "\n".join(report["lines"])
    assert "no changes made" in joined.lower()
    assert "10 of 10" in joined
    assert "1,234" in joined


def test_preview_partial_boundary_counts():
    hist = _history(5)
    report = summarize_compress_preview(hist, True, 2, None, 999)
    # Keeping last 2 exchanges = 4 tail messages, 6 head messages.
    assert report["head_count"] == 6
    assert report["tail_count"] == 4
    assert report["partial"] is True
    joined = "\n".join(report["lines"])
    assert "last 2 exchange" in joined


def test_preview_partial_degenerate_falls_back_to_full():
    hist = _history(2)  # keep_last=5 would swallow everything
    report = summarize_compress_preview(hist, True, 5, None, 100)
    assert report["partial"] is False
    assert report["head_count"] == 4
    joined = "\n".join(report["lines"])
    assert "falling back to full compression" in joined


def test_preview_includes_focus_topic():
    hist = _history(4)
    report = summarize_compress_preview(hist, False, DEFAULT_KEEP_LAST, "db schema", 50)
    assert 'Focus topic: "db schema"' in "\n".join(report["lines"])


def test_preview_is_side_effect_free():
    hist = _history(4)
    before = [dict(m) for m in hist]
    summarize_compress_preview(hist, True, 1, None, 10)
    assert hist == before
