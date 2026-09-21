"""Contract tests for the compact, text-only approval observer."""
import importlib.util
import json
from pathlib import Path
from types import ModuleType


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "approval_requester.py"
_spec = importlib.util.spec_from_file_location("approval_requester", SCRIPT)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)
assert isinstance(mod, ModuleType)


def _entry(slug: str, title: str, mdx_path: Path, *, status: str = "pending", created_at: str = "2026-07-18T08:00:00Z") -> dict:
    return {
        "slug": slug,
        "title": title,
        "stream": "ai",
        "tier": "ai",
        "status": status,
        "created_at": created_at,
        "mdx_path": str(mdx_path),
        "preview_path": "/intentionally/not/exposed.html",
    }


def _write_mdx(path: Path, approved: bool) -> None:
    path.write_text(f"---\ntitle: Test\napproved: {'true' if approved else 'false'}\n---\nBody\n")


def _write_tracker(path: Path, entries: list[dict]) -> None:
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))


def test_observe_removes_pending_rows_for_approved_mdx(tmp_path):
    approved = tmp_path / "approved.mdx"
    pending = tmp_path / "pending.mdx"
    _write_mdx(approved, True)
    _write_mdx(pending, False)
    tracker = tmp_path / "pending_approvals.jsonl"
    _write_tracker(tracker, [_entry("approved", "Approved draft", approved), _entry("pending", "Pending draft", pending)])

    message = mod.observe(tracker=tracker)

    assert "approved" not in message.lower()
    assert "pending" in message
    rows = [json.loads(line) for line in tracker.read_text().splitlines()]
    assert [row["slug"] for row in rows] == ["pending"]


def test_build_message_is_text_only_compact_stable_and_never_leaks_paths(tmp_path):
    entries = []
    for index in range(3):
        mdx = tmp_path / f"draft-{index}.mdx"
        _write_mdx(mdx, False)
        entries.append(_entry(f"slug-{index}", f"Title {index}", mdx, created_at=f"2026-07-18T0{3 - index}:00:00Z"))

    message = mod.build_message(entries, page=1, page_size=10)

    assert message.splitlines()[0] == "Blog approvals 1/1 (3 pending)"
    assert message.index("slug-2") < message.index("slug-1") < message.index("slug-0")
    assert len(message.splitlines()) <= 15
    assert "MEDIA:" not in message
    assert str(tmp_path) not in message
    assert "!approve <slug>" in message


def test_build_message_redacts_absolute_paths_from_untrusted_metadata(tmp_path):
    draft = tmp_path / "draft.mdx"
    _write_mdx(draft, False)

    message = mod.build_message([_entry("slug", "Review /tmp/private.mdx", draft)])

    assert "/tmp/private.mdx" not in message
    assert "<path>" in message


def test_observe_pages_pending_entries_with_at_most_fifteen_lines(tmp_path):
    tracker = tmp_path / "pending_approvals.jsonl"
    entries = []
    for index in range(12):
        mdx = tmp_path / f"draft-{index}.mdx"
        _write_mdx(mdx, False)
        entries.append(_entry(f"slug-{index:02}", f"Title {index}", mdx))
    _write_tracker(tracker, entries)

    first = mod.observe(tracker=tracker, page=1, page_size=10)
    second = mod.observe(tracker=tracker, page=2, page_size=10)

    assert first.splitlines()[0] == "Blog approvals 1/2 (12 pending)"
    assert second.splitlines()[0] == "Blog approvals 2/2 (12 pending)"
    assert "slug-00" in first and "slug-10" not in first
    assert "slug-10" in second
    assert len(first.splitlines()) <= 15
    assert len(second.splitlines()) <= 15


def test_observe_is_silent_when_no_pending_rows(tmp_path):
    tracker = tmp_path / "pending_approvals.jsonl"
    _write_tracker(tracker, [])

    assert mod.observe(tracker=tracker) == "[SILENT]"
