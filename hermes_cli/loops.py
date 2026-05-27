"""Fruit-Loop state helpers for the Hermes /loop command.

V1 intentionally mirrors Aaron Prins' Claude Loop mechanics: a small durable
state folder, a PRD, a progress log, status, and archive support. Execution
routing comes later; these helpers stay deterministic and file-backed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from typing import Any

DEFAULT_SLUG = "fruit-loop"


@dataclass(frozen=True)
class LoopResult:
    slug: str
    path: Path
    text: str


def slugify(value: str | None) -> str:
    raw = (value or DEFAULT_SLUG).strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")[:64]
    return slug or DEFAULT_SLUG


def loop_dir(slug: str | None = None, *, root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else Path.cwd()
    return base / ".hermes" / "loops" / slugify(slug)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _empty_prd(slug: str, title: str | None = None) -> dict[str, Any]:
    return {
        "project": title or slug,
        "slug": slug,
        "status": "draft",
        "branchName": "",
        "description": "",
        "userStories": [],
        "createdAt": _now(),
        "updatedAt": _now(),
    }


def _load_prd(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _counts(prd: dict[str, Any]) -> tuple[int, int, int]:
    stories = prd.get("userStories")
    if not isinstance(stories, list):
        return 0, 0, 0
    total = len(stories)
    passed = sum(1 for story in stories if isinstance(story, dict) and story.get("passes") is True)
    blocked = sum(1 for story in stories if isinstance(story, dict) and story.get("status") == "blocked")
    return total, total - passed - blocked, passed


def _blocked_count(prd: dict[str, Any]) -> int:
    stories = prd.get("userStories")
    if not isinstance(stories, list):
        return 0
    return sum(1 for story in stories if isinstance(story, dict) and story.get("status") == "blocked")


def _render(slug: str, path: Path, prd: dict[str, Any], status: str = "active") -> str:
    total, pending, passed = _counts(prd)
    project = str(prd.get("project") or slug)
    running = next(
        (
            story for story in prd.get("userStories", [])
            if isinstance(story, dict) and story.get("status") == "running"
        ),
        None,
    )
    lines = [
        f"Loop: {slug}",
        f"Status: {status}",
        f"Project: {project}",
        f"Stories: {total} total / {pending} pending / {passed} passed",
    ]
    blocked = _blocked_count(prd)
    if blocked:
        lines.append(f"Blocked: {blocked}")
    if isinstance(running, dict):
        lines.append(f"Running: {running.get('id', '?')} — {running.get('title', 'untitled')}")
    lines.extend([
        "",
        f"State: {path}",
    ])
    if status == "closed":
        lines.append(f"Next: /loop status {slug}")
    elif pending:
        lines.append(f"Next: /loop run {slug}")
    elif total:
        lines.append(f"Next: /loop close {slug}")
    else:
        lines.append(f"Next: edit {path / 'prd.json'} then /loop run {slug}")
    return "\n".join(lines)


def init_loop(slug: str | None = None, *, root: str | Path | None = None, title: str | None = None) -> LoopResult:
    name = slugify(slug)
    path = loop_dir(name, root=root)
    path.mkdir(parents=True, exist_ok=True)
    (path / "archive").mkdir(exist_ok=True)

    prd_path = path / "prd.json"
    if prd_path.exists():
        prd = _load_prd(prd_path)
    else:
        prd = _empty_prd(name, title)
        prd_path.write_text(json.dumps(prd, indent=2) + "\n")

    progress = path / "progress.md"
    if not progress.exists():
        progress.write_text("## Codebase Patterns\n\n## Progress\n")

    text = _render(name, path, prd, status="initialized")
    (path / "status.md").write_text(text + "\n")
    return LoopResult(name, path, text)


def status_loop(slug: str | None = None, *, root: str | Path | None = None) -> LoopResult:
    name = slugify(slug)
    path = loop_dir(name, root=root)
    prd_path = path / "prd.json"
    if not prd_path.exists():
        text = "\n".join([
            f"Loop: {name}",
            "Status: not started",
            "Stories: 0 total / 0 pending / 0 passed",
            "",
            f"State: {path}",
            f"Next: /loop init {name}",
        ])
        return LoopResult(name, path, text)

    prd = _load_prd(prd_path)
    status = "closed" if prd.get("status") == "closed" else "active"
    text = _render(name, path, prd, status=status)
    (path / "status.md").write_text(text + "\n")
    return LoopResult(name, path, text)


def _pending_stories(prd: dict[str, Any]) -> list[dict[str, Any]]:
    stories = prd.get("userStories")
    if not isinstance(stories, list):
        return []
    return [
        story
        for story in stories
        if isinstance(story, dict)
        and story.get("passes") is not True
        and story.get("status") != "blocked"
    ]


def _all_stories(prd: dict[str, Any]) -> list[dict[str, Any]]:
    stories = prd.get("userStories")
    if not isinstance(stories, list):
        return []
    return [story for story in stories if isinstance(story, dict)]


def _select_story(prd: dict[str, Any], story_id: str | None = None) -> dict[str, Any] | None:
    stories = _all_stories(prd)
    if story_id:
        wanted = story_id.casefold()
        return next((story for story in stories if str(story.get("id", "")).casefold() == wanted), None)
    running = next((story for story in stories if story.get("status") == "running"), None)
    if running is not None:
        return running
    pending = sorted(_pending_stories(prd), key=lambda s: (s.get("priority", 999), str(s.get("id", ""))))
    return pending[0] if pending else None


def _append_progress(path: Path, heading: str, story: dict[str, Any], note: str) -> None:
    progress = path / "progress.md"
    if not progress.exists():
        progress.write_text("## Codebase Patterns\n\n## Progress\n", encoding="utf-8")
    story_ref = f"{story.get('id', '?')} — {story.get('title', 'untitled')}"
    message = note.strip() if note.strip() else "No note provided."
    with progress.open("a", encoding="utf-8") as handle:
        handle.write(f"\n### {heading}: {story_ref}\n")
        handle.write(f"- {_now()}: {message}\n")


def _story_prompt(slug: str, path: Path, prd: dict[str, Any], story: dict[str, Any]) -> str:
    criteria = story.get("acceptanceCriteria")
    acceptance = criteria if isinstance(criteria, list) else []
    checks = "\n".join(f"- {item}" for item in acceptance) or "- No acceptance criteria listed; define the blocker before implementing."
    return "\n".join([
        f"Loop: {slug}",
        f"Project: {prd.get('project') or slug}",
        f"State: {path}",
        f"Story: {story.get('id', '?')} — {story.get('title', 'untitled')}",
        "",
        str(story.get("description") or prd.get("description") or ""),
        "",
        "Acceptance:",
        checks,
        "",
        "Do one story only. Use a fresh context boundary, run the relevant verification, then update prd.json, progress.md, and status.md. Stop on blocker; do not auto-retry.",
    ])


def run_loop(slug: str | None = None, *, root: str | Path | None = None) -> LoopResult:
    name = slugify(slug)
    path = loop_dir(name, root=root)
    prd_path = path / "prd.json"
    if not prd_path.exists():
        return status_loop(name, root=root)

    prd = _load_prd(prd_path)
    if prd.get("status") == "closed":
        text = _render(name, path, prd, status="closed")
        (path / "status.md").write_text(text + "\n", encoding="utf-8")
        return LoopResult(name, path, text)
    stories = sorted(_pending_stories(prd), key=lambda s: (s.get("priority", 999), str(s.get("id", ""))))
    if not stories:
        text = _render(name, path, prd, status="complete")
        (path / "status.md").write_text(text + "\n")
        return LoopResult(name, path, text)

    story = stories[0]
    story["status"] = "running"
    prd["updatedAt"] = _now()
    prd_path.write_text(json.dumps(prd, indent=2) + "\n")
    (path / "status.md").write_text(_render(name, path, prd, status="running") + "\n")
    return LoopResult(name, path, _story_prompt(name, path, prd, story))


def complete_story(
    slug: str | None = None,
    *,
    root: str | Path | None = None,
    story_id: str | None = None,
    note: str = "",
) -> LoopResult:
    name = slugify(slug)
    path = loop_dir(name, root=root)
    prd_path = path / "prd.json"
    if not prd_path.exists():
        return status_loop(name, root=root)

    prd = _load_prd(prd_path)
    if prd.get("status") == "closed":
        text = _render(name, path, prd, status="closed")
        (path / "status.md").write_text(text + "\n", encoding="utf-8")
        return LoopResult(name, path, text)
    story = _select_story(prd, story_id)
    if story is None:
        text = _render(name, path, prd, status="no runnable story")
        (path / "status.md").write_text(text + "\n", encoding="utf-8")
        return LoopResult(name, path, text)

    story["passes"] = True
    story["status"] = "passed"
    story.pop("blockReason", None)
    story.pop("blockedAt", None)
    story["completedAt"] = _now()
    prd["updatedAt"] = _now()
    prd_path.write_text(json.dumps(prd, indent=2) + "\n", encoding="utf-8")
    _append_progress(path, "Completed", story, note)
    status = "complete" if not _pending_stories(prd) and _blocked_count(prd) == 0 else "active"
    text = _render(name, path, prd, status=status)
    (path / "status.md").write_text(text + "\n", encoding="utf-8")
    return LoopResult(name, path, text)


def block_story(
    slug: str | None = None,
    *,
    root: str | Path | None = None,
    story_id: str | None = None,
    note: str = "",
) -> LoopResult:
    name = slugify(slug)
    path = loop_dir(name, root=root)
    prd_path = path / "prd.json"
    if not prd_path.exists():
        return status_loop(name, root=root)

    prd = _load_prd(prd_path)
    if prd.get("status") == "closed":
        text = _render(name, path, prd, status="closed")
        (path / "status.md").write_text(text + "\n", encoding="utf-8")
        return LoopResult(name, path, text)
    story = _select_story(prd, story_id)
    if story is None:
        text = _render(name, path, prd, status="no runnable story")
        (path / "status.md").write_text(text + "\n", encoding="utf-8")
        return LoopResult(name, path, text)

    reason = note.strip() or "Blocked without a note."
    story["passes"] = False
    story["status"] = "blocked"
    story["blockReason"] = reason
    story["blockedAt"] = _now()
    prd["updatedAt"] = _now()
    prd_path.write_text(json.dumps(prd, indent=2) + "\n", encoding="utf-8")
    _append_progress(path, "Blocked", story, reason)
    text = _render(name, path, prd, status="blocked")
    (path / "status.md").write_text(text + "\n", encoding="utf-8")
    return LoopResult(name, path, text)


def close_loop(slug: str | None = None, *, root: str | Path | None = None) -> LoopResult:
    name = slugify(slug)
    path = loop_dir(name, root=root)
    prd_path = path / "prd.json"
    if not prd_path.exists():
        return status_loop(name, root=root)

    prd = _load_prd(prd_path)
    prd["status"] = "closed"
    prd["updatedAt"] = _now()
    prd_path.write_text(json.dumps(prd, indent=2) + "\n")
    archive_dir = path / "archive" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("prd.json", "progress.md", "status.md"):
        source = path / filename
        if source.exists():
            copy2(source, archive_dir / filename)
    text = _render(name, path, prd, status="closed") + f"\nArchive: {archive_dir}"
    (path / "status.md").write_text(text + "\n")
    return LoopResult(name, path, text)


def loop_text(args: str, *, root: str | Path | None = None) -> str:
    parts = args.split()
    command = parts[0] if parts else "status"
    slug = parts[1] if len(parts) > 1 else DEFAULT_SLUG
    if command == "init":
        return init_loop(slug, root=root).text
    if command == "status":
        return status_loop(slug, root=root).text
    if command == "run":
        return run_loop(slug, root=root).text
    if command == "complete":
        story_id = parts[2] if len(parts) > 2 else None
        note = " ".join(parts[3:]) if len(parts) > 3 else ""
        return complete_story(slug, root=root, story_id=story_id, note=note).text
    if command == "block":
        story_id = parts[2] if len(parts) > 2 else None
        note = " ".join(parts[3:]) if len(parts) > 3 else ""
        return block_story(slug, root=root, story_id=story_id, note=note).text
    if command == "close":
        return close_loop(slug, root=root).text
    return "Usage: /loop [init|run|status|complete|block|close] <slug>"
