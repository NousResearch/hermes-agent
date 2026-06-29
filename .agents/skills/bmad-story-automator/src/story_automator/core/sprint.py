from __future__ import annotations

import re
from dataclasses import dataclass

from .story_keys import StoryKey, normalize_story_key, normalize_story_key_for_epic, sprint_status_file
from .utils import file_exists, read_text, trim_lines


@dataclass(frozen=True)
class SprintStatus:
    found: bool
    story: str
    status: str
    done: bool
    reason: str = ""


def sprint_status_get(project_root: str, story_key: str) -> SprintStatus:
    status_file = sprint_status_file(project_root)
    if not file_exists(status_file):
        return SprintStatus(False, story_key, "unknown", False, "sprint-status.yaml not found")
    content = read_text(status_file)
    norm = normalize_story_key(project_root, story_key)
    if norm is not None:
        result = _best_status_match(project_root, content, story_key, norm)
        if result is not None:
            return result
    match = _exact_status_match(content, story_key)
    if match:
        status = match.group(1).strip()
        return SprintStatus(True, story_key, status, status == "done")
    return SprintStatus(False, story_key, "not_found", False)


def sprint_status_epic(project_root: str, epic: str) -> tuple[list[str], int]:
    status_file = sprint_status_file(project_root)
    if not file_exists(status_file):
        return ([], 0)
    story_order: list[str] = []
    story_rows: dict[str, tuple[int, str, str]] = {}
    for line in trim_lines(read_text(status_file)):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":", 1)
        if len(parts) < 2:
            continue
        key = parts[0].strip()
        norm = normalize_story_key_for_epic(project_root, epic, key)
        if norm is None or norm.id.rsplit(".", 1)[0] != epic:
            continue
        status = parts[1].strip().split()
        rank = _status_key_rank(key, norm)
        if norm.id not in story_rows:
            story_order.append(norm.id)
        previous = story_rows.get(norm.id)
        if previous is None or rank >= previous[0]:
            story_rows[norm.id] = (rank, key, status[0] if status else "")
    stories = [story_rows[story_id][1] for story_id in story_order]
    done_count = sum(1 for story_id in story_order if story_rows[story_id][2] == "done")
    return (stories, done_count)


def _status_key_rank(key: str, norm: StoryKey) -> int:
    if key == norm.key and key not in {norm.id, norm.prefix}:
        return 2
    return 1


def _best_status_match(project_root: str, content: str, story_key: str, norm: StoryKey) -> SprintStatus | None:
    candidates: list[tuple[int, int, str, str]] = []
    explicit_full_key = story_key == norm.key and story_key not in {norm.id, norm.prefix}
    requested_rank = 5 if explicit_full_key else 3
    for rank, key in ((requested_rank, story_key), (2, norm.id), (2, norm.prefix)):
        match = _exact_status_match(content, key)
        if match:
            candidates.append((rank, len(candidates), key, match.group(1).strip()))
    for key, status in _status_rows(content):
        if explicit_full_key and key != story_key:
            continue
        if key.startswith(f"{norm.prefix}-") and _status_key_matches_story(project_root, key, norm.id):
            candidates.append((4, len(candidates), key, status))
    if not candidates:
        return None
    _, _, key, status = max(candidates)
    return SprintStatus(True, key, status, status == "done")


def _exact_status_match(content: str, key: str) -> re.Match[str] | None:
    return re.search(rf"(?m)^\s*{re.escape(key)}:\s*(\S+)", content)


def _status_rows(content: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in trim_lines(content):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, status = stripped.split(":", 1)
        parts = status.strip().split()
        if parts:
            rows.append((key.strip(), parts[0]))
    return rows


def _status_key_matches_story(project_root: str, key: str, story_id: str) -> bool:
    norm = normalize_story_key(project_root, key)
    if norm is not None:
        return norm.id == story_id
    prefix = story_id.replace(".", "-")
    if not key.startswith(f"{prefix}-"):
        return False
    story_num = story_id.rsplit(".", 1)[-1]
    remainder = key[len(prefix) + 1 :]
    first_segment = remainder.split("-", 1)[0]
    if not first_segment.isdigit():
        return not _has_ambiguous_later_boundary(key, story_id)
    return len(story_num) >= 4 and int(first_segment) <= 99


def _has_ambiguous_later_boundary(key: str, story_id: str) -> bool:
    story_num = story_id.rsplit(".", 1)[-1]
    if len(story_num) < 4:
        return False
    prefix = story_id.replace(".", "-")
    if not key.startswith(f"{prefix}-"):
        return False
    remainder = key[len(prefix) + 1 :]
    return re.search(r"^[^-]+-\d+-\d+-", remainder) is not None
