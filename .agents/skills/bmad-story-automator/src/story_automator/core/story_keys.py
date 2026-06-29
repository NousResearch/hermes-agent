from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .artifact_paths import implementation_artifacts_dir, sprint_status_path
from .utils import file_exists, read_text


@dataclass(frozen=True)
class StoryKey:
    id: str
    prefix: str
    key: str


def sprint_status_file(project_root: str) -> str:
    return str(sprint_status_path(project_root))


def normalize_story_key(project_root: str, value: str) -> StoryKey | None:
    if re.fullmatch(r"\d+\.\d+", value):
        story_id = value
        prefix = value.replace(".", "-")
        key = ""
    elif re.fullmatch(r"\d+-\d+", value):
        prefix = value
        story_id = value.replace("-", ".")
        key = ""
    elif re.fullmatch(r"\d+-\d+-.+", value):
        key = value
        prefix = "-".join(value.split("-", 2)[:2])
        story_id = prefix.replace("-", ".")
    elif re.fullmatch(r"[A-Za-z][\w-]*\.\d+", value):
        story_id = value
        epic_part, _, story_num = value.partition(".")
        prefix = f"{epic_part}-{story_num}"
        key = ""
    elif re.fullmatch(r"[A-Za-z][\w-]*-\d+", value):
        prefix = value
        epic_part, _, story_num = value.rpartition("-")
        story_id = f"{epic_part}.{story_num}"
        key = ""
    elif re.fullmatch(r"[A-Za-z][\w-]*-\d+-.+", value):
        split = _split_non_numeric_full_key(project_root, value)
        if split is None:
            return None
        epic_part, story_num = split
        prefix = f"{epic_part}-{story_num}"
        story_id = f"{epic_part}.{story_num}"
        key = value
    else:
        return None

    return _complete_story_key(project_root, story_id, prefix, key)


def normalize_story_key_for_epic(project_root: str, epic: str, value: str) -> StoryKey | None:
    if "." in value:
        norm = normalize_story_key(project_root, value)
        if norm is None or norm.id.rsplit(".", 1)[0] != epic:
            return None
        return norm

    dotted = re.fullmatch(rf"{re.escape(epic)}\.(\d+)", value)
    if dotted:
        story_num = dotted.group(1)
        return _complete_story_key(project_root, f"{epic}.{story_num}", f"{epic}-{story_num}", "")

    dashed = re.fullmatch(rf"{re.escape(epic)}-(\d+)(?:-.+)?", value)
    if dashed:
        if _has_known_longer_epic(project_root, epic, value) or _story_prefix_claimed_by_parent_epic(project_root, epic, value):
            return None
        story_num = dashed.group(1)
        prefix = f"{epic}-{story_num}"
        key = value if value != prefix else ""
        return _complete_story_key(project_root, f"{epic}.{story_num}", prefix, key)

    return normalize_story_key(project_root, value)


def _complete_story_key(project_root: str, story_id: str, prefix: str, key: str) -> StoryKey:
    artifacts = implementation_artifacts_dir(project_root)
    if not key:
        for match in sorted(artifacts.glob(f"{prefix}-*.md")):
            if _full_key_matches_story(project_root, match.stem, story_id, allow_ambiguous_same_id=False):
                key = match.stem
                break
    if not key:
        status_file = sprint_status_file(project_root)
        if file_exists(status_file):
            for status_key in _status_keys(read_text(status_file)):
                if status_key.startswith(f"{prefix}-") and _full_key_matches_story(project_root, status_key, story_id, allow_ambiguous_same_id=True):
                    key = status_key
                    break
    if not key:
        key = prefix
    return StoryKey(id=story_id, prefix=prefix, key=key)


def _split_non_numeric_full_key(project_root: str, value: str) -> tuple[str, str] | None:
    matches = list(re.finditer(r"(?=-(\d+)-)", value))
    if not matches:
        return None
    single_story = [
        match
        for match in matches[1:]
        if _is_single_story_key(project_root, value, match)
    ]
    if single_story:
        match = max(single_story, key=lambda item: item.start())
        return value[: match.start()], match.group(1)
    known = [match for match in matches if _epic_exists(project_root, value[: match.start()])]
    if known:
        match = max(known, key=lambda item: item.start())
        return value[: match.start()], match.group(1)
    match = _numeric_epic_segment_match(matches) or _default_story_epic_match(value, matches)
    return value[: match.start()], match.group(1)


def _has_known_longer_epic(project_root: str, epic: str, value: str) -> bool:
    for match in re.finditer(r"(?=-(\d+)-)", value):
        candidate_epic = value[: match.start()]
        if candidate_epic == epic or not candidate_epic.startswith(f"{epic}-"):
            continue
        if _epic_exists(project_root, candidate_epic) or _is_single_story_key(project_root, value, match):
            return True
    return False


def _status_keys(content: str) -> list[str]:
    keys: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        keys.append(stripped.split(":", 1)[0].strip())
    return keys


def _full_key_matches_story(project_root: str, key: str, story_id: str, *, allow_ambiguous_same_id: bool) -> bool:
    norm = normalize_story_key(project_root, key)
    if norm is not None:
        if not allow_ambiguous_same_id and _has_ambiguous_later_boundary(key, story_id):
            return False
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


def _epic_exists(project_root: str, epic: str) -> bool:
    if _epic_file_exists(project_root, epic):
        return True
    status_file = sprint_status_file(project_root)
    if not file_exists(status_file):
        return False
    pattern = re.compile(rf"(?m)^\s*{re.escape(epic)}-(\d+)(?:-[^:\s]+)?\s*:")
    story_nums = {match.group(1) for match in pattern.finditer(read_text(status_file))}
    return len(story_nums) > 1


def _is_single_story_key(project_root: str, value: str, match: re.Match[str]) -> bool:
    candidate_epic = value[: match.start()]
    story_num = match.group(1)
    if story_num != "1" and _known_parent_epic(project_root, candidate_epic):
        return False
    return (
        _has_exact_story_key(project_root, value)
        and _last_epic_segment_is_numeric(candidate_epic)
        and _single_story_numeric_epic(candidate_epic)
        and (
            story_num == "1"
            or (
                story_num.isdigit()
                and 10 <= int(story_num) <= 99
                and _year_like_numeric_suffix_epic(candidate_epic)
            )
        )
    )


def _default_story_epic_match(value: str, matches: list[re.Match[str]]) -> re.Match[str]:
    for match in reversed(matches[1:]):
        candidate_epic = value[: match.start()]
        story_num = match.group(1)
        if story_num.isdigit() and 10 <= int(story_num) <= 99 and _year_like_numeric_suffix_epic(candidate_epic):
            return match
    return matches[0]


def _numeric_epic_segment_match(matches: list[re.Match[str]]) -> re.Match[str] | None:
    if len(matches) < 2 or len(matches[0].group(1)) < 4 or len(matches[1].group(1)) < 3:
        return None
    return matches[1]


def _has_exact_story_key(project_root: str, value: str) -> bool:
    artifacts = implementation_artifacts_dir(project_root)
    if (artifacts / f"{value}.md").is_file():
        return True
    status_file = sprint_status_file(project_root)
    return file_exists(status_file) and re.search(rf"(?m)^\s*{re.escape(value)}\s*:", read_text(status_file)) is not None


def _known_parent_epic(project_root: str, candidate_epic: str) -> bool:
    parent_epic, _, story_num = candidate_epic.rpartition("-")
    return bool(parent_epic and story_num.isdigit() and _epic_exists(project_root, parent_epic))


def _story_prefix_claimed_by_parent_epic(project_root: str, epic: str, value: str) -> bool:
    parent_epic, _, story_num = epic.rpartition("-")
    return bool(
        parent_epic
        and story_num.isdigit()
        and not _epic_exists(project_root, epic)
        and normalize_story_key_for_epic(project_root, parent_epic, value) is not None
    )


def _last_epic_segment_is_numeric(epic: str) -> bool:
    return epic.rsplit("-", 1)[-1].isdigit()


def _year_like_numeric_suffix_epic(epic: str) -> bool:
    segment = epic.rsplit("-", 1)[-1]
    return segment.isdigit() and int(segment) >= 1000


def _single_story_numeric_epic(epic: str) -> bool:
    segments = epic.split("-")
    numeric_indexes = [index for index, segment in enumerate(segments) if segment.isdigit()]
    if len(numeric_indexes) == 1:
        return True
    if len(numeric_indexes) == 2:
        first, second = numeric_indexes
        return int(segments[second]) <= 99 and any(not segment.isdigit() for segment in segments[first + 1 : second])
    return False


def _epic_file_exists(project_root: str, epic: str) -> bool:
    root = Path(project_root)
    for base in (implementation_artifacts_dir(root), root / "docs" / "epics"):
        if (base / f"epic-{epic}.md").is_file() or next(base.glob(f"epic-{epic}-*.md"), None) is not None:
            return True
    return False
