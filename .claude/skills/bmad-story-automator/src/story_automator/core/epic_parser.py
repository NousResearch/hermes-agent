from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .common import read_text, trim_lines
from .story_keys import normalize_story_key, normalize_story_key_for_epic


def parse_epic_file(epic_file: str | Path) -> dict[str, Any]:
    content = read_text(epic_file)
    lines = trim_lines(content)
    project_root = _project_root_for_epic_file(epic_file)
    epic_title = ""
    for line in lines:
        if line.startswith("# "):
            epic_title = line.removeprefix("# ").strip()
            break
    story_re = re.compile(r"^###\s+Story\s+([^:]+):\s*(.*)$")
    epic_re = re.compile(r"^##\s+Epic\s+([A-Za-z][\w-]*|\d+):\s*(.*)$")
    current_epic = ""
    current_epic_title = ""
    stories: list[dict[str, str]] = []
    for line in lines:
        epic_match = epic_re.match(line)
        if epic_match:
            current_epic = epic_match.group(1).strip()
            current_epic_title = epic_match.group(2).strip()
            continue
        story_match = story_re.match(line)
        if story_match:
            raw_story, title = story_match.groups()
            story_key = _normalize_header_story(project_root, current_epic, raw_story.strip())
            if story_key is None:
                continue
            story_id = story_key.id
            epic_num, _, story_num = story_id.rpartition(".")
            stories.append(
                {
                    "epicNum": epic_num,
                    "epicTitle": current_epic_title,
                    "storyNum": story_num,
                    "storyId": story_id,
                    "storyKey": story_key.key,
                    "headerStory": raw_story.strip(),
                    "title": title.strip(),
                }
            )
    return {"ok": True, "epicTitle": epic_title, "stories": stories, "count": len(stories), "file": str(epic_file)}


def parse_story(epic_file: str | Path, story_id: str, rules_file: str | Path) -> dict[str, Any]:
    content = read_text(epic_file)
    lines = trim_lines(content)
    project_root = _project_root_for_epic_file(epic_file)
    epic_re = re.compile(r"^##\s+Epic\s+([A-Za-z][\w-]*|\d+):")
    header_re = re.compile(r"^###\s+Story\s+([^:]+):\s*(.*)$")
    target_id = story_id
    start_index = -1
    title = ""
    current_epic = ""
    for index, line in enumerate(lines):
        epic_match = epic_re.match(line)
        if epic_match:
            current_epic = epic_match.group(1).strip()
            continue
        match = header_re.match(line)
        if match:
            raw_story, raw_title = match.groups()
            story_key = _normalize_header_story(project_root, current_epic, raw_story.strip())
            if story_key is None:
                continue
            if target_id not in {raw_story.strip(), story_key.id, story_key.prefix, story_key.key}:
                continue
            start_index = index
            target_id = story_key.id
            title = raw_title.strip()
            break
    if start_index < 0:
        raise ValueError("story_not_found")
    description_lines: list[str] = []
    acceptance_criteria: list[str] = []
    dependencies = ""
    in_ac = False
    for line in lines[start_index + 1 :]:
        if line.startswith("### Story ") or line.startswith("## Epic "):
            break
        if "Acceptance Criteria" in line:
            in_ac = True
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if "Dependencies:" in line or "**Dependencies**:" in line:
            dep = line.replace("**Dependencies**:", "").replace("Dependencies:", "").strip()
            if not dependencies:
                dependencies = dep
        if in_ac:
            acceptance_criteria.append(stripped)
        else:
            description_lines.append(stripped)
    description = " ".join(" ".join(description_lines).split())
    rules = json.loads(read_text(rules_file))
    content_for_score = " ".join(part for part in [title, description, " ".join(acceptance_criteria)] if part).strip()
    score = 0
    reasons: list[str] = []
    for rule in rules.get("rules", []):
        pattern = rule.get("pattern", "")
        if pattern and re.search(pattern, content_for_score, re.IGNORECASE):
            score += int(rule.get("score", 0))
            reasons.append(str(rule.get("label", "")))
    structural = rules.get("structural_rules", {})
    ac_count = len(acceptance_criteria)
    if structural.get("ac_count_high", 0) and ac_count > int(structural["ac_count_high"]):
        score += int(structural.get("ac_count_high_score", 0))
        reasons.append(f"High AC count ({ac_count})")
    elif structural.get("ac_count_medium", 0) and ac_count > int(structural["ac_count_medium"]):
        score += int(structural.get("ac_count_medium_score", 0))
        reasons.append(f"Elevated AC count ({ac_count})")
    if structural.get("dependency_score", 0) and dependencies and dependencies.lower() != "none":
        score += int(structural.get("dependency_score", 0))
        reasons.append("Has explicit dependencies")
    word_threshold = int(structural.get("large_story_word_threshold", 0))
    if word_threshold:
        word_count = len(content_for_score.split())
        if word_count > word_threshold:
            score += int(structural.get("large_story_score", 0))
            reasons.append(f"Large story ({word_count} words)")
    low_max = int(rules.get("thresholds", {}).get("low_max", 0))
    medium_max = int(rules.get("thresholds", {}).get("medium_max", low_max))
    level = "High"
    if score <= low_max:
        level = "Low"
    elif score <= medium_max:
        level = "Medium"
    return {
        "ok": True,
        "storyId": target_id,
        "title": title,
        "description": description,
        "acceptanceCriteria": acceptance_criteria,
        "dependencies": dependencies,
        "complexity": {"score": score, "level": level, "reasons": reasons},
    }


def _project_root_for_epic_file(epic_file: str | Path) -> str:
    path = Path(epic_file).resolve()
    parts = path.parts
    if "_bmad-output" in parts:
        return str(Path(*parts[: parts.index("_bmad-output")]))
    if "docs" in parts:
        return str(Path(*parts[: parts.index("docs")]))
    return str(path.parent)


def _normalize_header_story(project_root: str, current_epic: str, raw_story: str):
    if current_epic:
        story_key = normalize_story_key_for_epic(project_root, current_epic, raw_story)
        if story_key is not None:
            return story_key
    return normalize_story_key(project_root, raw_story)


def parse_story_range(user_input: str, total: int, ids_csv: str = "") -> dict[str, Any]:
    if not user_input or total <= 0:
        raise ValueError("missing_input_or_total")
    ids = [part.strip() for part in ids_csv.split(",")] if ids_csv else []
    selected: set[int] = set()
    normalized = user_input.lower().replace(" ", "")
    if normalized == "all":
        selected = set(range(1, total + 1))
    else:
        for part in normalized.split(","):
            if not part:
                continue
            if "-" in part:
                start_raw, end_raw = part.split("-", 1)
                if start_raw.isdigit() and end_raw.isdigit():
                    start = int(start_raw)
                    end = int(end_raw)
                    low, high = sorted((start, end))
                    selected.update(range(low, high + 1))
            elif part.isdigit():
                selected.add(int(part))
    indices = sorted(index for index in selected if 1 <= index <= total)
    story_ids = [ids[index - 1] for index in indices if index - 1 < len(ids)]
    return {"ok": True, "indices": indices, "storyIds": story_ids, "count": len(indices)}


def epic_complete(epic_file: str | Path, range_csv: str) -> dict[str, Any]:
    stories = parse_epic_file(epic_file)["stories"]
    story_ids = [story["storyId"] for story in stories]
    if not story_ids:
        raise ValueError("no_stories_found")
    max_epic_story = max(story_ids, key=_story_sort_key)
    selected = [_canonical_story_id(part.strip(), stories) for part in range_csv.split(",") if part.strip()]
    max_range_story = max(selected, key=_story_sort_key) if selected else "0.0"
    return {"ok": True, "epicComplete": max_range_story == max_epic_story, "maxEpicStory": max_epic_story}


def _canonical_story_id(value: str, stories: list[dict[str, str]]) -> str:
    for story in stories:
        story_id = story["storyId"]
        if value in _story_aliases(story):
            return story_id
    return value


def _story_aliases(story: dict[str, str]) -> set[str]:
    story_id = story["storyId"]
    aliases = {story_id}
    header_story = story.get("headerStory", "")
    if header_story:
        aliases.add(header_story)
    story_key = story.get("storyKey", "")
    if story_key:
        aliases.add(story_key)
    if not _is_explicit_header_story(header_story, story_id):
        epic, _, story_num = story_id.rpartition(".")
        prefix = f"{epic}-{story_num}"
        aliases.add(prefix)
        title_slug = _slugify_title(story.get("title", ""))
        if title_slug:
            aliases.add(f"{prefix}-{title_slug}")
    return aliases


def _is_explicit_header_story(header_story: str, story_id: str) -> bool:
    if not header_story:
        return False
    epic, _, story_num = story_id.rpartition(".")
    return header_story not in {story_id, f"{epic}-{story_num}"}


def _slugify_title(title: str) -> str:
    return "-".join(part for part in re.split(r"[^A-Za-z0-9]+", title.lower()) if part)


def _story_sort_key(value: str) -> tuple[int, int, str, int, str]:
    epic, _, story_num = value.rpartition(".")
    if epic.isdigit():
        return (0, int(epic), "", int(story_num) if story_num.isdigit() else 0, value)
    return (1, 0, epic, int(story_num) if story_num.isdigit() else 0, value)
