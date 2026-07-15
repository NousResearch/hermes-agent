"""Shared review projections for StudyOS HTTP and model-tool adapters.

``StudyReviewReadModel`` is the module interface. It owns review queue
selection, concept projections, and cached spacing statistics so transports do
not reconstruct those rules independently.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from plugins.study_os.notes import (
    StudyNoteCatalog,
    _read_text,
    _strip_wikilink,
    _write_text,
)
from plugins.study_os.workspace import study_state_dir as _study_dir


_EBBINGHAUS_BASE = [1, 2, 4, 7, 15, 30, 60, 120]
_REVIEW_LEVEL_WEIGHT = {0: 0.5, 1: 0.7, 2: 1.0, 3: 1.3, 4: 1.6, 5: 2.5}
_LEARNING_STATES = ("未开始", "学习中", "已理解", "已掌握")
_GRAPH_CACHE_TTL_HOURS = 1


def _parse_date(value: Any, default: date | None = None) -> date:
    if not value:
        return default or date.today()
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def upsert_frontmatter_field(path: Path, field: str, value: Any) -> None:
    """Add or update one YAML frontmatter field without reformatting the note."""

    raw = _read_text(path)
    lines = raw.splitlines()
    if isinstance(value, bool):
        serialized = "true" if value else "false"
    elif isinstance(value, date):
        serialized = value.isoformat()
    elif isinstance(value, datetime):
        serialized = value.strftime("%Y-%m-%d")
    else:
        serialized = str(value)

    if not lines or lines[0].strip() != "---":
        _write_text(path, f"---\n{field}: {serialized}\n---\n\n{raw}")
        return

    end_idx = next(
        (index for index in range(1, len(lines)) if lines[index].strip() == "---"),
        None,
    )
    if end_idx is None:
        return
    field_re = re.compile(rf"^{re.escape(field)}\s*:.*$")
    for index in range(1, end_idx):
        if field_re.match(lines[index]):
            lines[index] = f"{field}: {serialized}"
            _write_text(path, "\n".join(lines) + "\n")
            return
    lines.insert(end_idx, f"{field}: {serialized}")
    _write_text(path, "\n".join(lines) + "\n")


def calculate_next_review(
    review_count: int,
    review_level: int,
    passed: bool,
) -> tuple[int, date]:
    if not passed:
        return 0, date.today() + timedelta(days=1)
    new_count = min(review_count + 1, len(_EBBINGHAUS_BASE) - 1)
    base = (
        _EBBINGHAUS_BASE[review_count]
        if review_count < len(_EBBINGHAUS_BASE)
        else _EBBINGHAUS_BASE[-1]
    )
    interval = max(1, int(base * _REVIEW_LEVEL_WEIGHT.get(review_level, 1.0)))
    return new_count, date.today() + timedelta(days=interval)


def read_review_state(note: dict[str, Any]) -> dict[str, Any]:
    frontmatter = note.get("frontmatter", {})
    return {
        "review_count": int(frontmatter.get("review_count", 0)),
        "last_reviewed_at": str(frontmatter.get("last_reviewed_at", "")),
        "next_review_at": str(frontmatter.get("next_review_at", "")),
    }


def is_due(note: dict[str, Any], today: date) -> bool:
    if note.get("layer") != "example":
        return False
    next_at = read_review_state(note)["next_review_at"]
    if not next_at:
        return True
    try:
        return _parse_date(next_at) <= today
    except Exception:
        return True


def concept_learning_state(note: dict[str, Any]) -> str:
    state = str(note.get("frontmatter", {}).get("learning_state", "未开始")).strip()
    return state if state in _LEARNING_STATES else "未开始"


def build_concept_graph(vault: Path) -> dict[str, Any]:
    prerequisites: dict[str, set[str]] = {}
    exercised_by: dict[str, list[str]] = {}
    review_by_concept: dict[str, list[int]] = {}
    catalog = StudyNoteCatalog(vault)
    for path in catalog.iter():
        note, _warnings = catalog.parse(path)
        concepts = [_strip_wikilink(item) for item in note.get("concepts", [])]
        if not concepts:
            continue
        for concept in concepts:
            exercised_by.setdefault(concept, []).append(note["path"])
            review_level = note.get("frontmatter", {}).get("review_level")
            if isinstance(review_level, (int, float)):
                review_by_concept.setdefault(concept, []).append(int(review_level))
        if note.get("layer", "note") in ("concept", "pattern"):
            for concept in concepts:
                dependencies = [item for item in concepts if item != concept]
                if dependencies:
                    prerequisites.setdefault(concept, set()).update(dependencies)

    dependents: dict[str, set[str]] = {}
    for concept, dependencies in prerequisites.items():
        for dependency in dependencies:
            dependents.setdefault(dependency, set()).add(concept)
    review_levels = {
        concept: {
            "min": min(levels),
            "avg": round(sum(levels) / len(levels), 1),
            "max": max(levels),
            "count": len(levels),
        }
        for concept, levels in review_by_concept.items()
    }
    return {
        "prerequisites": {
            key: sorted(values) for key, values in prerequisites.items()
        },
        "dependents": {key: sorted(values) for key, values in dependents.items()},
        "exercised_by": exercised_by,
        "review_levels": review_levels,
        "note_count": {
            concept: len(paths) for concept, paths in exercised_by.items()
        },
    }


def graph_cache_path(vault: Path) -> Path:
    return _study_dir(vault) / "concept_graph.json"


def _load_graph_cache(vault: Path) -> dict[str, Any] | None:
    path = graph_cache_path(vault)
    if not path.exists():
        return None
    try:
        data = json.loads(_read_text(path))
        built_at = data.get("built_at", "")
        if built_at and datetime.now() - datetime.fromisoformat(built_at) > timedelta(
            hours=_GRAPH_CACHE_TTL_HOURS
        ):
            return None
        graph = data.get("graph")
        return graph if isinstance(graph, dict) else None
    except Exception:
        return None


def _save_graph_cache(vault: Path, graph: dict[str, Any]) -> None:
    _write_text(
        graph_cache_path(vault),
        json.dumps(
            {"built_at": datetime.now().isoformat(), "graph": graph},
            ensure_ascii=False,
        ),
    )


def get_concept_graph(vault: Path, rebuild: bool = False) -> dict[str, Any]:
    if not rebuild and (cached := _load_graph_cache(vault)) is not None:
        return cached
    graph = build_concept_graph(vault)
    _save_graph_cache(vault, graph)
    return graph


def concept_ancestors(
    concept: str,
    graph: dict[str, Any],
    max_depth: int = 5,
) -> list[list[str]]:
    prerequisites = graph["prerequisites"]
    chains: list[list[str]] = []

    def walk(current: str, path: list[str], depth: int) -> None:
        if depth > max_depth:
            return
        dependencies = prerequisites.get(current, [])
        if not dependencies:
            chains.append(path + [current])
            return
        for dependency in dependencies:
            if dependency in path:
                chains.append(path + [current, f"(cycle→{dependency})"])
            else:
                walk(dependency, path + [current], depth + 1)

    walk(concept, [], 0)
    return chains


def concept_descendants(
    concept: str,
    graph: dict[str, Any],
    max_depth: int = 5,
) -> list[list[str]]:
    dependents = graph["dependents"]
    chains: list[list[str]] = []

    def walk(current: str, path: list[str], depth: int) -> None:
        if depth > max_depth:
            return
        children = dependents.get(current, [])
        if not children:
            chains.append(path + [current])
            return
        for child in children:
            if child in path:
                chains.append(path + [current, f"(cycle→{child})"])
            else:
                walk(child, path + [current], depth + 1)

    walk(concept, [], 0)
    return chains


def topological_order(
    concepts: list[str],
    graph: dict[str, Any],
) -> list[str]:
    prerequisites = graph["prerequisites"]
    relevant: set[str] = set()

    def collect(concept: str) -> None:
        if concept in relevant:
            return
        relevant.add(concept)
        for dependency in prerequisites.get(concept, []):
            collect(dependency)

    for concept in concepts:
        collect(concept)
    in_degree = {concept: 0 for concept in relevant}
    adjacent = {concept: [] for concept in relevant}
    for concept in relevant:
        for dependency in prerequisites.get(concept, []):
            if dependency in relevant and concept != dependency:
                adjacent.setdefault(dependency, []).append(concept)
                in_degree[concept] = in_degree.get(concept, 0) + 1
    queue = [concept for concept in relevant if in_degree.get(concept, 0) == 0]
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in adjacent.get(node, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    return order


def _stats_cache_path(vault: Path) -> Path:
    return _study_dir(vault) / "review_stats.json"


def invalidate_review_stats(vault: Path) -> None:
    _stats_cache_path(vault).unlink(missing_ok=True)


def build_review_stats(vault: Path) -> dict[str, Any]:
    today = date.today()
    total = 0
    by_level: Counter[int] = Counter()
    concept_levels: dict[str, list[int]] = {}
    concept_due: dict[str, int] = {}
    last_reviewed_dates: list[date] = []
    reviewed_count = 0
    due_count = 0
    catalog = StudyNoteCatalog(vault)
    # Multi-subject Vaults commonly store examples under Math/examples,
    # OS/examples, and similar roots. The layer check below is authoritative;
    # restricting discovery to a root-level examples/ directory undercounts
    # the same queue that ``due`` correctly discovers.
    for path in catalog.iter():
        note, _warnings = catalog.parse(path)
        if note.get("layer") != "example":
            continue
        total += 1
        frontmatter = note.get("frontmatter", {})
        level = int(frontmatter.get("review_level", 0))
        by_level[level] += 1
        if int(frontmatter.get("review_count", 0) or 0) > 0:
            reviewed_count += 1
        if last_reviewed := frontmatter.get("last_reviewed_at"):
            try:
                last_reviewed_dates.append(_parse_date(str(last_reviewed)))
            except Exception:
                pass
        for concept in note.get("concepts", []):
            name = _strip_wikilink(concept)
            concept_levels.setdefault(name, []).append(level)
        if is_due(note, today):
            due_count += 1
            for concept in note.get("concepts", []):
                name = _strip_wikilink(concept)
                concept_due[name] = concept_due.get(name, 0) + 1
    coverage = round(reviewed_count / total * 100, 1) if total else 0.0
    concept_stats = {
        concept: {
            "avg": round(sum(levels) / len(levels), 1),
            "min": min(levels),
            "max": max(levels),
            "count": len(levels),
            "due": concept_due.get(concept, 0),
        }
        for concept, levels in concept_levels.items()
    }
    streak = 0
    if last_reviewed_dates:
        check = today
        for reviewed_on in sorted(set(last_reviewed_dates), reverse=True):
            if reviewed_on == check or reviewed_on == check - timedelta(days=1):
                if reviewed_on == check - timedelta(days=1):
                    check = reviewed_on
                streak += 1
            elif reviewed_on < check - timedelta(days=1):
                break
    return {
        "semantics": "spacing_coverage.v1",
        "built_at": datetime.now().isoformat(),
        "total_examples": total,
        "by_review_level": {
            str(level): count for level, count in sorted(by_level.items())
        },
        "reviewed_examples": reviewed_count,
        "spacing_coverage_pct": coverage,
        "progress_pct": coverage,
        "due_today": due_count,
        "review_streak_days": streak,
        "concepts": concept_stats,
    }


def load_review_stats(vault: Path) -> dict[str, Any] | None:
    path = _stats_cache_path(vault)
    if not path.exists():
        return None
    try:
        value = json.loads(_read_text(path))
        if not isinstance(value, dict) or value.get("semantics") != "spacing_coverage.v1":
            return None
        return value
    except Exception:
        return None


def save_review_stats(vault: Path, stats: dict[str, Any]) -> None:
    _write_text(
        _stats_cache_path(vault),
        json.dumps(stats, ensure_ascii=False),
    )


@dataclass(frozen=True)
class StudyReviewReadModel:
    """Build all Vault-scoped review views from one set of rules."""

    vault: Path

    @property
    def notes(self) -> StudyNoteCatalog:
        return StudyNoteCatalog(self.vault)

    def due(
        self,
        *,
        as_of: date | None = None,
        subject: str = "",
        level: int | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        today = as_of or date.today()
        bounded_limit = max(1, min(int(limit), 500))
        subject_query = subject.strip().casefold()
        due: list[dict[str, Any]] = []
        subjects: set[str] = set()
        for path in self.notes.iter():
            note, _warnings = self.notes.parse(path)
            if note.get("layer") != "example" or not is_due(note, today):
                continue
            note_subject = self.notes.subject(note)
            if note_subject:
                subjects.add(note_subject)
            if subject_query:
                tags = {
                    str(tag).lstrip("#").casefold()
                    for tag in note.get("tags", [])
                }
                concepts = {
                    str(concept).casefold() for concept in note.get("concepts", [])
                }
                if (
                    subject_query != (note_subject or "").casefold()
                    and subject_query not in tags
                    and not any(subject_query in concept for concept in concepts)
                ):
                    continue
            frontmatter = note.get("frontmatter", {})
            review_level = int(frontmatter.get("review_level", 0))
            if level is not None and review_level != int(level):
                continue
            state = read_review_state(note)
            due.append(
                {
                    "path": note["path"],
                    "title": note["title"],
                    "review_level": review_level,
                    "review_count": state["review_count"],
                    "last_reviewed_at": state["last_reviewed_at"] or None,
                    "next_review_at": state["next_review_at"] or None,
                    "concepts": note.get("concepts", []),
                    "tags": note.get("tags", []),
                    "difficulty": frontmatter.get("difficulty"),
                    "subject": note_subject,
                }
            )
        due.sort(
            key=lambda item: (
                item["review_level"],
                item["last_reviewed_at"] or "0000-00-00",
                item["path"],
            )
        )
        selected = due[:bounded_limit]
        return {
            "vault_path": str(self.vault),
            "date": today.isoformat(),
            "count": len(selected),
            "subjects": sorted(subjects),
            "due": selected,
        }

    def raw_stats(self, *, rebuild: bool = False) -> tuple[dict[str, Any], bool]:
        stats = None if rebuild else load_review_stats(self.vault)
        cached = stats is not None
        if stats is None:
            stats = build_review_stats(self.vault)
            save_review_stats(self.vault, stats)
        return stats, cached

    def stats(self, *, rebuild: bool = False) -> dict[str, Any]:
        stats, cached = self.raw_stats(rebuild=rebuild)
        return {
            "vault_path": str(self.vault),
            "total": stats.get("total_examples", 0),
            "by_level": {
                int(key): value
                for key, value in stats.get("by_review_level", {}).items()
            },
            "spacing_coverage": stats.get("spacing_coverage_pct", 0.0),
            "reviewed_count": stats.get("reviewed_examples", 0),
            "progress": stats.get("progress_pct", 0.0),
            "concept_stats": stats.get("concepts", {}),
            "review_streak": stats.get("review_streak_days", 0),
            "due_count": stats.get("due_today", 0),
            "cached": cached,
        }

    def queue(self, *, state: str = "", limit: int = 30) -> dict[str, Any]:
        graph = get_concept_graph(self.vault)
        state_filter = state.strip()
        bounded_limit = max(1, min(int(limit), 500))
        new_concepts: list[dict[str, Any]] = []
        new_examples: list[dict[str, Any]] = []
        for path in self.notes.iter():
            note, _warnings = self.notes.parse(path)
            layer = note.get("layer", "note")
            frontmatter = note.get("frontmatter", {})
            if layer in ("concept", "pattern"):
                learning_state = concept_learning_state(note)
                if (
                    state_filter and learning_state != state_filter
                ) or learning_state == "已掌握":
                    continue
                new_concepts.append(
                    {
                        "path": note["path"],
                        "title": note["title"],
                        "learning_state": learning_state,
                        "prerequisites": graph.get("prerequisites", {}).get(
                            _strip_wikilink(note.get("title", "")), []
                        ),
                        "tags": note.get("tags", []),
                    }
                )
            elif layer == "example":
                review_count = int(frontmatter.get("review_count", 0))
                if review_count > 0:
                    continue
                if state_filter:
                    review_level = int(frontmatter.get("review_level", 0))
                    if state_filter == "学习中" and review_level != 0:
                        continue
                    if state_filter == "已理解" and review_level == 0:
                        continue
                new_examples.append(
                    {
                        "path": note["path"],
                        "title": note["title"],
                        "review_level": int(frontmatter.get("review_level", 0)),
                        "difficulty": frontmatter.get("difficulty"),
                        "concepts": note.get("concepts", []),
                        "tags": note.get("tags", []),
                        "source": frontmatter.get("source"),
                    }
                )
        new_examples.sort(
            key=lambda item: (
                {"easy": 1, "medium": 2, "hard": 3}.get(
                    str(item.get("difficulty", "")).lower(), 2
                ),
                item["title"],
            )
        )
        new_concepts.sort(
            key=lambda item: (
                max(
                    (
                        len(chain)
                        for chain in concept_ancestors(item["title"], graph)
                    ),
                    default=0,
                ),
                item["title"],
            )
        )
        return {
            "vault_path": str(self.vault),
            "new_concepts": new_concepts[:bounded_limit],
            "new_concepts_total": len(new_concepts),
            "new_examples": new_examples[:bounded_limit],
            "new_examples_total": len(new_examples),
        }

    def concepts(self) -> dict[str, Any]:
        graph = get_concept_graph(self.vault)
        names = sorted(
            set(graph.get("prerequisites", {}))
            | set(graph.get("dependents", {}))
            | set(graph.get("exercised_by", {}))
        )
        states: dict[str, str] = {}
        for path in self.notes.iter():
            note, _warnings = self.notes.parse(path)
            if note.get("layer") in ("concept", "pattern"):
                states[str(note.get("title") or "")] = concept_learning_state(note)
        concepts = []
        for name in names:
            review_info = graph.get("review_levels", {}).get(name, {})
            concepts.append(
                {
                    "title": name,
                    "learning_state": states.get(name, "未开始"),
                    "prerequisites": graph.get("prerequisites", {}).get(name, []),
                    "example_count": graph.get("note_count", {}).get(name, 0),
                    "avg_level": review_info.get("avg"),
                }
            )
        return {"vault_path": str(self.vault), "concepts": concepts}
