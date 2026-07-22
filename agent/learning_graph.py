"""Assemble the "learning made visible" graph for desktop.

This graph is intentionally scoped to what a user actually learns over time:
- non-base, learned/profile skills (agent-created or used),
- memory chunks from ``MEMORY.md`` / ``USER.md`` as first-class nodes.

Skill links come from declared ``related_skills``. Memory-to-skill links are
derived from lexical overlap so the graph can answer "which learned skills are
connected to the things I remember?".

Run as a module to print edge-density stats against real data:

    python -m agent.learning_graph
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home
from agent.skill_utils import is_excluded_skill_path


@dataclass
class SkillNode:
    name: str
    category: str
    source: str = "profile"
    timestamp: Optional[int] = None
    use_count: int = 0
    state: str = "active"
    created_by: Optional[str] = None
    pinned: bool = False
    related: list[str] = field(default_factory=list)


def _frontmatter(text: str) -> dict[str, Any]:
    try:
        from agent.skill_utils import parse_frontmatter

        fm, _ = parse_frontmatter(text)
        return fm or {}
    except Exception:
        return {}


def _hermes_meta(fm: dict[str, Any]) -> dict[str, Any]:
    """``metadata.hermes`` as a dict, tolerant of the string-valued frontmatter
    that ``parse_frontmatter``'s malformed-YAML fallback produces."""
    meta = fm.get("metadata")
    hermes = meta.get("hermes") if isinstance(meta, dict) else None
    return hermes if isinstance(hermes, dict) else {}


def _related(fm: dict[str, Any]) -> list[str]:
    raw = fm.get("related_skills") or _hermes_meta(fm).get("related_skills")
    if isinstance(raw, list):
        return [str(r).strip() for r in raw if str(r).strip()]
    if isinstance(raw, str):
        return [r.strip() for r in raw.strip("[]").split(",") if r.strip()]
    return []


def _category(fm: dict[str, Any], skill_md: Path) -> str:
    cat = fm.get("category") or _hermes_meta(fm).get("category")
    if cat:
        return str(cat)
    # …/skills/<category>/<skill>/SKILL.md
    parts = skill_md.parts
    return parts[-3] if len(parts) >= 3 else "general"


def _iter_skill_files(roots: list[tuple[str, Path]]):
    for source, root in roots:
        if root.exists():
            for path in root.rglob("SKILL.md"):
                yield source, path


def _load_usage() -> dict[str, dict[str, Any]]:
    try:
        from tools.skill_usage import load_usage

        return load_usage()
    except Exception:
        path = get_hermes_home() / "skills" / ".usage.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}


def _to_int_ts(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value)
        s = str(value).strip()
        if not s:
            return None
        try:
            return int(float(s))
        except ValueError:
            parsed = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return int(parsed.timestamp())
    except Exception:
        return None


def _usage_timestamp(rec: dict[str, Any]) -> Optional[int]:
    for key in ("last_activity_at", "last_used_at", "last_viewed_at", "last_patched_at", "created_at"):
        ts = _to_int_ts(rec.get(key))
        if ts is not None:
            return ts
    return None


def build_skill_nodes(skill_roots: list[tuple[str, Path]]) -> dict[str, SkillNode]:
    usage = _load_usage()
    nodes: dict[str, SkillNode] = {}

    for source, skill_md in _iter_skill_files(skill_roots):
        if is_excluded_skill_path(skill_md):
            continue
        try:
            fm = _frontmatter(skill_md.read_text(encoding="utf-8")[:4000])
        except OSError:
            continue
        name = str(fm.get("name") or skill_md.parent.name).strip()
        if not name or name in nodes:
            continue
        rec = usage.get(name, {})
        last_activity = _usage_timestamp(rec)
        file_ts = _to_int_ts(skill_md.stat().st_mtime)
        nodes[name] = SkillNode(
            name=name,
            category=_category(fm, skill_md),
            source=source,
            timestamp=last_activity or file_ts,
            use_count=int(rec.get("use_count", 0) or 0),
            state=str(rec.get("state", "active") or "active"),
            created_by=rec.get("created_by"),
            pinned=bool(rec.get("pinned", False)),
            related=_related(fm),
        )
    return nodes


def build_edges(nodes: dict[str, SkillNode]) -> list[tuple[str, str]]:
    """Undirected related_skills edges where BOTH endpoints exist (deduped)."""
    seen: set[tuple[str, str]] = set()
    edges: list[tuple[str, str]] = []
    for node in nodes.values():
        for target in node.related:
            if target in nodes and target != node.name:
                a, b = sorted((node.name, target))
                key = (a, b)
                if key not in seen:
                    seen.add(key)
                    edges.append(key)
    return edges


def density_stats(nodes: dict[str, SkillNode], edges: list[tuple[str, str]]) -> dict[str, Any]:
    linked: set[str] = set()
    for a, b in edges:
        linked.add(a)
        linked.add(b)
    cats: dict[str, int] = {}
    for n in nodes.values():
        cats[n.category] = cats.get(n.category, 0) + 1
    n = len(nodes) or 1
    return {
        "nodes": len(nodes),
        "related_edges": len(edges),
        "edges_per_node": round(len(edges) / n, 3),
        "linked_nodes": len(linked),
        "isolated_pct": round(100 * (n - len(linked)) / n, 1),
        "categories": len(cats),
        "agent_created": sum(1 for x in nodes.values() if x.created_by == "agent"),
        "used": sum(1 for x in nodes.values() if x.use_count > 0),
        "top_categories": sorted(cats.items(), key=lambda kv: -kv[1])[:8],
    }


def _memory_cards() -> list[dict[str, Any]]:
    """Freeform memory as readable cards.

    ``MEMORY.md`` / ``USER.md`` are prose split on bare ``§`` separators; each
    chunk becomes one card. Every chunk is surfaced — the graph shows everything.
    """
    base = get_hermes_home() / "memories"
    cards: list[dict[str, Any]] = []
    for fname, source in (("MEMORY.md", "memory"), ("USER.md", "profile")):
        path = base / fname
        try:
            text = path.read_text(encoding="utf-8").strip()
            file_ts = _to_int_ts(path.stat().st_mtime)
        except OSError:
            continue
        for chunk_idx, chunk in enumerate(c.strip() for c in text.split("\n§\n")):
            if not chunk:
                continue
            first = chunk.splitlines()[0].strip().lstrip("# ").strip()
            cards.append(
                {
                    "source": source,
                    "timestamp": file_ts + chunk_idx if file_ts is not None else None,
                    "title": (first[:80] + "…") if len(first) > 80 else first,
                    "body": chunk[:1200],
                }
            )
    return cards


def _tokenize(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) >= 3}


def _memory_skill_edges(memory_cards: list[dict[str, Any]], skills: list[SkillNode]) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    skill_meta = [(s, _tokenize(s.name), s.name.lower()) for s in skills]
    for idx, card in enumerate(memory_cards):
        mem_id = f"memory:{card['source']}:{idx}"
        text = f"{card.get('title', '')}\n{card.get('body', '')}".lower()
        text_tokens = _tokenize(text)
        scored: list[tuple[int, str]] = []
        for skill, tokens, skill_name_lower in skill_meta:
            score = 0
            if skill_name_lower in text:
                score += 6
            score += len(tokens & text_tokens)
            if score > 0:
                scored.append((score, skill.name))
        scored.sort(key=lambda x: (-x[0], x[1]))
        for _, skill_name in scored[:4]:
            edges.append((mem_id, skill_name))
    return edges


def _skill_roots() -> list[tuple[str, Path]]:
    repo = Path(__file__).resolve().parent.parent
    home_skills = get_hermes_home() / "skills"
    return [("base", repo / "skills"), ("profile", home_skills)]


def build_learning_graph() -> dict[str, Any]:
    """Full payload for the desktop learning panel.

    Focus on what is profile-learned and actionable:
    - skills that are NOT base-installed and show real learning signal
      (agent-created or used),
    - memory chunks as first-class graph nodes connected to those learned skills.
    """
    all_skills = build_skill_nodes(_skill_roots())
    learned_skills = {
        name: node
        for name, node in all_skills.items()
        if node.source != "base" and (node.created_by == "agent" or node.use_count > 0)
    }
    skill_edges = build_edges(learned_skills)
    memory_cards = _memory_cards()
    memory_edges = _memory_skill_edges(memory_cards, list(learned_skills.values()))

    edges = skill_edges + memory_edges
    clusters: dict[str, int] = {}
    for node in learned_skills.values():
        clusters[node.category] = clusters.get(node.category, 0) + 1
    if memory_cards:
        clusters["memory"] = len(memory_cards)

    graph_nodes = [
        {
            "id": n.name,
            "label": n.name,
            "kind": "skill",
            "timestamp": n.timestamp,
            "category": n.category,
            "useCount": n.use_count,
            "state": n.state,
            "createdBy": n.created_by,
            "pinned": n.pinned,
        }
        for n in learned_skills.values()
    ]
    for i, card in enumerate(memory_cards):
        graph_nodes.append(
            {
                "id": f"memory:{card['source']}:{i}",
                "label": card["title"],
                "kind": "memory",
                "memorySource": card["source"],
                "timestamp": card.get("timestamp"),
                "category": "memory",
                "useCount": 0,
                "state": "active",
                "createdBy": "memory",
                "pinned": False,
            }
        )

    return {
        "nodes": graph_nodes,
        "edges": [{"source": a, "target": b} for a, b in edges],
        "clusters": [
            {"category": c, "count": n}
            for c, n in sorted(clusters.items(), key=lambda kv: -kv[1])
        ],
        "memory": memory_cards,
        "stats": {
            **density_stats(learned_skills, skill_edges),
            "memory_nodes": len(memory_cards),
            "memory_skill_edges": len(memory_edges),
            "learned_skills": len(learned_skills),
        },
    }


if __name__ == "__main__":
    nodes = build_skill_nodes(_skill_roots())
    print(json.dumps(density_stats(nodes, build_edges(nodes)), indent=2))
