"""Knowledge intake routing for Skill/Agent/KB candidates.

This module classifies raw knowledge snippets into the next best home:
skill candidate, agent candidate, shared domain knowledge, workspace-level
knowledge, playbook, or project-local note. The Obsidian source of truth is
the recovered HermesNous graph renamed as the HermesAgent vault: MOC,
AI_MEMORY, OBSIDIAN_LINK_INDEX, AI_SKILL_ROUTER, SKILL_GRAPH, and the existing
sources/knowledge/lessons/patterns/playbooks/review-queue/skills layers.
"""

from __future__ import annotations

import json
import os
import re
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from hermes_constants import get_hermes_home


DESTINATIONS = {
    "skill_candidate",
    "agent_candidate",
    "domain_knowledge",
    "workspace_knowledge",
    "playbook_candidate",
    "project_note",
}

_INTAKE_DIRS = {
    "skill_candidate": "skill-candidates",
    "agent_candidate": "agent-candidates",
    "domain_knowledge": "domain-kb-candidates",
    "workspace_knowledge": "workspace-knowledge",
    "playbook_candidate": "playbook-candidates",
    "project_note": "project-notes",
}

_PRIMARY_LAYER_DIRS = [
    "sources",
    "knowledge",
    "lessons",
    "patterns",
    "playbooks",
    "skills",
    "training-packs",
    "docs",
    "review-queue",
]

_DESTINATION_MERGE_DIRS = {
    "skill_candidate": ["skills", "docs", "review-queue"],
    "agent_candidate": ["docs", "skills/conductor", "review-queue"],
    "domain_knowledge": ["knowledge", "patterns", "lessons", "sources", "review-queue"],
    "workspace_knowledge": ["knowledge", "docs", "patterns", "sources", "review-queue"],
    "playbook_candidate": ["playbooks", "patterns", "lessons", "review-queue"],
    "project_note": ["sources", "knowledge", "lessons", "reports", "review-queue"],
}

_DESTINATION_KEYWORDS = {
    "skill_candidate": [
        "skill", "when to use", "prerequisites", "how to run", "procedure",
        "pitfalls", "verification", "script", "template", "tool", "api",
        "repeatable", "workflow", "steps", "checklist", "command",
    ],
    "agent_candidate": [
        "agent", "role", "persona", "profile", "responsibility", "owner",
        "handoff", "delegate", "routing", "orchestrator", "worker",
        "dispatcher", "kanban lane", "team", "subagent",
    ],
    "playbook_candidate": [
        "playbook", "runbook", "sop", "standard operating", "incident",
        "rollback", "phase", "gate", "operating loop", "escalation",
        "recovery", "release process",
    ],
    "domain_knowledge": [
        "pattern", "lesson", "architecture", "gotcha", "best practice",
        "anti-pattern", "configuration", "integration", "migration",
        "workaround", "cross-project", "reusable", "shared",
    ],
    "workspace_knowledge": [
        "portfolio", "workspace", "roadmap", "strategy", "vision",
        "business model", "project cluster", "ecosystem", "dependency",
        "dependencies", "priority", "priorities", "principle",
        "decision principle", "glossary", "taxonomy", "positioning",
        "all projects", "across projects", "ภาพรวม", "กลยุทธ์",
        "โปรเจกต์ทั้งหมด", "ความสัมพันธ์", "roadmap",
    ],
    "project_note": [
        "project", "repo", "repository", "local path", "this app",
        "current project", "specific", "customer", "environment",
    ],
}


@dataclass
class IntakeClassification:
    title: str
    destination: str
    confidence: float
    domains: List[str]
    source_project: str
    rationale: List[str]
    scores: Dict[str, float]
    related_files: List[Dict[str, str]]
    recommended_action: str
    merge_candidates: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_vault_path(vault_path: Optional[Path] = None) -> Path:
    """Resolve the HermesAgent Obsidian vault path."""
    if vault_path is not None:
        return Path(vault_path)
    env_path = os.environ.get("OBSIDIAN_VAULT_PATH")
    if env_path:
        return Path(env_path)
    return Path.home() / "ObsidianVault" / "HermesAgent"


def resolve_skills_root(skills_root: Optional[Path] = None) -> Path:
    """Resolve the runtime skills root."""
    if skills_root is not None:
        return Path(skills_root)
    local = Path.cwd() / ".hermes" / "skills"
    if local.exists():
        return local
    return get_hermes_home() / "skills"


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "untitled"


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _source_block(source_url: Optional[str], source_title: Optional[str], content: str) -> Dict[str, str]:
    if not source_url:
        return {}
    parsed = urlparse(source_url)
    platform = parsed.netloc.lower().replace("www.", "")
    return {
        "type": "url",
        "url": source_url,
        "platform": platform,
        "title": source_title or "",
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "content_hash": _content_hash(content),
        "access_status": "pasted_by_user" if content.strip() else "url_only",
    }


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text.lower()) if len(t) > 2}


def _keyword_score(text: str, keywords: Iterable[str]) -> Tuple[float, List[str]]:
    text_lower = text.lower()
    hits = [kw for kw in keywords if kw in text_lower]
    if not hits:
        return 0.0, []
    return min(1.0, len(hits) / 4.0), hits


def _vault_link(vault: Path, path: Path, label: Optional[str] = None) -> str:
    try:
        rel = path.relative_to(vault).with_suffix("")
        target = str(rel).replace("\\", "/")
    except ValueError:
        target = path.stem
    return f"[[{target}|{label or path.stem}]]"


def _read_title(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return path.stem
    match = re.search(r"^title:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip().strip('"') if match else path.stem


def _skill_metadata(skill_file: Path) -> Dict[str, str]:
    text = skill_file.read_text(encoding="utf-8", errors="ignore")
    name = skill_file.parent.name
    desc = ""
    category = skill_file.parent.parent.name if skill_file.parent.parent.name != "skills" else "uncategorized"
    for key, value in re.findall(r"^(name|description|category):\s*(.+)$", text, re.MULTILINE):
        value = value.strip().strip('"').strip("'")
        if key == "name" and value:
            name = value
        elif key == "description":
            desc = value
        elif key == "category":
            category = value
    return {"name": name, "description": desc, "category": category}


def _find_related_files(
    *,
    vault: Path,
    skills_root: Path,
    content: str,
    title: str,
    source_project: str,
    domains: List[str],
    destination: str,
) -> List[Dict[str, str]]:
    """Find related vault and skill files with simple lexical matching."""
    related: List[Dict[str, str]] = []
    query_tokens = _tokens(f"{title}\n{content}")

    seed_files = [
        vault / "MOC.md",
        vault / "AI_MEMORY.md",
        vault / "docs" / "OBSIDIAN_LINK_INDEX.md",
        vault / "docs" / "AI_SKILL_ROUTER.md",
        vault / "docs" / "SKILL_GRAPH.md",
        vault / "knowledge" / "Knowledge Operating Rules.md",
    ]
    if destination == "agent_candidate":
        seed_files.extend([
            vault / "docs" / "SYSTEM_AGENTS_AND_FLOW.md",
            vault / "skills" / "conductor" / "routing-table.md",
            vault / "skills" / "conductor" / "nous-conductor.md",
        ])
    if destination == "workspace_knowledge":
        seed_files.extend([
            vault / "docs" / "HERMESNOUS_ALL_PROJECTS_PLAN.md",
            vault / "knowledge" / "owner-context-v0.1.md",
            vault / "knowledge" / "README.md",
        ])
    if destination == "playbook_candidate":
        seed_files.append(vault / "playbooks" / "README.md")

    for seed in seed_files:
        if seed.exists():
            related.append({
                "kind": "hub",
                "path": str(seed),
                "link": _vault_link(vault, seed),
                "reason": "source_of_truth_hub",
            })

    layer_hits: List[Tuple[int, Path]] = []
    for dirname in _DESTINATION_MERGE_DIRS.get(destination, _PRIMARY_LAYER_DIRS):
        directory = vault / dirname
        if not directory.exists():
            continue
        for note_file in sorted(directory.glob("**/*.md")):
            try:
                text = note_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            overlap = len(query_tokens & _tokens(f"{note_file.stem}\n{text}"))
            if overlap or (source_project and source_project.lower() in note_file.stem.lower()):
                layer_hits.append((overlap, note_file))
    for _, note_file in sorted(layer_hits, key=lambda row: row[0], reverse=True)[:8]:
        related.append({
            "kind": "vault_note",
            "path": str(note_file),
            "link": _vault_link(vault, note_file),
            "reason": "knowledge_graph_overlap",
        })

    # Backward-compatible support for the short-lived standalone vault shape.
    if destination == "agent_candidate":
        for role_note in sorted((vault / "roles").glob("*.md")) if (vault / "roles").exists() else []:
            if role_note.name == "README.md":
                continue
            role_text = role_note.read_text(encoding="utf-8", errors="ignore").lower()
            if role_note.stem in content.lower() or query_tokens & _tokens(role_text):
                related.append({
                    "kind": "role",
                    "path": str(role_note),
                    "link": _vault_link(vault, role_note, role_note.stem),
                    "reason": "role_keyword_overlap",
                })

    if destination == "workspace_knowledge":
        workspace_dir = vault / "workspace"
        if workspace_dir.exists():
            workspace_hits: List[Tuple[int, Path]] = []
            for note_file in sorted(workspace_dir.glob("*.md")):
                text = note_file.read_text(encoding="utf-8", errors="ignore")
                overlap = len(query_tokens & _tokens(f"{note_file.stem}\n{text}"))
                if overlap or note_file.name == "README.md":
                    workspace_hits.append((overlap, note_file))
            for _, note_file in sorted(workspace_hits, key=lambda row: row[0], reverse=True)[:8]:
                related.append({
                    "kind": "workspace",
                    "path": str(note_file),
                    "link": _vault_link(vault, note_file),
                    "reason": "workspace_keyword_overlap",
                })

    skill_hits: List[Tuple[int, Path, Dict[str, str]]] = []
    if skills_root.exists():
        for skill_file in skills_root.glob("**/SKILL.md"):
            try:
                meta = _skill_metadata(skill_file)
                skill_text = f"{meta['name']} {meta['description']} {skill_file.parent.name}"
                overlap = len(query_tokens & _tokens(skill_text))
                if overlap:
                    skill_hits.append((overlap, skill_file, meta))
            except OSError:
                continue
    for _, skill_file, meta in sorted(skill_hits, key=lambda row: row[0], reverse=True)[:8]:
        related.append({
            "kind": "skill",
            "path": str(skill_file),
            "link": f"[[skills/{meta['category']}/{meta['name']}|{meta['name']}]]",
            "reason": "skill_keyword_overlap",
        })

    return related[:20]


def find_merge_candidates(
    classification: IntakeClassification,
    content: str,
    source_url: Optional[str] = None,
    vault_path: Optional[Path] = None,
    threshold: float = 0.35,
) -> List[Dict[str, Any]]:
    """Find existing notes that should receive this knowledge instead of a new file."""
    vault = resolve_vault_path(vault_path)
    query = _tokens(f"{classification.title}\n{content}")
    if not query and not source_url:
        return []

    candidate_dirs = [
        vault / dirname
        for dirname in _DESTINATION_MERGE_DIRS.get(classification.destination, ["review-queue"])
    ]
    # Always compare against review-queue notes to prevent duplicates while an
    # item is still awaiting review.
    candidate_dirs.append(vault / "review-queue")

    candidates: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for directory in candidate_dirs:
        if not directory.exists():
            continue
        for note in sorted(directory.glob("*.md")):
            if str(note) in seen:
                continue
            seen.add(str(note))
            try:
                text = note.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            same_source = bool(source_url and source_url in text)
            note_tokens = _tokens(f"{note.stem}\n{text}")
            overlap = len(query & note_tokens)
            union = max(len(query | note_tokens), 1)
            jaccard = overlap / union
            score = 1.0 if same_source else jaccard

            if classification.destination == "workspace_knowledge" and note.parts[-2] in {"knowledge", "docs"}:
                score += 0.12
            if note.parent.name == "review-queue":
                score += 0.08
            score = min(1.0, score)

            if score >= threshold:
                candidates.append({
                    "path": str(note),
                    "link": _vault_link(vault, note),
                    "score": round(score, 3),
                    "reason": "same_source_url" if same_source else "content_similarity",
                    "recommended_merge": score >= 0.8,
                })

    return sorted(candidates, key=lambda item: item["score"], reverse=True)[:8]


def classify_knowledge(
    title: str,
    content: str,
    source_project: str = "",
    preferred_destination: Optional[str] = None,
    source_url: Optional[str] = None,
    source_title: Optional[str] = None,
    vault_path: Optional[Path] = None,
    skills_root: Optional[Path] = None,
) -> IntakeClassification:
    """Classify knowledge into the next best destination."""
    vault = resolve_vault_path(vault_path)
    skills_dir = resolve_skills_root(skills_root)
    combined = f"{title}\n{source_title or ''}\n{source_url or ''}\n{content}"
    rationale: List[str] = []
    scores: Dict[str, float] = {}
    keyword_hits: Dict[str, List[str]] = {}

    for destination, keywords in _DESTINATION_KEYWORDS.items():
        score, hits = _keyword_score(combined, keywords)
        scores[destination] = score
        keyword_hits[destination] = hits

    domains: List[str] = []
    try:
        from agent.knowledge_domains import DomainRelevanceMatcher
        matcher = DomainRelevanceMatcher(vault_path=vault)
        domains = matcher.match_knowledge(combined)
        if source_project:
            project_domains = matcher.classify(source_project)
            domains = list(dict.fromkeys(project_domains + domains))
    except Exception:
        domains = []

    if domains:
        scores["domain_knowledge"] = max(scores["domain_knowledge"], min(1.0, 0.35 + 0.1 * len(domains)))
        rationale.append(f"matched domains: {', '.join(domains[:5])}")

    if source_project:
        scores["project_note"] = max(scores["project_note"], 0.35)

    # Explicit mentions should dominate close scores.
    text_lower = combined.lower()
    if "skill" in text_lower or "skill.md" in text_lower:
        scores["skill_candidate"] += 0.35
    if "agent" in text_lower or "profile" in text_lower:
        scores["agent_candidate"] += 0.35
    if "playbook" in text_lower or "runbook" in text_lower:
        scores["playbook_candidate"] += 0.3
    if (
        "portfolio" in text_lower
        or "roadmap" in text_lower
        or "strategy" in text_lower
        or "all projects" in text_lower
        or "ภาพรวม" in text_lower
        or "โปรเจกต์ทั้งหมด" in text_lower
    ):
        scores["workspace_knowledge"] += 0.35

    for key in list(scores):
        scores[key] = min(1.0, scores[key])

    if preferred_destination in DESTINATIONS:
        destination = preferred_destination
        scores[destination] = max(scores[destination], 0.95)
        rationale.append(f"preferred destination override: {destination}")
    else:
        destination = max(scores, key=scores.get)

    confidence = max(scores[destination], 0.2)
    hits = keyword_hits.get(destination) or []
    if hits:
        rationale.append(f"{destination} keywords: {', '.join(hits[:8])}")
    if not rationale:
        rationale.append("no strong reusable pattern detected; keep as project-local note")
        destination = "project_note"
        confidence = max(confidence, 0.25)

    related = _find_related_files(
        vault=vault,
        skills_root=skills_dir,
        content=content,
        title=title,
        source_project=source_project,
        domains=domains,
        destination=destination,
    )

    action_map = {
        "skill_candidate": "queue for human review, then create or patch a Hermes skill with skill_manage",
        "agent_candidate": "queue for human review, then create or update an agent/profile/role card",
        "domain_knowledge": "add to knowledge review queue for promote_knowledge approval",
        "workspace_knowledge": "queue for review, then merge into workspace portfolio/strategy/roadmap knowledge",
        "playbook_candidate": "queue as Obsidian playbook candidate",
        "project_note": "store as project-local context unless it becomes reusable later",
    }

    classification = IntakeClassification(
        title=title.strip() or "Untitled knowledge",
        destination=destination,
        confidence=round(confidence, 3),
        domains=domains[:8],
        source_project=source_project,
        rationale=rationale,
        scores={k: round(v, 3) for k, v in scores.items()},
        related_files=related,
        recommended_action=action_map[destination],
        merge_candidates=[],
    )
    classification.merge_candidates = find_merge_candidates(
        classification,
        content=content,
        source_url=source_url,
        vault_path=vault,
    )
    return classification


def write_intake_note(
    classification: IntakeClassification,
    content: str,
    source_url: Optional[str] = None,
    source_title: Optional[str] = None,
    vault_path: Optional[Path] = None,
) -> Path:
    """Write or merge a pending intake note into the recovered Obsidian graph."""
    vault = resolve_vault_path(vault_path)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date = datetime.now().strftime("%Y-%m-%d")
    queue_dir = vault / "review-queue"
    queue_dir.mkdir(parents=True, exist_ok=True)

    domains = "[" + ", ".join(classification.domains) + "]"
    related_lines = "\n".join(
        f"- {item['link']} ({item['kind']}; {item['reason']})"
        for item in classification.related_files
    ) or "- None"
    rationale_lines = "\n".join(f"- {r}" for r in classification.rationale)
    score_lines = "\n".join(f"- `{k}`: {v}" for k, v in classification.scores.items())
    source = _source_block(source_url, source_title, content)
    source_yaml = ""
    source_section = "- None"
    if source:
        source_yaml = (
            "source:\n"
            f"  type: {source['type']}\n"
            f"  url: {source['url']}\n"
            f"  platform: {source['platform']}\n"
            f"  title: {source['title']}\n"
            f"  fetched_at: {source['fetched_at']}\n"
            f"  content_hash: {source['content_hash']}\n"
            f"  access_status: {source['access_status']}\n"
        )
        source_section = "\n".join(f"- `{key}`: {value}" for key, value in source.items())

    merge_lines = "\n".join(
        f"- {item['link']} score `{item['score']}` ({item['reason']})"
        for item in classification.merge_candidates
    ) or "- None"

    for candidate in classification.merge_candidates:
        if not candidate.get("recommended_merge"):
            continue
        note_path = Path(candidate["path"])
        if not note_path.exists():
            continue
        source = _source_block(source_url, source_title, content)
        source_lines = "\n".join(f"- `{key}`: {value}" for key, value in source.items()) or "- None"
        update = (
            f"\n\n## Hermes Intake Update - {date} - {classification.title}\n\n"
            f"- Intake type: `{classification.destination}`\n"
            f"- Confidence: `{classification.confidence}`\n"
            f"- Recommended action: {classification.recommended_action}\n\n"
            "### Source\n\n"
            f"{source_lines}\n\n"
            "### Captured Content\n\n"
            f"{content.strip()}\n\n"
            f"_Merged by Hermes Knowledge Intake Router at {now}._\n"
        )
        note_path.write_text(
            note_path.read_text(encoding="utf-8", errors="ignore").rstrip() + update,
            encoding="utf-8",
        )
        _update_intake_index(vault)
        return note_path

    base_slug = _slugify(classification.title)
    note_path = queue_dir / f"intake-{classification.destination}-{base_slug}.md"
    counter = 1
    while note_path.exists():
        note_path = queue_dir / f"intake-{classification.destination}-{base_slug}-{counter}.md"
        counter += 1

    body = (
        "---\n"
        f"title: {classification.title}\n"
        "tags:\n"
        "  - hermes-agent/intake\n"
        f"  - hermes-agent/{classification.destination}\n"
        "status: pending\n"
        f"intake_type: {classification.destination}\n"
        f"source_project: {classification.source_project or ''}\n"
        f"domains: {domains}\n"
        f"confidence: {classification.confidence}\n"
        f"created: {date}\n"
        f"updated: {date}\n"
        f"{source_yaml}"
        "---\n\n"
        f"# {classification.title}\n\n"
        "## Classification\n\n"
        f"- Destination: `{classification.destination}`\n"
        f"- Confidence: `{classification.confidence}`\n"
        f"- Recommended action: {classification.recommended_action}\n\n"
        "## Rationale\n\n"
        f"{rationale_lines}\n\n"
        "## Scores\n\n"
        f"{score_lines}\n\n"
        "## Related Files\n\n"
        f"{related_lines}\n\n"
        "## Source\n\n"
        f"{source_section}\n\n"
        "## Merge Candidates\n\n"
        f"{merge_lines}\n\n"
        "## Captured Content\n\n"
        f"{content.strip()}\n\n"
        f"_Captured by Hermes Knowledge Intake Router at {now}._\n"
    )
    note_path.write_text(body, encoding="utf-8")
    _update_intake_index(vault)
    return note_path


def _update_intake_index(vault: Path) -> None:
    queue_root = vault / "review-queue"
    queue_root.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        "title: Knowledge Intake Index",
        "tags:",
        "  - hermes-agent/intake",
        "status: active",
        f"updated: {datetime.now().strftime('%Y-%m-%d')}",
        "---",
        "",
        "# Knowledge Intake Index",
        "",
        "Pending routing index for knowledge that may become a Skill, Agent, knowledge note, playbook, pattern, source, or project note.",
        "",
        "Source of truth stays in [[MOC]], [[AI_MEMORY]], [[docs/OBSIDIAN_LINK_INDEX]], [[docs/AI_SKILL_ROUTER]], and [[docs/SKILL_GRAPH]].",
        "",
        "| Queue | Notes |",
        "|---|---:|",
    ]
    for destination in sorted(DESTINATIONS):
        count = len(list(queue_root.glob(f"intake-{destination}-*.md")))
        lines.append(f"| `{destination}` | {count} |")
    lines.append("")
    (queue_root / "intake-index.md").write_text("\n".join(lines), encoding="utf-8")


def list_intake_notes(
    destination: Optional[str] = None,
    vault_path: Optional[Path] = None,
) -> List[Dict[str, str]]:
    vault = resolve_vault_path(vault_path)
    queue_root = vault / "review-queue"
    destinations = [destination] if destination in _INTAKE_DIRS else list(_INTAKE_DIRS)
    notes: List[Dict[str, str]] = []
    for dest in destinations:
        for note in sorted(queue_root.glob(f"intake-{dest}-*.md")) if queue_root.exists() else []:
            notes.append({
                "destination": dest,
                "title": _read_title(note),
                "path": str(note),
                "link": _vault_link(vault, note),
            })
    return notes


def sync_obsidian_maps(
    vault_path: Optional[Path] = None,
    skills_root: Optional[Path] = None,
    write_runtime_db: bool = True,
) -> Dict[str, Any]:
    """Sync lightweight route metadata into the recovered Obsidian graph."""
    vault = resolve_vault_path(vault_path)
    skills_dir = resolve_skills_root(skills_root)
    vault.mkdir(parents=True, exist_ok=True)

    skill_cards = _existing_skill_cards(vault, skills_dir)
    agent_cards = _existing_agent_cards(vault)
    _update_intake_index(vault)
    relation_path = _write_relation_index(vault, skill_cards, agent_cards)

    route_db = {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "vault": str(vault),
        "skills_root": str(skills_dir),
        "skill_cards": skill_cards,
        "agent_cards": agent_cards,
        "workspace_index": str(vault / "docs" / "OBSIDIAN_LINK_INDEX.md"),
        "relation_index": str(relation_path),
        "routing_destinations": sorted(DESTINATIONS),
    }
    db_path = vault / "review-queue" / "knowledge-routes.json"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_text(json.dumps(route_db, indent=2), encoding="utf-8")

    if write_runtime_db:
        runtime_path = get_hermes_home() / "knowledge_routes.json"
        runtime_path.write_text(json.dumps(route_db, indent=2), encoding="utf-8")
        route_db["runtime_db"] = str(runtime_path)

    _ensure_moc_links(vault)
    return route_db


def _existing_skill_cards(vault: Path, skills_dir: Path) -> List[Dict[str, str]]:
    cards: List[Dict[str, str]] = []
    for note in sorted((vault / "skills").glob("**/*.md")) if (vault / "skills").exists() else []:
        if note.name == "README.md":
            continue
        cards.append({"name": note.stem, "path": str(note), "source_path": str(note)})
    if skills_dir.exists():
        for skill_file in sorted(skills_dir.glob("**/SKILL.md")):
            meta = _skill_metadata(skill_file)
            cards.append({"name": meta["name"], "path": str(skill_file), "source_path": str(skill_file)})
    return cards


def _existing_agent_cards(vault: Path) -> List[Dict[str, str]]:
    candidates = [
        vault / "docs" / "SYSTEM_AGENTS_AND_FLOW.md",
        vault / "docs" / "AI_SKILL_ROUTER.md",
        vault / "docs" / "SKILL_GRAPH.md",
    ]
    return [
        {"name": path.stem, "path": str(path), "role_path": str(path)}
        for path in candidates
        if path.exists()
    ]


def _skill_role_map(vault: Path) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    roles_dir = vault / "roles"
    if not roles_dir.exists():
        return mapping
    for role_note in sorted(roles_dir.glob("*.md")):
        if role_note.name == "README.md":
            continue
        text = role_note.read_text(encoding="utf-8", errors="ignore")
        for skill in re.findall(r"\[\[([^|\]]+)(?:\|[^\]]+)?\]\]", text):
            skill_name = skill.split("/")[-1]
            mapping.setdefault(skill_name, []).append(role_note.stem)
    return mapping


def _sync_skill_cards(vault: Path, skills_dir: Path, skill_to_roles: Dict[str, List[str]]) -> List[Dict[str, str]]:
    cards: List[Dict[str, str]] = []
    skills_index: Dict[str, List[Dict[str, str]]] = {}
    for skill_file in sorted(skills_dir.glob("**/SKILL.md")) if skills_dir.exists() else []:
        meta = _skill_metadata(skill_file)
        category = _slugify(meta["category"])
        name = _slugify(meta["name"])
        card_dir = vault / "skills" / category
        card_dir.mkdir(parents=True, exist_ok=True)
        card = card_dir / f"{name}.md"
        roles = skill_to_roles.get(meta["name"], []) + skill_to_roles.get(skill_file.parent.name, [])
        roles = sorted(set(roles))
        role_links = ", ".join(f"[[agents/{r}|{r}]]" for r in roles) or "None"
        source = str(skill_file)
        body = (
            "---\n"
            f"title: {meta['name']}\n"
            "tags:\n"
            "  - hermes-agent/skill\n"
            f"  - skill/{category}\n"
            "status: active\n"
            f"skill_name: {meta['name']}\n"
            f"category: {category}\n"
            f"source_path: {source}\n"
            f"updated: {datetime.now().strftime('%Y-%m-%d')}\n"
            "---\n\n"
            f"# {meta['name']}\n\n"
            f"{meta['description'] or 'No description found.'}\n\n"
            "## Runtime Source\n\n"
            f"`{source}`\n\n"
            "## Agent Links\n\n"
            f"{role_links}\n"
        )
        card.write_text(body, encoding="utf-8")
        record = {
            "name": meta["name"],
            "category": category,
            "path": str(card),
            "source_path": source,
        }
        cards.append(record)
        skills_index.setdefault(category, []).append(record)

    lines = [
        "---",
        "title: Hermes Runtime Skills",
        "tags:",
        "  - hermes-agent/skills",
        "status: active",
        f"updated: {datetime.now().strftime('%Y-%m-%d')}",
        "---",
        "",
        "# Hermes Runtime Skills",
        "",
        "| Skill | Category | Source |",
        "|---|---|---|",
    ]
    for card in sorted(cards, key=lambda row: (row["category"], row["name"])):
        link = _vault_link(vault, Path(card["path"]), card["name"])
        lines.append(f"| {link} | `{card['category']}` | `{card['source_path']}` |")
    (vault / "skills").mkdir(exist_ok=True)
    (vault / "skills" / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cards


def _sync_agent_cards(vault: Path, skill_to_roles: Dict[str, List[str]]) -> List[Dict[str, str]]:
    agents_dir = vault / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    roles_dir = vault / "roles"
    cards: List[Dict[str, str]] = []
    for role_note in sorted(roles_dir.glob("*.md")) if roles_dir.exists() else []:
        if role_note.name == "README.md":
            continue
        text = role_note.read_text(encoding="utf-8", errors="ignore")
        skills = re.findall(r"\[\[([^|\]]+)(?:\|[^\]]+)?\]\]", text)
        skill_links = []
        for skill in skills:
            skill_name = skill.split("/")[-1]
            skill_links.append(f"[[{skill_name}]]")
        card = agents_dir / f"{role_note.stem}.md"
        body = (
            "---\n"
            f"title: {role_note.stem}\n"
            "tags:\n"
            "  - hermes-agent/agent\n"
            "status: active\n"
            f"role_note: roles/{role_note.stem}\n"
            f"updated: {datetime.now().strftime('%Y-%m-%d')}\n"
            "---\n\n"
            f"# {role_note.stem}\n\n"
            f"Role source: [[roles/{role_note.stem}|{role_note.stem}]]\n\n"
            "## Default Skills\n\n"
            + ("\n".join(f"- {link}" for link in skill_links) if skill_links else "- None")
            + "\n\n## Source Summary\n\n"
            + text.split("---", 2)[-1].strip()
            + "\n"
        )
        card.write_text(body, encoding="utf-8")
        cards.append({"name": role_note.stem, "path": str(card), "role_path": str(role_note)})

    lines = [
        "---",
        "title: Hermes Agent Profiles",
        "tags:",
        "  - hermes-agent/agents",
        "status: active",
        f"updated: {datetime.now().strftime('%Y-%m-%d')}",
        "---",
        "",
        "# Hermes Agent Profiles",
        "",
        "| Agent | Role Source |",
        "|---|---|",
    ]
    for card in cards:
        lines.append(f"| {_vault_link(vault, Path(card['path']), card['name'])} | {_vault_link(vault, Path(card['role_path']))} |")
    (agents_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cards


def _write_relation_index(vault: Path, skill_cards: List[Dict[str, str]], agent_cards: List[Dict[str, str]]) -> Path:
    path = vault / "docs" / "OBSIDIAN_LINK_INDEX.md"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Obsidian Link Index\n", encoding="utf-8")
    existing = path.read_text(encoding="utf-8", errors="ignore").rstrip()
    marker = "\n## Knowledge Intake Routing\n"
    body = (
        marker +
        "\n"
        "This section connects intake decisions to the existing HermesAgent source-of-truth graph.\n\n"
        "| Destination | Meaning | Next Action |\n"
        "|---|---|---|\n"
        "| `skill_candidate` | Repeatable procedural capability | Review, then create/patch a Skill |\n"
        "| `agent_candidate` | Role/profile/orchestration behavior | Review, then create/update Agent profile |\n"
        "| `domain_knowledge` | Reusable cross-project technical knowledge | Review, then promote to domain KB |\n"
        "| `workspace_knowledge` | Portfolio, strategy, roadmap, glossary, dependency, or ecosystem knowledge | Review, then merge into workspace notes |\n"
        "| `playbook_candidate` | Global operating procedure | Review, then write playbook |\n"
        "| `project_note` | Project-specific context | Keep with project card/context pack |\n\n"
        "## Indexes\n\n"
        "- [[review-queue/intake-index|Knowledge Intake Index]]\n"
        "- [[review-queue/queue|Review Queue]]\n"
        "- [[skills/README|Runtime Skills]]\n"
        "- [[docs/AI_SKILL_ROUTER|AI Skill Router]]\n"
        "- [[docs/SKILL_GRAPH|Skill Graph]]\n"
        "- [[knowledge/README|Knowledge]]\n"
        "- [[sources/README|Sources]]\n"
        "- [[patterns/README|Patterns]]\n"
        "- [[playbooks/README|Playbooks]]\n\n"
        f"Synced skill cards: {len(skill_cards)}\n\n"
        f"Synced agent/router hubs: {len(agent_cards)}\n"
    )
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip()
    path.write_text(existing + "\n" + body, encoding="utf-8")
    return path


def _sync_workspace_index(vault: Path) -> Path:
    """Create the workspace-level knowledge index and standard note slots."""
    workspace_dir = vault / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y-%m-%d")
    slots = {
        "portfolio-map.md": "Portfolio Map",
        "strategy.md": "Strategy",
        "project-clusters.md": "Project Clusters",
        "roadmap.md": "Roadmap",
        "decision-principles.md": "Decision Principles",
        "glossary.md": "Glossary",
        "dependencies.md": "Dependencies",
        "open-questions.md": "Open Questions",
    }
    for filename, title in slots.items():
        path = workspace_dir / filename
        if path.exists():
            continue
        path.write_text(
            "---\n"
            f"title: {title}\n"
            "tags:\n"
            "  - hermes-agent/workspace-knowledge\n"
            "status: draft\n"
            "scope: workspace\n"
            f"updated: {date}\n"
            "---\n\n"
            f"# {title}\n\n"
            "Pending reviewed workspace knowledge.\n",
            encoding="utf-8",
        )

    index = workspace_dir / "README.md"
    lines = [
        "---",
        "title: Workspace Knowledge",
        "tags:",
        "  - hermes-agent/workspace-knowledge",
        "status: active",
        "scope: workspace",
        f"updated: {date}",
        "---",
        "",
        "# Workspace Knowledge",
        "",
        "Portfolio-level knowledge for the full HermesAgent project workspace.",
        "",
        "| Note | Purpose |",
        "|---|---|",
        "| [[workspace/portfolio-map|Portfolio Map]] | How projects relate across the workspace |",
        "| [[workspace/strategy|Strategy]] | Strategic direction and positioning |",
        "| [[workspace/project-clusters|Project Clusters]] | Product/customer/tooling clusters |",
        "| [[workspace/roadmap|Roadmap]] | Cross-project sequencing and priorities |",
        "| [[workspace/decision-principles|Decision Principles]] | Reusable decision rules |",
        "| [[workspace/glossary|Glossary]] | Vocabulary and shared concepts |",
        "| [[workspace/dependencies|Dependencies]] | Cross-project dependencies |",
        "| [[workspace/open-questions|Open Questions]] | Questions awaiting user review |",
        "",
        "New unreviewed items arrive in [[intake/workspace-knowledge|workspace_knowledge intake]].",
    ]
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def _ensure_moc_links(vault: Path) -> None:
    moc = vault / "MOC.md"
    if not moc.exists():
        return
    text = moc.read_text(encoding="utf-8", errors="ignore")
    additions = [
        "- [[review-queue/intake-index|Knowledge Intake Index]]",
        "- [[review-queue/queue|Review Queue]]",
        "- [[skills/README|Runtime Skills]]",
        "- [[docs/AI_SKILL_ROUTER|AI Skill Router]]",
        "- [[docs/SKILL_GRAPH|Skill Graph]]",
        "- [[docs/OBSIDIAN_LINK_INDEX|Knowledge Center / Link Index]]",
    ]
    missing = [line for line in additions if line not in text]
    if not missing:
        return
    marker = "## Entry Points\n"
    if marker in text:
        text = text.replace(marker, marker + "\n".join(missing) + "\n", 1)
    else:
        text = text.rstrip() + "\n\n## Knowledge Routing\n\n" + "\n".join(missing) + "\n"
    moc.write_text(text, encoding="utf-8")
