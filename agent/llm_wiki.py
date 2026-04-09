from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any

import yaml

from hermes_constants import get_hermes_home

DEFAULT_WIKI_PATH = Path("~/wiki").expanduser()
_PAGE_DIRS = ("entities", "concepts", "comparisons", "queries")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "i", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "what", "when", "where", "which", "who", "why", "with", "you", "your",
}
_KIND_ORDER = [
    "preferences",
    "prohibitions",
    "project-conventions",
    "environment-facts",
    "workflow-rules",
]
_KIND_SPECS: dict[str, dict[str, Any]] = {
    "preferences": {
        "title": "Preferences",
        "summary": "Durable operator and user preferences that shape tone, routing, and presentation.",
        "tags": ["memory", "preference", "profile"],
        "links": ["[[user-profile]]", "[[llm-wiki-memory-lane]]", "[[workflow-rules]]"],
    },
    "prohibitions": {
        "title": "Prohibitions",
        "summary": "Durable things Hermes should not do, skip, or quietly route around.",
        "tags": ["memory", "constraint", "prohibition"],
        "links": ["[[user-profile]]", "[[persistent-memory-notes]]", "[[workflow-rules]]"],
    },
    "project-conventions": {
        "title": "Project Conventions",
        "summary": "Project-specific conventions, routing rules, naming constraints, and architecture decisions.",
        "tags": ["memory", "project", "convention"],
        "links": ["[[persistent-memory-notes]]", "[[memory-v2]]", "[[environment-facts]]"],
    },
    "environment-facts": {
        "title": "Environment Facts",
        "summary": "Concrete facts about installed tools, paths, versions, runtimes, hosts, and machine state.",
        "tags": ["memory", "environment", "fact"],
        "links": ["[[persistent-memory-notes]]", "[[memory-v2]]", "[[project-conventions]]"],
    },
    "workflow-rules": {
        "title": "Workflow Rules",
        "summary": "Operational rules for how Hermes should inspect, build, verify, and report work.",
        "tags": ["memory", "workflow", "rule"],
        "links": ["[[user-profile]]", "[[persistent-memory-notes]]", "[[prohibitions]]"],
    },
}
_TOPIC_LINK_STOPWORDS = _STOPWORDS | {
    "always", "answers", "avoid", "concise", "default", "durable", "entry", "facts",
    "first", "guidance", "hermes", "memory", "operator", "persistent", "prefer",
    "prefers", "preference", "preferences", "project", "rule", "rules", "sessions",
    "should", "user", "users", "uses", "using", "wiki", "workflow",
}


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def get_configured_wiki_path() -> Path:
    config_path = get_hermes_home() / "config.yaml"
    if config_path.exists():
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            config = {}
        skills = config.get("skills") or {}
        skill_config = skills.get("config") or {}
        wiki_cfg = skill_config.get("wiki") or {}
        raw_path = wiki_cfg.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            return Path(raw_path).expanduser()
    return DEFAULT_WIKI_PATH


def wiki_exists(wiki_path: Path | None = None) -> bool:
    path = (wiki_path or get_configured_wiki_path()).expanduser()
    return path.exists() and path.is_dir()


def _wiki_schema_file(wiki_path: Path) -> Path:
    return wiki_path / "SCHEMA.md"


def _wiki_index_file(wiki_path: Path) -> Path:
    return wiki_path / "index.md"


def _wiki_log_file(wiki_path: Path) -> Path:
    return wiki_path / "log.md"


def _memory_notes_page(wiki_path: Path) -> Path:
    return wiki_path / "concepts" / "persistent-memory-notes.md"


def _user_profile_page(wiki_path: Path) -> Path:
    return wiki_path / "entities" / "user-profile.md"


def _memory_lane_page(wiki_path: Path) -> Path:
    return wiki_path / "concepts" / "llm-wiki-memory-lane.md"


def _memory_v2_page(wiki_path: Path) -> Path:
    return wiki_path / "concepts" / "memory-v2.md"


def _kind_page(wiki_path: Path, kind: str) -> Path:
    return wiki_path / "concepts" / f"{kind}.md"


def _topic_page(wiki_path: Path, slug: str) -> Path:
    return wiki_path / "queries" / f"memory-topic-{slug}.md"


def _raw_distill_page(wiki_path: Path, slug: str) -> Path:
    return wiki_path / "queries" / f"raw-distill-{slug}.md"


def _tokenize(text: str) -> list[str]:
    return [
        token for token in re.split(r"[^a-z0-9]+", (text or "").lower())
        if token and token not in _STOPWORDS and len(token) > 1
    ]


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug or "untitled"


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        return text
    return parts[2]


def _frontmatter(text: str) -> dict[str, Any]:
    if not text.startswith("---\n"):
        return {}
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        return {}
    try:
        data = yaml.safe_load(parts[1]) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _first_summary_line(text: str) -> str:
    body = _strip_frontmatter(text)
    for raw in body.splitlines():
        line = " ".join(raw.strip().lstrip("#>-*• ").split())
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith(("## ", "links", "source", "sources:")):
            continue
        return line[:180]
    return ""


def _body_excerpt(text: str, query_tokens: set[str]) -> str:
    lines = []
    for raw in _strip_frontmatter(text).splitlines():
        line = " ".join(raw.strip().lstrip("#>-*• ").split())
        if not line:
            continue
        lines.append(line)
    for line in lines:
        hay = set(_tokenize(line))
        if hay & query_tokens:
            return line[:220]
    return (lines[0][:220] if lines else "")


def _provenance_score(text: str) -> int:
    score = 0
    lowered = text.lower()
    active = lowered.count("status=active")
    superseded = lowered.count("status=superseded")
    forgotten = lowered.count("status=forgotten")
    hard = lowered.count("strength=hard_rule")
    soft = lowered.count("strength=soft_rule")
    hint = lowered.count("strength=hint")

    score += active * 40
    score -= superseded * 30
    score -= forgotten * 60
    score += hard * 20
    score += soft * 5
    score -= hint * 2
    return score


def _page_files(wiki_path: Path) -> list[Path]:
    pages: list[Path] = []
    for dirname in _PAGE_DIRS:
        directory = wiki_path / dirname
        if directory.exists():
            pages.extend(sorted(directory.glob("*.md")))
    return pages


def _wikilinks(text: str) -> list[str]:
    return [match.group(1).strip() for match in re.finditer(r"\[\[([^\]]+)\]\]", text)]


def _ensure_dirs(wiki_path: Path) -> None:
    wiki_path.mkdir(parents=True, exist_ok=True)
    (wiki_path / "raw" / "articles").mkdir(parents=True, exist_ok=True)
    (wiki_path / "raw" / "papers").mkdir(parents=True, exist_ok=True)
    (wiki_path / "raw" / "transcripts").mkdir(parents=True, exist_ok=True)
    (wiki_path / "raw" / "assets").mkdir(parents=True, exist_ok=True)
    for dirname in _PAGE_DIRS:
        (wiki_path / dirname).mkdir(parents=True, exist_ok=True)


def _write_if_changed(path: Path, content: str) -> bool:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if old == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def _render_frontmatter(
    *,
    title: str,
    created: str,
    page_type: str,
    tags: list[str],
    sources: list[str],
    generated_by: str | None = None,
    kind: str | None = None,
) -> list[str]:
    lines = [
        "---",
        f"title: {title}",
        f"created: {created}",
        f"updated: {_today()}",
        f"type: {page_type}",
        f"tags: [{', '.join(tags)}]",
        f"sources: [{', '.join(sources)}]",
    ]
    if generated_by:
        lines.append(f"generated_by: {generated_by}")
    if kind:
        lines.append(f"memory_kind: {kind}")
    lines.append("---")
    return lines


def _topic_title_from_entry(entry: str) -> str:
    words = [token for token in _tokenize(entry) if token not in _TOPIC_LINK_STOPWORDS]
    if not words:
        words = _tokenize(entry)
    if not words:
        return "Memory Topic"
    display = words[:4]
    return "Topic: " + " ".join(word.upper() if word.isdigit() else word.capitalize() for word in display)


def _topic_slug_from_entry(entry: str) -> str:
    words = [token for token in _tokenize(entry) if token not in _TOPIC_LINK_STOPWORDS]
    if not words:
        words = _tokenize(entry)
    base = "-".join(words[:4]).strip("-")
    digest = hashlib.sha1(entry.strip().encode("utf-8")).hexdigest()[:8]
    if not base:
        base = "entry"
    slug = _slugify(base)
    if len(slug) > 48:
        slug = slug[:48].rstrip("-")
    return f"{slug}-{digest}"


def _rebuild_index(wiki_path: Path) -> None:
    _ensure_dirs(wiki_path)
    sections = [
        ("Entities", wiki_path / "entities"),
        ("Concepts", wiki_path / "concepts"),
        ("Comparisons", wiki_path / "comparisons"),
        ("Queries", wiki_path / "queries"),
    ]
    entries: list[Path] = []
    lines = [
        "# Wiki Index",
        "",
        "> Content catalog for Hermes local knowledge and persistent memory.",
        f"> Last updated: {_today()} | Total pages: 0",
        "",
    ]
    for label, directory in sections:
        lines.append(f"## {label}")
        section_entries = []
        for path in sorted(directory.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            summary = _first_summary_line(text) or "No summary yet."
            section_entries.append(f"- [[{path.stem}]] — {summary}")
            entries.append(path)
        if section_entries:
            lines.extend(section_entries)
        lines.append("")
    lines[3] = f"> Last updated: {_today()} | Total pages: {len(entries)}"
    _wiki_index_file(wiki_path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _append_log(wiki_path: Path, action: str, subject: str, details: list[str] | None = None) -> None:
    _ensure_dirs(wiki_path)
    path = _wiki_log_file(wiki_path)
    if not path.exists():
        path.write_text(
            "# Wiki Log\n\n"
            "> Chronological record of wiki actions.\n\n"
            f"## [{_today()}] create | Wiki initialized\n"
            "- Domain: Hermes local wiki, durable memory, and user profile recall\n",
            encoding="utf-8",
        )
    lines = ["", f"## [{_today()}] {action} | {subject}"]
    for detail in details or []:
        clean = " ".join(str(detail).split())
        if clean:
            lines.append(f"- {clean}")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _raw_source_files(wiki_path: Path) -> list[Path]:
    files: list[Path] = []
    for lane in (wiki_path / "raw" / "articles", wiki_path / "raw" / "transcripts"):
        if lane.exists():
            files.extend(sorted(lane.glob("*.md")))
    return files


def _title_from_markdown(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("# ").strip()[:120]
    return path.stem.replace("-", " ").title()


def _distilled_takeaways(text: str, limit: int = 4) -> list[str]:
    takeaways: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith(("- ", "* ")):
            takeaways.append(stripped[2:].strip()[:220])
        elif stripped.endswith(":"):
            continue
        elif len(takeaways) < limit:
            takeaways.append(stripped[:220])
        if len(takeaways) >= limit:
            break
    return takeaways[:limit]


def _distill_implications(path: Path, title: str, takeaways: list[str]) -> list[str]:
    text = f"{title}\n" + "\n".join(takeaways)
    lowered = text.lower()
    if "open-source" in lowered or "open weight" in lowered or "llm" in lowered or "model" in lowered:
        return [
            "Map the source claims against existing Hermes routing before treating them as doctrine.",
            "Promote only the deployment or evaluation implications that survive a crew pressure-test.",
        ]
    if "tim dillon" in lowered or "ray kump" in lowered or "cadence" in lowered:
        return [
            "Use this source as cadence/voice reference, not as factual AI signal.",
            "Extract persona texture into query or concept pages before it influences live shell behavior.",
        ]
    lane = path.parent.name
    return [
        f"Distill this {lane} source into one durable claim before promoting it beyond raw intake.",
        "Add cross-links to the existing wiki only after the implication is explicit.",
    ]


def distill_raw_sources_to_wiki(wiki_path: Path | None = None, limit_per_lane: int | None = None) -> list[Path]:
    path = ensure_wiki_scaffold(wiki_path)
    changed: list[Path] = []
    raw_files = _raw_source_files(path)
    lane_counts: dict[str, int] = {}
    for raw_path in raw_files:
        lane = raw_path.parent.name
        lane_counts[lane] = lane_counts.get(lane, 0) + 1
        if limit_per_lane is not None and lane_counts[lane] > limit_per_lane:
            continue
        text = raw_path.read_text(encoding="utf-8")
        title = _title_from_markdown(raw_path, text)
        slug = _slugify(raw_path.stem)
        page = _raw_distill_page(path, slug)
        existing = page.read_text(encoding="utf-8") if page.exists() else ""
        created = str(_frontmatter(existing).get("created") or _today())
        takeaways = _distilled_takeaways(text)
        implications = _distill_implications(raw_path, title, takeaways)
        source_rel = str(raw_path.relative_to(path)).replace("\\", "/")
        body = _render_frontmatter(
            title=f"Distillation: {title}",
            created=created,
            page_type="query",
            tags=["wiki", "distillation", lane[:-1] if lane.endswith("s") else lane],
            sources=[source_rel],
            generated_by="raw-distiller-v1",
        )
        body.extend([
            "",
            f"First-class distilled page generated from raw/{lane} intake.",
            "",
            "## Distilled takeaways",
            "",
        ])
        if takeaways:
            body.extend(f"- {item}" for item in takeaways)
        else:
            body.append("- No distilled takeaways yet.")
        body.extend([
            "",
            "## Routing/build implications",
            "",
        ])
        body.extend(f"- {item}" for item in implications)
        body.extend([
            "",
            "## Provenance",
            "",
            f"- source_file={source_rel}",
            f"- lane={lane}",
            f"- title={title}",
            "",
            "## Links",
            "",
            "- [[llm-wiki-memory-lane]]",
            "- [[memory-v2]]",
        ])
        content = "\n".join(body) + "\n"
        if _write_if_changed(page, content):
            changed.append(page)
    if changed:
        _rebuild_index(path)
        _append_log(path, "update", "raw source distillation", [f"Updated {p.relative_to(path)}" for p in changed])
    return changed


def ensure_wiki_scaffold(wiki_path: Path | None = None) -> Path:
    path = (wiki_path or get_configured_wiki_path()).expanduser()
    _ensure_dirs(path)

    _write_if_changed(
        _wiki_schema_file(path),
        "# Wiki Schema\n\n"
        "## Domain\n"
        "Hermes local knowledge base: durable memory, user profile, and compiled wiki recall.\n\n"
        "## Conventions\n"
        "- File names: lowercase, hyphens, no spaces\n"
        "- Every wiki page starts with YAML frontmatter\n"
        "- Use [[wikilinks]] for internal links\n"
        "- Update index.md and log.md whenever a page changes\n"
        "- Keep pages concise and operational\n\n"
        "## Typed Memory Kinds\n"
        "- preferences\n- prohibitions\n- project-conventions\n- environment-facts\n- workflow-rules\n\n"
        "## Tag Taxonomy\n"
        "- memory\n- user\n- wiki\n- recall\n- profile\n- preference\n- workflow\n- constraint\n- prohibition\n- project\n- convention\n- environment\n- context\n",
    )

    core_pages = {
        _memory_v2_page(path): (
            "Memory V2",
            "Memory v2 is the durable local memory lane for Hermes, with markdown exports and a sqlite-backed source of truth when enabled.",
            ["[[persistent-memory-notes]]", "[[user-profile]]", "[[llm-wiki-memory-lane]]", "[[project-conventions]]", "[[environment-facts]]"],
            ["memory", "wiki", "context"],
        ),
        _memory_lane_page(path): (
            "LLM Wiki Memory Lane",
            "The llm-wiki lane compiles durable memory and user profile facts into typed markdown pages for local recall before model calls.",
            ["[[memory-v2]]", "[[persistent-memory-notes]]", "[[user-profile]]", "[[preferences]]", "[[workflow-rules]]"],
            ["wiki", "recall", "context"],
        ),
    }
    for page, (title, summary, links, tags) in core_pages.items():
        existing = page.read_text(encoding="utf-8") if page.exists() else ""
        created = str(_frontmatter(existing).get("created") or _today())
        content = (
            "---\n"
            f"title: {title}\n"
            f"created: {created}\n"
            f"updated: {_today()}\n"
            "type: concept\n"
            f"tags: [{', '.join(tags)}]\n"
            "sources: []\n"
            "---\n\n"
            f"{summary}\n\n"
            "## Links\n\n" + "\n".join(f"- {link}" for link in links) + "\n"
        )
        _write_if_changed(page, content)

    if not _wiki_log_file(path).exists():
        _append_log(path, "create", "Wiki initialized", ["Domain: Hermes local wiki, durable memory, and user profile recall"])
    _rebuild_index(path)
    return path


def _normalize_memory_kind(kind: str | None, entry: str, target: str) -> str:
    raw = (kind or "").strip().lower()
    mapping = {
        "preference": "preferences",
        "preferences": "preferences",
        "profile": "preferences",
        "prohibition": "prohibitions",
        "prohibitions": "prohibitions",
        "constraint": "prohibitions",
        "project": "project-conventions",
        "project-convention": "project-conventions",
        "project-conventions": "project-conventions",
        "environment": "environment-facts",
        "environment-fact": "environment-facts",
        "environment-facts": "environment-facts",
        "workflow": "workflow-rules",
        "workflow-rule": "workflow-rules",
        "workflow-rules": "workflow-rules",
        "rule": "workflow-rules",
    }
    normalized = mapping.get(raw)
    if normalized in _KIND_ORDER:
        return normalized
    return classify_memory_entry(entry, target=target)


def classify_memory_entry(entry: str, target: str = "memory") -> str:
    text = " ".join((entry or "").strip().lower().split())
    if not text:
        return "environment-facts" if target == "memory" else "preferences"

    score = {kind: 0 for kind in _KIND_ORDER}
    if target == "user":
        score["preferences"] += 2
        score["workflow-rules"] += 1
    else:
        score["environment-facts"] += 1
        score["project-conventions"] += 1

    if re.search(r"\b(do not|don't|never|must not|cannot|can't|no hidden|forbid|forbidden|prohibit|prohibited|without)\b", text):
        score["prohibitions"] += 5
    if re.search(r"\b(prefers?|likes?|wants?|trusts?|favors?|skew|tone|voice|style|communication|brief|concise)\b", text):
        score["preferences"] += 4
    if re.search(r"\b(if|when|before|after|always|first|then|by default|check .* first|run .* first|verify|background .* by default|show .* preview|do not ask twice)\b", text):
        score["workflow-rules"] += 4
    if re.search(r"\b(convention|conventions|routing|route|provider|api|project|repo|repository|crew|tars|outside-hull|identity|support-groups|main api|main provider)\b", text):
        score["project-conventions"] += 3
    if re.search(r"\b(installed|available|runs|running|version|python|path|located|under|uses sqlite|configured|nginx|windows|linux|rtx|model|home is|project path)\b", text):
        score["environment-facts"] += 5

    if target == "user" and re.search(r"\b(use|show|check|background|verify|preview|inspect|read|test)\b", text):
        score["workflow-rules"] += 2
    if target == "memory" and re.search(r"\b(prefers?|wants?|never use|do not)\b", text):
        score["preferences"] += 1
        score["prohibitions"] += 1

    return max(_KIND_ORDER, key=lambda kind: (score[kind], -_KIND_ORDER.index(kind)))


def _render_kind_links(kind: str) -> list[str]:
    links = list(_KIND_SPECS[kind]["links"])
    sibling_links = [f"[[{other}]]" for other in _KIND_ORDER if other != kind]
    return links + sibling_links[:2]


def _format_provenance_bits(item: dict[str, Any]) -> str:
    bits = []
    if item.get("row_id") is not None:
        bits.append(f"row_id={item['row_id']}")
    if item.get("status"):
        bits.append(f"status={item['status']}")
    if item.get("strength"):
        bits.append(f"strength={item['strength']}")
    if item.get("created_in_session_id"):
        bits.append(f"session={item['created_in_session_id']}")
    if item.get("replaced_by"):
        bits.append(f"replaced_by={item['replaced_by']}")
    if item.get("forgotten_by"):
        bits.append(f"forgotten_by={item['forgotten_by']}")
    if item.get("source"):
        bits.append(f"source_file={item['source']}")
    return " | ".join(bits) if bits else "No sqlite provenance available."


def _render_kind_page(
    path: Path,
    kind: str,
    items: list[dict[str, str]],
    changed: list[Path],
) -> None:
    page = _kind_page(path, kind)
    existing = page.read_text(encoding="utf-8") if page.exists() else ""
    created = str(_frontmatter(existing).get("created") or _today())
    spec = _KIND_SPECS[kind]

    lines = _render_frontmatter(
        title=spec["title"],
        created=created,
        page_type="concept",
        tags=spec["tags"],
        sources=["memories/MEMORY.md", "memories/USER.md"],
        generated_by="memory-mirror-v2",
        kind=kind,
    )
    lines.extend(["", spec["summary"], "", "## Classified entries", ""])
    if items:
        for item in items:
            lines.append(f"- {item['entry']} ([[memory-topic-{item['topic_slug']}]] | source: {item['target']}])")
    else:
        lines.append("- No classified entries yet.")
    lines.extend(["", "## Provenance", ""])
    if items:
        for item in items:
            lines.append(f"- {item['entry']} :: {_format_provenance_bits(item)}")
    else:
        lines.append("- No sqlite provenance available.")
    lines.extend(["", "## Links", ""])
    lines.extend(f"- {link}" for link in _render_kind_links(kind))
    content = "\n".join(lines) + "\n"
    if _write_if_changed(page, content):
        changed.append(page)


def _render_topic_page(
    path: Path,
    slug: str,
    topic: dict[str, Any],
    changed: list[Path],
) -> None:
    page = _topic_page(path, slug)
    existing = page.read_text(encoding="utf-8") if page.exists() else ""
    created = str(_frontmatter(existing).get("created") or _today())
    kinds = sorted(topic["kinds"], key=lambda kind: _KIND_ORDER.index(kind))
    tags = ["memory", "wiki", "topic"] + [kind.replace("-", "_") for kind in kinds]

    lines = _render_frontmatter(
        title=topic["title"],
        created=created,
        page_type="query",
        tags=tags,
        sources=sorted(topic["sources"]),
        generated_by="memory-mirror-v2",
        kind=kinds[0] if len(kinds) == 1 else "mixed",
    )
    lines.extend([
        "",
        f"Generated topic page for durable memory entries about {topic['title'].replace('Topic: ', '').lower()}.",
        "",
        "## Related entries",
        "",
    ])
    for item in topic["entries"]:
        lines.append(f"- {item['entry']} (kind: [[{item['kind']}]] | source: {item['target']}])")
    lines.extend(["", "## Provenance", ""])
    for item in topic["entries"]:
        lines.append(f"- {item['entry']} :: {_format_provenance_bits(item)}")
    lines.extend(["", "## Links", ""])
    kind_links = [f"[[{kind}]]" for kind in kinds]
    lines.extend(f"- {link}" for link in kind_links + ["[[persistent-memory-notes]]", "[[user-profile]]"])
    content = "\n".join(lines) + "\n"
    if _write_if_changed(page, content):
        changed.append(page)


def _cleanup_generated_pages(path: Path, keep_paths: set[Path], changed: list[Path]) -> None:
    removable = set(_kind_page(path, kind) for kind in _KIND_ORDER)
    removable.update((path / "queries").glob("memory-topic-*.md"))
    for page in sorted(removable):
        if page in keep_paths or not page.exists():
            continue
        page.unlink()
        changed.append(page)


def sync_memory_store_to_wiki(
    memory_entries: list[str],
    user_entries: list[str],
    wiki_path: Path | None = None,
    *,
    memory_records: list[dict[str, Any]] | None = None,
    user_records: list[dict[str, Any]] | None = None,
) -> list[Path]:
    path = ensure_wiki_scaffold(wiki_path)
    changed: list[Path] = []
    memory_source = str((get_hermes_home() / "memories" / "MEMORY.md").relative_to(get_hermes_home())).replace("\\", "/")
    user_source = str((get_hermes_home() / "memories" / "USER.md").relative_to(get_hermes_home())).replace("\\", "/")

    classified: list[dict[str, str]] = []
    grouped: dict[str, list[dict[str, str]]] = {kind: [] for kind in _KIND_ORDER}
    topics: dict[str, dict[str, Any]] = {}

    record_map = {
        "memory": list(memory_records or []),
        "user": list(user_records or []),
    }

    for target, entries, source in (
        ("memory", memory_entries, memory_source),
        ("user", user_entries, user_source),
    ):
        records_by_content: dict[str, list[dict[str, Any]]] = {}
        for record in record_map[target]:
            content = str(record.get("content") or "")
            records_by_content.setdefault(content, []).append(record)
        for entry in entries:
            matched_record = None
            bucket = records_by_content.get(entry) or []
            if bucket:
                matched_record = bucket.pop(0)
            kind = _normalize_memory_kind((matched_record or {}).get("kind"), entry, target)
            topic_slug = _topic_slug_from_entry(entry)
            item = {
                "entry": entry,
                "target": target,
                "kind": kind,
                "topic_slug": topic_slug,
                "topic_title": _topic_title_from_entry(entry),
                "source": source,
                "row_id": (matched_record or {}).get("id"),
                "strength": (matched_record or {}).get("strength"),
                "created_in_session_id": (matched_record or {}).get("created_in_session_id"),
                "replaced_by": (matched_record or {}).get("replaced_by"),
                "forgotten_by": (matched_record or {}).get("forgotten_by"),
                "status": (matched_record or {}).get("status") or "active",
            }
            classified.append(item)
            grouped[kind].append(item)
            topic = topics.setdefault(topic_slug, {
                "title": item["topic_title"],
                "entries": [],
                "kinds": set(),
                "sources": set(),
            })
            topic["entries"].append(item)
            topic["kinds"].add(kind)
            topic["sources"].add(source)

    keep_paths: set[Path] = set()

    page_specs = [
        (
            _memory_notes_page(path),
            "Persistent Memory Notes",
            "concept",
            ["memory", "context", "project"],
            [memory_source],
            "Persistent memory notes compile durable local facts and project conventions remembered across sessions.",
            memory_entries,
            ["[[memory-v2]]", "[[llm-wiki-memory-lane]]", "[[user-profile]]", "[[project-conventions]]", "[[environment-facts]]"],
            "No durable memory notes saved yet.",
            "memory",
        ),
        (
            _user_profile_page(path),
            "User Profile",
            "entity",
            ["user", "profile", "preference"],
            [user_source],
            "The user profile compiles durable preferences, workflow rules, and operator guidance remembered across sessions.",
            user_entries,
            ["[[memory-v2]]", "[[persistent-memory-notes]]", "[[llm-wiki-memory-lane]]", "[[preferences]]", "[[prohibitions]]", "[[workflow-rules]]"],
            "No durable user profile facts saved yet.",
            "user",
        ),
    ]

    for page, title, page_type, tags, sources, summary, entries, links, empty_line, target in page_specs:
        keep_paths.add(page)
        existing = page.read_text(encoding="utf-8") if page.exists() else ""
        created = str(_frontmatter(existing).get("created") or _today())
        target_items = [item for item in classified if item["target"] == target]
        body = _render_frontmatter(
            title=title,
            created=created,
            page_type=page_type,
            tags=tags,
            sources=sources,
            generated_by="memory-mirror-v2",
        )
        body.extend(["", summary, "", "## Current compiled entries", ""])
        if entries:
            body.extend(f"- {entry}" for entry in entries)
        else:
            body.append(f"- {empty_line}")
        body.extend(["", "## Typed routing", ""])
        if target_items:
            for kind in _KIND_ORDER:
                kind_items = [item for item in target_items if item["kind"] == kind]
                if not kind_items:
                    continue
                body.append(f"### {_KIND_SPECS[kind]['title']}")
                body.append("")
                for item in kind_items:
                    body.append(f"- {item['entry']} ([[{kind}]] | [[memory-topic-{item['topic_slug']}]]])")
                body.append("")
        else:
            body.append("- No typed entries yet.")
            body.append("")
        body.extend(["## Provenance", ""])
        if target_items:
            for item in target_items:
                body.append(f"- {item['entry']} :: {_format_provenance_bits(item)}")
        else:
            body.append("- No sqlite provenance available.")
        body.extend(["", "## Links", ""])
        body.extend(f"- {link}" for link in links)
        content = "\n".join(body) + "\n"
        if _write_if_changed(page, content):
            changed.append(page)

    for kind in _KIND_ORDER:
        page = _kind_page(path, kind)
        keep_paths.add(page)
        _render_kind_page(path, kind, grouped[kind], changed)

    for slug, topic in sorted(topics.items()):
        page = _topic_page(path, slug)
        keep_paths.add(page)
        _render_topic_page(path, slug, topic, changed)

    _cleanup_generated_pages(path, keep_paths, changed)

    if changed:
        _rebuild_index(path)
        detail_lines = []
        for p in changed:
            try:
                detail_lines.append(f"Updated {p.relative_to(path)}")
            except ValueError:
                detail_lines.append(f"Updated {p}")
        _append_log(path, "update", "memory mirror sync", detail_lines)
    return changed


def retrieve_relevant_wiki_pages(query: str, wiki_path: Path | None = None, limit: int = 3) -> list[dict[str, str]]:
    path = (wiki_path or get_configured_wiki_path()).expanduser()
    if not query.strip() or not wiki_exists(path):
        return []
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return []

    pages = _page_files(path)
    page_map: dict[str, tuple[Path, str]] = {}
    for page in pages:
        page_map[page.stem] = (page, page.read_text(encoding="utf-8"))

    scored: list[tuple[int, str, dict[str, str]]] = []
    direct_hits: dict[str, int] = {}
    for page, text in page_map.values():
        frontmatter = _frontmatter(text)
        title = str(frontmatter.get("title") or page.stem.replace("-", " ").title())
        body = _strip_frontmatter(text)
        page_tokens = set(_tokenize(f"{title}\n{body}"))
        overlap = len(query_tokens & page_tokens)
        if overlap <= 0:
            continue
        score = overlap * 100
        parent = page.parent.name
        if page.stem in _KIND_SPECS:
            score += 45
        if page.name.startswith("memory-topic-"):
            score += 30
        if parent == "entities":
            score += 20
        elif parent == "queries":
            score += 10
        score += _provenance_score(text)
        summary = _first_summary_line(text)
        excerpt = _body_excerpt(text, query_tokens)
        scored.append((score, page.name, {
            "title": title,
            "path": str(page.relative_to(path)).replace("\\", "/"),
            "summary": summary,
            "excerpt": excerpt,
        }))
        direct_hits[page.stem] = score

    for stem, base_score in direct_hits.items():
        _, text = page_map[stem]
        for linked_stem in _wikilinks(text):
            linked = page_map.get(linked_stem)
            if not linked or linked_stem in direct_hits:
                continue
            linked_page, linked_text = linked
            linked_frontmatter = _frontmatter(linked_text)
            linked_title = str(linked_frontmatter.get("title") or linked_page.stem.replace("-", " ").title())
            summary = _first_summary_line(linked_text)
            excerpt = _body_excerpt(linked_text, query_tokens)
            score = max(1, base_score - 35)
            if linked_page.stem in _KIND_SPECS:
                score += 10
            score += _provenance_score(linked_text)
            scored.append((score, linked_page.name, {
                "title": linked_title,
                "path": str(linked_page.relative_to(path)).replace("\\", "/"),
                "summary": summary,
                "excerpt": excerpt,
            }))

    deduped: dict[str, tuple[int, str, dict[str, str]]] = {}
    for score, name, item in scored:
        existing = deduped.get(item["path"])
        if existing is None or score > existing[0]:
            deduped[item["path"]] = (score, name, item)

    ranked = sorted(deduped.values(), key=lambda item: (item[0], item[1]), reverse=True)
    return [item for _, _, item in ranked[:limit]]


def render_wiki_prefetch(query: str, wiki_path: Path | None = None, limit: int = 3) -> str:
    hits = retrieve_relevant_wiki_pages(query, wiki_path=wiki_path, limit=limit)
    if not hits:
        return ""
    lines = ["## LLM Wiki Recall"]
    for hit in hits:
        summary = hit.get("summary") or hit.get("excerpt") or ""
        suffix = f" — {summary}" if summary else ""
        lines.append(f"- {hit.get('title', 'Untitled')} ({hit.get('path', '')}){suffix}")
    return "\n".join(lines)
