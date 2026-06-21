"""
Skill Graph plugin — knowledge graph for skills discovery.

Builds a SQLite graph from SKILL.md relations, exposes:
- ``skill_graph_search`` — find skills by intent
- ``skill_load`` — load a skill's full content from graph-managed dirs

Maintains the graph incrementally across sessions.

SKILL.md relations format (frontmatter):
    metadata:
      hermes:
        relations:
          - type: depends_on
            target: another-skill
            properties:
              reason: "why"
              strength: strong|medium|weak

Config (in config.yaml):
    skills:
      config:
        skill-graph:
          source_dirs:
            - ~/path/to/extra/skills
"""

from __future__ import annotations
import json
import logging
import os
import re
import sqlite3
import threading
import time
import yaml
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_RELATION_TYPES = {
    "depends_on", "supported_by", "alternative_to", "complemented_by",
    "similar_to", "belongs_to_domain", "used_in_workflow", "supersedes",
}

GRAPH_DB_FILENAME = "skill-graph.db"
GRAPH_LOCK = threading.Lock()

# ── SQLite schema ──────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS skill_nodes (
    name        TEXT PRIMARY KEY,
    category    TEXT DEFAULT '',
    description TEXT DEFAULT '',
    tags        TEXT DEFAULT '[]',      -- JSON array
    file_path   TEXT DEFAULT '',
    content_hash TEXT DEFAULT '',
    last_parsed REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS skill_edges (
    source      TEXT NOT NULL REFERENCES skill_nodes(name),
    target      TEXT NOT NULL REFERENCES skill_nodes(name),
    rel_type    TEXT NOT NULL,
    properties  TEXT DEFAULT '{}',       -- JSON dict
    PRIMARY KEY (source, target, rel_type)
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON skill_edges(source);
CREATE INDEX IF NOT EXISTS idx_edges_target ON skill_edges(target);
CREATE INDEX IF NOT EXISTS idx_edges_type   ON skill_edges(rel_type);

CREATE VIRTUAL TABLE IF NOT EXISTS skill_fts USING fts5(
    name, category, description, tags,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS skill_terms (
    term        TEXT NOT NULL,
    skill_name  TEXT NOT NULL REFERENCES skill_nodes(name),
    strength    REAL DEFAULT 1.0,
    source      TEXT DEFAULT 'tag',   -- 'name' | 'tag' | 'description'
    PRIMARY KEY (term, skill_name)
);

CREATE INDEX IF NOT EXISTS idx_terms_term ON skill_terms(term);
CREATE INDEX IF NOT EXISTS idx_terms_skill ON skill_terms(skill_name);

CREATE TABLE IF NOT EXISTS skill_term_stats (
    skill_name   TEXT NOT NULL REFERENCES skill_nodes(name),
    term         TEXT NOT NULL,
    search_count INTEGER DEFAULT 1,
    load_count   INTEGER DEFAULT 0,
    PRIMARY KEY (skill_name, term)
);
"""

# ── Database helpers ────────────────────────────────────────────────────────


def _db_path() -> Path:
    """Return path to graph DB under the active Hermes home."""
    hermes_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    return hermes_home / "personal" / GRAPH_DB_FILENAME


def _get_conn() -> sqlite3.Connection:
    """Get a thread-safe connection."""
    db_path = _db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    """Ensure schema exists."""
    conn.executescript(SCHEMA_SQL)
    conn.commit()


# ── Skill directory discovery ──────────────────────────────────────────────


def _read_source_dirs_from_config() -> list[Path]:
    """Read ``skills.config.skill-graph.source_dirs`` from config.yaml."""
    dirs: list[Path] = []
    try:
        from hermes_cli.config import load_config
        config = load_config()
        sg_config = (
            config
            .get("skills", {})
            .get("config", {})
            .get("skill-graph", {})
        )
        raw = sg_config.get("source_dirs", [])
        if isinstance(raw, list):
            for entry in raw:
                p = Path(os.path.expandvars(os.path.expanduser(str(entry))))
                if p.exists():
                    dirs.append(p.resolve())
    except Exception:
        pass
    return dirs


def _find_all_skills_dirs() -> list[Path]:
    """Return list of directories to scan for SKILL.md files.

    Always includes:
      1. The global ~/.hermes/skills/ (Hermes built-in + any PS symlinks)
      2. The global ~/.hermes/hermes-agent/skills/ (Hermes built-in source)
      3. The current profile's skills/ dir (via HERMES_HOME)
      4. Configured source_dirs (user's extra paths, e.g. PS repo)
      5. Hermes config's external_dirs
    """
    global_hermes = Path.home() / ".hermes"
    hermes_home = Path(os.environ.get("HERMES_HOME", global_hermes))
    dirs: list[Path] = []

    # 1. Global ~/.hermes/skills/ (always scanned, not profile-relative)
    global_skills = global_hermes / "skills"
    if global_skills.exists():
        dirs.append(global_skills)

    # 2. Hermes Agent built-in skills (global)
    agent_skills = global_hermes / "hermes-agent" / "skills"
    if agent_skills.exists():
        dirs.append(agent_skills)

    # 3. Current profile's skills/ dir (if inside a named profile,
    #    HERMES_HOME != global_hermes, this picks up profile-specific skills)
    profile_skills = hermes_home / "skills"
    if profile_skills.exists() and str(profile_skills) != str(global_skills):
        dirs.append(profile_skills)

    # 4. Hermes-agent-created skills (discoverable via graph, not in prompt)
    agent_created_dir = hermes_home / "skill-graph" / "agent-created"
    if agent_created_dir.exists():
        dirs.append(agent_created_dir)

    # 5. Configured source dirs (skill-graph's own extra paths)
    source_dirs = _read_source_dirs_from_config()
    dirs.extend(source_dirs)

    # 6. External skill dirs from Hermes config
    try:
        from hermes_cli.config import load_config
        config = load_config()
        ext_dirs = config.get("skills", {}).get("external_dirs", [])
        for ed in ext_dirs:
            p = Path(os.path.expandvars(os.path.expanduser(str(ed))))
            if p.exists():
                dirs.append(p.resolve())
    except Exception:
        pass

    return dirs


def _find_skill_path(name: str) -> Path | None:
    """Find a SKILL.md by name across all configured dirs.

    Returns the first match, preferring ``~/.hermes/skills/`` when duplicates exist.
    """
    skill_dirs = _find_all_skills_dirs()
    skills = _scan_skill_mds(skill_dirs)
    found: Path | None = None
    primary_hint = str(Path.home() / ".hermes" / "skills")
    for n, path in skills:
        if n == name:
            if found is None:
                found = path
            elif str(path).startswith(primary_hint):
                found = path
                break
    return found


# ── SKILL.md scanner & parser ──────────────────────────────────────────────


def _scan_skill_mds(skill_dirs: list[Path]) -> list[tuple[str, Path]]:
    """Scan all skill directories for SKILL.md files.

    Returns list of (skill_name, skill_md_path).
    """
    results: list[tuple[str, Path]] = []
    seen_names: set[str] = set()

    for base_dir in skill_dirs:
        if not base_dir.exists():
            continue
        for cat_dir in base_dir.iterdir():
            if not cat_dir.is_dir() or cat_dir.name.startswith("."):
                continue
            # Flat layout: <name>/SKILL.md
            skill_md = cat_dir / "SKILL.md"
            if skill_md.exists():
                name = cat_dir.name
                if name not in seen_names:
                    seen_names.add(name)
                    results.append((name, skill_md))
                continue
            # Nested layout: <cat>/<name>/SKILL.md
            for name_dir in cat_dir.iterdir():
                if not name_dir.is_dir() or name_dir.name.startswith("."):
                    continue
                skill_md = name_dir / "SKILL.md"
                if skill_md.exists():
                    name = name_dir.name
                    if name not in seen_names:
                        seen_names.add(name)
                        results.append((name, skill_md))
    return results


def _parse_skill_md(path: Path) -> dict[str, Any]:
    """Parse SKILL.md and extract metadata for the graph.

    Returns dict with keys: name, category, description, tags, relations, content_hash
    """
    result: dict[str, Any] = {
        "name": path.parent.name,
        "category": "",
        "description": "",
        "tags": [],
        "relations": [],
        "content_hash": "",
    }

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        result["content_hash"] = str(hash(content))

        content_str = content.lstrip("\ufeff")
        if content_str.startswith("---"):
            end = content_str.find("---", 3)
            if end != -1:
                frontmatter = content_str[3:end].strip()
                try:
                    meta = yaml.safe_load(frontmatter) or {}
                except yaml.YAMLError:
                    meta = {}

                result["name"] = meta.get("name", result["name"])
                result["category"] = meta.get("category", "") or \
                    meta.get("metadata", {}).get("hermes", {}).get("category", "")
                result["description"] = meta.get("description", "")

                tags = meta.get("metadata", {}).get("hermes", {}).get("tags", [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",") if t.strip()]
                result["tags"] = list(tags) if isinstance(tags, list) else []

                relations = meta.get("metadata", {}).get("hermes", {}).get("relations", [])
                if isinstance(relations, list):
                    result["relations"] = relations

                related = meta.get("metadata", {}).get("hermes", {}).get("related_skills", [])
                if isinstance(related, str):
                    related = [t.strip() for t in related.split(",") if t.strip()]
                if isinstance(related, list):
                    for rs in related:
                        if not any(r.get("target") == rs for r in result["relations"]):
                            result["relations"].append({
                                "type": "similar_to",
                                "target": rs,
                                "properties": {"source": "legacy_related_skills"},
                            })
    except Exception as e:
        logger.debug("Failed to parse %s: %s", path, e)

    return result


# ── Term extraction ─────────────────────────────────────────────────────────

# English stop words filtered from description terms
_STOP_WORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "and", "but", "or", "if", "because", "until", "while", "about",
    "using", "via", "its", "their", "your", "this", "that", "these",
    "those", "which", "what", "who", "whom", "make", "made", "set", "get",
}


def _extract_skill_terms(name: str, tags: list[str], description: str) -> list[tuple[str, float, str]]:
    """Extract (term, strength, source) triples from a skill's metadata.

    Sources:
      - name: split on ``-_``, each part strength=1.0
      - tags: each tag verbatim, strength=1.0
      - description: English words (3+ chars) + Chinese phrases (2-6 chars), strength=0.7
    """
    seen: dict[str, tuple[float, str]] = {}

    def add(t: str, s: float, src: str):
        t = t.strip().lower()
        if t and (t not in seen or seen[t][0] < s):
            seen[t] = (s, src)

    # From name
    clean = name.replace("_", "-")
    for part in clean.split("-"):
        if len(part) > 1:
            add(part, 1.0, "name")

    # From tags
    for tag in tags:
        t = tag.strip().lower().replace("_", "-")
        if len(t) > 0:
            add(t, 1.0, "tag")

    # From description
    if description:
        for w in re.findall(r"[a-zA-Z][a-zA-Z]{2,}", description):
            w = w.lower()
            if w not in _STOP_WORDS and len(w) > 2:
                add(w, 0.7, "description")
        for c in re.findall(r"[\u4e00-\u9fff]{2,6}", description):
            if len(c) > 1:
                add(c, 0.7, "description")

    return [(t, s, src) for t, (s, src) in seen.items()]


# ── Graph sync ──────────────────────────────────────────────────────────────


def _dedup_skills(skills: list[tuple[str, Path]]) -> dict[str, Path]:
    """Deduplicate skills by name, preferring ~/.hermes/skills/ paths."""
    deduped: dict[str, Path] = {}
    primary_hint = str(Path.home() / ".hermes" / "skills")
    for name, path in skills:
        if name not in deduped:
            deduped[name] = path
        else:
            if str(path).startswith(primary_hint) and \
               not str(deduped[name]).startswith(primary_hint):
                deduped[name] = path
    return deduped


def _upsert_skill(conn: sqlite3.Connection, name: str, path: Path, now: float) -> dict[str, Any]:
    """Parse a SKILL.md and upsert its node + edges + FTS into the graph."""
    info = _parse_skill_md(path)
    tags_json = json.dumps(info["tags"], ensure_ascii=False)

    conn.execute(
        """INSERT OR REPLACE INTO skill_nodes
           (name, category, description, tags, file_path, content_hash, last_parsed)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (name, info["category"], info["description"],
         tags_json, str(path), info["content_hash"], now),
    )
    conn.execute("DELETE FROM skill_edges WHERE source = ?", (name,))

    for rel in info.get("relations", []):
        rel_type = rel.get("type", "similar_to")
        target = rel.get("target", "")
        props = rel.get("properties", {})
        if not target:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO skill_edges (source, target, rel_type, properties) VALUES (?, ?, ?, ?)",
            (name, target, rel_type, json.dumps(props, ensure_ascii=False)),
        )
        reverse_type = _reverse_type(rel_type)
        if reverse_type:
            reverse_props = {"inferred": True, "reason": f"reverse of {rel_type}"}
            conn.execute(
                "INSERT OR IGNORE INTO skill_edges (source, target, rel_type, properties) VALUES (?, ?, ?, ?)",
                (target, name, reverse_type, json.dumps(reverse_props)),
            )

    tags_text = " ".join(info.get("tags", []))
    conn.execute("DELETE FROM skill_fts WHERE name = ?", (name,))
    conn.execute(
        "INSERT INTO skill_fts (name, category, description, tags) VALUES (?, ?, ?, ?)",
        (name, info.get("category", ""), info.get("description", ""), tags_text),
    )

    # Upsert terms
    conn.execute("DELETE FROM skill_terms WHERE skill_name = ?", (name,))
    terms = _extract_skill_terms(name, info.get("tags", []), info.get("description", ""))
    for term_text, strength, source in terms:
        conn.execute(
            "INSERT OR IGNORE INTO skill_terms (term, skill_name, strength, source) VALUES (?, ?, ?, ?)",
            (term_text, name, strength, source),
        )

    return info


def _full_rebuild(conn: sqlite3.Connection) -> int:
    """Full rebuild: scan all skills dirs, rebuild graph from scratch."""
    skill_dirs = _find_all_skills_dirs()
    skills = _scan_skill_mds(skill_dirs)
    deduped = _dedup_skills(skills)

    now = time.time()
    conn.execute("DELETE FROM skill_edges")
    conn.execute("DELETE FROM skill_nodes")
    conn.execute("DELETE FROM skill_fts")
    conn.execute("DELETE FROM skill_terms")

    # Drop and recreate FTS table to ensure correct schema
    # (old DBs may have content='' which breaks JOIN queries)
    conn.execute("DROP TABLE IF EXISTS skill_fts")
    conn.execute("DROP TABLE IF EXISTS skill_fts_data")
    conn.execute("DROP TABLE IF EXISTS skill_fts_idx")
    conn.execute("DROP TABLE IF EXISTS skill_fts_docsize")
    conn.execute("DROP TABLE IF EXISTS skill_fts_config")
    conn.executescript("""
        CREATE VIRTUAL TABLE IF NOT EXISTS skill_fts USING fts5(
            name, category, description, tags,
            tokenize='porter unicode61'
        );
    """)

    for name, path in deduped.items():
        _upsert_skill(conn, name, path, now)
    conn.commit()

    logger.info("Skill graph rebuilt: %d skills", len(deduped))
    return len(deduped)


def _incremental_sync(conn: sqlite3.Connection) -> int:
    """Incremental sync: only re-parse skills whose mtime changed."""
    skill_dirs = _find_all_skills_dirs()
    skills = _scan_skill_mds(skill_dirs)
    deduped = _dedup_skills(skills)

    db_nodes: dict[str, dict[str, Any]] = {}
    for row in conn.execute(
        "SELECT name, content_hash, last_parsed, file_path FROM skill_nodes"
    ):
        db_nodes[row["name"]] = {
            "content_hash": row["content_hash"],
            "last_parsed": row["last_parsed"],
            "file_path": row["file_path"],
        }

    now = time.time()
    parsed_count = 0
    skipped_count = 0

    for name, path in deduped.items():
        existing = db_nodes.get(name)

        # Condition 1: not in DB → must upsert
        if existing is None:
            _upsert_skill(conn, name, path, now)
            parsed_count += 1
            continue

        # Condition 2: in DB but file changed (mtime newer or path relocated) → upsert
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0
        if mtime > existing["last_parsed"] or existing["file_path"] != str(path):
            _upsert_skill(conn, name, path, now)
            parsed_count += 1
            continue

        # Condition 3: in DB and unchanged → skip
        skipped_count += 1

    current_names = set(deduped.keys())
    db_names = set(db_nodes.keys())
    stale = db_names - current_names
    for name in stale:
        conn.execute("DELETE FROM skill_edges WHERE source = ? OR target = ?", (name, name))
        conn.execute("DELETE FROM skill_nodes WHERE name = ?", (name,))
        conn.execute("DELETE FROM skill_fts WHERE name = ?", (name,))
        conn.execute("DELETE FROM skill_terms WHERE skill_name = ?", (name,))

    conn.commit()
    logger.info(
        "Skill graph synced: %d parsed, %d unchanged, %d removed, %d total",
        parsed_count, skipped_count, len(stale), len(deduped),
    )
    return len(deduped)


def _sync_graph(conn: sqlite3.Connection) -> int:
    """Sync the graph with the filesystem. Full if empty, incremental otherwise."""
    count = conn.execute("SELECT COUNT(*) FROM skill_nodes").fetchone()[0]
    if count == 0:
        return _full_rebuild(conn)
    return _incremental_sync(conn)


def _update_single_skill(conn: sqlite3.Connection, skill_name: str) -> bool:
    """Re-parse a single skill and update its node + edges + FTS."""
    skill_path = _find_skill_path(skill_name)
    if skill_path is None:
        logger.debug("skill-graph: skill '%s' not found on disk, skipping", skill_name)
        return False
    now = time.time()
    _upsert_skill(conn, skill_name, skill_path, now)
    conn.commit()
    logger.debug("skill-graph: updated single skill '%s' (%s)", skill_name, skill_path)
    return True


def _reverse_type(rel_type: str) -> str | None:
    """Return the reverse relation type, or None if symmetric."""
    mapping = {
        "depends_on": "supported_by",
        "supported_by": "depends_on",
        "supersedes": "superseded_by",
        "superseded_by": "supersedes",
    }
    return mapping.get(rel_type)


# ── Graph search ────────────────────────────────────────────────────────────


def _search_graph(query: str, conn: sqlite3.Connection, limit: int = 10) -> list[dict[str, Any]]:
    """Search the skill graph by intent query."""
    results: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()

    # Phase 1: FTS5 direct search — include BM25 rank for normalized scoring
    fts_query = _fts_query(query)
    if fts_query:
        cursor = conn.execute(
            """SELECT n.name, n.category, n.description, n.tags, n.file_path, f.rank
               FROM skill_fts f
               JOIN skill_nodes n ON f.name = n.name
               WHERE skill_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (fts_query, limit * 2),
        )
        for row in cursor:
            name = row["name"]
            seen.add(name)
            tags = json.loads(row["tags"]) if row["tags"] else []
            # Normalize BM25: rank is negative (closer to 0 = better)
            bm25_score = 1.0 / (1.0 + abs(row["rank"]))
            results[name] = {
                "name": name,
                "category": row["category"],
                "description": row["description"],
                "tags": tags,
                "file_path": row["file_path"],
                "relevance": "direct",
                "relationship_chain": [],
                "score": bm25_score,
            }

    # Phase 2: Graph expansion — only fills gaps not found by FTS5
    expansion_queue = list(seen)
    while expansion_queue and len(results) < limit * 3:
        current = expansion_queue.pop(0)
        cursor = conn.execute(
            """SELECT e.target, e.rel_type, e.properties, n.category, n.description
               FROM skill_edges e
               JOIN skill_nodes n ON e.target = n.name
               WHERE e.source = ?
               ORDER BY e.rel_type
               LIMIT 5""",
            (current,),
        )
        for row in cursor:
            target = row["target"]
            if target in seen:
                continue
            seen.add(target)
            props = json.loads(row["properties"]) if row["properties"] else {}
            rel_type = row["rel_type"]
            reason = props.get("reason", f"via {rel_type}")
            _score = 0.8 if rel_type == "supersedes" else 0.5
            results[target] = {
                "name": target,
                "category": row["category"] or "",
                "description": row["description"] or "",
                "tags": [],
                "file_path": "",
                "relevance": "expansion",
                "relationship_chain": [f"{current} --({rel_type})--> {target}: {reason}"],
                "score": _score,
            }
            expansion_queue.append(target)

    # Phase 3: Tag match (existing)
    terms = _extract_terms(query)
    for term in terms:
        cursor = conn.execute(
            """SELECT name FROM skill_nodes WHERE instr(tags, ?) > 0""",
            (json.dumps(term),),
        )
        for row in cursor:
            if row["name"] not in seen:
                seen.add(row["name"])
                info = _get_node_info(conn, row["name"])
                if info:
                    info["relevance"] = "tag_match"
                    info["score"] = 0.7
                    results[info["name"]] = info

    # Phase 4: Term table match — direct term→skill lookup from the
    # skill_terms table (auto-extracted from name, tags, description).
    # This catches Chinese terms and split-name parts that FTS5 misses.
    for term in terms:
        cursor = conn.execute(
            """SELECT t.skill_name, t.strength, t.source, n.category, n.description
               FROM skill_terms t
               JOIN skill_nodes n ON t.skill_name = n.name
               WHERE t.term = ? AND t.skill_name NOT IN ({})
               ORDER BY t.strength DESC
               LIMIT 5""".format(",".join("?" for _ in seen)) if seen else "NULL",
            (term.lower(),) + (tuple(seen) if seen else ()),
        )
        for row in cursor:
            sname = row["skill_name"]
            seen.add(sname)
            results[sname] = {
                "name": sname,
                "category": row["category"] or "",
                "description": row["description"] or "",
                "tags": [],
                "file_path": "",
                "relevance": "term_match",
                "relationship_chain": [f"term[{term}] → {sname} (strength={row['strength']}, source={row['source']})"],
                "score": 0.8 * row["strength"],
            }

    # Phase 5: Term-based scoring boost + search stats
    # Apply per-(skill, term) load-ratio boost from skill_term_stats.
    # A skill that gets loaded more often for a given term ranks higher.
    for sname, r in results.items():
        matched_terms = [t for t in terms if t.lower() in (r.get("name", "") + r.get("description", "") + str(r.get("tags", ""))).lower()]
        if matched_terms:
            for mt in matched_terms:
                try:
                    conn.execute(
                        """INSERT INTO skill_term_stats (skill_name, term, search_count, load_count)
                           VALUES (?, ?, 1, 0)
                           ON CONFLICT(skill_name, term) DO UPDATE SET search_count = search_count + 1""",
                        (sname, mt),
                    )
                except Exception:
                    pass
            # Compute average load ratio for matched terms
            try:
                rows = conn.execute(
                    """SELECT term, load_count, search_count FROM skill_term_stats
                       WHERE skill_name = ? AND term IN ({})"""
                    .format(",".join("?" for _ in matched_terms)),
                    (sname,) + tuple(mt.lower() for mt in matched_terms),
                ).fetchall()
                if rows:
                    avg_ratio = sum(r["load_count"] / max(r["search_count"], 1) for r in rows) / len(rows)
                    r["score"] *= (1.0 + 0.15 * avg_ratio)
            except Exception:
                pass
    conn.commit()

    sorted_results = sorted(results.values(), key=lambda r: -r["score"])
    return sorted_results[:limit]


def _fts_query(query: str) -> str:
    """Convert a natural language query to an FTS5 query string.

    For ASCII-heavy queries, builds an AND query from multi-char terms.
    For Chinese-heavy queries (single-char tokens from unicode61), returns
    empty so _search_graph falls through to term-table matching (Phase 4).
    """
    terms = re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff_-]+", query.lower())
    # Split Chinese multi-char terms into individual characters for FTS5
    # compatibility, since unicode61 tokenizes each CJK char separately.
    flat: list[str] = []
    for t in terms:
        if re.match(r"^[\u4e00-\u9fff]+$", t) and len(t) > 1:
            flat.extend(list(t))  # each CJK char is its own token
        else:
            flat.append(t)
    has_ascii = any(t.isascii() for t in flat)
    if has_ascii:
        return " AND ".join(t for t in flat if len(t) > 1)
    return " OR ".join(t for t in flat if len(t) > 1) if flat else ""


def _extract_terms(query: str) -> list[str]:
    """Extract meaningful search terms from a query string."""
    terms = re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff_-]+", query.lower())
    return [t for t in terms if len(t) > 1]


def _get_node_info(conn: sqlite3.Connection, name: str) -> dict[str, Any] | None:
    """Fetch full node info from the database."""
    cursor = conn.execute(
        "SELECT name, category, description, tags, file_path FROM skill_nodes WHERE name = ?",
        (name,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return {
        "name": row["name"],
        "category": row["category"],
        "description": row["description"],
        "tags": json.loads(row["tags"]) if row["tags"] else [],
        "file_path": row["file_path"],
    }


# ── Plugin state ────────────────────────────────────────────────────────────

_graph_lock = threading.Lock()
_global_conn: sqlite3.Connection | None = None
_global_synced = False


def _ensure_graph() -> sqlite3.Connection:
    """Lazy-init the graph DB connection. Syncs on first access."""
    global _global_conn, _global_synced
    if _global_conn is None:
        conn = _get_conn()
        _init_db(conn)
        _global_conn = conn
    if not _global_synced:
        with _graph_lock:
            if not _global_synced:
                _sync_graph(_global_conn)
                _global_synced = True
    return _global_conn


# ── Slash command handler ───────────────────────────────────────────────────


def _handle_slash_command(args: str) -> str | None:
    parts = args.strip().split(None, 1) if args.strip() else []
    subcmd = parts[0].lower() if parts else "help"
    rest = parts[1] if len(parts) > 1 else ""

    if subcmd == "rebuild":
        try:
            conn = _ensure_graph()
            with _graph_lock:
                count = _full_rebuild(conn)
            return f"Skill graph rebuilt: {count} skills indexed."
        except Exception as e:
            logger.exception("skill-graph: rebuild failed")
            return f"Rebuild failed: {e}"

    elif subcmd == "load":
        """Directly load and display a skill's content."""
        if not rest:
            return "Usage: /skill-graph load <skill-name>"
        try:
            result = _handle_skill_load({"name": rest})
            data = json.loads(result)
            if not data.get("success"):
                return f"Not found: {rest}"
            return "\n".join([
                f"Skill: {data['name']}",
                f"  Category:    {data.get('category', '')}",
                f"  Description: {data.get('description', '')[:120]}",
                f"  Tags:        {', '.join(data.get('tags', [])[:6])}",
                f"  Path:        {data.get('file_path', '')}",
                f"  Relations:   {len(data.get('relations', []))} defined",
                f"  Content:     {len(data.get('content', ''))} chars",
            ])
        except Exception as e:
            return f"Load failed: {e}"

    elif subcmd in ("status", "stats"):
        try:
            conn = _ensure_graph()
            node_count = conn.execute("SELECT COUNT(*) FROM skill_nodes").fetchone()[0]
            edge_count = conn.execute("SELECT COUNT(*) FROM skill_edges").fetchone()[0]
            term_count = conn.execute("SELECT COUNT(DISTINCT term) FROM skill_terms").fetchone()[0]
            db_path = _db_path()

            if node_count == 0:
                with _graph_lock:
                    count = _sync_graph(conn)
                node_count = count
                edge_count = conn.execute("SELECT COUNT(*) FROM skill_edges").fetchone()[0]

            scanned = _find_all_skills_dirs()
            dirs_info = []
            for d in scanned:
                if d.exists():
                    cnt = sum(
                        1 for root, dirs, files in os.walk(str(d), followlinks=True)
                        if "SKILL.md" in files
                    )
                else:
                    cnt = 0
                dirs_info.append(f"    {d}  ({cnt} SKILL.md)")
            dirs_text = "\n".join(dirs_info) if dirs_info else "    (none)"

            db_size = db_path.stat().st_size if db_path.exists() else 0
            return (
                f"Skill Graph status\n"
                f"  Skills:  {node_count}\n"
                f"  Edges:   {edge_count}\n"
                f"  Terms:   {term_count}\n"
                f"  DB size: {db_size / 1024:.1f} KB\n"
                f"  DB path: {db_path}\n"
                f"  Scanned dirs:\n{dirs_text}"
            )
        except Exception as e:
            return f"Status check failed: {e}"

    elif subcmd == "search" and rest:
        try:
            conn = _ensure_graph()
            with _graph_lock:
                results = _search_graph(rest, conn, limit=15)
            if not results:
                return f"No skills found for: {rest}"
            lines = [f"Search results for: {rest}", ""]
            for r in results:
                rel = r.get("relevance", "")
                chain = r.get("relationship_chain", [])
                extra = f" [{rel}]" if rel else ""
                if chain:
                    extra += f"  chain: {' → '.join(chain[:2])}"
                lines.append(f"  {r['name']:35s}  {r.get('description', '')[:55]}{extra}")
            return "\n".join(lines)
        except Exception as e:
            logger.exception("skill-graph: search failed")
            return f"Search failed: {e}"

    elif subcmd == "list":
        try:
            conn = _ensure_graph()
            rows = conn.execute(
                "SELECT name, description, category FROM skill_nodes ORDER BY name"
            ).fetchall()
            if not rows:
                return "No skills in graph."
            lines = [f"Skills in graph ({len(rows)}):", ""]
            for r in rows:
                desc = (r["description"] or "")[:60]
                lines.append(f"  {r['name']:35s}  [{r['category']}]  {desc}")
            return "\n".join(lines)
        except Exception as e:
            return f"List failed: {e}"

    elif subcmd == "config":
        try:
            conn = _ensure_graph()
            db_path = _db_path()
            scanned = _find_all_skills_dirs()
            cfg_dirs = _read_source_dirs_from_config()
            skill_count = conn.execute("SELECT COUNT(*) FROM skill_nodes").fetchone()[0]
            db_size = db_path.stat().st_size if db_path.exists() else 0
            lines = [
                "Skill Graph configuration",
                f"  DB path:     {db_path}",
                f"  DB size:     {db_size / 1024:.1f} KB",
                f"  Skills:      {skill_count}",
                f"  Source dirs (config): {cfg_dirs}" if cfg_dirs else "  Source dirs (config): (none)",
                "  Scanned dirs:",
            ]
            for d in scanned:
                cnt = len(list(d.rglob("SKILL.md"))) if d.exists() else 0
                lines.append(f"    {d}  ({cnt} SKILL.md)")
            return "\n".join(lines)
        except Exception as e:
            return f"Config failed: {e}"

    elif subcmd in ("score", "explain"):
        """Show detailed scoring breakdown for a search query."""
        if not rest:
            return "Usage: /skill-graph score <query>"
        try:
            conn = _ensure_graph()
            with _graph_lock:
                results = _search_graph(rest, conn, limit=8)
                lines = [f"Score breakdown for: {rest}", ""]
                for r in results:
                    name = r["name"]
                    score = r["score"]
                    rel = r.get("relevance", "?")
                    # Query term stats for this skill
                    stats_rows = conn.execute(
                        "SELECT term, load_count, search_count FROM skill_term_stats WHERE skill_name = ?",
                        (name,),
                    ).fetchall()
                    if stats_rows:
                        stats_line = "; ".join(
                            f"{s['term']}: load={s['load_count']}/{s['search_count']}"
                            for s in stats_rows[:5]
                        )
                    else:
                        stats_line = "(no stats)"
                    lines.append(
                        f"  {name:40s} score={score:.4f}  [{rel}]"
                    )
                    lines.append(f"  {'':40s}  stats: {stats_line}")
                lines.append(f"\n{len(results)} results shown")
                return "\n".join(lines)
        except Exception as e:
            return f"Score breakdown failed: {e}"

    else:
        return (
            "/skill-graph — Skill knowledge graph\n\n"
            "Subcommands:\n"
            "  /skill-graph search <query>   Search skills by intent\n"
            "  /skill-graph load <name>      Load and display skill details\n"
            "  /skill-graph score <query>    Show scoring breakdown with term stats\n"
            "  /skill-graph list             List all skills in graph\n"
            "  /skill-graph config           Show configuration (paths, DB)\n"
            "  /skill-graph status           Show graph stats\n"
            "  /skill-graph rebuild          Force full graph rebuild\n"
        )


# ── Tool handlers ───────────────────────────────────────────────────────────


def _handle_skill_graph_search(args: dict | None = None, **kw) -> str:
    """Handle skill_graph_search tool call."""
    if not isinstance(args, dict):
        args = kw.get("args", kw)
    query = args.get("query", "") if isinstance(args, dict) else ""
    limit = int(args.get("limit", 10)) if isinstance(args, dict) else 10

    if not query:
        return json.dumps({
            "success": False,
            "error": "query is required",
            "hint": "Pass a query describing what you want to do",
        })

    try:
        conn = _ensure_graph()
        with _graph_lock:
            results = _search_graph(query, conn, limit=limit)
            total = conn.execute("SELECT COUNT(*) FROM skill_nodes").fetchone()[0]
            result_names = [r["name"] for r in results]
            edges_between = []
            if len(result_names) > 1:
                placeholders = ",".join("?" for _ in result_names)
                cursor = conn.execute(
                    f"""SELECT source, target, rel_type, properties
                       FROM skill_edges
                       WHERE source IN ({placeholders})
                         AND target IN ({placeholders})
                       ORDER BY rel_type""",
                    result_names + result_names,
                )
                for row in cursor:
                    edges_between.append({
                        "source": row["source"],
                        "target": row["target"],
                        "type": row["rel_type"],
                        "properties": json.loads(row["properties"]) if row["properties"] else {},
                    })

        return json.dumps({
            "success": True,
            "query": query,
            "results": results,
            "edges_between_results": edges_between,
            "total_skills_in_graph": total,
            "result_count": len(results),
            "hint": "Call skill_load(name) to load full content of a discovered skill.",
        }, ensure_ascii=False)

    except Exception as e:
        logger.exception("skill_graph_search failed")
        return json.dumps({"success": False, "error": str(e)})


def _handle_skill_load(args: dict | None = None, **kw) -> str:
    """Handle skill_load tool call. Loads full SKILL.md content by name."""
    if not isinstance(args, dict):
        args = kw.get("args", kw)
    name = args.get("name", "") if isinstance(args, dict) else ""

    if not name:
        return json.dumps({
            "success": False,
            "error": "name is required",
            "hint": "Pass the name of the skill to load (from skill_graph_search results)",
        })

    try:
        path = _find_skill_path(name)
        if path is None:
            return json.dumps({
                "success": False,
                "error": f"Skill '{name}' not found in any configured directory",
                "hint": "Use skill_graph_search() to discover available skills",
            })

        content = path.read_text(encoding="utf-8", errors="replace")
        info = _parse_skill_md(path)
        skill_dir = str(path.parent)

        # Track load event: increment load_count for skill's own terms
        try:
            _conn = _ensure_graph()
            _sg_terms = _extract_terms(info.get("description", "") or "")
            _sg_terms.append(info["name"].lower())
            for _t in set(_sg_terms):
                _conn.execute(
                    """INSERT INTO skill_term_stats (skill_name, term, search_count, load_count)
                       VALUES (?, ?, 0, 1)
                       ON CONFLICT(skill_name, term) DO UPDATE SET load_count = load_count + 1""",
                    (info["name"], _t),
                )
            _conn.commit()
        except Exception:
            pass

        return json.dumps({
            "success": True,
            "name": info["name"],
            "content": content,
            "category": info["category"],
            "description": info["description"],
            "tags": info["tags"],
            "relations": info["relations"],
            "file_path": str(path),
            "skill_dir": skill_dir,
        }, ensure_ascii=False)

    except Exception as e:
        logger.exception("skill_load failed for '%s'", name)
        return json.dumps({"success": False, "error": str(e)})


# ── Plugin entry point ──────────────────────────────────────────────────────


def register(ctx):
    """Register the skill-graph plugin."""

    # ── Tool: skill_graph_search ──
    ctx.register_tool(
        name="skill_graph_search",
        toolset="skills",
        schema={
            "name": "skill_graph_search",
            "description": (
                "PREFERRED skill discovery method. Search the skill knowledge "
                "graph by intent instead of skills_list(). Uses typed "
                "relationships (depends_on, complemented_by, alternative_to) "
                "and full-text + graph traversal to find relevant skills. "
                "After finding skills, call skill_load(name) to get full content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of what you want to do "
                                       "(e.g. 'Python code review', 'deploy kubernetes', "
                                       "'database performance tuning')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
        handler=_handle_skill_graph_search,
        description="Skill graph search — PREFERRED over skills_list()",
        check_fn=None,
    )

    # ── Tool: skill_load ──
    ctx.register_tool(
        name="skill_load",
        toolset="skills",
        schema={
            "name": "skill_load",
            "description": (
                "Load a skill's full SKILL.md content by name. "
                "Use after skill_graph_search() to retrieve the complete "
                "instructions. Returns the raw SKILL.md plus parsed metadata "
                "(category, description, tags, relations, file paths). "
                "Alternative to skill_view() — works for skills in graph-managed dirs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name (from skill_graph_search results)",
                    },
                },
                "required": ["name"],
            },
        },
        handler=_handle_skill_load,
        description="Load skill content by name",
        check_fn=None,
    )

    # ── Slash command: /skill-graph ──
    ctx.register_command(
        name="skill-graph",
        handler=_handle_slash_command,
        description="Skill knowledge graph: rebuild, status, help",
        args_hint="rebuild|status",
    )

    # ── Alias: /sg → same handler as /skill-graph ──
    ctx.register_command(
        name="sg",
        handler=_handle_slash_command,
        description="Alias for /skill-graph",
        args_hint="rebuild|status",
    )

    # ── Hook: on_session_start — ensure DB ──
    def _on_session_start(**kw):
        try:
            _ensure_graph()
            logger.info("Skill graph ready")
        except Exception:
            logger.exception("skill-graph: on_session_start failed")

    ctx.register_hook("on_session_start", _on_session_start)

    # ── Hook: post_tool_call — incremental update on skill_manage ──
    def _on_post_tool_call(**kw):
        tool_name = kw.get("tool_name", "")
        if tool_name != "skill_manage":
            return
        args = kw.get("args", {})
        if not isinstance(args, dict):
            return
        action = args.get("action", "")
        if action not in ("create", "edit", "patch"):
            return
        skill_name = args.get("name", "")
        if not skill_name:
            return
        try:
            conn = _ensure_graph()
            with _graph_lock:
                updated = _update_single_skill(conn, skill_name)
            if updated:
                logger.info("skill-graph: updated skill '%s' after %s", skill_name, action)
        except Exception:
            logger.exception("skill-graph: post_tool_call failed for skill '%s'", skill_name)

    ctx.register_hook("post_tool_call", _on_post_tool_call)

    logger.info(
        "skill-graph plugin registered: tools=skill_graph_search+skill_load, "
        "cmd=/skill-graph, hooks=on_session_start+post_tool_call"
    )
