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
from typing import Any, Mapping

from hermes_constants import (
    get_bundled_skills_dir,
    get_hermes_home,
    get_skills_dir,
)

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_RELATION_TYPES = {
    "depends_on", "supported_by", "alternative_to", "complemented_by",
    "similar_to", "belongs_to_domain", "used_in_workflow", "supersedes",
}

GRAPH_DB_FILENAME = "skill-graph.db"
_RUNTIME_SOURCE_DIRS: list[Path] = []


def _skill_graph_config() -> dict[str, Any]:
    """Return the active profile's skill-graph settings mapping."""
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
        value = (((config.get("skills") or {}).get("config") or {}).get("skill-graph") or {})
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _resolve_config_path(value: str | os.PathLike[str]) -> Path:
    """Expand a configured path; relative paths are scoped to HERMES_HOME."""
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (path if path.is_absolute() else get_hermes_home() / path).resolve()


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path.absolute())
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _is_under(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError):
        return False

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
    success_count INTEGER DEFAULT 0,
    last_searched TEXT,
    last_loaded   TEXT,
    PRIMARY KEY (skill_name, term)
);
"""

# ── Database helpers ────────────────────────────────────────────────────────


def _db_path() -> Path:
    """Return path to graph DB under the active Hermes home.

    Priority:
    1. skills.config.skill-graph.db_path from config.yaml (explicit override)
    2. HERMES_BUNDLED_PLUGINS → root level (profiles/<name>/skill-graph.db)
    3. Default → under personal/ (profiles/<name>/personal/skill-graph.db)
    """
    # Priority 1: config.yaml override
    try:
        raw = _skill_graph_config().get("db_path")
        if raw:
            return _resolve_config_path(raw)
    except Exception:
        pass

    hermes_home = get_hermes_home()
    if os.environ.get("HERMES_BUNDLED_PLUGINS"):
        return hermes_home / GRAPH_DB_FILENAME
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
        raw = _skill_graph_config().get("source_dirs", [])
        if isinstance(raw, list):
            for entry in raw:
                p = _resolve_config_path(entry)
                if p.is_dir():
                    dirs.append(p)
    except Exception:
        pass
    return dirs


def _find_all_skills_dirs() -> list[Path]:
    """Return list of directories to scan for SKILL.md files.

    Precedence is profile-local, bundled, agent-created, then explicitly
    configured sources. No implicit global/default-profile directory is read.
    """
    hermes_home = get_hermes_home()
    bundled_default = Path(__file__).resolve().parents[2] / "skills"
    dirs = [
        get_skills_dir(),
        get_bundled_skills_dir(bundled_default),
        hermes_home / "skill-graph" / "agent-created",
        *_read_source_dirs_from_config(),
        *_RUNTIME_SOURCE_DIRS,
    ]

    try:
        from agent.skill_utils import get_external_skills_dirs

        dirs.extend(get_external_skills_dirs())
    except Exception:
        pass
    return [path for path in _dedupe_paths(dirs) if path.is_dir()]


def _find_skill_path(name: str) -> Path | None:
    """Find a SKILL.md by name across all configured dirs.

    Returns the first match in :func:`_find_all_skills_dirs` precedence order.
    """
    skill_dirs = _find_all_skills_dirs()
    skills = _scan_skill_mds(skill_dirs)
    for n, path in skills:
        if n == name:
            return path
    return None


# ── SKILL.md scanner & parser ──────────────────────────────────────────────


def _scan_skill_mds(skill_dirs: list[Path]) -> list[tuple[str, Path]]:
    """Scan all skill directories for SKILL.md files.

    Deduplicates by real path (resolving symlinks) so the same skill
    discovered via different routes (symlink vs original, multiple
    base_dirs) is only indexed once.

    Returns list of (skill_name, skill_md_path).
    """
    results: list[tuple[str, Path]] = []
    seen_realpaths: set[str] = set()
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
                real = os.path.realpath(skill_md)
                dedup_key = f"{name}\x00{real}"
                if dedup_key not in seen_realpaths:
                    seen_realpaths.add(dedup_key)
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
                    real = os.path.realpath(skill_md)
                    dedup_key = f"{name}\x00{real}"
                    if dedup_key not in seen_realpaths:
                        seen_realpaths.add(dedup_key)
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
                result["tags"] = [str(tag) for tag in tags] if isinstance(tags, list) else []

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
    """Deduplicate by name and real path, preserving source precedence."""
    deduped: dict[str, Path] = {}
    seen_realpaths: set[str] = set()
    for name, path in skills:
        real = os.path.realpath(path)
        if name in deduped or real in seen_realpaths:
            continue
        seen_realpaths.add(real)
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
    # Mark active-profile skills as needing external organization.
    if _is_under(skill_path, get_skills_dir()):
        conn.execute(
            "UPDATE skill_nodes SET needs_organizing = 1 WHERE name = ? AND (needs_organizing IS NULL OR needs_organizing = 0)",
            (skill_name,),
        )
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
                 AND (n.is_deleted IS NULL OR n.is_deleted = 0)
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

    # Phase 3: Tag match (existing) — uses its own dedup set so it doesn't
    # block Phase 4 from finding higher-scoring term matches.
    _tag_seen: set[str] = set()
    terms = _extract_terms(query)
    for term in terms:
        cursor = conn.execute(
            """SELECT name FROM skill_nodes WHERE instr(tags, ?) > 0 AND (is_deleted IS NULL OR is_deleted = 0)""",
            (json.dumps(term),),
        )
        for row in cursor:
            if row["name"] not in _tag_seen:
                _tag_seen.add(row["name"])
                info = _get_node_info(conn, row["name"])
                if info:
                    info["relevance"] = "tag_match"
                    info["score"] = 0.7
                    results[info["name"]] = info

    # Phase 4: Term table match — direct term→skill lookup from the
    # skill_terms table (auto-extracted from name, tags, description).
    # This catches Chinese terms and split-name parts that FTS5 misses.
    for term in terms:
        if seen:
            cursor = conn.execute(
                """SELECT t.skill_name, t.strength, t.source, n.category, n.description
                   FROM skill_terms t
                   JOIN skill_nodes n ON t.skill_name = n.name AND (n.is_deleted IS NULL OR n.is_deleted = 0)
                   WHERE t.term = ? AND t.skill_name NOT IN ({})
                   ORDER BY t.strength DESC
                   LIMIT 5""".format(",".join("?" for _ in seen)),
                (term.lower(),) + tuple(seen),
            )
        else:
            cursor = conn.execute(
                """SELECT t.skill_name, t.strength, t.source, n.category, n.description
                   FROM skill_terms t
                   JOIN skill_nodes n ON t.skill_name = n.name AND (n.is_deleted IS NULL OR n.is_deleted = 0)
                   WHERE t.term = ?
                   ORDER BY t.strength DESC
                   LIMIT 5""",
                (term.lower(),),
            )
        for row in cursor:
            sname = row["skill_name"]
            _term_score = 0.8 * row["strength"]
            if sname in results:
                # Don't overwrite — take the higher score
                if results[sname]["score"] < _term_score:
                    results[sname]["score"] = _term_score
                    results[sname]["relevance"] = "term_match"
                    results[sname]["relationship_chain"] = [
                        f"term[{term}] → {sname} (strength={row['strength']}, source={row['source']})"
                    ]
                continue
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
    # Uses per-(skill, term) stats with confidence-weighted S-curve.
    _norm_terms = [t.lower() for t in terms]
    for sname, r in results.items():
        _placeholders = ",".join("?" for _ in _norm_terms)
        _term_rows = conn.execute(
            f"SELECT term FROM skill_terms WHERE skill_name = ? AND term IN ({_placeholders})",
            (sname,) + tuple(_norm_terms),
        ).fetchall()
        matched_terms = [row["term"] for row in _term_rows]
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
            try:
                rows = conn.execute(
                    """SELECT term, load_count, search_count, success_count FROM skill_term_stats
                       WHERE skill_name = ? AND term IN ({})"""
                    .format(",".join("?" for _ in matched_terms)),
                    (sname,) + tuple(mt.lower() for mt in matched_terms),
                ).fetchall()
                if rows:
                    import math as _m
                    _avg_eff = sum(
                        (r["success_count"] * 2 + r["load_count"]) / max(r["search_count"] * 3, 1)
                        for r in rows
                    ) / len(rows)
                    _confidence = 1 - _m.pow(0.5, sum(r["search_count"] for r in rows) / max(len(rows), 1) / 5)
                    _adj = (_avg_eff - 0.5) * 2
                    _tanh = _adj / (1 + abs(_adj) * 0.5)  # tanh approximation
                    r["score"] *= (1.0 + 0.1 * _tanh * _confidence)
            except Exception:
                pass
    conn.commit()

    sorted_results = sorted(results.values(), key=lambda r: -r["score"])
    if not sorted_results:
        return _fallback_search(query, conn, limit)
    return sorted_results[:limit]


def _fallback_search(query: str, conn: sqlite3.Connection, limit: int = 10) -> list[dict[str, Any]]:
    """Broad fallback when the primary search returns nothing.

    Returns skills whose name, tags, or description contain any of the
    query terms, ordered by term strength. This catches skills that FTS5
    and exact-term matching miss (e.g. stemming mismatches, partial words).
    """
    terms = _extract_terms(query)
    if not terms:
        # No parseable terms — return top skills by name
        cursor = conn.execute(
            "SELECT name, category, description, tags, file_path FROM skill_nodes WHERE (is_deleted IS NULL OR is_deleted = 0) ORDER BY name LIMIT ?",
            (limit,),
        )
        fallback = []
        for row in cursor:
            fallback.append({
                "name": row["name"],
                "category": row["category"] or "",
                "description": row["description"] or "",
                "tags": json.loads(row["tags"]) if row["tags"] else [],
                "file_path": row["file_path"],
                "relevance": "fallback",
                "score": 0.1,
            })
        return fallback

    results: dict[str, dict[str, Any]] = {}
    for term in terms:
        cursor = conn.execute(
            """SELECT n.name, n.category, n.description, n.tags, n.file_path
               FROM skill_nodes n
               WHERE (n.is_deleted IS NULL OR n.is_deleted = 0)
                 AND (instr(n.name, ?) > 0
                  OR instr(n.description, ?) > 0
                  OR instr(n.tags, ?) > 0)
               LIMIT ?""",
            (term, term, json.dumps(term), limit),
        )
        for row in cursor:
            if row["name"] not in results:
                results[row["name"]] = {
                    "name": row["name"],
                    "category": row["category"] or "",
                    "description": row["description"] or "",
                    "tags": json.loads(row["tags"]) if row["tags"] else [],
                    "file_path": row["file_path"],
                    "relevance": "fallback",
                    "score": 0.2,
                }
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
        # Quote each term so FTS5 treats hyphens and other special chars
        # as literal text, not column-filter operators.
        return " AND ".join(f'"{t}"' for t in flat if len(t) > 1)
    return " OR ".join(t for t in flat if len(t) > 1) if flat else ""


def _extract_terms(query: str) -> list[str]:
    """Extract meaningful search terms from a query string."""
    terms = re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff_-]+", query.lower())
    return [t for t in terms if len(t) > 1]


def _get_node_info(conn: sqlite3.Connection, name: str) -> dict[str, Any] | None:
    """Fetch full node info from the database."""
    cursor = conn.execute(
        """SELECT name, category, description, tags, file_path, needs_organizing
           FROM skill_nodes WHERE name = ? AND (is_deleted IS NULL OR is_deleted = 0)""",
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
        "needs_organizing": bool(dict(row).get("needs_organizing")) or False,
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
        _migrate_db(conn)
        _global_conn = conn
    if not _global_synced:
        with _graph_lock:
            if not _global_synced:
                _sync_graph(_global_conn)
                _global_synced = True
    return _global_conn


# ── Schema migration helper ──────────────────────────────────────────────────

def _migrate_db(conn: sqlite3.Connection) -> None:
    """Apply schema changes that can't be done via CREATE TABLE IF NOT EXISTS."""
    # v2: add success_count column
    try:
        conn.execute("ALTER TABLE skill_term_stats ADD COLUMN success_count INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # column already exists
    try:
        conn.execute("ALTER TABLE skill_term_stats ADD COLUMN last_searched TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE skill_term_stats ADD COLUMN last_loaded TEXT")
    except sqlite3.OperationalError:
        pass
    # v3: soft delete + needs_organizing for lifecycle management
    for col, col_type in (
        ("is_deleted", "INTEGER DEFAULT 0"),
        ("deleted_at", "TEXT"),
        ("needs_organizing", "INTEGER DEFAULT 0"),
    ):
        try:
            conn.execute(f"ALTER TABLE skill_nodes ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass



def _persist_source_dirs(source_dirs: list[Any]) -> None:
    """Atomically update only the active profile's graph source list."""
    from hermes_cli.config import save_config

    path = ("skills", "config", "skill-graph", "source_dirs")
    partial = {"skills": {"config": {"skill-graph": {"source_dirs": source_dirs}}}}
    save_config(partial, preserve_keys={path}, merge_existing=True)


def _change_source_dir(action: str, path_str: str, *, persist: bool) -> tuple[Path, int, str]:
    """Apply one source-dir change and rebuild; returns target, count, note."""
    target = _resolve_config_path(path_str)
    if action == "add_dir" and not target.is_dir():
        raise ValueError(f"Not a directory: {target}")

    note = ""
    if persist:
        raw = _skill_graph_config().get("source_dirs", [])
        source_dirs = list(raw) if isinstance(raw, list) else []
        resolved = [_resolve_config_path(entry) for entry in source_dirs]
        if action == "add_dir":
            if target in resolved:
                note = "already present"
            else:
                source_dirs.append(path_str)
                _persist_source_dirs(source_dirs)
        elif target in resolved:
            source_dirs = [
                entry for entry in source_dirs
                if _resolve_config_path(entry) != target
            ]
            _persist_source_dirs(source_dirs)
        else:
            note = "not present"
    elif action == "add_dir":
        if target not in _RUNTIME_SOURCE_DIRS:
            _RUNTIME_SOURCE_DIRS.append(target)
        else:
            note = "already present"
    elif target in _RUNTIME_SOURCE_DIRS:
        _RUNTIME_SOURCE_DIRS.remove(target)
    else:
        note = "not present"

    conn = _ensure_graph()
    with _graph_lock:
        count = _full_rebuild(conn)
    return target, count, note

def _show_graph_config() -> str:
    """Return current graph config (slash command 'config' default action)."""
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

def _handle_source_dir_config(action: str, path_str: str) -> str:
    """Add or remove a source_dir at runtime and persist to config.yaml."""
    if not path_str:
        return f"Usage: /sg config {action} <path>"
    try:
        mapped = {"add": "add_dir", "remove": "remove_dir"}.get(action)
        if mapped is None:
            return f"Unknown action: {action}"
        target, count, note = _change_source_dir(mapped, path_str, persist=True)
        suffix = f" ({note})" if note else ""
        return f"✅ {action}ed {target}{suffix}\n   Graph rebuilt: {count} skills indexed."
    except Exception as e:
        return f"Config {action} failed: {e}"


def _handle_skill_graph_config(args: dict | None = None, **kw) -> str:
    """Handle skill_graph_config tool — add/remove/list source_dirs."""
    if not isinstance(args, dict):
        return json.dumps({"success": False, "error": "args must be dict"})
    action = args.get("action", "")
    path_str = args.get("path", "")
    persist = args.get("persist", True)
    try:
        if action == "list_dirs":
            cfg_dirs = _read_source_dirs_from_config()
            return json.dumps({"success": True, "source_dirs": [str(d) for d in cfg_dirs],
                               "runtime_source_dirs": [str(d) for d in _RUNTIME_SOURCE_DIRS],
                               "scanned_dirs": [str(d) for d in _find_all_skills_dirs() if d.exists()],
                               "persisted": persist}, default=str)
        if action in ("add_dir", "remove_dir"):
            if not path_str:
                return json.dumps({"success": False, "error": "path required"})
            target, count, note = _change_source_dir(action, path_str, persist=bool(persist))
            result = {"success": True, "action": action, "path": str(target),
                      "skills_indexed": count, "persisted": bool(persist)}
            if note:
                result["note"] = note
            return json.dumps(result)
        return json.dumps({"success": False, "error": f"Unknown action: {action}. Use add_dir, remove_dir, or list_dirs."})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

# ── Slash command handler ───────────────────────────────────────────────────


def _format_edges(skill_name: str) -> str:
    """Query and format graph edges only."""
    try:
        conn = _ensure_graph()
        rows = conn.execute(
            """SELECT source, target, rel_type, properties FROM skill_edges
               WHERE source = ? OR target = ?
               ORDER BY rel_type, source""",
            (skill_name, skill_name),
        ).fetchall()
        if not rows:
            return f"No relations defined for: {skill_name}"
        seen: set[tuple[str, str, str]] = set()
        parts = []
        for src, tgt, rel, props in rows:
            key = (src, tgt, rel)
            if key in seen:
                continue
            seen.add(key)
            arrow = f"  {src} ──({rel})──> {tgt}"
            reason = ""
            if isinstance(props, str) and props:
                import json as _j
                try:
                    reason = _j.loads(props).get("reason", "")
                except Exception:
                    reason = props[:40]
            elif isinstance(props, dict):
                reason = props.get("reason", "")
            if reason:
                parts.append(f"{arrow:55s} {reason[:50]}")
            else:
                parts.append(arrow)
        return "Edges:\n" + "\n".join(parts) + "\n"
    except Exception:
        return ""


def _format_terms(skill_name: str) -> str:
    """Query and format term associations with inline stats."""
    try:
        conn = _ensure_graph()
        parts = []

        # Skill's own terms
        terms = conn.execute(
            "SELECT t.term, t.strength, t.source, "
            "COALESCE(s.search_count,0) AS sc, COALESCE(s.load_count,0) AS lc, "
            "COALESCE(s.success_count,0) AS suc "
            "FROM skill_terms t "
            "LEFT JOIN skill_term_stats s ON t.skill_name = s.skill_name AND t.term = s.term "
            "WHERE t.skill_name = ? ORDER BY t.strength DESC, t.source",
            (skill_name,),
        ).fetchall()
        if terms:
            term_lines = ["", "  Terms:"]
            for t in terms:
                _sc, _lc, _suc = t['sc'], t['lc'], t['suc']
                _eff = (_suc * 2 + _lc) / max(_sc * 3, 1)
                _conf = 1 - __import__("math").pow(0.5, _sc / 5)
                _adj = (_eff - 0.5) * 2
                _th = _adj / (1 + abs(_adj) * 0.5)
                _boost = 0.1 * _th * _conf
                _sign = "+" if _boost > 0 else ""
                _stats = f"s={_sc}/l={_lc}/ok={_suc}/b={_sign}{_boost:.3f}".replace("+-", "")
                term_lines.append(
                    f"    {skill_name} ──({t['source']})──> {t['term']}  [{_stats}]"
                )
            parts.append("\n".join(term_lines))

        # Reverse lookup
        rev = conn.execute(
            "SELECT t.skill_name, t.strength, t.source, "
            "COALESCE(s.search_count,0) AS sc, COALESCE(s.load_count,0) AS lc, "
            "COALESCE(s.success_count,0) AS suc "
            "FROM skill_terms t "
            "LEFT JOIN skill_term_stats s ON t.skill_name = s.skill_name AND t.term = s.term "
            "WHERE t.term = ? ORDER BY t.strength DESC",
            (skill_name,),
        ).fetchall()
        if rev:
            rev_lines = ["", "  Skills with this term:"]
            for sn, s, src, sc, lc, suc in rev:
                _eff2 = (suc * 2 + lc) / max(sc * 3, 1)
                _conf2 = 1 - __import__("math").pow(0.5, sc / 5)
                _adj2 = (_eff2 - 0.5) * 2
                _th2 = _adj2 / (1 + abs(_adj2) * 0.5)
                _boost2 = 0.1 * _th2 * _conf2
                _sign2 = "+" if _boost2 > 0 else ""
                rev_lines.append(f"    {sn:40s} ──({src})──> {skill_name}  [s={sc}/l={lc}/ok={suc}/b={_sign2}{_boost2:.3f}]".replace("+-", ""))
            parts.append("\n".join(rev_lines))

        return "\n".join(parts) if parts else ""
    except Exception:
        return ""


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

    elif subcmd == "show":
        """Show full skill content (preview)."""
        if not rest:
            return "Usage: /skill-graph show <skill-name>"
        try:
            result = _handle_skill_load({"name": rest})
            data = json.loads(result)
            if not data.get("success"):
                return f"Not found: {rest}"
            content = data.get("content", "")
            return (
                f"Skill: {data['name']} ({len(content)} chars)\n"
                f"  Description: {data.get('description', '')}\n"
                f"  Category:    {data.get('category', '')}\n"
                f"\n{content[:2000]}"
            )
        except Exception as e:
            return f"Show failed: {e}"

    elif subcmd == "info":
        """Show skill metadata only."""
        if not rest:
            return "Usage: /skill-graph info <skill-name>"
        try:
            conn = _ensure_graph()
            node = conn.execute(
                "SELECT name, category, description, tags, file_path FROM skill_nodes WHERE name = ?",
                (rest,),
            ).fetchone()
            if not node:
                return f"Not found: {rest}  (try /sg list)"
            return "\n".join([
                f"Node: {node['name']}",
                f"  Category:    {node['category'] or ''}",
                f"  Description: {node['description'] or ''}",
                f"  Tags:        {node['tags'] or ''}",
                f"  Path:        {node['file_path'] or ''}",
            ])
        except Exception as e:
            return f"Info failed: {e}"

    elif subcmd == "terms":
        """Show term associations with stats."""
        if not rest:
            return "Usage: /skill-graph terms <skill-name>"
        try:
            return _format_terms(rest)
        except Exception as e:
            return f"Terms failed: {e}"

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
        rest_parts = rest.strip().split(None, 1) if rest.strip() else []
        config_action = rest_parts[0].lower() if rest_parts else "show"
        config_arg = rest_parts[1] if len(rest_parts) > 1 else ""
        if config_action in ("add", "remove"):
            return _handle_source_dir_config(config_action, config_arg)
        return _show_graph_config()

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
        # Unknown command — try proxying to a skill in the graph
        if subcmd:
            try:
                conn = _ensure_graph()
                _node = conn.execute(
                    "SELECT file_path FROM skill_nodes WHERE name = ?", (subcmd,)
                ).fetchone()
                if _node:
                    _result = _handle_skill_load({"name": subcmd})
                    _data = json.loads(_result)
                    if _data.get("success"):
                        _content = _data.get("content", "")
                        return (
                            f"Loaded skill: {subcmd}\n"
                            f"  Description: {_data.get('description', '')}\n"
                            f"  Category:    {_data.get('category', '')}\n"
                            f"  Content ({len(_content)} chars):\n"
                            f"{_content[:500]}\n"
                            f"...\n"
                            f"(Use /sg info {subcmd} for metadata, "
                            f"/sg terms {subcmd} for term details)"
                        )
            except Exception:
                pass
        return (
            "/skill-graph — Skill knowledge graph\n\n"
            "Subcommands:\n"
            "  /skill-graph search <query>   Search skills by intent\n"
            "  /skill-graph show <name>      Show full skill content (preview)\n"
            "  /skill-graph info <name>      Show skill metadata\n"
            "  /skill-graph terms <name>     Show term associations with stats\n"
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
    list_all = args.get("list_all", False) if isinstance(args, dict) else False

    if not query and not list_all:
        return json.dumps({
            "success": False,
            "error": "query is required",
            "hint": "Pass a query describing what you want to do, "
                    "or set list_all=True to browse all skills.",
        })

    try:
        conn = _ensure_graph()
        with _graph_lock:
            if list_all:
                cursor = conn.execute(
                    """SELECT name, category, description, tags, file_path, needs_organizing
                       FROM skill_nodes
                       WHERE (is_deleted IS NULL OR is_deleted = 0)
                       ORDER BY name"""
                )
                results = []
                for row in cursor:
                    results.append({
                        "name": row["name"],
                        "category": row["category"] or "",
                        "description": row["description"] or "",
                        "tags": json.loads(row["tags"]) if row["tags"] else [],
                        "file_path": row["file_path"],
                        "relevance": "listed",
                        "score": 0.0,
                        "needs_organizing": bool(dict(row).get("needs_organizing")) or False,
                    })
                total = len(results)
                hint = "All skills listed by name. Call skill_load(name) to load full content."
                logger.info("skill-graph: search list_all=True → %d results (total: %d)", total, total)
                logger.debug("skill-graph: search list_all=True → skills: %s", [r["name"] for r in results])
                return json.dumps({
                    "success": True,
                    "query": "",
                    "results": results,
                    "edges_between_results": [],
                    "total_skills_in_graph": total,
                    "result_count": len(results),
                    "hint": hint,
                    "note": "list_all=True — results sorted by name, not by relevance score.",
                }, ensure_ascii=False)

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

        if results and results[0].get("score", 0) < 0.3:
            hint = (
                "Top results have low confidence. "
                "Retry skill_graph_search() with different keywords, "
                "or use skill_graph_search(list_all=True) to browse all skills."
            )
        else:
            hint = "Call skill_load(name) to load full content of a discovered skill."
        logger.info(
            "skill-graph: search query=%r limit=%d → %d results (total: %d)",
            query, limit, len(results), total,
        )
        logger.debug("skill-graph: search query=%r → skills: %s", query, result_names)
        return json.dumps({
            "success": True,
            "query": query,
            "results": results,
            "edges_between_results": edges_between,
            "total_skills_in_graph": total,
            "result_count": len(results),
            "hint": hint,
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


def _skill_graph_mode_enabled() -> bool:
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
        return bool((config.get("agent") or {}).get("skill_graph_mode", False))
    except Exception:
        return False


def _gateway_extension_skills() -> list[tuple[str, str]]:
    """Read the declarative gateway list from the configured extensions file."""
    raw_path = _skill_graph_config().get("extensions_file")
    if not raw_path:
        return []
    try:
        path = _resolve_config_path(raw_path)
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, TypeError, ValueError):
        return []

    extras: list[tuple[str, str]] = []
    in_gateways = False
    for line in lines:
        line = line.rstrip()
        if line.startswith("## Pre-installed Gateways"):
            in_gateways = True
            continue
        if in_gateways and line.startswith("## "):
            break
        if not in_gateways or not line.startswith("| `"):
            continue
        cells = line.split("|")
        if len(cells) < 3:
            continue
        name = cells[1].strip().strip("`")
        description = cells[2].strip()
        if name and not name.startswith("-"):
            extras.append((name, description))
    return extras


def _render_skill_graph_prompt(session_info: Mapping[str, Any]) -> str:
    """Render graph discovery guidance only when mode and both tools are live."""
    tools = set(session_info.get("valid_tool_names") or ())
    if not session_info.get("skill_graph_mode") or not {
        "skill_graph_search", "skill_load"
    }.issubset(tools):
        return ""

    from agent.prompt_builder import SKILL_GRAPH_GUIDANCE, SKILL_GRAPH_IDENTITY

    description = "Skill knowledge graph — discover and load skills by intent"
    companion = _find_skill_path("skill-graph")
    if companion is not None:
        parsed = _parse_skill_md(companion)
        description = parsed.get("description") or description

    available = [f"- skill-graph — {str(description)[:100]}"]
    available.extend(f"- {name} — {desc[:100]}" for name, desc in _gateway_extension_skills())
    available_block = "Available Skills\n" + "\n".join(available)
    return "\n\n".join((SKILL_GRAPH_IDENTITY, available_block, SKILL_GRAPH_GUIDANCE))


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
                    "list_all": {
                        "type": "boolean",
                        "description": "List all available skills by name (bypasses scoring). Use when search results have low confidence.",
                        "default": False,
                    },
                },
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


    # ── Tool: skill_graph_config ──
    ctx.register_tool(
        name="skill_graph_config",
        toolset="skills",
        schema={
            "name": "skill_graph_config",
            "description": (
                "Manage skill-graph source directories at runtime without restarting Hermes. "
                "Add or remove directories for skill discovery, or list current configuration. "
                "Changes persist to config.yaml when persist=true (default). "
                "The graph is automatically rebuilt after add/remove."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "add_dir, remove_dir, or list_dirs",
                        "enum": ["add_dir", "remove_dir", "list_dirs"],
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory path (required for add_dir/remove_dir)",
                    },
                    "persist": {
                        "type": "boolean",
                        "description": "Save to config.yaml (default: true). Set false for ephemeral changes.",
                        "default": True,
                    },
                },
                "required": ["action"],
            },
        },
        handler=_handle_skill_graph_config,
        description="Manage skill-graph source directories at runtime",
        check_fn=None,
    )

    # Cache-safe, profile-scoped prompt guidance. Core persists the rendered
    # bytes for resume and re-renders only at the normal invalidation boundary.
    ctx.register_system_prompt_section(
        "skill-graph.discovery",
        _render_skill_graph_prompt,
        position="after_memory",
        max_chars=8000,
    )

    # ── Slash command: /skill-graph ──
    ctx.register_command(
        name="skill-graph",
        handler=_handle_slash_command,
        description="Skill knowledge graph: rebuild, status, help",
        args_hint="rebuild|status|config [add|remove] <path>",
    )

    # ── Alias: /sg → same handler as /skill-graph ──
    ctx.register_command(
        name="sg",
        handler=_handle_slash_command,
        description="Alias for /skill-graph",
        args_hint="rebuild|status|config [add|remove] <path>",
    )

    # ── Hook: pre_tool_call — gate search_files behind skill_graph_search ──
    # search_files: prevents the model from bypassing skill discovery by
    #   going straight to filesystem search (e.g. "find project files").
    # read_file is intentionally excluded — it's the execution step after a
    #   skill has been loaded (or reading a known config/source file) and
    #   should not be gated.
    # find is not a real Hermes tool; search_files replaced it.
    _gated_tools = frozenset({"search_files"})
    _graph_searched: bool = False  # per-turn flag
    _last_turn_id: str = ""  # for per-turn reset detection

    # Resolve mode once for the session's tool-gating policy.
    _graph_mode = _skill_graph_mode_enabled()

    def _on_pre_tool_call(tool_name: str, args: dict | None = None, **kw: Any) -> dict | str | None:
        nonlocal _graph_searched, _graph_mode, _last_turn_id
        turn_id = kw.get("turn_id", "")

        # Per-turn reset: when turn_id changes, clear the flag
        if turn_id and turn_id != _last_turn_id:
            _graph_searched = False
            _last_turn_id = turn_id

        # A graph search or a pre-injected-candidate load satisfies discovery.
        if tool_name in {"skill_graph_search", "skill_load"}:
            _graph_searched = True
            return None

        # Check gating: skill-graph mode + restricted tool + not yet searched
        if (
            _graph_mode
            and tool_name in _gated_tools
            and not _graph_searched
        ):
            return {"action": "block", "message":
                f"Tool '{tool_name}' is blocked until you call "
                f"skill_graph_search() first. This profile requires graph "
                f"discovery before filesystem searches."
            }
        return None

    ctx.register_hook("pre_tool_call", _on_pre_tool_call)

    # ── Hook: on_session_start — ensure DB ──
    def _on_session_start(**kw):
        try:
            _ensure_graph()
            logger.info("Skill graph ready")
        except Exception:
            logger.exception("skill-graph: on_session_start failed")

    ctx.register_hook("on_session_start", _on_session_start)

    _last_loaded_skill: str | None = None

    def _on_post_tool_call(**kw):
        nonlocal _last_loaded_skill
        tool_name = kw.get("tool_name", "")

        # Track skill_load → when quality-gate loads, mark the previous skill as successful
        if tool_name == "skill_load":
            skill_name = (kw.get("args", {}) or {}).get("name", "") or ""
            if not skill_name:
                return
            if skill_name == "quality-gate" and _last_loaded_skill:
                try:
                    conn = _get_conn()
                    conn.execute(
                        "UPDATE skill_term_stats SET success_count = success_count + 1 WHERE skill_name = ?",
                        (_last_loaded_skill,),
                    )
                    conn.commit()
                except Exception:
                    pass
                _last_loaded_skill = None
            else:
                _last_loaded_skill = skill_name
            return

        # Handle successful skill_manage batches → update affected graph nodes.
        if tool_name != "skill_manage":
            return
        args = kw.get("args", {})
        if not isinstance(args, dict):
            return
        result = kw.get("result")
        if isinstance(result, str):
            try:
                parsed_result = json.loads(result)
            except (TypeError, ValueError):
                parsed_result = None
            if isinstance(parsed_result, dict) and parsed_result.get("success") is False:
                return

        operations = args.get("operations")
        if not isinstance(operations, list):
            # Compatibility with the legacy single-operation shape.
            operations = [args]
        changes = [
            (str(op.get("action") or ""), str(op.get("name") or ""))
            for op in operations
            if isinstance(op, dict) and op.get("name")
        ]
        if not changes:
            return
        try:
            conn = _ensure_graph()
            with _graph_lock:
                for action, skill_name in changes:
                    if action == "delete":
                        conn.execute(
                            "UPDATE skill_nodes SET is_deleted = 1, "
                            "deleted_at = datetime('now') WHERE name = ?",
                            (skill_name,),
                        )
                        logger.info("skill-graph: soft-deleted skill '%s'", skill_name)
                    elif action in {"create", "patch", "write_file", "remove_file", "edit"}:
                        updated = _update_single_skill(conn, skill_name)
                        if updated:
                            logger.info(
                                "skill-graph: updated skill '%s' after %s",
                                skill_name, action,
                            )
                conn.commit()
        except Exception:
            logger.exception("skill-graph: post_tool_call failed for %s", changes)

    ctx.register_hook("post_tool_call", _on_post_tool_call)

    logger.info(
        "skill-graph plugin registered: tools=skill_graph_search+skill_load+skill_graph_config, "
        "cmd=/skill-graph, hooks=on_session_start+post_tool_call+pre_tool_call"
    )
