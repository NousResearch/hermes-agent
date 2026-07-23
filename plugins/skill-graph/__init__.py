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
    scenes      TEXT DEFAULT '[]',      -- JSON array, from metadata.hermes.scenes
    auto_tagged INTEGER DEFAULT 0,      -- 1 if tags/scenes were LLM-generated
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
    name, category, description, tags, scenes,
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
        from hermes_cli.config import load_config
        config = load_config()
        raw = (
            config
            .get("skills", {})
            .get("config", {})
            .get("skill-graph", {})
            .get("db_path")
        )
        if raw:
            return Path(raw).expanduser().resolve()
    except Exception:
        pass

    hermes_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
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
    """Read ``skills.config.skill-graph.source_dirs`` from config.yaml.

    Supports ``:read-only`` suffix per entry (e.g. ``~/bundled/skills:read-only``).
    The suffix is stripped from the returned path but tracked separately via
    ``_get_read_only_source_dirs()``.
    """
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
                entry_str = str(entry)
                # Strip :read-only suffix for path resolution
                clean = entry_str.rsplit(":read-only", 1)[0].strip()
                p = Path(os.path.expandvars(os.path.expanduser(clean)))
                if p.exists():
                    dirs.append(p.resolve())
    except Exception:
        pass
    return dirs


def _get_read_only_source_dirs() -> set[Path]:
    """Return the set of source_dirs marked ``:read-only``.

    Skills under these dirs should never have their SKILL.md modified
    by automated processes (LLM auto-tag, etc.).
    """
    ro_dirs: set[Path] = set()
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
                entry_str = str(entry)
                if ":read-only" in entry_str:
                    clean = entry_str.rsplit(":read-only", 1)[0].strip()
                    p = Path(os.path.expandvars(os.path.expanduser(clean)))
                    if p.exists():
                        ro_dirs.add(p.resolve())
    except Exception:
        pass
    return ro_dirs


def _is_read_only_skill(skill_path: str) -> bool:
    """Check if a skill's file path is under a read-only source dir."""
    ro_dirs = _get_read_only_source_dirs()
    if not ro_dirs:
        return False
    resolved = os.path.realpath(skill_path)
    for ro_dir in ro_dirs:
        ro_str = str(ro_dir)
        if resolved.startswith(ro_str + os.sep) or resolved == ro_str:
            return True
    return False


def _find_all_skills_dirs() -> list[Path]:
    """Return list of directories to scan for SKILL.md files.

    Scans three tiers:
      1. Current profile's skills/ dir (profile-specific skills)
      2. Global ~/.hermes/skills/ (shared across profiles — built-in
         skills, symlinked suites, etc.)
      3. External skill dirs from Hermes config + source_dirs
    """
    global_hermes = Path.home() / ".hermes"
    hermes_home = Path(os.environ.get("HERMES_HOME", global_hermes))
    dirs: list[Path] = []

    # 1. Current profile's skills/ dir (profile-local)
    profile_skills = hermes_home / "skills"
    if profile_skills.exists():
        dirs.append(profile_skills)

    # 2. Global ~/.hermes/skills/ (shared across profiles)
    global_skills = global_hermes / "skills"
    if global_skills.exists() and str(global_skills) != str(profile_skills):
        dirs.append(global_skills)

    # 3. Agent-created skills (discoverable via graph, not in prompt)
    agent_created_dir = hermes_home / "skill-graph" / "agent-created"
    if agent_created_dir.exists():
        dirs.append(agent_created_dir)

    # 4. Configured source dirs (user's extra paths, e.g. PS repo)
    source_dirs = _read_source_dirs_from_config()
    dirs.extend(source_dirs)

    # 5. External skill dirs from Hermes config
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
        "scenes": [],
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

                scenes = meta.get("metadata", {}).get("hermes", {}).get("scenes", [])
                if isinstance(scenes, str):
                    scenes = [s.strip() for s in scenes.split(",") if s.strip()]
                result["scenes"] = list(scenes) if isinstance(scenes, list) else []

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
    """Deduplicate skills by real path (resolving symlinks).

    When the same SKILL.md is reachable via multiple paths (e.g. a symlink
    in ~/.hermes/skills/ and the original in a source_dir), only one entry
    is kept.  Prefers paths under ~/.hermes/skills/ when duplicates exist,
    so profile-local overrides take precedence.
    """
    deduped: dict[str, Path] = {}
    real_to_name: dict[str, str] = {}
    primary_hint = str(Path.home() / ".hermes" / "skills")
    for name, path in skills:
        real = os.path.realpath(path)
        existing_name = real_to_name.get(real)
        if existing_name is None:
            # First time seeing this real path
            real_to_name[real] = name
            deduped[name] = path
        else:
            # Same real file — prefer ~/.hermes/skills/ paths
            if str(path).startswith(primary_hint) and \
               not str(deduped[existing_name]).startswith(primary_hint):
                # Replace: current path is in primary dir, old one wasn't
                del deduped[existing_name]
                real_to_name[real] = name
                deduped[name] = path
    return deduped


def _upsert_skill(conn: sqlite3.Connection, name: str, path: Path, now: float) -> dict[str, Any]:
    """Parse a SKILL.md and upsert its node + edges + FTS into the graph."""
    info = _parse_skill_md(path)
    tags_json = json.dumps(info["tags"], ensure_ascii=False)

    conn.execute(
        """INSERT OR REPLACE INTO skill_nodes
           (name, category, description, tags, scenes, file_path, content_hash, last_parsed)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (name, info["category"], info["description"],
         tags_json, json.dumps(info["scenes"], ensure_ascii=False), str(path), info["content_hash"], now),
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
    scenes_text = " ".join(info.get("scenes", []))
    conn.execute("DELETE FROM skill_fts WHERE name = ?", (name,))
    conn.execute(
        "INSERT INTO skill_fts (name, category, description, tags, scenes) VALUES (?, ?, ?, ?, ?)",
        (name, info.get("category", ""), info.get("description", ""), tags_text, scenes_text),
    )

    # Upsert terms
    conn.execute("DELETE FROM skill_terms WHERE skill_name = ?", (name,))
    terms = _extract_skill_terms(name, info.get("tags", []), info.get("description", ""))
    for term_text, strength, source in terms:
        conn.execute(
            "INSERT OR IGNORE INTO skill_terms (term, skill_name, strength, source) VALUES (?, ?, ?, ?)",
            (term_text, name, strength, source),
        )


    # Auto-tag detection: mark skills missing tags or scenes
    if _needs_auto_tag(info):
        conn.execute(
            "UPDATE skill_nodes SET auto_tagged = 0 WHERE name = ?",
            (name,),
        )
    else:
        conn.execute(
            "UPDATE skill_nodes SET auto_tagged = 1 WHERE name = ?",
            (name,),
        )

    return info


# ── LLM Auto-Tag ────────────────────────────────────────────────────────────

# Scene vocabulary — fixed set of valid scene values for LLM auto-tagging
SCENE_VOCABULARY = [
    "coding",       # Writing/fixing/reviewing code, PRs, git operations
    "writing",      # Composing text — papers, articles, docs, fiction
    "research",     # Investigating, exploring, searching, auditing
    "design",       # Architecture, system design, planning, prototyping
    "devops",       # Deploying, configuring, docker, infrastructure
    "hermes",       # Hermes agent config, plugins, skills, meta-work
    "media",        # Audio, video, image production
    "common",       # Cross-scene, applicable in any context
]


def _needs_auto_tag(info: dict[str, Any]) -> bool:
    """Check if a skill needs LLM auto-tagging (missing tags or scenes)."""
    return not info.get("tags") or not info.get("scenes")


def _build_autotag_prompt(skill_name: str, content: str) -> str:
    """Build the LLM prompt for auto-tagging a skill."""
    scene_desc = "\n".join(
        f"  - {s}: {SCENE_VOCABULARY_DESC.get(s, '')}"
        for s in SCENE_VOCABULARY
    )
    return f"""You are a skill metadata tagger. Analyze this SKILL.md and produce:
1. tags: 3-8 lowercase descriptive tags (domain, tech stack, activity type)
2. scenes: 1-3 scene values from the fixed scene vocabulary below
3. suggestions: any scene candidates NOT in the vocabulary that might apply

Scene vocabulary (ONLY use these for the "scenes" field):
{scene_desc}

Rules for scenes:
- Pick 1-3 most applicable scenes from the vocabulary above
- Include "common" only if the skill truly applies across many scenes
- Be conservative — if unsure, prefer fewer scenes

Rules for tags:
- Lowercase, descriptive, 3-8 tags
- Include: domain, primary tech/tool, activity type
- Example: ["python", "fastapi", "backend", "code-review"]

Respond with ONLY a JSON object (no markdown, no explanation):
{{"tags": ["tag1", "tag2", ...], "scenes": ["scene1", ...], "suggestions": []}}

Skill: {skill_name}
Content:
{content[:8000]}"""


SCENE_VOCABULARY_DESC = {
    "coding": "Writing/fixing/reviewing code, PRs, git operations",
    "writing": "Composing text — papers, articles, docs, fiction",
    "research": "Investigating, exploring, searching, auditing",
    "design": "Architecture, system design, planning, prototyping",
    "devops": "Deploying, configuring, docker, infrastructure",
    "hermes": "Hermes agent config, plugins, skills, meta-work",
    "media": "Audio, video, image production",
    "common": "Cross-scene, applicable in any context",
}


def _call_llm_for_tagging(prompt: str) -> dict[str, Any] | None:
    """Call the configured LLM provider for auto-tagging.

    Uses the same provider/model as the current agent session.
    Returns parsed JSON response or None on failure.
    """
    try:
        import requests
        from hermes_cli.auth import PROVIDER_REGISTRY, has_usable_secret
        from hermes_cli.config import load_config

        config = load_config()
        sg_config = config.get("skills", {}).get("config", {}).get("skill-graph", {})
        auto_tag_cfg = sg_config.get("auto_tag", {}) if isinstance(sg_config, dict) else {}
        preferred_provider = auto_tag_cfg.get("provider", "")

        def _resolve_provider(pid: str) -> tuple[str, str] | None:
            """Return (api_key, base_url) for a provider, or None."""
            pconfig = PROVIDER_REGISTRY.get(pid)
            if not pconfig:
                return None
            for env_var in pconfig.api_key_env_vars:
                key_val = os.environ.get(env_var, "")
                if has_usable_secret(key_val):
                    url = os.environ.get(pconfig.base_url_env_var, "") if pconfig.base_url_env_var else ""
                    if not url:
                        url = pconfig.inference_base_url or ""
                    return key_val, url
            return None

        provider_name = ""
        api_key = ""
        base_url = ""
        if preferred_provider:
            resolved = _resolve_provider(preferred_provider)
            if resolved:
                provider_name = preferred_provider
                api_key, base_url = resolved
            else:
                logger.warning(
                    "skill-graph: auto-tag skipped — preferred provider '%s' has no API key configured",
                    preferred_provider,
                )
                return None
        else:
            # Fallback: first provider with a usable API key
            for pid in PROVIDER_REGISTRY:
                if pid in ("copilot", "lmstudio"):
                    continue
                resolved = _resolve_provider(pid)
                if resolved:
                    provider_name = pid
                    api_key, base_url = resolved
                    break

        if not provider_name:
            logger.warning("skill-graph: auto-tag skipped — no provider with API key configured")
            return None

        # Resolve model: config model.default > hardcoded fallback
        try:
            config = load_config()
            model_cfg = config.get("model", {})
            model_name = model_cfg.get("default", "") if isinstance(model_cfg, dict) else ""
        except Exception:
            model_name = ""
        if not model_name:
            model_name = "deepseek-chat"

        # Normalise base_url: strip /v1 suffix if present (we add it below)
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 512,
        }

        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()

        # Extract JSON from possible markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:]) if len(lines) > 1 else text
            if text.endswith("```"):
                text = text[:-3]
        if text.startswith("json"):
            text = text[4:].strip()

        result = json.loads(text)
        if not isinstance(result, dict):
            return None
        return result

    except Exception as e:
        logger.warning("skill-graph: auto-tag LLM call failed: %s", e)
        return None


def _patch_skill_frontmatter(skill_path: str, tags: list[str], scenes: list[str]) -> bool:
    """Write tags and scenes back to the SKILL.md frontmatter.

    Only called for editable source dirs (non-read-only).
    Returns True on success.
    """
    try:
        p = Path(skill_path)
        content = p.read_text(encoding="utf-8", errors="replace")

        # Find frontmatter boundaries
        content_str = content.lstrip("\ufeff")
        if not content_str.startswith("---"):
            return False
        end = content_str.find("---", 3)
        if end == -1:
            return False

        # Parse existing frontmatter
        frontmatter = content_str[3:end].strip()
        try:
            meta = yaml.safe_load(frontmatter) or {}
        except yaml.YAMLError:
            return False

        # Update metadata.hermes.tags and metadata.hermes.scenes
        hermes_meta = meta.setdefault("metadata", {}).setdefault("hermes", {})
        hermes_meta["tags"] = tags
        hermes_meta["scenes"] = scenes

        # Re-serialize frontmatter
        new_fm = yaml.dump(meta, default_flow_style=False, allow_unicode=True).strip()
        new_content = f"---\n{new_fm}\n---{content_str[end + 3:]}"

        p.write_text(new_content, encoding="utf-8")
        return True

    except Exception as e:
        logger.warning("skill-graph: failed to patch frontmatter for %s: %s", skill_path, e)
        return False


def _auto_tag_skill(conn: sqlite3.Connection, skill_name: str) -> bool:
    """Run LLM auto-tag for a single skill. Updates DB. Writes back to SKILL.md if editable.

    Returns True on success.
    """
    # Fetch skill content
    node = conn.execute(
        "SELECT file_path, content_hash FROM skill_nodes WHERE name = ?",
        (skill_name,),
    ).fetchone()
    if not node or not node["file_path"]:
        return False

    skill_path = node["file_path"]
    try:
        content = Path(skill_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False

    # Build prompt + call LLM
    prompt = _build_autotag_prompt(skill_name, content)
    result = _call_llm_for_tagging(prompt)
    if result is None:
        return False

    tags = result.get("tags", [])
    scenes = result.get("scenes", [])
    suggestions = result.get("suggestions", [])

    if not isinstance(tags, list) or not isinstance(scenes, list):
        return False

    # Filter scenes to vocabulary only
    valid_scenes = [s for s in scenes if s in SCENE_VOCABULARY]
    if suggestions:
        logger.info(
            "skill-graph: auto-tag suggestions for '%s': %s",
            skill_name, suggestions,
        )

    # Update DB
    conn.execute(
        "UPDATE skill_nodes SET tags = ?, scenes = ?, auto_tagged = 1 WHERE name = ?",
        (json.dumps(tags, ensure_ascii=False),
         json.dumps(valid_scenes, ensure_ascii=False),
         skill_name),
    )
    # Update FTS
    tags_text = " ".join(tags)
    scenes_text = " ".join(valid_scenes)
    conn.execute("DELETE FROM skill_fts WHERE name = ?", (skill_name,))
    conn.execute(
        "INSERT INTO skill_fts (name, category, description, tags, scenes) VALUES (?, ?, ?, ?, ?)",
        (skill_name, "", "", tags_text, scenes_text),
    )
    # Update terms
    conn.execute("DELETE FROM skill_terms WHERE skill_name = ?", (skill_name,))
    terms = _extract_skill_terms(skill_name, tags, "")
    for term_text, strength, source in terms:
        conn.execute(
            "INSERT OR IGNORE INTO skill_terms (term, skill_name, strength, source) VALUES (?, ?, ?, ?)",
            (term_text, skill_name, strength, source),
        )

    # Write back to SKILL.md if not read-only
    if not _is_read_only_skill(skill_path):
        wrote = _patch_skill_frontmatter(skill_path, tags, valid_scenes)
        if wrote:
            logger.info("skill-graph: auto-tagged '%s' → tags=%s scenes=%s (wrote SKILL.md)",
                        skill_name, tags, valid_scenes)
        else:
            logger.info("skill-graph: auto-tagged '%s' → tags=%s scenes=%s (SKILL.md write failed)",
                        skill_name, tags, valid_scenes)
    else:
        logger.info("skill-graph: auto-tagged '%s' → tags=%s scenes=%s (read-only, DB only)",
                    skill_name, tags, valid_scenes)

    conn.commit()
    return True





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
            name, category, description, tags, scenes,
            tokenize='porter unicode61'
        );
    """)

    for name, path in deduped.items():
        _upsert_skill(conn, name, path, now)
    conn.commit()

    # Auto-tag skills missing tags or scenes
    auto_count = 0
    pending = conn.execute(
        "SELECT name FROM skill_nodes WHERE auto_tagged = 0 AND (is_deleted IS NULL OR is_deleted = 0)"
    ).fetchall()
    for row in pending:
        if _auto_tag_skill(conn, row["name"]):
            auto_count += 1

    logger.info("Skill graph rebuilt: %d skills, %d auto-tagged", len(deduped), auto_count)
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

    # Auto-tag newly parsed skills missing tags or scenes
    auto_count = 0
    pending = conn.execute(
        "SELECT name FROM skill_nodes WHERE auto_tagged = 0 AND (is_deleted IS NULL OR is_deleted = 0)"
    ).fetchall()
    for row in pending:
        if _auto_tag_skill(conn, row["name"]):
            auto_count += 1

    logger.info(
        "Skill graph synced: %d parsed, %d unchanged, %d removed, %d total, %d auto-tagged",
        parsed_count, skipped_count, len(stale), len(deduped), auto_count,
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
    # Mark skills in main ~/.hermes/skills/ dir as needing external organization
    main_skills = str((Path.home() / ".hermes" / "skills").resolve())
    if str(skill_path.resolve()).startswith(main_skills):
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


def _search_graph(query: str, conn: sqlite3.Connection, limit: int = 10,
                  scenes: list[str] | None = None) -> list[dict[str, Any]]:
    """Search the skill graph by intent query.

    If ``scenes`` is provided, skill results matching those scenes get a
    ×1.3 score boost in Phase 5 (soft weighting, not hard filtering).
    """
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

    # Phase 5b: Scene-based scoring boost (soft weighting, not hard filtering)
    if scenes:
        _scene_set = set(s.lower() for s in scenes)
        for sname, r in results.items():
            _skill_scenes_json = conn.execute(
                "SELECT scenes FROM skill_nodes WHERE name = ?", (sname,),
            ).fetchone()
            if _skill_scenes_json:
                try:
                    _skill_scenes = json.loads(_skill_scenes_json[0])
                    if isinstance(_skill_scenes, list) and _skill_scenes:
                        _skill_set = set(s.lower() for s in _skill_scenes)
                        # "common" is neutral — no boost, no penalty
                        if _skill_set == {"common"}:
                            continue
                        # If any search scene matches any skill scene → boost
                        if _scene_set & _skill_set:
                            r["score"] *= 1.3
                            r.setdefault("_scene_boost", True)
                except (json.JSONDecodeError, TypeError):
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
    # v4: scenes + auto_tagged for LLM-powered scene/tag generation
    for col, col_type in (
        ("scenes", "TEXT DEFAULT '[]'"),
        ("auto_tagged", "INTEGER DEFAULT 0"),
    ):
        try:
            conn.execute(f"ALTER TABLE skill_nodes ADD COLUMN {col} {col_type}")
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

    # v5: FTS5 scenes column — virtual tables can't be ALTERed, must rebuild
    _fts_cols = conn.execute("PRAGMA table_info(skill_fts)").fetchall()
    _fts_names = {row[1] for row in _fts_cols}
    if "scenes" not in _fts_names:
        logger.info("skill-graph: migrating FTS table — adding scenes column")
        # Save existing data (fetch all + commit to release read lock before DROP)
        _rows = conn.execute(
            "SELECT name, category, description, tags FROM skill_fts"
        ).fetchall()
        conn.commit()
        # Drop old FTS + shadow tables
        for _t in ("skill_fts", "skill_fts_data", "skill_fts_idx",
                    "skill_fts_docsize", "skill_fts_config"):
            conn.execute(f"DROP TABLE IF EXISTS {_t}")
        # Recreate with scenes column
        conn.executescript("""
            CREATE VIRTUAL TABLE skill_fts USING fts5(
                name, category, description, tags, scenes,
                tokenize='porter unicode61'
            );
        """)
        # Repopulate (scenes data may not exist yet, so use empty string)
        for _r in _rows:
            _skill_scenes = conn.execute(
                "SELECT scenes FROM skill_nodes WHERE name = ?", (_r["name"],)
            ).fetchone()
            _scenes_text = " ".join(
                json.loads(_skill_scenes[0])
            ) if _skill_scenes and _skill_scenes[0] else ""
            conn.execute(
                "INSERT INTO skill_fts (name, category, description, tags, scenes) "
                "VALUES (?, ?, ?, ?, ?)",
                (_r["name"], _r["category"], _r["description"], _r["tags"],
                 _scenes_text),
            )
        conn.commit()
        logger.info("skill-graph: FTS migration complete — %d rows repopulated", len(_rows))



def _read_config_path() -> Path | None:
    """Read config.yaml path from HERMES_HOME."""
    hermes_home = os.environ.get("HERMES_HOME", "")
    if hermes_home:
        cfg = Path(hermes_home) / "config.yaml"
        if cfg.exists():
            return cfg
    cfg = Path.home() / ".hermes" / "config.yaml"
    return cfg if cfg.exists() else None

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
    target = os.path.abspath(os.path.expanduser(path_str))
    if not os.path.isdir(target):
        return f"Not a directory: {target}"
    try:
        cfg_path = _read_config_path()
        if not cfg_path:
            return "❌ Cannot find config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        sg_cfg = cfg.setdefault("skills", {}).setdefault("config", {}).setdefault("skill-graph", {})
        source_dirs = sg_cfg.get("source_dirs", [])
        resolved = [os.path.abspath(os.path.expanduser(str(d))) for d in source_dirs]
        if action == "add":
            if target in resolved:
                return f"Already in source_dirs: {target}"
            source_dirs.append(path_str)
        elif action == "remove":
            if target not in resolved:
                return f"Not in source_dirs: {target}"
            source_dirs = [d for d in source_dirs if os.path.abspath(os.path.expanduser(str(d))) != target]
        else:
            return f"Unknown action: {action}"
        sg_cfg["source_dirs"] = source_dirs
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        try:
            conn = _ensure_graph()
            with _graph_lock:
                count = _full_rebuild(conn)
            return f"✅ {action}ed {target}\n   Graph rebuilt: {count} skills indexed."
        except Exception as e:
            return f"✅ Config updated but graph rebuild failed: {e}"
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
                               "scanned_dirs": [str(d) for d in _find_all_skills_dirs() if d.exists()],
                               "persisted": persist}, default=str)
        if action in ("add_dir", "remove_dir"):
            if not path_str:
                return json.dumps({"success": False, "error": "path required"})
            target = os.path.abspath(os.path.expanduser(path_str))
            if not os.path.isdir(target):
                return json.dumps({"success": False, "error": f"Not a directory: {target}"})
            if persist:
                cfg_path = _read_config_path()
                if not cfg_path:
                    return json.dumps({"success": False, "error": "Cannot find config.yaml"})
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f) or {}
                sg_cfg = cfg.setdefault("skills", {}).setdefault("config", {}).setdefault("skill-graph", {})
                source_dirs = sg_cfg.get("source_dirs", [])
                resolved = [os.path.abspath(os.path.expanduser(str(d))) for d in source_dirs]
                if action == "add_dir":
                    if target in resolved:
                        return json.dumps({"success": True, "action": "add_dir", "path": target, "note": "already present"})
                    source_dirs.append(path_str)
                else:
                    source_dirs = [d for d in source_dirs if os.path.abspath(os.path.expanduser(str(d))) != target]
                sg_cfg["source_dirs"] = source_dirs
                with open(cfg_path, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
            conn = _ensure_graph()
            with _graph_lock:
                count = _full_rebuild(conn)
            return json.dumps({"success": True, "action": action, "path": target, "skills_indexed": count, "persisted": persist})
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
    scenes = args.get("scenes") if isinstance(args, dict) else None
    if isinstance(scenes, str):
        scenes = [s.strip() for s in scenes.split(",") if s.strip()]
    elif not isinstance(scenes, list):
        scenes = None

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

            results = _search_graph(query, conn, limit=limit, scenes=scenes)
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
                    "scenes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional scene filter — skills matching these scenes get ×1.3 score boost. Use scene vocabulary: coding, writing, research, design, devops, hermes, media, common. Soft weighting, not hard filtering.",
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

    # ── Hook: pre_tool_call — block find/read_file/recall if graph not searched ──
    _gated_tools = frozenset({"find", "read_file"})
    _graph_searched_turn: dict[str, bool] = {}  # turn_id → searched

    # Resolve skill_graph_mode from config at registration time (startup constant)
    _graph_mode = False
    try:
        from hermes_cli.config import load_config
        _graph_mode = load_config().get("agent", {}).get("skill_graph_mode", False)
    except Exception:
        pass

    def _on_pre_tool_call(tool_name: str, args: dict | None = None, **kw: Any) -> dict | str | None:
        nonlocal _graph_searched_turn, _graph_mode
        turn_id = kw.get("turn_id", "")
        if not turn_id:
            return None

        # Turn boundary: reset flag for new turns
        if turn_id not in _graph_searched_turn:
            _graph_searched_turn.clear()
            _graph_searched_turn[turn_id] = False

        # If this IS skill_graph_search, mark it and allow
        if tool_name == "skill_graph_search":
            _graph_searched_turn[turn_id] = True
            return None

        # Check gating: skill-graph mode + restricted tool + not yet searched
        if (
            _graph_mode
            and tool_name in _gated_tools
            and not _graph_searched_turn.get(turn_id, False)
        ):
            return {"action": "block", "message":
                f"Tool '{tool_name}' is blocked until you call "
                f"skill_graph_search() first. This profile requires graph "
                f"discovery before filesystem or session searches."
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

        # ── Register proxy commands from graph-discovered skills ──
    _main_skills_dir = str((Path.home() / ".hermes" / "skills").resolve())

    def _register_graph_commands():
        """Scan all skills' SKILL.md frontmatter for commands: and register proxy handlers."""
        try:
            skill_dirs = _find_all_skills_dirs()
            skills = _scan_skill_mds(skill_dirs)
            deduped = _dedup_skills(skills)
            _registered = 0
            for _name, _path in deduped.items():
                _path_str = str(_path.resolve())
                _unresolved_str = str(_path)
                _is_in_main_dir = (
                    _path_str.startswith(_main_skills_dir) or
                    _unresolved_str.startswith(_main_skills_dir)
                )
                # Register /<skill_name> ONLY for skills outside the main skills dir.
                # Skills in ~/.hermes/skills/ are already handled natively by Hermes'
                # scan_skill_commands() — a plugin proxy would block the native handler
                # that loads SKILL.md as instructions and continues the conversation.
                if not _is_in_main_dir:
                    ctx.register_command(
                        name=_name,
                        handler=_make_proxy(_name),
                        description=f"Proxy to graph-discovered skill: {_name}",
                    )
                    _registered += 1
                # Register any explicit commands: from frontmatter for ALL skills
                try:
                    _text = _path.read_text(encoding="utf-8", errors="replace")
                    _text = _text.lstrip("\ufeff")
                    if _text.startswith("---"):
                        _end = _text.find("---", 3)
                        if _end != -1:
                            _fm = yaml.safe_load(_text[3:_end].strip()) or {}
                            _cmds = _fm.get("metadata", {}).get("hermes", {}).get("commands", [])
                            if isinstance(_cmds, str):
                                _cmds = [c.strip() for c in _cmds.split(",") if c.strip()]
                            if isinstance(_cmds, list):
                                for _cmd in _cmds:
                                    _cmd = _cmd.lstrip("/").strip()
                                    if _cmd and _cmd != _name:
                                        ctx.register_command(
                                            name=_cmd,
                                            handler=_make_proxy(_name),
                                            description=f"Proxy to graph-discovered skill: {_name}",
                                        )
                                        _registered += 1
                except Exception:
                    pass
            if _registered:
                logger.info("skill-graph: registered %d proxy slash commands from graph skills", _registered)
        except Exception:
            logger.exception("skill-graph: failed to register proxy commands")

    def _make_proxy(skill_name: str):
        """Create a proxy handler that loads the skill and injects it as agent input."""
        def _proxy(_args: str) -> dict | None:
            try:
                _result = _handle_skill_load({"name": skill_name})
                _data = json.loads(_result)
                if not _data.get("success"):
                    return None
                _content = _data.get("content", "")
                _skill_dir = _data.get("skill_dir", "")
                _name = _data.get("name", skill_name)

                parts = [
                    f"[IMPORTANT: The user has invoked the {_name} skill. "
                    f"The full skill content is loaded below.]",
                    "",
                    _content.strip(),
                ]
                if _skill_dir:
                    parts.extend(["", f"[Skill directory: {_skill_dir}]"])
                if _args:
                    parts.extend(["",
                        f"The user has provided the following instruction "
                        f"alongside the skill invocation: {_args}"])

                return {"action": "inject", "content": "\n".join(parts)}
            except Exception:
                pass
            return None
        return _proxy

    _register_graph_commands()
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

        # Handle skill_manage → update graph
        if tool_name != "skill_manage":
            return
        args = kw.get("args", {})
        if not isinstance(args, dict):
            return
        action = args.get("action", "")
        if action not in ("create", "edit", "patch", "delete"):
            return
        skill_name = args.get("name", "")
        if not skill_name:
            return
        try:
            conn = _ensure_graph()
            with _graph_lock:
                if action == "delete":
                    conn.execute(
                        "UPDATE skill_nodes SET is_deleted = 1, deleted_at = datetime('now') WHERE name = ?",
                        (skill_name,),
                    )
                    conn.commit()
                    logger.info("skill-graph: soft-deleted skill '%s'", skill_name)
                else:
                    updated = _update_single_skill(conn, skill_name)
                    if updated:
                        logger.info("skill-graph: updated skill '%s' after %s", skill_name, action)
        except Exception:
            logger.exception("skill-graph: post_tool_call failed for skill '%s'", skill_name)

    ctx.register_hook("post_tool_call", _on_post_tool_call)

    # ── Monkey-patch skill_view to fall back to skill-graph ──
    try:
        import tools.skills_tool as _st
        _orig_skill_view = _st.skill_view

        def _patched_skill_view(
            name: str,
            file_path: str = None,
            task_id: str = None,
            preprocess: bool = True,
        ) -> str:
            result = _orig_skill_view(
                name, file_path=file_path,
                task_id=task_id, preprocess=preprocess,
            )
            data = json.loads(result)
            if data.get("success") or file_path:
                return result
            sg = _handle_skill_load({"name": name})
            sg_data = json.loads(sg)
            return sg if sg_data.get("success") else result

        _st.skill_view = _patched_skill_view
        logger.info("skill-graph: patched skill_view with graph fallback")
    except Exception:
        logger.exception("skill-graph: failed to patch skill_view")

    logger.info(
        "skill-graph plugin registered: tools=skill_graph_search+skill_load+skill_graph_config, "
        "cmd=/skill-graph, hooks=on_session_start+post_tool_call+pre_tool_call"
    )
