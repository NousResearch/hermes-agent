"""
Wiki Builder — converts pending sessions into structured wiki pages.

Runs as a background daemon thread inside the holographic memory plugin.
Reads from wiki_pending_queue, calls the LLM for analysis, writes wiki_pages.
"""

import json
import logging
import re
import sqlite3
import datetime
from typing import Any, Dict, List, Optional

# i18n: section headers by language
_I18N_HEADERS = {
    "en": {"bg": "Background", "dec": "Key Decisions", "prob": "Problems & Solutions", "res": "Result", "timeline": "Session Timeline", "overview": "Overview"},
    "zh": {"bg": "背景", "dec": "关键决策", "prob": "问题与解决", "res": "结果", "timeline": "Session 时间线", "overview": "概述"},
    "ja": {"bg": "背景", "dec": "重要な決定", "prob": "問題と解決", "res": "結果", "timeline": "セッションタイムライン", "overview": "概要"},
    "ko": {"bg": "배경", "dec": "주요 결정", "prob": "문제와 해결", "res": "결과", "timeline": "세션 타임라인", "overview": "개요"},
    "de": {"bg": "Hintergrund", "dec": "Wichtige Entscheidungen", "prob": "Probleme & Lösungen", "res": "Ergebnis", "timeline": "Sitzungsverlauf", "overview": "Übersicht"},
    "fr": {"bg": "Contexte", "dec": "Décisions clés", "prob": "Problèmes et solutions", "res": "Résultat", "timeline": "Chronologie", "overview": "Aperçu"},
    "es": {"bg": "Contexto", "dec": "Decisiones clave", "prob": "Problemas y soluciones", "res": "Resultado", "timeline": "Cronología", "overview": "Resumen"},
}

logger = logging.getLogger(__name__)

# Truncation limits
_MAX_MESSAGE_LEN = 3000
_MAX_MESSAGES_FOR_LLM = 80  # cap messages sent to LLM


class WikiBuilder:
    """Process pending sessions into wiki pages via LLM analysis."""

    def __init__(self, store, config: dict):
        self._store = store
        self._config = config
        self._conn = store._conn  # shared connection
        self._lock = store._lock

    def process_pending(self) -> int:
        """Process all pending wiki jobs. Returns count processed."""
        processed = 0
        while True:
            items = self._store.dequeue_wiki_pending(limit=1)
            if not items:
                break
            item = items[0]
            try:
                self._process_session(item)
                self._store.mark_wiki_done(item["id"], "done")
                processed += 1
            except Exception as e:
                logger.error("WikiBuilder failed for %s: %s", item.get("session_id"), e)
                self._store.mark_wiki_done(item["id"], "failed")
        return processed

    def _process_session(self, item: dict) -> None:
        """Process a single session into wiki page + facts."""
        session_id = item["session_id"]
        title = item.get("title", "") or ""
        source = item.get("source", "")
        messages = item.get("messages", [])

        # Check if already processed
        with self._lock:
            existing = self._conn.execute(
                "SELECT page_id FROM wiki_pages WHERE source_session_id = ?",
                (session_id,),
            ).fetchone()
        if existing:
            logger.debug("Session %s already in wiki_pages, skipping", session_id)
            return

        # Filter to user + assistant only, truncate long messages
        filtered = []
        for msg in messages:
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue
            content = msg.get("content", "")
            if not content or not isinstance(content, str):
                continue
            if len(content) > _MAX_MESSAGE_LEN:
                content = content[: _MAX_MESSAGE_LEN // 2] + "\n...[truncated]...\n" + content[-_MAX_MESSAGE_LEN // 2 :]
            filtered.append({"role": role, "content": content})

        if len(filtered) < 2:
            logger.debug("Session %s too few messages (%d), skipping", session_id, len(filtered))
            return

        # Cap total messages
        if len(filtered) > _MAX_MESSAGES_FOR_LLM:
            filtered = filtered[:_MAX_MESSAGES_FOR_LLM]

        # Call LLM for analysis
        analysis = self._call_llm(filtered, title, source)
        if not analysis:
            logger.warning("LLM returned no analysis for %s, using defaults", session_id)
            analysis = self._default_analysis(title)

        language = analysis.get("language", "en")
        quality = analysis.get("quality", 2)
        if quality < 1 or quality > 5:
            quality = 2

        # Build date from session_id
        date = self._extract_date(session_id)

        # Build slug
        slug = self._build_slug(date, analysis.get("title", title))

        # Build wiki page content
        full_content = self._build_wiki_page(
            session_id=session_id,
            date=date,
            title=analysis.get("title", title),
            language=language,
            quality=quality,
            content_type=analysis.get("content_type", "discussion"),
            topics=analysis.get("topics", []),
            entities=analysis.get("entities", []),
            keywords=analysis.get("keywords", []),
            result=analysis.get("result", ""),
            background=analysis.get("background", ""),
            decisions=analysis.get("decisions", []),
            problems=analysis.get("problems", []),
        )

        summary = analysis.get("result", "")[:200] or analysis.get("background", "")[:200]

        topics = analysis.get("topics", [])
        entities = analysis.get("entities", [])
        keywords = analysis.get("keywords", [])

        # Write to SQLite
        with self._lock:
            # Insert wiki page
            cur = self._conn.execute(
                """INSERT OR REPLACE INTO wiki_pages
                   (page_type, slug, title, date, quality, content_type,
                    topics, keywords, summary, full_content, source_session_id)
                   VALUES ('session', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    slug,
                    analysis.get("title", title),
                    date,
                    quality,
                    analysis.get("content_type", "discussion"),
                    json.dumps(topics, ensure_ascii=False),
                    json.dumps(keywords, ensure_ascii=False),
                    summary,
                    full_content,
                    session_id,
                ),
            )
            page_id = cur.lastrowid

            # Link entities
            for ename in entities:
                entity_id = self._ensure_entity(ename)
                if entity_id:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO wiki_page_entities (page_id, entity_id) VALUES (?, ?)",
                        (page_id, entity_id),
                    )

            # Update topic pages
            for topic in topics:
                self._update_topic_page(topic, date, analysis.get("title", title), quality)

            # Extract facts (only for quality >= 3)
            if quality >= 3:
                self._extract_facts(analysis, slug, entities)

            self._conn.commit()

        logger.info(
            "Wiki page created: %s (quality=%d, topics=%s)",
            slug, quality, topics,
        )

    def _call_llm(self, messages: list, title: str, source: str) -> Optional[dict]:
        """Call the configured LLM provider for session analysis."""
        import os
        import httpx

        # Get provider config: plugin config → config.yaml → env vars
        provider = self._config.get("provider", "")
        model = self._config.get("model", "")
        api_key = self._config.get("api_key", "")
        base_url = self._config.get("base_url", "")

        # If not in plugin config, read from config.yaml model section
        if not model or not provider:
            try:
                import yaml
                from hermes_constants import get_hermes_home
                cfg_path = get_hermes_home() / "config.yaml"
                if cfg_path.exists():
                    with open(cfg_path) as f:
                        cfg = yaml.safe_load(f) or {}
                    model_cfg = cfg.get("model", {})
                    if not model:
                        model = model_cfg.get("default", "") or model_cfg.get("model", "")
                    if not provider:
                        provider = model_cfg.get("provider", "")
                    if not base_url:
                        base_url = model_cfg.get("base_url", "")
            except Exception:
                pass

        if not model:
            logger.warning("No model configured for wiki builder, skipping LLM call")
            return None

        # Resolve base_url from provider
        if not base_url:
            base_url = self._resolve_base_url(provider)
        if not base_url:
            logger.warning("Cannot resolve base_url for provider %s", provider)
            return None

        # Resolve API key
        if not api_key:
            api_key = self._resolve_api_key(provider)
        if not api_key:
            api_key = "no-key"  # some providers don't need a key

        # Build conversation for analysis
        analysis_messages = [
            {
                "role": "system",
                "content": self._analysis_prompt(),
            },
            {
                "role": "user",
                "content": self._format_session_for_llm(messages, title, source),
            },
        ]

        try:
            url = f"{base_url.rstrip('/')}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            payload = {
                "model": model,
                "messages": analysis_messages,
                "max_tokens": 2000,
                "temperature": 0.3,
            }

            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            content = data["choices"][0]["message"]["content"]
            return self._extract_json(content)

        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return None

    def _analysis_prompt(self) -> str:
        # Use English for broader compatibility
        return """You are a memory-wiki agent. Analyze the following session and return a JSON analysis.

Return strict JSON only, no extra text:
{
  "title": "Session title (≤30 chars, summarize core content)",
  "language": "en",
  "quality": 3,
  "content_type": "troubleshooting",
  "topics": ["topic-slug-1", "topic-slug-2"],
  "entities": ["Entity Name 1", "Entity Name 2"],
  "keywords": ["keyword1", "keyword2"],
  "result": "One-sentence summary of the final outcome",
  "background": "Brief user goal and context (≤100 chars)",
  "decisions": ["Decision 1 and reason", "Decision 2 and reason"],
  "problems": ["Problem 1 → Solution", "Problem 2 → Solution"]
}

CRITICAL: Respond in the SAME LANGUAGE as the session content.
The "language" field must be the ISO 639-1 code (e.g. "en", "zh", "ja", "ko", "de", "fr", "es").
All text fields (title, result, background, decisions, problems) must be in that language.
Only topic slugs, entity names, and content_type remain in English."""

    def _format_session_for_llm(self, messages: list, title: str, source: str) -> str:
        """Format messages into a readable prompt for the LLM."""
        lines = []
        if title:
            lines.append(f"Title: {title}")
        if source:
            lines.append(f"Source: {source}")
        lines.append(f"Messages: {len(messages)}")
        lines.append("---")
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"[{prefix}]: {content[:1500]}")
        return "\n".join(lines)

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON from LLM response, handling markdown code blocks and extra text."""
        text = text.strip()
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try extracting from markdown code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        # Try finding first { ... } block
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        logger.warning("Failed to extract JSON from LLM response: %s...", text[:200])
        return None

    def _default_analysis(self, title: str) -> dict:
        """Fallback analysis when LLM is unavailable."""
        return {
            "title": title or "Untitled Session",
            "language": "en",
            "quality": 2,
            "content_type": "discussion",
            "topics": [],
            "entities": [],
            "keywords": [],
            "result": "",
            "background": "",
            "decisions": [],
            "problems": [],
        }

    def _resolve_base_url(self, provider: str) -> Optional[str]:
        """Resolve provider name to API base URL using PROVIDER_REGISTRY."""
        import os
        # 1. Check env var overrides first
        for env_key in ("OPENAI_BASE_URL", "CUSTOM_BASE_URL", "XIAOMI_BASE_URL"):
            val = os.environ.get(env_key, "")
            if val:
                return val
        # 2. Look up PROVIDER_REGISTRY
        try:
            from hermes_cli.auth import PROVIDER_REGISTRY
            cfg = PROVIDER_REGISTRY.get(provider)
            if cfg and cfg.inference_base_url:
                # Check provider-specific env var override
                if cfg.base_url_env_var:
                    env_val = os.environ.get(cfg.base_url_env_var, "")
                    if env_val:
                        return env_val
                return cfg.inference_base_url
        except ImportError:
            pass
        return None

    def _resolve_api_key(self, provider: str) -> Optional[str]:
        """Resolve provider name to API key using PROVIDER_REGISTRY."""
        import os
        # 1. Check generic env vars
        for env_key in ("CUSTOM_API_KEY", "OPENAI_API_KEY"):
            val = os.environ.get(env_key, "")
            if val:
                return val
        # 2. Look up PROVIDER_REGISTRY
        try:
            from hermes_cli.auth import PROVIDER_REGISTRY
            cfg = PROVIDER_REGISTRY.get(provider)
            if cfg and cfg.api_key_env_vars:
                for env_var in cfg.api_key_env_vars:
                    val = os.environ.get(env_var, "")
                    if val:
                        return val
        except ImportError:
            pass
        return None

    def _ensure_entity(self, name: str) -> Optional[int]:
        """Get or create entity, return entity_id."""
        row = self._conn.execute(
            "SELECT entity_id FROM entities WHERE name = ?", (name,)
        ).fetchone()
        if row:
            return row[0]
        try:
            cur = self._conn.execute(
                "INSERT INTO entities (name, entity_type, created_at) VALUES (?, 'wiki', ?)",
                (name, datetime.datetime.now().isoformat()),
            )
            return cur.lastrowid
        except Exception:
            return None

    def _update_topic_page(self, topic: str, date: str, title: str, quality: int) -> None:
        """Create or update a topic aggregation page."""
        existing = self._conn.execute(
            "SELECT page_id, full_content FROM wiki_pages WHERE slug = ? AND page_type = 'topic'",
            (topic,),
        ).fetchone()

        if existing:
            content = existing["full_content"] or ""
            new_entry = f"- [{date}] {title} — quality: {quality}"
            if new_entry not in content:
                updated = content.replace(
                    f"## {_I18N_HEADERS['en']['timeline']}\n",
                    f"## {_I18N_HEADERS['en']['timeline']}\n{new_entry}\n",
                )
                self._conn.execute(
                    "UPDATE wiki_pages SET full_content = ?, updated_at = ? WHERE page_id = ?",
                    (updated, datetime.datetime.now().isoformat(), existing["page_id"]),
                )
        else:
            topic_content = (
                f"---\ntopic: {topic}\npage_count: 1\n"
                f"first_seen: {date}\nlast_updated: {date}\n---\n\n"
                f"# {topic}\n\n## {_I18N_HEADERS['en']['overview']}\n\n## {_I18N_HEADERS['en']['timeline']}\n"
                f"- [{date}] {title} — quality: {quality}\n"
            )
            self._conn.execute(
                """INSERT INTO wiki_pages (page_type, slug, title, date, topics, summary, full_content)
                   VALUES ('topic', ?, ?, ?, ?, ?, ?)""",
                (topic, topic, date, json.dumps([topic], ensure_ascii=False), f"Topic: {topic}", topic_content),
            )

    def _extract_facts(self, analysis: dict, slug: str, entities: list) -> None:
        """Extract stable knowledge from analysis into facts table."""
        facts_to_add = []

        # Background → general fact
        bg = analysis.get("background", "")
        if bg and len(bg) > 20:
            facts_to_add.append((bg[:80], "general"))

        # Result → reference fact
        result = analysis.get("result", "")
        if result and len(result) > 10:
            facts_to_add.append((result[:80], "reference"))

        # Decisions → reference facts
        for dec in analysis.get("decisions", [])[:2]:
            if len(dec) > 10:
                facts_to_add.append((dec[:80], "reference"))

        for content, category in facts_to_add:
            # Deduplicate
            existing = self._conn.execute(
                "SELECT fact_id FROM facts WHERE content = ?", (content,)
            ).fetchone()
            if existing:
                continue
            try:
                now = datetime.datetime.now().isoformat()
                cur = self._conn.execute(
                    """INSERT INTO facts (content, category, tags, trust_score, retrieval_count, helpful_count, created_at, updated_at)
                       VALUES (?, ?, ?, ?, 0, 0, ?, ?)""",
                    (content, category, f"wiki:{slug}", 0.5, now, now),
                )
                fact_id = cur.lastrowid
                # Link to first entity
                if entities:
                    entity_id = self._ensure_entity(entities[0])
                    if entity_id:
                        self._conn.execute(
                            "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                            (fact_id, entity_id),
                        )
            except Exception:
                pass

    def _extract_date(self, session_id: str) -> str:
        """Extract YYYY-MM-DD from session_id like 20260718_000446_83fbd4."""
        match = re.match(r"(\d{4})(\d{2})(\d{2})", session_id)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        return datetime.date.today().isoformat()

    def _build_slug(self, date: str, title: str) -> str:
        """Build a filesystem-safe slug from date and title."""
        # Clean title
        slug = re.sub(r"[^\w\s-]", "", title.lower())
        slug = re.sub(r"[\s_]+", "-", slug).strip("-")
        slug = slug[:40]
        if not slug:
            slug = "session"
        return f"{date}_{slug}"

    def _build_wiki_page(self, **kwargs) -> str:
        """Build full markdown content for a wiki page."""
        lang = kwargs.get("language", "en")
        h = _I18N_HEADERS.get(lang, _I18N_HEADERS["en"])

        lines = [
            "---",
            f'session_id: "{kwargs["session_id"]}"',
            f'date: {kwargs["date"]}',
            f'language: {lang}',
            f'quality: {kwargs["quality"]}',
            f'content_type: {kwargs["content_type"]}',
            f'topics: {json.dumps(kwargs["topics"], ensure_ascii=False)}',
            f'entities: {json.dumps(kwargs["entities"], ensure_ascii=False)}',
            f'keywords: {json.dumps(kwargs["keywords"], ensure_ascii=False)}',
            f'result: "{kwargs["result"]}"',
            "---",
            "",
            f"# {kwargs['title']} ({kwargs['date']})",
            "",
        ]

        if kwargs.get("background"):
            lines += [f"## {h['bg']}", kwargs["background"], ""]

        if kwargs.get("decisions"):
            lines += [f"## {h['dec']}"]
            for d in kwargs["decisions"]:
                lines.append(f"- {d}")
            lines.append("")

        if kwargs.get("problems"):
            lines += [f"## {h['prob']}"]
            for p in kwargs["problems"]:
                lines.append(f"- {p}")
            lines.append("")

        if kwargs.get("result"):
            lines += [f"## {h['res']}", kwargs["result"], ""]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Maintenance: capacity gate, classification audit, index, export
    # ------------------------------------------------------------------

    def run_maintenance(self) -> dict:
        """Run all maintenance tasks. Called from background thread."""
        report = {}
        report["capacity"] = self._capacity_gate()
        report["audit"] = self._classification_audit()
        report["index"] = self._rebuild_index()
        report["export"] = self._export_md_files()
        return report

    def _capacity_gate(self) -> dict:
        """Check and compress MEMORY.md / USER.md if over threshold."""
        result = {"memory_size": 0, "user_size": 0, "compressed": False}
        try:
            from pathlib import Path
            from hermes_constants import get_hermes_home
            mem_dir = get_hermes_home() / "memories"

            for fname, limit in [("MEMORY.md", 1760), ("USER.md", 1100)]:
                fpath = mem_dir / fname
                if not fpath.exists():
                    continue
                content = fpath.read_text(encoding="utf-8")
                size = len(content)
                key = "memory_size" if "MEMORY" in fname else "user_size"
                result[key] = size

                if size > limit:
                    lines = content.split("\n")
                    keep = max(10, int(len(lines) * 0.8))
                    compressed = "\n".join(lines[:keep])
                    fpath.write_text(compressed, encoding="utf-8")
                    result["compressed"] = True
                    logger.info("Compressed %s: %d -> %d chars", fname, size, len(compressed))
        except Exception as e:
            logger.warning("Capacity gate failed: %s", e)
        return result

    def _classification_audit(self) -> dict:
        """Periodic audit: merge duplicate topics, handle orphans. Runs every 4 calls."""
        result = {"merged": 0, "split": 0, "orphans": 0, "skipped": False}
        try:
            counter_path = self._get_wiki_path("audit_counter.txt")
            counter = 0
            if counter_path and counter_path.exists():
                try:
                    counter = int(counter_path.read_text().strip())
                except ValueError:
                    counter = 0

            if counter < 3:
                if counter_path:
                    counter_path.parent.mkdir(parents=True, exist_ok=True)
                    counter_path.write_text(str(counter + 1))
                result["skipped"] = True
                return result

            if counter_path:
                counter_path.write_text("0")

            with self._lock:
                topics = self._conn.execute(
                    "SELECT slug, title FROM wiki_pages WHERE page_type='topic'"
                ).fetchall()

                seen = {}
                for row in topics:
                    d = dict(row)
                    title_key = (d["title"] or "").lower().strip()
                    if title_key in seen:
                        self._merge_topics(seen[title_key], d["slug"])
                        result["merged"] += 1
                    else:
                        seen[title_key] = d["slug"]

                orphans = self._conn.execute("""
                    SELECT page_id FROM wiki_pages
                    WHERE page_type='session'
                    AND (topics IS NULL OR topics = '[]')
                    AND NOT EXISTS (
                        SELECT 1 FROM wiki_page_entities WHERE wiki_page_entities.page_id = wiki_pages.page_id
                    )
                """).fetchall()
                result["orphans"] = len(orphans)
                self._conn.commit()
        except Exception as e:
            logger.warning("Classification audit failed: %s", e)
        return result

    def _merge_topics(self, keep_slug: str, remove_slug: str) -> None:
        """Merge remove_slug into keep_slug."""
        sessions = self._conn.execute(
            "SELECT page_id, topics FROM wiki_pages WHERE page_type='session'"
        ).fetchall()
        for row in sessions:
            d = dict(row)
            try:
                topics = json.loads(d["topics"]) if d["topics"] else []
            except (json.JSONDecodeError, TypeError):
                continue
            if remove_slug in topics:
                topics = [keep_slug if t == remove_slug else t for t in topics]
                self._conn.execute(
                    "UPDATE wiki_pages SET topics=? WHERE page_id=?",
                    (json.dumps(topics), d["page_id"]),
                )
        self._conn.execute(
            "DELETE FROM wiki_pages WHERE slug=? AND page_type='topic'", (remove_slug,)
        )

    def _rebuild_index(self) -> dict:
        """Rebuild INDEX.md from SQLite."""
        result = {"sessions": 0, "topics": 0, "entities": 0, "written": False}
        try:
            wiki_path = self._get_wiki_path("INDEX.md")
            if not wiki_path:
                return result

            with self._lock:
                pages = self._conn.execute(
                    "SELECT slug, title, date, quality, page_type, topics FROM wiki_pages ORDER BY date DESC"
                ).fetchall()

            by_topic = {}
            timeline = []
            for row in pages:
                d = dict(row)
                pt = d["page_type"]
                if pt == "session":
                    result["sessions"] += 1
                    timeline.append(d)
                    try:
                        topics = json.loads(d["topics"]) if d["topics"] else []
                    except (json.JSONDecodeError, TypeError):
                        topics = []
                    for t in topics:
                        by_topic.setdefault(t, []).append(d)
                elif pt == "topic":
                    result["topics"] += 1
                elif pt == "entity":
                    result["entities"] += 1

            lines = [
                "# Memory Wiki Index", "",
                f"Last updated: {datetime.date.today().isoformat()}", "",
                "## Stats",
                f"- Sessions: {result['sessions']}",
                f"- Topics: {result['topics']}",
                f"- Entities: {result['entities']}", "",
                "## By Topic",
            ]
            for topic, slist in sorted(by_topic.items(), key=lambda x: -len(x[1])):
                lines.append(f"### {topic} ({len(slist)} pages)")
                for s in slist[:5]:
                    q = s.get("quality", "?")
                    lines.append(f"- [{s.get('date', '?')}] {s.get('title', '?')} - quality: {q}")
                lines.append("")

            lines.append("## Timeline")
            for s in timeline[:20]:
                q = s.get("quality", "?")
                lines.append(f"- [{s.get('date', '?')}] {s.get('title', '?')} - quality: {q}")
            lines.append("")

            wiki_path.parent.mkdir(parents=True, exist_ok=True)
            wiki_path.write_text("\n".join(lines), encoding="utf-8")
            result["written"] = True
        except Exception as e:
            logger.warning("INDEX.md rebuild failed: %s", e)
        return result

    def _export_md_files(self) -> dict:
        """Export wiki_pages to .md files. Clean stale files."""
        result = {"exported": 0, "cleaned": 0}
        try:
            base = self._get_wiki_path("")
            if not base:
                return result

            with self._lock:
                pages = self._conn.execute(
                    "SELECT slug, page_type, full_content FROM wiki_pages"
                ).fetchall()

            exported = set()
            subdir_map = {"session": "sessions", "topic": "topics", "entity": "entities"}
            for row in pages:
                d = dict(row)
                subdir = subdir_map.get(d["page_type"])
                if not subdir or not d["full_content"]:
                    continue
                fpath = base / subdir / f"{d['slug']}.md"
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(d["full_content"], encoding="utf-8")
                exported.add(str(fpath))
                result["exported"] += 1

            for subdir in ("sessions", "topics", "entities"):
                dir_path = base / subdir
                if not dir_path.exists():
                    continue
                for f in dir_path.iterdir():
                    if f.suffix == ".md" and str(f) not in exported:
                        f.unlink()
                        result["cleaned"] += 1
        except Exception as e:
            logger.warning("Export failed: %s", e)
        return result

    def _get_wiki_path(self, filename: str):
        """Get path to wiki directory file."""
        try:
            from pathlib import Path
            from hermes_constants import get_hermes_home
            wiki_dir = get_hermes_home() / "wiki"
            return wiki_dir / filename if filename else wiki_dir
        except Exception:
            return None
