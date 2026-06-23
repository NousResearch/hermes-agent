"""Split Hermes memories into an Obsidian-compatible LLM Wiki.

This plugin intentionally exports *sanitized summaries* instead of raw session logs
or secret-bearing memory databases.  Raw source snapshots are redacted and kept in
``raw/memory/``; wiki pages under ``concepts/`` hold the usable semantic graph.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover - direct ad-hoc import fallback
    def get_hermes_home() -> Path:  # type: ignore
        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))

ENTRY_DELIMITER = "\n§\n"
DEFAULT_SUBDIR = "Hermes-Memory-Wiki"

SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password|passwd|authorization|bearer|cookie|session)"
    r"\s*[:=]\s*[^\s`\"']{6,}"
)
LONG_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_\-]{32,}\b")
EMAIL_RE = re.compile(r"\b[\w.%-]+@[\w.-]+\.[A-Za-z]{2,}\b")
WIN_HOME_RE = re.compile(r"C:\\Users\\[^\\\s`]+", re.I)
MSYS_HOME_RE = re.compile(r"C:/Users/[^\s`]+", re.I)
IP_RE = re.compile(r"\b(?:10|127|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b")

CATEGORIES: list[tuple[str, str, list[str]]] = [
    ("user-profile", "ユーザー像と好み", ["User", "ユーザー", "prefers", "ボブ", "nickname", "呼称", "ギャル", "English X"]),
    ("identity-persona", "はくあ/同一性", ["はくあ", "Hakua", "Persona", "Identity", "同一性", "中枢", "外殻", "声帯", "VRChat Harness"]),
    ("communication-channels", "通信経路とチャンネル", ["LINE", "Discord", "Telegram", "gateway", "line-personal", "replyToMessageId", "#hermesagent"]),
    ("memory-operations", "記憶運用", ["memory", "Ebbinghaus", "Obsidian", "睡眠", "LLM Wiki", "vault", "記憶", "セッションログ"]),
    ("hermes-environment", "Hermes環境とルーティング", ["Hermes", "plugin", "config", "GPT", "fallback", "llama-server", "tool", "provider", "QuestFrame"]),
    ("workflows-procedures", "手順とワークフロー", ["skill", "workflow", "cron", "OSINT", "commit", "push", "Calendar", "Gmail", "appointment"]),
    ("safety-boundaries", "安全境界と公開制御", ["approval", "push", "secret", "PII", "公開", "remote", "credential", "Bitwarden", "安全"]),
]

STATUS_SCHEMA = {
    "name": "memory_llm_wiki_status",
    "description": "Inspect Hermes memory sources and the target LLM Wiki path without writing files.",
    "parameters": {
        "type": "object",
        "properties": {
            "wiki_root": {"type": "string", "description": "Optional explicit wiki root path."},
        },
    },
}

EXPORT_SCHEMA = {
    "name": "memory_llm_wiki_export",
    "description": "Export sanitized Hermes curated/Ebbinghaus memories as split Obsidian-compatible LLM Wiki pages.",
    "parameters": {
        "type": "object",
        "properties": {
            "wiki_root": {"type": "string", "description": "Optional explicit wiki root. Defaults to Obsidian vault/Hermes-Memory-Wiki when discoverable."},
            "include_curated": {"type": "boolean", "description": "Include MEMORY.md and USER.md entries. Default true."},
            "include_ebbinghaus": {"type": "boolean", "description": "Include high-salience Ebbinghaus memories. Default true."},
            "max_ebbinghaus": {"type": "integer", "minimum": 1, "maximum": 500, "description": "Max Ebbinghaus rows to export. Default 80."},
            "max_entries_per_page": {"type": "integer", "minimum": 10, "maximum": 200, "description": "Split pages after this many entries. Default 80."},
            "dry_run": {"type": "boolean", "description": "Plan the export without writing. Default false."},
        },
    },
}

@dataclass
class MemoryItem:
    source: str
    category: str
    title: str
    content: str
    salience: float | None = None
    retention: float | None = None
    tags: str = ""


def check_available() -> bool:
    return True


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _today() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d")


def _now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def sanitize(text: str) -> str:
    text = str(text or "")
    text = SECRET_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
    text = EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    text = WIN_HOME_RE.sub("~", text)
    text = MSYS_HOME_RE.sub("~", text)
    text = IP_RE.sub("[PRIVATE_IP_REDACTED]", text)
    text = LONG_TOKEN_RE.sub("[LONG_TOKEN_REDACTED]", text)
    return text.strip()


def _slug(text: str) -> str:
    text = text.lower().replace("/", "-").replace("_", "-")
    text = re.sub(r"[^a-z0-9\-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "memory"


def _path_from_env(name: str) -> Path | None:
    val = os.environ.get(name)
    if not val:
        return None
    p = Path(val).expanduser()
    return p if p.exists() else None


def _obsidian_vault() -> Path | None:
    for env_name in ("WIKI_PATH", "OBSIDIAN_VAULT_PATH"):
        p = _path_from_env(env_name)
        if p:
            return p
    reg = Path.home() / "AppData" / "Roaming" / "obsidian" / "obsidian.json"
    if reg.exists():
        try:
            data = json.loads(reg.read_text(encoding="utf-8"))
            vaults = data.get("vaults", {}) if isinstance(data, dict) else {}
            candidates = []
            for vault in vaults.values():
                path = Path(vault.get("path", "")) if isinstance(vault, dict) else None
                if path and path.exists():
                    score = int((path / ".git").exists()) + int((path / "Hermes-Sessions").exists())
                    candidates.append((score, path))
            if candidates:
                return sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
        except Exception:
            pass
    fallback = Path.home() / "Documents" / "ObsidianVault"
    return fallback if fallback.exists() else None


def resolve_wiki_root(args: dict[str, Any] | None = None) -> Path:
    args = args or {}
    if args.get("wiki_root"):
        return Path(str(args["wiki_root"])).expanduser()
    base = _obsidian_vault()
    if base:
        return base / DEFAULT_SUBDIR
    return Path(os.environ.get("WIKI_PATH", str(Path.home() / "wiki"))).expanduser()


def _read_entries(path: Path) -> list[str]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return []
    return [sanitize(x) for x in raw.split(ENTRY_DELIMITER) if x.strip()]


def _classify(content: str) -> str:
    low = content.lower()
    for slug, _label, words in CATEGORIES:
        if any(w.lower() in low for w in words):
            return slug
    return "misc"


def _category_label(slug: str) -> str:
    for s, label, _ in CATEGORIES:
        if s == slug:
            return label
    return "その他"


def read_curated_items() -> list[MemoryItem]:
    mem_dir = get_hermes_home() / "memories"
    items: list[MemoryItem] = []
    for store, filename in (("memory", "MEMORY.md"), ("user", "USER.md")):
        for i, entry in enumerate(_read_entries(mem_dir / filename), start=1):
            items.append(MemoryItem(store, _classify(entry), f"{store}-{i}", entry))
    return items


def _retention_from_row(row: sqlite3.Row, schema: str) -> float | None:
    now = time.time()
    try:
        if schema == "new":
            anchor = row["last_rehearsed_at"] or row["last_retrieved_at"] or row["updated_at"] or row["created_at"] or now
            elapsed_days = max(0.0, (now - float(anchor)) / 86400.0)
            stability = max(0.25, float(row["strength"] or 1.0) * (1.0 + 0.25 * float(row["rehearsal_count"] or 0) + 0.10 * float(row["retrieval_count"] or 0)))
            return max(0.0, min(1.0, math.exp(-elapsed_days / stability)))
        anchor = row["last_rehearsed_at"] or row["created_at"] or now
        elapsed_days = max(0.0, (now - float(anchor)) / 86400.0)
        decay_rate = max(0.001, float(row["decay_rate"] or 0.1))
        return max(0.0, min(1.0, math.exp(-decay_rate * elapsed_days)))
    except Exception:
        return None


def read_ebbinghaus_items(limit: int = 80) -> list[MemoryItem]:
    dbs = [get_hermes_home() / "ebbinghaus_memory.db", get_hermes_home() / "memories" / "ebbinghaus.db"]
    out: list[MemoryItem] = []
    for db in dbs:
        if not db.exists() or db.stat().st_size == 0:
            continue
        con = sqlite3.connect(db)
        con.row_factory = sqlite3.Row
        try:
            cols = {r[1] for r in con.execute("pragma table_info(memories)")}
            if "memory_id" in cols:
                rows = con.execute(
                    "select * from memories order by salience desc, retrieval_count desc, updated_at desc limit ?",
                    (limit,),
                ).fetchall()
                for row in rows:
                    content = sanitize(row["content"] or "")
                    out.append(MemoryItem("ebbinghaus", _classify(content), f"ebbinghaus-{row['memory_id']}", content, row["salience"], _retention_from_row(row, "new"), sanitize(row["tags"] or "")))
            elif "id" in cols:
                rows = con.execute("select * from memories order by importance desc, rehearsal_count desc, created_at desc limit ?", (limit,)).fetchall()
                for row in rows:
                    content = sanitize(row["content"] or "")
                    out.append(MemoryItem("ebbinghaus-legacy", _classify(content), f"ebbinghaus-legacy-{row['id']}", content, row["importance"], _retention_from_row(row, "legacy"), sanitize(row["tags"] or "")))
        finally:
            con.close()
    out.sort(key=lambda x: ((x.salience or 0), (x.retention or 0)), reverse=True)
    return out[:limit]


def _schema_md() -> str:
    return f"""# Memory LLM Wiki Schema

## Domain
Hermes/Hakua の長期記憶を、LLM が読みやすい分割済み Markdown Wiki として保つ。

## Conventions
- Raw snapshots live under `raw/memory/` and are sanitized; do not store secrets or raw session logs.
- Agent-owned synthesized pages live under `concepts/`.
- Every concept page links back to [[index]] and at least one related memory concept when available.
- Use Japanese labels for Hakua/OpenClaw/VRChat memory layers.
- Update `index.md` and `log.md` on every export.

## Frontmatter
```yaml
title: Page Title
created: YYYY-MM-DD
updated: YYYY-MM-DD
type: concept | summary | raw-index
tags: [memory, hermes]
sources: [raw/memory/...]
confidence: medium
```

## Tag Taxonomy
- memory, hermes, hakua, user-profile, channel, workflow, safety, environment, ebbinghaus, obsidian

## Page Thresholds
- Split pages when a memory category exceeds the configured `max_entries_per_page`.
- Prefer summary pages over raw dumps; raw files are evidence only.

Last generated: {_now()}
"""


def _frontmatter(title: str, typ: str, tags: list[str], sources: list[str]) -> str:
    tag_s = ", ".join(tags)
    src_s = ", ".join(sources)
    today = _today()
    return f"---\ntitle: {title}\ncreated: {today}\nupdated: {today}\ntype: {typ}\ntags: [{tag_s}]\nsources: [{src_s}]\nconfidence: medium\n---\n\n"


def _concept_page(slug: str, label: str, items: list[MemoryItem], part: int | None, raw_sources: list[str]) -> str:
    title = label if part is None else f"{label} Part {part}"
    body = _frontmatter(title, "concept", ["memory", "hermes", slug], raw_sources)
    body += f"# {title}\n\n"
    body += "関連: [[index]] / [[memory-operations]] / [[identity-persona]]\n\n"
    body += "## 要約\n\n"
    body += f"- カテゴリ: **{label}**\n- 件数: **{len(items)}**\n- 内容はサニタイズ済み。生セッションログではなく、長期記憶の分割ビュー。\n\n"
    body += "## 記憶エントリ\n\n"
    for item in items:
        meta = []
        if item.salience is not None:
            meta.append(f"salience={item.salience:.2f}")
        if item.retention is not None:
            meta.append(f"retention={item.retention:.2f}")
        if item.tags:
            meta.append(f"tags={item.tags}")
        suffix = f" _({' / '.join(meta)})_" if meta else ""
        body += f"- **{item.source}**: {item.content}{suffix}\n"
    body += "\n## 更新方針\n\n- 新しい矛盾が見つかった場合は上書きせず、同じ箇条書き内に日付つきで併記する。\n- 手順化できるものは memory ではなく skill へ昇格する。\n"
    return body


def _write_export(wiki_root: Path, items: list[MemoryItem], max_entries_per_page: int, dry_run: bool) -> dict[str, Any]:
    today = _today()
    raw_dir = wiki_root / "raw" / "memory"
    concept_dir = wiki_root / "concepts"
    files: dict[str, str] = {}
    grouped: dict[str, list[MemoryItem]] = defaultdict(list)
    for item in items:
        grouped[item.category].append(item)

    raw_path = raw_dir / f"hermes-memory-sanitized-{today}.md"
    raw_body = _frontmatter("Hermes Memory Sanitized Snapshot", "raw-index", ["memory", "hermes"], [])
    raw_body += "# Hermes Memory Sanitized Snapshot\n\n"
    for item in items:
        raw_body += f"- [{item.source}] ({item.category}) {item.content}\n"
    files[str(raw_path)] = raw_body

    files[str(wiki_root / "SCHEMA.md")] = _schema_md()
    index_lines = [
        "# Memory LLM Wiki Index",
        "",
        f"> Last updated: {today} | Total concept groups: {len(grouped)}",
        "",
        "## Concepts",
        "",
    ]
    created_pages: list[str] = []
    for slug in sorted(grouped):
        label = _category_label(slug)
        chunks = [grouped[slug][i:i + max_entries_per_page] for i in range(0, len(grouped[slug]), max_entries_per_page)]
        for idx, chunk in enumerate(chunks, start=1):
            suffix = f"-part-{idx}" if len(chunks) > 1 else ""
            page_slug = f"{slug}{suffix}"
            page_path = concept_dir / f"{page_slug}.md"
            files[str(page_path)] = _concept_page(slug, label, chunk, idx if len(chunks) > 1 else None, [f"raw/memory/{raw_path.name}"])
            created_pages.append(str(page_path))
            index_lines.append(f"- [[{page_slug}]] — {label} ({len(chunk)} entries)")
    index_lines += ["", "## Raw", "", f"- [[raw/memory/{raw_path.stem}]] — Sanitized snapshot for {today}", ""]
    files[str(wiki_root / "index.md")] = "\n".join(index_lines)

    log_path = wiki_root / "log.md"
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else "# Memory LLM Wiki Log\n\n"
    log_entry = f"\n## [{today}] export | Hermes memory split\n- Items: {len(items)}\n- Groups: {len(grouped)}\n- Files: {len(files)}\n"
    files[str(log_path)] = existing.rstrip() + log_entry + "\n"

    if not dry_run:
        for path_s, content in files.items():
            path = Path(path_s)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    return {"wiki_root": str(wiki_root), "dry_run": dry_run, "items": len(items), "groups": {k: len(v) for k, v in sorted(grouped.items())}, "files": sorted(files), "concept_pages": created_pages}


def handle_status(args: dict[str, Any] | None = None, **_kwargs: Any) -> str:
    args = args or {}
    wiki_root = resolve_wiki_root(args)
    mem_dir = get_hermes_home() / "memories"
    curated = read_curated_items()
    ebb_count = 0
    for db in (get_hermes_home() / "ebbinghaus_memory.db", get_hermes_home() / "memories" / "ebbinghaus.db"):
        if db.exists() and db.stat().st_size:
            try:
                con = sqlite3.connect(db)
                ebb_count += int(con.execute("select count(*) from memories").fetchone()[0])
                con.close()
            except Exception:
                pass
    return _json({"success": True, "wiki_root": str(wiki_root), "wiki_exists": wiki_root.exists(), "memory_dir": str(mem_dir), "curated_entries": len(curated), "ebbinghaus_rows": ebb_count})


def handle_export(args: dict[str, Any] | None = None, **_kwargs: Any) -> str:
    args = args or {}
    include_curated = bool(args.get("include_curated", True))
    include_ebbinghaus = bool(args.get("include_ebbinghaus", True))
    max_ebbinghaus = int(args.get("max_ebbinghaus") or 80)
    max_entries_per_page = int(args.get("max_entries_per_page") or 80)
    dry_run = bool(args.get("dry_run", False))
    wiki_root = resolve_wiki_root(args)
    items: list[MemoryItem] = []
    if include_curated:
        items.extend(read_curated_items())
    if include_ebbinghaus:
        items.extend(read_ebbinghaus_items(max_ebbinghaus))
    if not items:
        return _json({"success": False, "error": "No memory items found", "wiki_root": str(wiki_root)})
    result = _write_export(wiki_root, items, max_entries_per_page, dry_run)
    result["success"] = True
    return _json(result)
