#!/usr/bin/env python3
"""
ses_bridge.py — мост SES ↔ Hermes.

  verify <snapshot.ses.json>      проверить целостность (canonical SHA-256)
  export-skills [learned_dir]     уроки (CORE/lesson) из графа → Hermes skill-стабы
  import-ocean <ocean_db.sqlite>  импорт узлов из настоящего Ocean (read-only)

Pure stdlib. Граф берётся из QCA_STORE (см. qca_engine.py).
"""

from __future__ import annotations
import json, os, re, sys, sqlite3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qca_engine import Graph, canonical_json, _now
import hashlib

LEARNED_DIR = os.path.expanduser(os.getenv("QCA_LEARNED_DIR", "~/.hermes/skills/learned"))


def verify(path: str) -> bool:
    doc = json.load(open(path))
    claimed = doc.get("provenance", {}).get("hash", "")
    actual = "sha256:" + hashlib.sha256(canonical_json(doc["body"]).encode()).hexdigest()
    ok = claimed == actual
    print(f"{'✅ целостность подтверждена' if ok else '❌ ТАМПЕР: хэш не совпадает'}")
    print(f"  claimed: {claimed}\n  actual:  {actual}")
    return ok


def export_skills(learned_dir: str = LEARNED_DIR):
    g = Graph()
    lessons = [n for n in g.nodes
               if n.get("layer") == "CORE" and n["meta"].get("kind") == "lesson"]
    if not lessons:
        print("уроков (CORE/lesson) в графе нет"); return
    os.makedirs(learned_dir, exist_ok=True)
    for n in lessons:
        slug = re.sub(r"[^a-z0-9а-яё]+", "-", n["_text"][:50].lower()).strip("-") or n["id"].lower()
        d = os.path.join(learned_dir, f"qca-lesson-{n['id'].lower()}")
        os.makedirs(d, exist_ok=True)
        conf = n["meta"].get("confidence", 0.5)
        with open(os.path.join(d, "SKILL.md"), "w") as f:
            f.write(f"""---
name: qca-lesson-{n['id'].lower()}
description: "{n['_text'][:140].replace('"', "'")}"
platforms: [linux, macos, windows]
metadata:
  source: ocean-qca-cycle
  confidence: {conf}
  node_id: {n['id']}
  exported: {_now()}
---

# Урок QCA: {slug}

{n['_text']}

> Авто-экспорт из графа QCA (H8-рефлексия). Confidence: {conf}.
> Curator может развить этот стаб в полноценный скилл.
""")
        print(f"→ {d}/SKILL.md (confidence={conf})")
    print(f"экспортировано уроков: {len(lessons)}")


def _clean_ocean_text(text: str) -> str:
    """Срезать служебную обёртку Ocean ('[ОПЫТ|...] Контекст: X → Действие: Y') —
    она разбавляет эмбеддинг и роняет recall ниже порога 0.70."""
    m = re.match(r"^\[ОПЫТ\|[^\]]*\]\s*Контекст:\s*(.*?)\s*→\s*Действие:\s*(.*)$",
                 text, flags=re.DOTALL)
    if m:
        ctx, action = m.group(1).strip(), m.group(2).strip()
        return f"{ctx}: {action}" if ctx and ctx.lower() not in ("other", "greeting") else action
    return text


def import_ocean(db_path: str):
    """Импорт активных узлов из SQLite настоящего Ocean. Источник не модифицируется."""
    src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    g = Graph()
    existing = {n["_text"] for n in g.nodes}
    rows = src.execute("SELECT * FROM nodes").fetchall()
    added = skipped = 0
    for r in rows:
        d = dict(r)
        text = d.get("_text") or d.get("text") or ""
        if not text.strip():
            # текст может лежать в JSON-поле
            for v in d.values():
                if isinstance(v, str) and v.startswith("{"):
                    try:
                        text = json.loads(v).get("_text", "")
                        if text: break
                    except ValueError: pass
        text = _clean_ocean_text(text)
        if not text.strip() or text in existing:
            skipped += 1; continue
        layer = d.get("layer") or "CONTEXT"
        status = d.get("status") or "active"
        if status == "archived":
            skipped += 1; continue
        g.add_node(text, layer=layer, role="system", meta={"imported_from": "ocean", "src_id": d.get("id")})
        existing.add(text); added += 1
    g.save()
    print(f"импортировано: {added}, пропущено: {skipped}, всего в графе: {len(g.nodes)}")


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "verify":
        sys.exit(0 if verify(args[0]) else 1)
    elif cmd == "export-skills":
        export_skills(args[0] if args else LEARNED_DIR)
    elif cmd == "import-ocean":
        import_ocean(args[0])
    else:
        print(f"неизвестная команда: {cmd}\n{__doc__}")


if __name__ == "__main__":
    main()
