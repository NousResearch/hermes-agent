#!/usr/bin/env python3
"""
ses_bridge.py — SES ↔ agent-framework bridge.

  verify <snapshot.ses.json>      verify integrity (canonical SHA-256)
  export-skills [learned_dir]     export lessons (CORE/lesson) as Hermes skill stubs
  import-ocean <ocean_db.sqlite>  import nodes from an original Ocean DB (read-only)

Pure stdlib. The graph store comes from QCA_STORE (see qca_engine.py).
"""

from __future__ import annotations
import json, os, re, sys, sqlite3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hermes_home import get_hermes_home
from qca_engine import Graph, canonical_json, _now
import hashlib

LEARNED_DIR = os.path.expanduser(os.getenv("QCA_LEARNED_DIR", "")) or \
              str(get_hermes_home() / "skills" / "learned")


def verify(path: str) -> bool:
    """Verify a snapshot. Supports canonical SES v5.1 (meta.hash over the whole
    snapshot, SES_CANON_JSON_v1 sorting) and the legacy body/provenance wrapper."""
    doc = json.load(open(path))
    canon_ok = True
    if "initiator" in doc:  # canonical v5.1
        from qca_engine import _snapshot_hash
        claimed = doc.get("meta", {}).get("hash", "")
        actual = _snapshot_hash(doc)
        # v5.1 canon lock (§12): a state must reference its kernel.
        # This is an integrity invariant, not a warning — it fails the check.
        if doc.get("snapshot_type") == "STATE_SNAPSHOT" and not (
                doc["meta"].get("kernel_ref") or doc["meta"].get("kernel_hash")):
            print("❌ canon violation: STATE_SNAPSHOT without kernel_ref/kernel_hash (SES v5.1 §12)")
            canon_ok = False
    else:  # legacy
        claimed = doc.get("provenance", {}).get("hash", "")
        actual = "sha256:" + hashlib.sha256(canonical_json(doc["body"]).encode()).hexdigest()
    hash_ok = claimed == actual
    if not hash_ok:
        print("❌ TAMPERED: hash mismatch")
    elif canon_ok:
        print("✅ integrity verified")
    print(f"  claimed: {claimed}\n  actual:  {actual}")
    return hash_ok and canon_ok


def export_skills(learned_dir: str = LEARNED_DIR):
    g = Graph()
    lessons = [n for n in g.nodes
               if n.get("layer") == "CORE" and n["meta"].get("kind") == "lesson"]
    if not lessons:
        print("no lessons (CORE/lesson) in the graph"); return
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
  source: qca-cycle
  confidence: {conf}
  node_id: {n['id']}
  exported: {_now()}
---

# QCA lesson: {slug}

{n['_text']}

> Auto-exported from the QCA graph (lesson extraction). Confidence: {conf}.
> The Curator may grow this stub into a full skill.
""")
        print(f"→ {d}/SKILL.md (confidence={conf})")
    print(f"lessons exported: {len(lessons)}")


def _clean_ocean_text(text: str) -> str:
    """Strip Ocean's service wrapper ('[ОПЫТ|...] Контекст: X → Действие: Y') —
    it dilutes the embedding and drops recall below the 0.70 threshold.
    (The wrapper is Russian because original Ocean memories are; keep as is.)"""
    m = re.match(r"^\[ОПЫТ\|[^\]]*\]\s*Контекст:\s*(.*?)\s*→\s*Действие:\s*(.*)$",
                 text, flags=re.DOTALL)
    if m:
        ctx, action = m.group(1).strip(), m.group(2).strip()
        return f"{ctx}: {action}" if ctx and ctx.lower() not in ("other", "greeting") else action
    return text


def import_ocean(db_path: str):
    """Import active nodes from an original Ocean SQLite DB. The source is opened
    read-only and never modified."""
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
            # the text may live inside a JSON field
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
    print(f"imported: {added}, skipped: {skipped}, total in graph: {len(g.nodes)}")


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
        print(f"unknown command: {cmd}\n{__doc__}")


if __name__ == "__main__":
    main()
