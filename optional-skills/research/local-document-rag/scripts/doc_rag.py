#!/usr/bin/env python3
"""doc_rag.py - per-project, offline RAG over local documents.

One SQLite file per project (sqlite-vec + FTS5). Indexes PDF (text layer, with
optional Tesseract OCR for scanned pages), DOCX, Markdown and plain text;
chunks per page so every hit carries a source file + page citation.
Retrieval is hybrid: vector cosine + FTS5 keywords, fused with Reciprocal
Rank Fusion (K=60). Prefix matching approximates stemming for inflected
languages (e.g. Polish, Czech, German).

Usage:
  python doc_rag.py ingest FILE... --db ./rag.db [--copy] [--project NAME]
  python doc_rag.py query "question" --db ./rag.db [-k 6] [--json]
  python doc_rag.py list  --db ./rag.db
  python doc_rag.py stats --db ./rag.db
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import struct
import sys
import time
from pathlib import Path

MODEL = os.environ.get("RAG_EMBEDDING_MODEL",
                       "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
DIM = int(os.environ.get("RAG_EMBEDDING_DIM", "384"))
CHUNK_CHARS = 1100      # ~280-350 tokens
CHUNK_OVERLAP = 200
OCR_LANG = os.environ.get("RAG_OCR_LANG", "eng")
OCR_MIN_CHARS = 20      # a page with less extractable text is treated as a scan
RRF_K = 60
# Opt-in JSONL log of ingest/query calls (which trigger invoked RAG). Off unless set.
USAGE_LOG = os.environ.get("RAG_USAGE_LOG", "")

_STOPWORDS = {
    "the", "and", "for", "with", "what", "which", "how", "are", "was", "that", "this",
    "jaki", "jaka", "jakie", "jest", "czy", "dla", "oraz", "jak", "ile", "musi", "przez",
    "przy", "który", "która", "które", "tego", "der", "die", "das", "und",
}

_embedder = None


def embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(model_name=MODEL)
    return _embedder


def embed(texts: list[str]) -> list[list[float]]:
    return [list(map(float, v)) for v in embedder().embed(texts)]


def pack(vec) -> bytes:
    return struct.pack("%sf" % DIM, *vec)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


# ---------- parsing ----------
def _tessdata() -> str:
    """First candidate dir that actually holds the OCR language model."""
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    for d in (os.environ.get("RAG_TESSDATA", ""), os.environ.get("TESSDATA_PREFIX", ""),
              str(Path(home) / "tessdata")):
        if d and (Path(d) / f"{OCR_LANG}.traineddata").is_file():
            return d
    return os.environ.get("TESSDATA_PREFIX", "")


def _ocr_page(page) -> str:
    try:
        tessdata = _tessdata()
        if tessdata:
            os.environ["TESSDATA_PREFIX"] = tessdata
        tp = page.get_textpage_ocr(language=OCR_LANG, dpi=200, full=True)
        return page.get_text(textpage=tp).strip()
    except Exception as exc:  # Tesseract or its language model missing
        print(f"    [OCR unavailable] {exc}", file=sys.stderr)
        return ""


def pdf_pages(path: Path) -> list[tuple[int, str]]:
    import pymupdf
    pages = []
    with pymupdf.open(str(path)) as doc:
        for i, page in enumerate(doc):
            txt = (page.get_text() or "").strip()
            if len(txt) < OCR_MIN_CHARS:
                ocr = _ocr_page(page)
                if ocr:
                    print(f"    [OCR] page {i + 1}: {len(ocr)} chars")
                    txt = ocr
            pages.append((i + 1, txt))
    return pages


def docx_pages(path: Path) -> list[tuple[int, str]]:
    from docx import Document
    return [(1, "\n".join(p.text for p in Document(str(path)).paragraphs))]


def text_pages(path: Path) -> list[tuple[int, str]]:
    return [(1, path.read_text(encoding="utf-8", errors="replace"))]


def load_pages(path: Path) -> list[tuple[int, str]]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return pdf_pages(path)
    if ext == ".docx":
        return docx_pages(path)
    if ext in (".md", ".markdown", ".txt"):
        return text_pages(path)
    raise ValueError(f"unsupported file type: {ext}")


def chunk_page(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


# ---------- storage ----------
def connect(db: str) -> sqlite3.Connection:
    import sqlite_vec
    con = sqlite3.connect(db)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(f"""
        CREATE TABLE IF NOT EXISTS docs(
            id INTEGER PRIMARY KEY, path TEXT, name TEXT, sha TEXT UNIQUE,
            project TEXT, pages INT, chunks INT, added_at TEXT);
        CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY, doc_id INT, page INT, ord INT, text TEXT);
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            embedding float[{DIM}] distance_metric=cosine);
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
            text, content='chunks', content_rowid='id');
    """)
    return con


def log_usage(cmd: str, via: str | None, db: str, extra: dict | None = None) -> None:
    if not USAGE_LOG:
        return
    try:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "cmd": cmd,
               "via": via or "unknown", "db": str(Path(db).resolve()), **(extra or {})}
        with open(USAGE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass  # telemetry must never break the real operation


# ---------- commands ----------
def cmd_ingest(a) -> int:
    log_usage("ingest", a.via, a.db, {"files": [Path(x).name for x in a.files]})
    con = connect(a.db)
    assets = Path(a.db).resolve().parent / "assets"
    failed = 0
    for f in map(Path, a.files):
        if not f.is_file():
            print(f"[SKIP] missing file: {f}")
            failed += 1
            continue
        digest = sha256(f)
        if con.execute("SELECT 1 FROM docs WHERE sha=?", (digest,)).fetchone():
            print(f"[SKIP] already indexed: {f.name}")
            continue
        try:
            pages = load_pages(f)
        except ValueError as exc:
            print(f"[SKIP] {f.name}: {exc}")
            failed += 1
            continue
        rows = [(pno, j, ch) for pno, ptext in pages for j, ch in enumerate(chunk_page(ptext))]
        if not rows:
            print(f"[SKIP] no text (scan without OCR?): {f.name}")
            failed += 1
            continue
        src = f
        if a.copy:
            assets.mkdir(parents=True, exist_ok=True)
            src = assets / f.name
            if not src.exists():
                shutil.copy2(f, src)
        vecs = embed([r[2] for r in rows])  # embed before any write: never a half-indexed doc
        cur = con.execute(
            "INSERT INTO docs(path,name,sha,project,pages,chunks,added_at) VALUES(?,?,?,?,?,?,?)",
            (str(src.resolve()), f.name, digest, a.project, len(pages), len(rows),
             time.strftime("%Y-%m-%dT%H:%M:%S")))
        doc_id = cur.lastrowid
        for (pno, j, ch), v in zip(rows, vecs):
            c = con.execute("INSERT INTO chunks(doc_id,page,ord,text) VALUES(?,?,?,?)",
                            (doc_id, pno, j, ch))
            con.execute("INSERT INTO vec_chunks(rowid,embedding) VALUES(?,?)", (c.lastrowid, pack(v)))
            con.execute("INSERT INTO fts_chunks(rowid,text) VALUES(?,?)", (c.lastrowid, ch))
        con.commit()
        print(f"[OK] {f.name}: {len(pages)} pages, {len(rows)} chunks" + (f" -> {src}" if a.copy else ""))
    con.close()
    return 1 if failed else 0


def fts_query(text: str) -> str:
    """Content tokens as an FTS5 OR query; long tokens become prefix terms (cheap stemming)."""
    terms = []
    for t in re.findall(r"\w+", text.lower(), flags=re.UNICODE):
        if len(t) <= 2 or t in _STOPWORDS:
            continue
        if len(t) <= 4:
            terms.append(f'"{t}"')
        else:
            terms.append(f'"{t[:max(4, int(len(t) * 0.75))]}"*')
    return " OR ".join(terms)


def rrf(*rankings: list[int], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion of ranked id lists, best first."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(score.items(), key=lambda x: -x[1])


def cmd_query(a) -> int:
    log_usage("query", a.via, a.db, {"text": a.text[:120]})
    con = connect(a.db)
    pool = max(a.k * 4, 20)
    vec = [r[0] for r in con.execute(
        "SELECT rowid FROM vec_chunks WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (pack(embed([a.text])[0]), pool))]
    fts: list[int] = []
    q = fts_query(a.text)
    if q:
        try:
            fts = [r[0] for r in con.execute(
                "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH ? ORDER BY rank LIMIT ?",
                (q, pool))]
        except sqlite3.OperationalError:
            fts = []
    top = rrf(vec, fts)[:a.k]
    if not top:
        con.close()
        print("(no hits)")
        return 0
    ids = [cid for cid, _ in top]
    meta = {row[0]: row for row in con.execute(
        "SELECT c.id, c.text, d.name, c.page, d.path FROM chunks c JOIN docs d ON d.id = c.doc_id "
        "WHERE c.id IN (%s)" % ",".join("?" * len(ids)), ids)}
    con.close()
    results = [(meta[cid], sc) for cid, sc in top if cid in meta]
    if a.json:
        print(json.dumps([{"text": m[1], "source": m[2], "page": m[3], "path": m[4],
                           "score": round(sc, 4)} for m, sc in results], ensure_ascii=False, indent=2))
        return 0
    for i, (m, sc) in enumerate(results, 1):
        print(f"\n[{i}] {m[2]} p.{m[3]} (RRF {sc:.4f})\n    source: {m[4]}")
        print("    " + m[1].strip()[:600])
    return 0


def cmd_list(a) -> int:
    con = connect(a.db)
    for r in con.execute("SELECT name, project, pages, chunks, added_at FROM docs ORDER BY id"):
        print(f"{r[0]:45} project={r[1] or '-':15} pages={r[2]:>4} chunks={r[3]:>4} {r[4]}")
    con.close()
    return 0


def cmd_stats(a) -> int:
    con = connect(a.db)
    docs = con.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    con.close()
    print(f"DB: {a.db}\nModel: {MODEL} ({DIM}-dim)\nDocuments: {docs}\nChunks: {chunks}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Per-project offline document RAG (sqlite-vec + fastembed)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest")
    p.add_argument("files", nargs="+")
    p.add_argument("--db", required=True)
    p.add_argument("--copy", action="store_true", help="copy sources into <db dir>/assets/")
    p.add_argument("--project")
    p.add_argument("--via", default="manual", help="trigger label for the opt-in usage log")
    p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("query")
    p.add_argument("text")
    p.add_argument("--db", required=True)
    p.add_argument("-k", type=int, default=6)
    p.add_argument("--json", action="store_true")
    p.add_argument("--via", default="manual")
    p.set_defaults(fn=cmd_query)
    for name, fn in (("list", cmd_list), ("stats", cmd_stats)):
        p = sub.add_parser(name)
        p.add_argument("--db", required=True)
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
