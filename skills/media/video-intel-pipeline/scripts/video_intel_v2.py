#!/usr/bin/env python3
import argparse
import json
import math
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional


def jprint(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def fail(msg: str, code: int = 1):
    jprint({"success": False, "error": msg})
    raise SystemExit(code)


def check_core_deps():
    missing = []
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg")
    try:
        import yt_dlp  # noqa
    except Exception:
        missing.append("yt-dlp")
    try:
        import faster_whisper  # noqa
    except Exception:
        missing.append("faster-whisper")
    if missing:
        fail("Missing dependencies: " + ", ".join(missing))


def optional_semantic_deps_ok() -> bool:
    try:
        import numpy  # noqa
        import fastembed  # noqa
        return True
    except Exception:
        return False


def sanitize_name(name: str) -> str:
    bad = '<>:"/\\|?*\n\r\t'
    out = "".join("_" if c in bad else c for c in name).strip()
    return out[:180] or "video"


def run_ffmpeg_to_wav(src: Path, dst: Path):
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", "-vn", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        fail(f"ffmpeg conversion failed: {proc.stderr[-1500:]}")


def download_audio(url: str, outdir: Path):
    from yt_dlp import YoutubeDL

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": str(outdir / "%(title).160s [%(id)s].%(ext)s"),
        "noplaylist": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = ydl.prepare_filename(info)
        return info, Path(filepath)


def transcribe(wav_path: Path, model_size: str, task: str, language: Optional[str]):
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(wav_path),
        task=task,
        language=language,
        vad_filter=True,
        beam_size=5,
    )

    out = []
    text_parts = []
    for i, s in enumerate(segments):
        text = (s.text or "").strip()
        seg = {
            "id": i,
            "start": round(float(s.start), 3),
            "end": round(float(s.end), 3),
            "text": text,
        }
        out.append(seg)
        if text:
            text_parts.append(text)

    return {
        "full_text": "\n".join(text_parts).strip(),
        "segments": out,
        "info": {
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "duration": getattr(info, "duration", None),
            "duration_after_vad": getattr(info, "duration_after_vad", None),
        },
    }


def hhmmss(seconds: float) -> str:
    s = max(0, int(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def build_keyword_index_sqlite(workdir: Path, segments: List[Dict[str, Any]]):
    db = workdir / "search.db"
    if db.exists():
        db.unlink()
    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    cur.execute("CREATE VIRTUAL TABLE seg_fts USING fts5(seg_id UNINDEXED, text, tokenize='unicode61')")
    cur.execute("CREATE TABLE seg_meta(seg_id INTEGER PRIMARY KEY, start REAL, end REAL, text TEXT)")
    for s in segments:
        cur.execute("INSERT INTO seg_fts(seg_id, text) VALUES (?,?)", (s["id"], s["text"]))
        cur.execute("INSERT INTO seg_meta(seg_id, start, end, text) VALUES (?,?,?,?)", (s["id"], s["start"], s["end"], s["text"]))
    conn.commit()
    conn.close()
    return db


def build_semantic_embeddings(workdir: Path, segments: List[Dict[str, Any]], model_name: str):
    import numpy as np
    from fastembed import TextEmbedding

    texts = [s.get("text", "") for s in segments]
    embedder = TextEmbedding(model_name=model_name)
    vectors = list(embedder.embed(texts))
    emb = np.vstack([np.asarray(v, dtype=np.float32) for v in vectors])
    # l2 normalize for cosine via dot-product
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms

    emb_path = workdir / "embeddings.npy"
    np.save(str(emb_path), emb)
    meta = {
        "model": model_name,
        "segments_count": len(segments),
        "embeddings_file": str(emb_path),
        "engine": "fastembed",
    }
    (workdir / "semantic_index.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return emb_path, meta


def cmd_ingest(args):
    check_core_deps()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    info, downloaded_audio = download_audio(args.url, outdir)
    title = info.get("title") or "video"
    vid = info.get("id")
    safe_base = sanitize_name(f"{title} [{vid}]" if vid else title)

    workdir = outdir / safe_base
    workdir.mkdir(parents=True, exist_ok=True)

    src_dir = workdir / "source_audio"
    src_dir.mkdir(parents=True, exist_ok=True)
    src_audio = src_dir / downloaded_audio.name
    if downloaded_audio.resolve() != src_audio.resolve():
        src_audio.write_bytes(downloaded_audio.read_bytes())

    wav_path = workdir / "audio_16k_mono.wav"
    run_ffmpeg_to_wav(src_audio, wav_path)

    tr = transcribe(wav_path, args.model, args.task, args.language)
    segments = tr["segments"]

    metadata = {
        "id": info.get("id"),
        "title": info.get("title"),
        "uploader": info.get("uploader"),
        "channel": info.get("channel"),
        "webpage_url": info.get("webpage_url"),
        "duration": info.get("duration"),
        "upload_date": info.get("upload_date"),
        "description": info.get("description"),
        "extractor": info.get("extractor"),
    }

    (workdir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (workdir / "segments.json").write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    (workdir / "transcript.txt").write_text(tr["full_text"] + "\n", encoding="utf-8")

    db_path = build_keyword_index_sqlite(workdir, segments)

    semantic_enabled = False
    semantic_error = None
    semantic_meta = None
    if args.semantic:
        try:
            if not optional_semantic_deps_ok():
                raise RuntimeError("sentence-transformers/numpy not available")
            _, semantic_meta = build_semantic_embeddings(workdir, segments, args.semantic_model)
            semantic_enabled = True
        except Exception as e:
            semantic_error = str(e)

    result = {
        "success": True,
        "mode": "ingest",
        "input_url": args.url,
        "output_dir": str(workdir),
        "files": {
            "metadata": str(workdir / "metadata.json"),
            "transcript": str(workdir / "transcript.txt"),
            "segments": str(workdir / "segments.json"),
            "search_db": str(db_path),
            "wav": str(wav_path),
        },
        "video": metadata,
        "asr": {
            "task": args.task,
            "model": args.model,
            "forced_language": args.language,
            "detected_language": tr["info"].get("language"),
            "detected_language_probability": tr["info"].get("language_probability"),
            "duration_seconds": tr["info"].get("duration"),
        },
        "segments_count": len(segments),
        "transcript_chars": len(tr["full_text"]),
        "semantic": {
            "enabled": semantic_enabled,
            "model": args.semantic_model if semantic_enabled else None,
            "error": semantic_error,
            "meta": semantic_meta,
        },
    }
    (workdir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    jprint(result)


def keyword_search(workdir: Path, query: str, top_k: int):
    db = workdir / "search.db"
    if not db.exists():
        fail(f"search.db not found in {workdir}. Run ingest first.")
    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    # bm25() lower is better -> convert to score by negating
    sql = """
    SELECT m.seg_id, m.start, m.end, m.text, bm25(seg_fts) AS bm
    FROM seg_fts
    JOIN seg_meta m ON m.seg_id = CAST(seg_fts.seg_id AS INTEGER)
    WHERE seg_fts MATCH ?
    ORDER BY bm ASC
    LIMIT ?
    """
    try:
        cur.execute(sql, (query, top_k))
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        # fallback for malformed MATCH query
        cur.execute(
            "SELECT seg_id, start, end, text FROM seg_meta WHERE text LIKE ? LIMIT ?",
            (f"%{query}%", top_k),
        )
        rows = [(r[0], r[1], r[2], r[3], 999.0) for r in cur.fetchall()]
    conn.close()

    out = []
    for seg_id, start, end, text, bm in rows:
        out.append({
            "seg_id": int(seg_id),
            "start": start,
            "end": end,
            "timestamp": hhmmss(start),
            "text": text,
            "score": float(-bm),
            "source": "keyword",
        })
    return out


def semantic_search(workdir: Path, query: str, top_k: int):
    import numpy as np
    from fastembed import TextEmbedding

    seg_file = workdir / "segments.json"
    emb_file = workdir / "embeddings.npy"
    idx_file = workdir / "semantic_index.json"

    if not seg_file.exists() or not emb_file.exists() or not idx_file.exists():
        return []

    segments = json.loads(seg_file.read_text(encoding="utf-8"))
    meta = json.loads(idx_file.read_text(encoding="utf-8"))
    model_name = meta.get("model", "BAAI/bge-small-en-v1.5")

    embedder = TextEmbedding(model_name=model_name)
    qv = np.asarray(list(embedder.embed([query]))[0], dtype=np.float32)
    qn = np.linalg.norm(qv)
    if qn == 0:
        return []
    qv = qv / qn

    emb = np.load(str(emb_file))
    sims = emb @ qv
    top_idx = np.argsort(-sims)[:top_k]

    out = []
    for i in top_idx:
        s = segments[int(i)]
        score = float(sims[int(i)])
        out.append({
            "seg_id": int(s["id"]),
            "start": s["start"],
            "end": s["end"],
            "timestamp": hhmmss(s["start"]),
            "text": s.get("text", ""),
            "score": score,
            "source": "semantic",
        })
    return out


def reciprocal_rank_fusion(results_lists: List[List[Dict[str, Any]]], k: int = 60):
    fused = {}
    for lst in results_lists:
        for rank, item in enumerate(lst, start=1):
            key = item["seg_id"]
            fused.setdefault(key, {**item, "rrf_score": 0.0})
            fused[key]["rrf_score"] += 1.0 / (k + rank)
            # keep highest confidence score seen
            if item.get("score", -1e9) > fused[key].get("score", -1e9):
                fused[key]["score"] = item.get("score")
                fused[key]["text"] = item.get("text")
                fused[key]["start"] = item.get("start")
                fused[key]["end"] = item.get("end")
                fused[key]["timestamp"] = item.get("timestamp")
                fused[key]["source"] = item.get("source")
    out = list(fused.values())
    out.sort(key=lambda x: x["rrf_score"], reverse=True)
    return out


def cmd_search(args):
    workdir = Path(args.workdir).expanduser().resolve()
    if not workdir.exists():
        fail(f"workdir not found: {workdir}")

    kw = keyword_search(workdir, args.query, args.top_k)

    sem = []
    sem_error = None
    if args.semantic:
        try:
            if not optional_semantic_deps_ok():
                raise RuntimeError("sentence-transformers/numpy not available")
            sem = semantic_search(workdir, args.query, args.top_k)
        except Exception as e:
            sem_error = str(e)

    if sem:
        merged = reciprocal_rank_fusion([kw, sem])[: args.top_k]
    else:
        merged = kw[: args.top_k]

    result = {
        "success": True,
        "mode": "search",
        "workdir": str(workdir),
        "query": args.query,
        "top_k": args.top_k,
        "semantic_requested": args.semantic,
        "semantic_error": sem_error,
        "counts": {
            "keyword": len(kw),
            "semantic": len(sem),
            "merged": len(merged),
        },
        "results": merged,
    }
    jprint(result)


def build_parser():
    p = argparse.ArgumentParser(description="Video Intel v2: ingest + hybrid search (keyword + semantic)")
    sub = p.add_subparsers(dest="command", required=True)

    i = sub.add_parser("ingest", help="Download, transcribe, and build search indexes")
    i.add_argument("--url", required=True)
    i.add_argument("--outdir", required=True)
    i.add_argument("--model", default="small", help="Whisper model: tiny/base/small/medium/large-v3")
    i.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    i.add_argument("--language", default=None)
    i.add_argument("--semantic", action="store_true", help="Build semantic embeddings index")
    i.add_argument("--semantic-model", default="BAAI/bge-small-en-v1.5")
    i.set_defaults(func=cmd_ingest)

    s = sub.add_parser("search", help="Query inside a previously indexed video")
    s.add_argument("--workdir", required=True, help="Path with segments.json/search.db")
    s.add_argument("--query", required=True)
    s.add_argument("--top-k", type=int, default=8)
    s.add_argument("--semantic", action="store_true", help="Use semantic search (if embeddings exist)")
    s.set_defaults(func=cmd_search)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        fail("Interrupted by user", 130)
