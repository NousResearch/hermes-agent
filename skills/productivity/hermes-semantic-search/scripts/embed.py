#!/usr/bin/env python3
"""
Hermes 语义搜索 —— 基于 ollama nomic-embed-text 的本地向量检索
对标 OpenClaw MemorySearch：混合检索 + MMR 重排 + 时间衰减

用法：
  python3 embed.py index                          # 索引全部知识库
  python3 embed.py index --path 系统/踩坑记录.md    # 索引指定文件
  python3 embed.py index-sessions                  # 索引最近 50 个 session 转录
  python3 embed.py index-sessions --limit 100      # 索引最近 N 个 session
  python3 embed.py search "截断API key的bug"       # 混合搜索（向量 + FTS5)
  python3 embed.py search "查询" --pure-vector      # 纯向量搜索
  python3 embed.py search "查询" --top-k 10         # 返回 top K
  python3 embed.py stats                          # 统计信息
  python3 embed.py cleanup                        # 清理已删除文件的向量

数据存储：~/.hermes/memory/vectors.db
"""

import sqlite3
import json
import sys
import os
import time
import re
import math
from pathlib import Path
from datetime import datetime

HOME = Path.home()
KB_DIR = HOME / ".hermes" / "knowledge"
STATE_DB = HOME / ".hermes" / "state.db"
VECTOR_DB = HOME / ".hermes" / "memory" / "vectors.db"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"
CHUNK_SIZE = 300  # 每块最多字符数
CHUNK_OVERLAP = 50

# 混合搜索权重（对标 OpenClaw: vector 0.68 / text 0.32）
VECTOR_WEIGHT = 0.68
TEXT_WEIGHT = 0.32
# MMR 多样性参数（lambda 越大越相关，越小越多样；OpenClaw 默认 0.7）
MMR_LAMBDA = 0.7
# 时间衰减半衰期（天）
TEMPORAL_HALF_LIFE_DAYS = 30
# 最低相似度阈值
MIN_SCORE = 0.45


def init_db():
    db = sqlite3.connect(str(VECTOR_DB))
    db.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            source_type TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now','localtime')),
            UNIQUE(source, chunk_index)
        )
    """)
    db.execute("CREATE INDEX IF NOT EXISTS idx_source ON chunks(source)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_type ON chunks(source_type)")
    db.commit()
    return db


def embed(text: str, retries: int = 3) -> list[float]:
    """调用 ollama 生成 embedding，HTTP 500 自动重试"""
    import urllib.request
    import urllib.error
    body = json.dumps({"model": MODEL, "prompt": text}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            return data["embedding"]
        except urllib.error.HTTPError as e:
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise
        except Exception:
            raise


def chunk_text(text: str, source: str) -> list[tuple[int, str]]:
    """将文本切块，返回 [(index, chunk_text), ...]"""
    # 按段落分，避免截断句子
    paragraphs = re.split(r'\n\n+', text)
    chunks = []
    idx = 0
    current = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) < CHUNK_SIZE:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append((idx, current))
                idx += 1
                # 重叠：保留最后一段
                overlap_start = max(0, len(current) - CHUNK_OVERLAP)
                current = current[overlap_start:] + "\n\n" + para
            else:
                current = para
    
    if current:
        chunks.append((idx, current))
    
    return chunks


def read_kb_file(path: Path) -> str:
    """读取知识库文件，只取 generated 区块"""
    text = path.read_text(encoding="utf-8")
    # 提取 hermes:generated 区块
    parts = []
    in_block = False
    for line in text.split("\n"):
        if "<!-- hermes:generated:start -->" in line:
            in_block = True
            continue
        if "<!-- hermes:generated:end -->" in line:
            in_block = False
            continue
        if in_block:
            parts.append(line)
    return "\n".join(parts)


def index_knowledge_base(db, path_filter: str = None):
    """索引知识库文件"""
    md_files = sorted(KB_DIR.rglob("*.md"))
    if path_filter:
        md_files = [f for f in md_files if path_filter in str(f.relative_to(KB_DIR))]
    
    total_chunks = 0
    for f in md_files:
        rel = str(f.relative_to(KB_DIR))
        try:
            text = read_kb_file(f)
            if not text.strip():
                continue
        except Exception as e:
            print(f"  ⚠️ 读取失败 {rel}: {e}")
            continue
        
        chunks = chunk_text(text, rel)
        for idx, chunk_text_content in chunks:
            # 检查是否已索引（内容未变则跳过）
            existing = db.execute(
                "SELECT content FROM chunks WHERE source=? AND chunk_index=?",
                (rel, idx)
            ).fetchone()
            if existing and existing[0] == chunk_text_content:
                continue
            
            try:
                vec = embed(chunk_text_content)
                db.execute(
                    "INSERT OR REPLACE INTO chunks (source, source_type, chunk_index, content, embedding) VALUES (?,?,?,?,?)",
                    (rel, "wiki", idx, chunk_text_content, json.dumps(vec))
                )
                total_chunks += 1
                print(f"  ✓ {rel} chunk {idx} ({len(chunk_text_content)}字)")
                time.sleep(0.1)  # 避免压垮 ollama
            except Exception as e:
                print(f"  ✗ {rel} chunk {idx}: {e}")
        
        db.commit()
    
    print(f"\n索引完成：{total_chunks} 个新/更新 chunk")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)


def temporal_decay(created_at_str: str) -> float:
    """时间衰减因子，半衰期 TEMPORAL_HALF_LIFE_DAYS 天"""
    try:
        created = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            created = datetime.strptime(created_at_str, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return 1.0  # 无法解析时间则不衰减
    
    age_days = (datetime.now() - created).total_seconds() / 86400
    decay = math.exp(-math.log(2) * age_days / TEMPORAL_HALF_LIFE_DAYS)
    return decay


def mmr_rerank(candidates: list[dict], lambda_param: float = MMR_LAMBDA) -> list[dict]:
    """
    MMR (Maximal Marginal Relevance) 多样性重排
    lambda 接近 1: 优先相关性
    lambda 接近 0: 优先多样性
    """
    if len(candidates) <= 1:
        return candidates
    
    selected = [candidates[0]]
    remaining = candidates[1:]
    
    while remaining:
        mmr_scores = []
        for item in remaining:
            # 相关性得分（向量得分或混合得分）
            relevance = item.get("hybrid_score", item.get("score", 0))
            # 多样性惩罚: 与已选结果的最大相似度
            max_sim = max(
                cosine_similarity(item["_vec"], s["_vec"])
                for s in selected
            )
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            mmr_scores.append((mmr, item))
        
        mmr_scores.sort(key=lambda x: x[0], reverse=True)
        best = mmr_scores[0][1]
        selected.append(best)
        remaining.remove(best)
    
    return selected


def fts5_search(query: str, limit: int = 20) -> list[dict]:
    """在 state.db 的 FTS5 索引中搜索 session 消息，返回 BM25-style 得分"""
    state = sqlite3.connect(str(STATE_DB))
    try:
        # 清理查询中的 FTS5 特殊字符（连字符等），否则会被解析为列运算符
        clean_query = re.sub(r'[-\~\(\)\"\^\*]', ' ', query)
        clean_query = ' '.join(clean_query.split())  # 合并多余空格
        if not clean_query.strip():
            clean_query = query.replace('-', ' ')
        
        rows = state.execute(
            """SELECT m.session_id, m.content, s.title, datetime(s.started_at, 'unixepoch') as created,
               rank FROM messages_fts f
               JOIN messages m ON f.rowid = m.rowid
               JOIN sessions s ON m.session_id = s.id
               WHERE messages_fts MATCH ?
               ORDER BY rank LIMIT ?""",
            (clean_query, limit)
        ).fetchall()
        
        results = []
        for session_id, content, title, created, rank in rows:
            # FTS5 rank 越负越相关，转换为 0-1 得分
            bm25_score = 1.0 / (1.0 - rank) if rank < 0 else 0.1
            results.append({
                "session_id": session_id,
                "content": content[:300] if content else "",
                "title": title or session_id,
                "created": created,
                "bm25_score": bm25_score,
                "source": f"session:{session_id[:16]}",
                "source_type": "session",
            })
    except Exception as e:
        print(f"  ⚠️ FTS5 搜索异常（可能查询词不在索引中）: {e}", file=sys.stderr)
        results = []
    finally:
        state.close()
    
    return results


def hybrid_search(db, query: str, top_k: int = 10, pure_vector: bool = False):
    """
    混合搜索：向量语义 + FTS5 文本 (0.68/0.32) → MMR 重排 → 时间衰减

    对标 OpenClaw MemorySearch 的完整管线
    """
    print(f"\n🔍 搜索：「{query}」\n")
    
    # 1. 向量搜索
    q_vec = embed(query)
    rows = db.execute(
        "SELECT id, source, source_type, content, embedding, created_at FROM chunks"
    ).fetchall()
    
    vector_results = []
    for row in rows:
        id_, source, stype, content, emb_json, created_at = row
        emb = json.loads(emb_json)
        score = cosine_similarity(q_vec, emb)
        if score >= MIN_SCORE:
            vector_results.append({
                "id": id_,
                "source": source,
                "source_type": stype,
                "content": content,
                "created": created_at,
                "score": score,
                "_vec": emb,
            })
    
    vector_results.sort(key=lambda x: x["score"], reverse=True)
    
    if pure_vector:
        results = vector_results
        print(f"  向量匹配: {len(results)} 条\n")
    else:
        # 2. FTS5 文本搜索
        fts5_results = fts5_search(query, limit=20)
        print(f"  向量匹配: {len(vector_results)} 条 | 文本匹配: {len(fts5_results)} 条\n")
        
        # 3. 合并得分（加权求和）
        merged = {}
        
        for item in vector_results:
            key = (item["source_type"], item["source"])
            merged[key] = {
                **item,
                "hybrid_score": item["score"] * VECTOR_WEIGHT,
                "_vec": item["_vec"],
            }
        
        for item in fts5_results:
            key = ("session", item["source"])
            if key in merged:
                # 已有向量结果，叠加文本得分
                merged[key]["hybrid_score"] += item["bm25_score"] * TEXT_WEIGHT
            else:
                # 仅 FTS5 命中，无向量得分（用 BM25 分数主导）
                merged[key] = {
                    **item,
                    "hybrid_score": item["bm25_score"] * TEXT_WEIGHT,
                    "_vec": q_vec,  # 用查询向量做 MMR fallback
                }
        
        results = list(merged.values())
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    
    # 4. MMR 多样性重排
    results = mmr_rerank(results)
    
    # 5. 时间衰减
    for item in results:
        decay = temporal_decay(item.get("created", ""))
        item["final_score"] = item.get("hybrid_score", item.get("score", 0)) * (0.8 + 0.2 * decay)
    
    results.sort(key=lambda x: x["final_score"], reverse=True)
    
    # 输出
    for i, item in enumerate(results[:top_k]):
        stype = item.get("source_type", "?")
        source = item.get("source", "?")
        content = item.get("content", "")[:200]
        score = item.get("final_score", item.get("hybrid_score", item.get("score", 0)))
        bar = "█" * min(int(score * 20), 20)
        
        print(f"{i+1}. [{stype}] {source}  (得分: {score:.3f})")
        print(f"   {bar}")
        print(f"   {content[:180]}...")
        print()
    
    return results[:top_k]


def index_sessions(db, limit: int = 50):
    """索引最近 N 个 session 的用户+助手对话转录"""
    state = sqlite3.connect(str(STATE_DB))
    MAX_CHUNKS_PER_SESSION = 30  # 超大 session 截断避免 ollama OOM
    
    session_ids = state.execute(
        """SELECT id, title, datetime(started_at, 'unixepoch') as created 
           FROM sessions 
           WHERE message_count > 2 AND source != 'cron'
           ORDER BY started_at DESC LIMIT ?""",
        (limit,)
    ).fetchall()
    
    if not session_ids:
        print("  没有找到可索引的 session")
        state.close()
        return
    
    print(f"  待索引: {len(session_ids)} 个 session\n")
    
    total_chunks = 0
    for sid, title, created in session_ids:
        msgs = state.execute(
            """SELECT role, content FROM messages 
               WHERE session_id=? AND role IN ('user','assistant') AND content IS NOT NULL
               ORDER BY id""",
            (sid,)
        ).fetchall()
        
        if not msgs:
            continue
        
        # 拼接对话为文本
        text_parts = []
        for role, content in msgs:
            prefix = "👤" if role == "user" else "🤖"
            text_parts.append(f"{prefix} {content}")
        
        full_text = "\n".join(text_parts)
        short_id = sid[:16]
        source_key = f"session:{short_id}"
        
        # 删除旧 chunks
        db.execute("DELETE FROM chunks WHERE source=?", (source_key,))
        
        # 切块 + 嵌入（超大 session 截断）
        chunks = chunk_text(full_text, source_key)
        chunks = chunks[:MAX_CHUNKS_PER_SESSION]
        for idx, chunk_text_content in chunks:
            try:
                vec = embed(chunk_text_content)
                db.execute(
                    "INSERT INTO chunks (source, source_type, chunk_index, content, embedding, created_at) VALUES (?,?,?,?,?,?)",
                    (source_key, "session", idx, 
                     f"[{title or short_id}] {chunk_text_content}", 
                     json.dumps(vec),
                     created)
                )
                total_chunks += 1
                time.sleep(0.1)
            except Exception as e:
                print(f"  ✗ {short_id} chunk {idx}: {e}")
        
        db.commit()
        print(f"  ✓ {short_id} ({len(msgs)} 条消息 → {len(chunks)} chunks) [{title or '无标题'}]")
    
    state.close()
    print(f"\n索引完成：{total_chunks} 个新 session chunk")


def stats(db):
    """统计信息"""
    total = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    wiki = db.execute("SELECT COUNT(*) FROM chunks WHERE source_type='wiki'").fetchone()[0]
    session = db.execute("SELECT COUNT(*) FROM chunks WHERE source_type='session'").fetchone()[0]
    sources = db.execute("SELECT COUNT(DISTINCT source) FROM chunks").fetchone()[0]
    
    print(f"向量数据库: {VECTOR_DB}")
    print(f"总 chunks:  {total}")
    print(f"  wiki:     {wiki}")
    print(f"  session:  {session}")
    print(f"源文件数:    {sources}")
    print(f"模型:       {MODEL}")


def cleanup(db):
    """清理已删除文件的向量 + 陈旧检测"""
    existing = set()
    for f in KB_DIR.rglob("*.md"):
        existing.add(str(f.relative_to(KB_DIR)))
    
    rows = db.execute("SELECT DISTINCT source FROM chunks WHERE source_type='wiki'").fetchall()
    removed = 0
    stale = []
    for (src,) in rows:
        if src not in existing:
            db.execute("DELETE FROM chunks WHERE source=?", (src,))
            removed += 1
            print(f"  🗑 清理: {src}")
        else:
            # 陈旧检测：检查最后更新时间
            fpath = KB_DIR / src
            mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
            age_days = (datetime.now() - mtime).days
            if age_days > 30:
                chunk_count = db.execute(
                    "SELECT COUNT(*) FROM chunks WHERE source=?", (src,)
                ).fetchone()[0]
                stale.append((src, age_days, chunk_count))
    
    if stale:
        print(f"\n⚠️ 陈旧页面（>30天未更新）：")
        for src, age, count in sorted(stale, key=lambda x: x[1], reverse=True):
            print(f"  📄 {src} — {age}天前更新，{count}个chunk")
    
    if removed == 0 and not stale:
        print("  无过期向量，无陈旧页面")
    
    db.commit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    cmd = sys.argv[1]
    db = init_db()
    
    if cmd == "index":
        path_filter = None
        if len(sys.argv) >= 4 and sys.argv[2] == "--path":
            path_filter = sys.argv[3]
        index_knowledge_base(db, path_filter)
    
    elif cmd == "index-sessions":
        limit = 50
        for i, arg in enumerate(sys.argv):
            if arg == "--limit" and i + 1 < len(sys.argv):
                limit = int(sys.argv[i + 1])
        index_sessions(db, limit)
    
    elif cmd == "search":
        if len(sys.argv) < 3:
            print("用法: python3 embed.py search <查询文本> [--top-k N] [--pure-vector]")
            sys.exit(1)
        query = sys.argv[2]
        top_k = 10
        pure_vector = False
        for i, arg in enumerate(sys.argv[3:], start=3):
            if arg == "--top-k" and i + 1 < len(sys.argv):
                top_k = int(sys.argv[i + 1])
            if arg == "--pure-vector":
                pure_vector = True
        hybrid_search(db, query, top_k, pure_vector)
    
    elif cmd == "stats":
        stats(db)
    
    elif cmd == "cleanup":
        cleanup(db)
    
    else:
        print(f"未知命令: {cmd}")
        print(__doc__)
    
    db.close()
