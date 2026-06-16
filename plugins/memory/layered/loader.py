import os
import time
import hashlib

CACHE_TTL = 300

L1_KEYWORDS = [
    "api", "key", "密钥", "token",
    "version", "版本",
    "skill", "技能",
    "publish", "发布",
    "credential", "credential",
]

L2_KEYWORDS = [
    "workflow", "工作流",
    "cron",
    "decision", "决策",
    "environment", "环境",
    "history", "历史",
    "archive", "归档",
    "deploy", "deployment", "部署",
]


class CacheManager:
    def __init__(self, ttl=CACHE_TTL):
        self._cache = {}
        self._ttl = ttl

    def get(self, key):
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.time() - entry["time"] > self._ttl:
            del self._cache[key]
            return None
        return entry["result"]

    def set(self, key, result):
        self._cache[key] = {"result": result, "time": time.time()}

    def invalidate(self, key=None):
        if key:
            self._cache.pop(key, None)
        else:
            self._cache.clear()

    @property
    def size(self):
        return len(self._cache)


_cache = CacheManager()


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def file_exists(path: str) -> bool:
    return os.path.isfile(path)


def needs_l1(query: str) -> bool:
    q = query.lower()
    return any(kw.lower() in q for kw in L1_KEYWORDS)


def needs_l2(query: str) -> bool:
    q = query.lower()
    return any(kw.lower() in q for kw in L2_KEYWORDS)


def load_l0(memory_dir: str) -> str:
    path = os.path.join(memory_dir, "L0_core.md")
    if not file_exists(path):
        return ""
    return read_file(path)


def load_l1(memory_dir: str) -> str:
    path = os.path.join(memory_dir, "L1_context.md")
    if not file_exists(path):
        return ""
    return read_file(path)


def load_l2_archive(memory_dir: str) -> str:
    l2_dir = os.path.join(memory_dir, "L2_archive")
    if not os.path.isdir(l2_dir):
        return ""
    parts = []
    for fname in sorted(os.listdir(l2_dir)):
        if fname.endswith(".md"):
            content = read_file(os.path.join(l2_dir, fname))
            parts.append(f"### {fname}\n{content}")
    return "\n\n".join(parts)


def load_for_query(memory_dir: str, query: str, use_cache: bool = True) -> str:
    cache_key = hashlib.sha256(query.encode()).hexdigest() if use_cache else ""

    if use_cache:
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

    parts = [load_l0(memory_dir)]

    if needs_l1(query):
        l1 = load_l1(memory_dir)
        if l1:
            parts.append(f"<!-- L1 context -->\n{l1}")

    if needs_l2(query):
        l2 = load_l2_archive(memory_dir)
        if l2:
            parts.append(f"<!-- L2 archive -->\n{l2}")

    result = "\n---\n".join(parts)

    if use_cache:
        _cache.set(cache_key, result)

    return result


def get_status(memory_dir: str) -> dict:
    result = {}
    for name in ["L0_core.md", "L1_context.md", "index.md"]:
        path = os.path.join(memory_dir, name)
        result[name] = {
            "exists": file_exists(path),
            "size": os.path.getsize(path) if file_exists(path) else 0,
        }
    l2_dir = os.path.join(memory_dir, "L2_archive")
    result["L2_archive"] = {
        "exists": os.path.isdir(l2_dir),
        "files": sorted(os.listdir(l2_dir)) if os.path.isdir(l2_dir) else [],
    }
    result["cache_entries"] = _cache.size
    return result
