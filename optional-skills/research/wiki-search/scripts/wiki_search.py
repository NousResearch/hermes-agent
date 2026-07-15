"""Semantic and keyword search for Markdown wikis using optional Ollama embeddings."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


CACHE_SCHEMA_VERSION = 1
SECTION_MIN_CHARS = 20
DEFAULT_MODEL = "all-minilm"


class PersistedEmbeddingDimensionChanged(Exception):
    """A reindex observed a new embedding dimension for an existing namespace."""


def log(message: str) -> None:
    print(f"[wiki-search] {message}", file=sys.stderr)


def ollama_embed(text: str, model: str) -> list[float] | None:
    """Request a local embedding, returning ``None`` when Ollama is unavailable."""
    payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    request = urllib.request.Request(
        f"{ollama_url}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            embedding = json.loads(response.read())["embedding"]
    except (KeyError, OSError, ValueError, urllib.error.HTTPError, urllib.error.URLError):
        return None
    if not isinstance(embedding, list) or not all(isinstance(value, (int, float)) for value in embedding):
        return None
    return [float(value) for value in embedding]


def _identity(root: Path, model: str) -> dict[str, object]:
    return {
        "root": str(root.expanduser().resolve()),
        "model": model,
        "schema": CACHE_SCHEMA_VERSION,
    }


def index_path(cache_dir: Path, root: Path, model: str) -> Path:
    """Return the index path unique to one canonical root, model, and schema."""
    encoded = json.dumps(_identity(root, model), sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return cache_dir / f"v{CACHE_SCHEMA_VERSION}" / digest / "index.json"


def fresh_index(root: Path, model: str) -> dict[str, object]:
    """Create an empty, validated index for one root and embedding model."""
    return {"metadata": {**_identity(root, model), "dimension": None}, "files": {}}


def _is_valid_index(index: object, root: Path, model: str) -> bool:
    if not isinstance(index, dict) or not isinstance(index.get("files"), dict):
        return False
    metadata = index.get("metadata")
    if not isinstance(metadata, dict):
        return False
    identity = _identity(root, model)
    if any(metadata.get(key) != value for key, value in identity.items()):
        return False
    dimension = metadata.get("dimension")
    if dimension is not None and (not isinstance(dimension, int) or dimension <= 0):
        return False
    for file_info in index["files"].values():
        if not isinstance(file_info, dict) or not isinstance(file_info.get("sections", []), list):
            return False
        for section in file_info.get("sections", []):
            if not isinstance(section, dict):
                return False
            embedding = section.get("embedding")
            if embedding is None:
                continue
            if (
                dimension is None
                or not isinstance(embedding, list)
                or len(embedding) != dimension
                or not all(isinstance(value, (int, float)) for value in embedding)
            ):
                return False
    return True


def load_index(cache_dir: Path, root: Path, model: str) -> dict[str, object]:
    """Load only a matching, current-schema index; otherwise start clean."""
    path = index_path(cache_dir, root, model)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fresh_index(root, model)
    return loaded if _is_valid_index(loaded, root, model) else fresh_index(root, model)


def save_index(cache_dir: Path, root: Path, model: str, index: dict[str, object]) -> Path:
    """Persist a validated index in its isolated namespace."""
    if not _is_valid_index(index, root, model):
        raise ValueError("refusing to save an index with incompatible metadata")
    path = index_path(cache_dir, root, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


def _without_frontmatter(text: str) -> str:
    return re.sub(r"\A---\s*\n.*?\n---\s*\n?", "", text, count=1, flags=re.DOTALL)


def parse_md_sections(text: str) -> list[dict[str, object]]:
    """Split Markdown into meaningful heading sections."""
    sections: list[dict[str, object]] = []
    heading = ""
    level = 0
    lines: list[str] = []

    def flush() -> None:
        content = "\n".join(lines).strip()
        if len(content) < SECTION_MIN_CHARS:
            return
        sections.append(
            {
                "heading": heading,
                "level": level,
                "content": content,
                "slug": hashlib.sha256(content.encode("utf-8")).hexdigest()[:12],
            }
        )

    for line in _without_frontmatter(text).splitlines():
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if match:
            flush()
            lines = []
            level = len(match.group(1))
            heading = match.group(2).strip()
        else:
            lines.append(line)
    flush()
    return sections


def extract_title(text: str, fallback: str) -> str:
    """Use the first H1 title when present, otherwise use the file stem."""
    match = re.search(r"^#\s+(.+)$", _without_frontmatter(text), re.MULTILINE)
    return match.group(1).strip() if match else fallback


def _file_changed(info: object, path: Path, force: bool) -> bool:
    if force or not isinstance(info, dict):
        return True
    stat = path.stat()
    return info.get("mtime_ns") != stat.st_mtime_ns or info.get("size") != stat.st_size


def _index_file(
    path: Path,
    relative_path: str,
    index: dict[str, Any],
    model: str,
    *,
    rebuild_on_dimension_change: bool = False,
) -> dict[str, object] | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    sections = parse_md_sections(text)
    if not sections:
        return None
    metadata = index["metadata"]
    dimension = metadata["dimension"]
    embedded_sections: list[dict[str, object]] = []
    for section in sections:
        query_text = str(section["content"])
        if section["heading"]:
            query_text = f"{section['heading']}: {query_text}"
        embedding = ollama_embed(query_text[:2000], model)
        if embedding is not None:
            if dimension is None:
                dimension = len(embedding)
                metadata["dimension"] = dimension
            if len(embedding) != dimension:
                if rebuild_on_dimension_change:
                    raise PersistedEmbeddingDimensionChanged
                log(f"incompatible embedding dimension for {relative_path}; keeping keyword-only")
                embedding = None
        embedded_sections.append({**section, "embedding": embedding})
    stat = path.stat()
    return {
        "title": extract_title(text, path.stem),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "sections": embedded_sections,
    }


def cmd_index(
    wiki_path: str | Path,
    *,
    model: str,
    cache_dir: Path,
    force: bool = False,
) -> dict[str, object]:
    """Create or incrementally update the isolated index for one Markdown wiki."""
    root = Path(wiki_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"wiki path does not exist: {root}")
    index: dict[str, Any] = load_index(cache_dir, root, model)
    markdown_files = [
        path for path in sorted(root.rglob("*.md")) if not path.relative_to(root).parts[0] == "raw"
    ]
    rebuild_on_dimension_change = index["metadata"]["dimension"] is not None
    while True:
        present_paths = {str(path.relative_to(root)) for path in markdown_files}
        files: dict[str, Any] = index["files"]
        removed = sum(1 for relative_path in list(files) if relative_path not in present_paths)
        for relative_path in list(files):
            if relative_path not in present_paths:
                del files[relative_path]

        indexed = 0
        skipped = 0
        try:
            for path in markdown_files:
                relative_path = str(path.relative_to(root))
                if not _file_changed(files.get(relative_path), path, force):
                    skipped += 1
                    continue
                entry = _index_file(
                    path,
                    relative_path,
                    index,
                    model,
                    rebuild_on_dimension_change=rebuild_on_dimension_change,
                )
                if entry is None:
                    files.pop(relative_path, None)
                else:
                    files[relative_path] = entry
                    indexed += 1
        except PersistedEmbeddingDimensionChanged:
            log("embedding dimension changed; rebuilding this index namespace")
            index = fresh_index(root, model)
            rebuild_on_dimension_change = False
            force = True
            continue
        break

    save_index(cache_dir, root, model, index)
    sections = sum(len(file_info["sections"]) for file_info in files.values())
    return {
        "command": "index",
        "wiki": str(root),
        "model": model,
        "indexed": indexed,
        "skipped": skipped,
        "removed": removed,
        "files": len(files),
        "sections": sections,
    }


def cosine_similarity(a: list[float], b: list[float]) -> float | None:
    """Return cosine similarity only when two non-zero vectors are compatible."""
    if not a or len(a) != len(b):
        return None
    dot_product = sum(left * right for left, right in zip(a, b))
    left_norm = sum(value * value for value in a) ** 0.5
    right_norm = sum(value * value for value in b) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return None
    return dot_product / (left_norm * right_norm)


def _result(relative_path: str, file_info: dict[str, Any], section: dict[str, Any], score: float) -> dict[str, object]:
    return {
        "file": relative_path,
        "title": file_info.get("title", Path(relative_path).stem),
        "heading": section.get("heading", ""),
        "content": str(section.get("content", ""))[:300],
        "score": score,
    }


def semantic_search(query: str, index: dict[str, Any], model: str, top_k: int = 10) -> tuple[list[dict[str, object]], bool]:
    """Return semantic candidates plus whether a query embedding was available."""
    query_embedding = ollama_embed(query, model)
    if query_embedding is None:
        return [], False
    expected_dimension = index["metadata"].get("dimension")
    if expected_dimension is not None and len(query_embedding) != expected_dimension:
        log("query embedding dimension does not match the index; using keywords")
        return [], False
    results: list[dict[str, object]] = []
    for relative_path, file_info in index["files"].items():
        for section in file_info.get("sections", []):
            embedding = section.get("embedding")
            if not isinstance(embedding, list):
                continue
            score = cosine_similarity(query_embedding, embedding)
            if score is not None:
                results.append(_result(relative_path, file_info, section, score))
    results.sort(key=lambda result: float(result["score"]), reverse=True)
    return results[:top_k], True


def keyword_search(query: str, index: dict[str, Any], top_k: int = 10) -> list[dict[str, object]]:
    """Return simple case-insensitive keyword candidates without external services."""
    terms = [term for term in query.lower().split() if term]
    if not terms:
        return []
    results: list[dict[str, object]] = []
    for relative_path, file_info in index["files"].items():
        for section in file_info.get("sections", []):
            content = str(section.get("content", ""))
            matches = sum(term in content.lower() for term in terms)
            if matches:
                results.append(_result(relative_path, file_info, section, matches / len(terms)))
    results.sort(key=lambda result: float(result["score"]), reverse=True)
    return results[:top_k]


def hybrid_search(
    semantic_results: list[dict[str, object]], keyword_results: list[dict[str, object]], top_k: int = 10
) -> list[dict[str, object]]:
    """Fuse semantic and keyword rankings using reciprocal-rank fusion."""
    scores: dict[tuple[str, str, str], float] = {}
    results: dict[tuple[str, str, str], dict[str, object]] = {}
    for ranking in (semantic_results, keyword_results):
        for rank, item in enumerate(ranking, start=1):
            key = (str(item["file"]), str(item["heading"]), str(item["content"]))
            scores[key] = scores.get(key, 0.0) + 1 / (60 + rank)
            results.setdefault(key, item)
    combined = [{**item, "score": scores[key]} for key, item in results.items()]
    combined.sort(key=lambda result: float(result["score"]), reverse=True)
    return combined[:top_k]


def cmd_search(
    query: str,
    wiki_path: str | Path,
    *,
    model: str,
    cache_dir: Path,
    mode: str = "semantic",
    top_k: int = 10,
) -> dict[str, object]:
    """Search exactly the index belonging to ``wiki_path`` and ``model``."""
    root = Path(wiki_path).expanduser().resolve()
    index: dict[str, Any] = load_index(cache_dir, root, model)
    semantic_results, semantic_available = semantic_search(query, index, model, top_k)
    keyword_results = keyword_search(query, index, top_k)
    if mode == "hybrid":
        results = hybrid_search(semantic_results, keyword_results, top_k)
        fallback = "semantic" if semantic_available else "keyword"
    elif semantic_results:
        results = semantic_results
        fallback = "semantic"
    else:
        results = keyword_results
        fallback = "keyword"
    return {
        "command": "search",
        "wiki": str(root),
        "model": model,
        "mode": mode,
        "fallback": fallback,
        "results": results,
    }


def cmd_clean(wiki_path: str | Path, *, model: str, cache_dir: Path) -> dict[str, object]:
    """Remove only the index namespace selected by root and model."""
    root = Path(wiki_path).expanduser().resolve()
    path = index_path(cache_dir, root, model)
    existed = path.exists()
    if existed:
        path.unlink()
    return {"command": "clean", "wiki": str(root), "model": model, "removed": existed}


def cmd_status(wiki_path: str | Path, *, model: str, cache_dir: Path) -> dict[str, object]:
    """Report the selected index without touching another wiki's cache."""
    root = Path(wiki_path).expanduser().resolve()
    index: dict[str, Any] = load_index(cache_dir, root, model)
    files: dict[str, Any] = index["files"]
    return {
        "command": "status",
        "wiki": str(root),
        "model": model,
        "index": str(index_path(cache_dir, root, model)),
        "files": len(files),
        "sections": sum(len(file_info.get("sections", [])) for file_info in files.values()),
        "dimension": index["metadata"]["dimension"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic search for Markdown wikis using local Ollama.")
    parser.add_argument("--wiki", default=os.environ.get("WIKI_PATH", "~/wiki"), help="Markdown wiki directory")
    parser.add_argument("--model", default=os.environ.get("WIKI_EMBED_MODEL", DEFAULT_MODEL), help="Ollama embedding model")
    parser.add_argument("--top-k", type=int, default=10, help="Maximum results to return")
    parser.add_argument("--hybrid", action="store_true", help="Fuse semantic and keyword rankings")
    parser.add_argument("--json", action="store_true", help="Emit stable machine-readable JSON")
    parser.add_argument("--status", action="store_true", help="Show index metadata")
    parser.add_argument("--reindex", action="store_true", help="Force a complete reindex")
    parser.add_argument("--clean", action="store_true", help="Delete only this root/model index")
    parser.add_argument("query", nargs="*", help="index, reindex, status, clean, or a search query")
    return parser


def emit(payload: dict[str, object], json_mode: bool) -> None:
    """Emit the versioned API response or a concise human-readable equivalent."""
    versioned = {"schema_version": 1, **payload}
    if json_mode:
        print(json.dumps(versioned, ensure_ascii=False, sort_keys=True))
        return
    command = str(payload["command"])
    if command == "search":
        results = payload["results"]
        if not results:
            print("No relevant results found.")
            return
        print(f"{len(results)} results ({payload['fallback']} search):")
        for number, item in enumerate(results, start=1):
            print(f"{number}. {item['file']} — {item['heading']}")
        return
    print(json.dumps(versioned, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    """Run the command-line interface."""
    args = build_parser().parse_args()
    cache_dir = Path(os.environ.get("WIKI_SEARCH_CACHE_DIR", "~/.cache/wiki-search")).expanduser()
    positional_command = args.query[0] if args.query else ""
    try:
        if args.clean or positional_command == "clean":
            payload = cmd_clean(args.wiki, model=args.model, cache_dir=cache_dir)
        elif args.status or positional_command == "status":
            payload = cmd_status(args.wiki, model=args.model, cache_dir=cache_dir)
        elif args.reindex or positional_command == "reindex":
            payload = cmd_index(args.wiki, model=args.model, cache_dir=cache_dir, force=True)
        elif positional_command == "index":
            payload = cmd_index(args.wiki, model=args.model, cache_dir=cache_dir)
        elif args.query:
            mode = "hybrid" if args.hybrid else "semantic"
            payload = cmd_search(
                " ".join(args.query),
                args.wiki,
                model=args.model,
                cache_dir=cache_dir,
                mode=mode,
                top_k=args.top_k,
            )
        else:
            build_parser().print_help()
            return
    except (FileNotFoundError, ValueError) as error:
        if args.json:
            print(json.dumps({"schema_version": 1, "error": str(error)}, ensure_ascii=False))
        else:
            print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    emit(payload, args.json)
