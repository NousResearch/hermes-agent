"""Graphify: code knowledge graph core.

Index a repo into symbols, imports, call edges, and file dependencies, then
persist/reload the graph as JSON. No runtime deps beyond the stdlib.

Graph shape
-----------
- **File nodes**   — one per source file
- **Symbol nodes** — top-level defs/classes (optionally functions inside classes)
- **Edges**        — IMPORTS, CONTAINS, CALLS, REFERENCES

A minimal JSON schema keeps indexes deterministic and easy to extend:

{
  "version": 1,
  "root": "/abs/path/to/repo",
  "generated_at": "<iso8601>",
  "files": {
    "<relative_path>": {
      "language": "python",
      "sha256": "<hex>",
      "symbols": ["<symbol_id>", ...]
    }
  },
  "symbols": {
    "<symbol_id>": {
      "name": "Foo",
      "kind": "class|function|method",
      "file": "<relative_path>",
      "line": 1,
      "signature": "Foo(Base)",
      "doc": "...",
      "meta": {}
    }
  },
  "edges": [
    {"source": "...", "target": "...", "type": "IMPORTS|CONTAINS|CALLS|REFERENCES"}
  ]
}

Stable symbol ids: ``file_relpath:symbol_name``.
For methods: ``file_relpath:symbol_name.method_name``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


# ---------------------------------------------------------------------------
# Graph schema
# ---------------------------------------------------------------------------

SYMBOL_KINDS = ("class", "function", "method", "attribute", "module")

EDGE_TYPES = ("IMPORTS", "CONTAINS", "CALLS", "REFERENCES")

SYMBOL_ID_DELIM = ":"
SYMBOL_ID_DELIM_NESTED = "."


def _safe_relpath(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace(os.sep, "/")
    except ValueError:
        return str(path).replace(os.sep, "/")


def _symbol_id(relpath: str, name: str) -> str:
    return f"{relpath}{SYMBOL_ID_DELIM}{name}"


def _split_symbol_id(symbol_id: str) -> tuple[str, str]:
    parts = symbol_id.split(SYMBOL_ID_DELIM, 1)
    if len(parts) != 2:
        return symbol_id, ""
    return parts[0], parts[1]


# ---------------------------------------------------------------------------
# Helpers / defaults
# ---------------------------------------------------------------------------

PYTHON_EXTS = (".py", ".pyi")
DEFAULT_IGNORE_DIRS = (".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache")
SUPPORTED_EXTS = PYTHON_EXTS


def _detect_language(path: Path) -> str:
    mapping = {
        ".py": "python",
        ".pyi": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".rb": "ruby",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".hpp": "cpp",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".cs": "csharp",
    }
    return mapping.get(path.suffix.lower(), "unknown")


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Python-specific extractors
# ---------------------------------------------------------------------------


class _SymbolExtractor(ast.NodeVisitor):
    def __init__(self, relpath: str) -> None:
        self.relpath = relpath
        self.symbols: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, str]] = []
        self._class_stack: list[str] = []

    # Helpers

    def _add_symbol(
        self,
        node_id: str,
        name: str,
        kind: str,
        lineno: int,
        signature: str = "",
        doc: str = "",
        meta: Optional[dict[str, Any]] = None,
    ) -> None:
        if node_id in self.symbols:
            return
        self.symbols[node_id] = {
            "name": name,
            "kind": kind,
            "file": self.relpath,
            "line": lineno,
            "signature": signature or name,
            "doc": doc,
            "meta": meta or {},
        }

    def _contain(self, parent_id: str, child_id: str) -> None:
        if parent_id and child_id and parent_id != child_id:
            self.edges.append(
                {"source": parent_id, "target": child_id, "type": "CONTAINS"}
            )

    def _imports(self, node: ast.AST) -> None:
        module: Optional[str] = None
        names: list[str] = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                asname = alias.asname or alias.name.split(".")[0]
                names.append(asname)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                names.append(alias.asname or alias.name)

        if not module:
            return
        for name in names:
            target = _symbol_id(self.relpath, f"{module}.{name}")
            source = _symbol_id(self.relpath, name)
            self.edges.append(
                {"source": source, "target": target, "type": "IMPORTS"}
            )

    def _calls(self, node: ast.AST, scope_id: str) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                name = ""
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    parts: list[str] = []
                    cur: ast.AST = func
                    while isinstance(cur, ast.Attribute):
                        parts.append(cur.attr)
                        cur = cur.value
                    if isinstance(cur, ast.Name):
                        parts.append(cur.id)
                    name = ".".join(reversed(parts))
                if name:
                    target = _symbol_id(self.relpath, name)
                    self.edges.append(
                        {"source": scope_id, "target": target, "type": "CALLS"}
                    )

    def visit_Module(self, node: ast.Module) -> None:
        module_id = _symbol_id(self.relpath, "__module__")
        self._add_symbol(module_id, Path(self.relpath).name, "module", 1)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_id = _symbol_id(self.relpath, node.name)
        signature = f"class {node.name}"
        if node.bases:
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute):
                    parts = []
                    cur = b
                    while isinstance(cur, ast.Attribute):
                        parts.append(cur.attr)
                        cur = cur.value
                    if isinstance(cur, ast.Name):
                        parts.append(cur.id)
                    bases.append(".".join(reversed(parts)))
            signature += "(" + ", ".join(bases) + ")"
        doc = ast.get_docstring(node) or ""
        self._add_symbol(class_id, node.name, "class", node.lineno, signature, doc)
        if self._class_stack:
            parent_id = _symbol_id(self.relpath, self._class_stack[-1])
            self._contain(parent_id, class_id)
        self._class_stack.append(node.name)
        self._calls(node, class_id)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._class_stack:
            kind = "method"
            prefix = f"{self._class_stack[-1]}{SYMBOL_ID_DELIM_NESTED}"
            full_name = f"{prefix}{node.name}"
        else:
            kind = "function"
            prefix = ""
            full_name = node.name
        func_id = _symbol_id(self.relpath, full_name)
        args = node.args
        signature = f"def {node.name}({ast.unparse(args) if hasattr(ast, 'unparse') else ''})"
        doc = ast.get_docstring(node) or ""
        self._add_symbol(func_id, full_name, kind, node.lineno, signature, doc)
        if self._class_stack:
            class_id = _symbol_id(self.relpath, self._class_stack[-1])
            self._contain(class_id, func_id)
        self._calls(node, func_id)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        node.__class__ = ast.FunctionDef  # type: ignore[misc]
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_Import(self, node: ast.Import) -> None:
        self._imports(node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._imports(node)
        self.generic_visit(node)


def _extract_python(file_path: Path, relpath: str) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, []
    extractor = _SymbolExtractor(relpath)
    extractor.visit(tree)
    return extractor.symbols, extractor.edges


# ---------------------------------------------------------------------------
# CodeGraphStore
# ---------------------------------------------------------------------------


@dataclass
class FileEntry:
    relative_path: str
    language: str = "unknown"
    sha256: str = ""
    symbols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "sha256": self.sha256,
            "symbols": list(self.symbols),
        }

    @classmethod
    def from_dict(cls, path: str, data: dict[str, Any]) -> "FileEntry":
        return cls(
            relative_path=path,
            language=str(data.get("language", "unknown")),
            sha256=str(data.get("sha256", "")),
            symbols=list(data.get("symbols", [])),
        )


class CodeGraphStore:
    """In-memory code graph with JSON persistence."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).expanduser().resolve() if root else Path.cwd().resolve()
        self.generated_at = _iso_now()
        self.files: dict[str, FileEntry] = {}
        self.symbols: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, str]] = []
        self._meta: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_path(self, path: str | Path) -> int:
        target = Path(path)
        if not target.exists():
            return 0
        if target.is_file():
            return self._maybe_index_file(target)
        if target.is_dir():
            count = 0
            for p in sorted(target.rglob("*")):
                if p.is_file():
                    count += self._maybe_index_file(p)
            return count
        return 0

    def _maybe_index_file(self, path: Path) -> int:
        if path.suffix.lower() not in SUPPORTED_EXTS:
            return 0
        return self._index_file(path)

    def _index_file(self, path: Path) -> int:
        relpath = _safe_relpath(self.root, path)
        if relpath in self.files:
            return 0
        language = _detect_language(path)
        sha = _sha256_file(path)
        symbols: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, str]] = []
        if language == "python":
            symbols, edges = _extract_python(path, relpath)
        self.files[relpath] = FileEntry(
            relative_path=relpath, language=language, sha256=sha, symbols=list(symbols.keys())
        )
        for sym_id, sym in symbols.items():
            self.symbols[sym_id] = sym
        self.edges.extend(edges)
        return 1

    def index_walk(
        self,
        root: str | Path,
        *,
        extensions: Optional[Iterable[str]] = None,
        ignore_dirs: Optional[Iterable[str]] = None,
    ) -> int:
        base = Path(root)
        exts = {e.lower() for e in (extensions or PYTHON_EXTS)}
        ignore = {d.lower() for d in (ignore_dirs or DEFAULT_IGNORE_DIRS)}
        if not base.exists():
            return 0
        count = 0
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if any(part.lower() in ignore for part in p.parts):
                continue
            if p.suffix.lower() not in exts:
                continue
            count += self._maybe_index_file(p)
        return count

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def nodes(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for rel, entry in sorted(self.files.items()):
            out.append(
                {
                    "id": rel,
                    "kind": "file",
                    "label": Path(rel).name,
                    "language": entry.language,
                    "sha256": entry.sha256,
                    "symbols": entry.symbols,
                }
            )
        for sym_id, sym in sorted(self.symbols.items()):
            node = dict(sym)
            node["id"] = sym_id
            out.append(node)
        return out

    def relationships(self) -> list[dict[str, str]]:
        rels: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for rel, entry in sorted(self.files.items()):
            module_id = _symbol_id(rel, "__module__")
            rels.append(
                {"source": module_id, "target": rel, "type": "CONTAINS"}
            )
            seen.add((module_id, rel, "CONTAINS"))
        for sym_id in sorted(self.symbols):
            rel, name = _split_symbol_id(sym_id)
            if rel:
                module_id = _symbol_id(rel, "__module__")
                key = (module_id, sym_id, "CONTAINS")
                if key not in seen:
                    rels.append({"source": module_id, "target": sym_id, "type": "CONTAINS"})
                    seen.add(key)
        for edge in self.edges:
            key = (edge["source"], edge["target"], edge["type"])
            if key not in seen:
                rels.append(edge)
                seen.add(key)
        return rels

    def references(self, symbol_id: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for e in self.edges:
            if e["source"] == symbol_id or e["target"] == symbol_id:
                other = e["target"] if e["source"] == symbol_id else e["source"]
                out.append(
                    {
                        "symbol_id": other,
                        "direction": "out" if e["source"] == symbol_id else "in",
                        "type": e["type"],
                    }
                )
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "version": 1,
            "root": str(self.root),
            "generated_at": self.generated_at,
            "files": {k: v.to_dict() for k, v in sorted(self.files.items())},
            "symbols": dict(sorted(self.symbols.items())),
            "edges": list(self.edges),
            "meta": dict(sorted(self._meta.items())),
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_json()
        tmp = target.with_suffix(target.suffix + f".tmp-{uuid.uuid4().hex}")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        tmp.replace(target)
        return target

    @classmethod
    def read_json(cls, path: str | Path) -> "CodeGraphStore":
        target = Path(path)
        data = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("graphify payload must be a JSON object")
        store = cls(root=data.get("root") or target.parent)
        store.generated_at = str(data.get("generated_at") or _iso_now())
        store.files = {
            k: FileEntry.from_dict(k, v) for k, v in data.get("files", {}).items()
        }
        store.symbols = dict(data.get("symbols", {}))
        store.edges = [dict(e) for e in data.get("edges", [])]
        store._meta = dict(data.get("meta", {}))
        return store

    def with_meta(self, **kwargs: Any) -> "CodeGraphStore":
        self._meta.update(kwargs)
        return self

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        kind_counts: dict[str, int] = {}
        for sym in self.symbols.values():
            kind_counts[sym.get("kind", "unknown")] = kind_counts.get(sym.get("kind", "unknown"), 0) + 1
        return {
            "files": len(self.files),
            "symbols": len(self.symbols),
            "edges": len(self.edges),
            "by_kind": dict(sorted(kind_counts.items())),
        }


# ---------------------------------------------------------------------------
# Helpers / defaults
# ---------------------------------------------------------------------------


def _tokenize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.strip().lower()).strip()


def build_graph(
    root: str | Path,
    *,
    extensions: Optional[Iterable[str]] = None,
    ignore_dirs: Optional[Iterable[str]] = None,
) -> CodeGraphStore:
    """Index ``root`` and return a populated ``CodeGraphStore``."""
    store = CodeGraphStore(root=root)
    store.index_walk(root, extensions=extensions, ignore_dirs=ignore_dirs)
    return store


# ---------------------------------------------------------------------------
# CLI / module entrypoint
# ---------------------------------------------------------------------------

def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Graphify code indexer")
    parser.add_argument("path", nargs="?", default=".", help="Repo path to index")
    parser.add_argument("--out", default="graphify.json", help="Output JSON path")
    parser.add_argument("--ext", nargs="*", default=None, help="File extensions to index")
    parser.add_argument("--ignore-dir", nargs="*", default=None, help="Directories to ignore")
    args = parser.parse_args()

    store = build_graph(
        args.path,
        extensions=args.ext,
        ignore_dirs=args.ignore_dir,
    )
    out = store.write_json(args.out)
    stats = store.stats()
    print(f"wrote {out}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
