#!/usr/bin/env python3
"""classify_imports.py — Phase 4 D1: import/dependency classification.

Reads a scan JSON and an imports JSON (produced by extract_imports.py),
classifies each raw import string into one of: stdlib, third_party,
local, relative, unknown.  Emits a classified-imports JSON to stdout.

Usage:
    python classify_imports.py <scan.json> <imports.json> > <classified.json>

Stdlib only — no external dependencies.
"""

import argparse
import json
import os
import sys
from typing import Any

# ── Stdlib maps ──────────────────────────────────────────────────────

_PYTHON_STDLIB: set[str] | None = None
if hasattr(sys, "stdlib_module_names"):
    _PYTHON_STDLIB = set(sys.stdlib_module_names)

_GO_STDLIB: set[str] = {
    "archive", "bufio", "bytes", "cmp", "compress", "container", "context",
    "crypto", "database", "debug", "embed", "encoding", "errors", "expvar",
    "flag", "fmt", "go", "hash", "html", "image", "index", "io", "log",
    "maps", "math", "mime", "net", "os", "path", "plugin", "reflect",
    "regexp", "runtime", "slices", "sort", "strconv", "strings",
    "sync", "syscall", "testing", "text", "time", "unicode", "unsafe",
}

_RUST_STDLIB: set[str] = {
    "std", "core", "alloc", "proc_macro",
}

_JS_STD_NODES: set[str] = {
    "fs", "path", "http", "https", "os", "crypto", "stream", "util",
    "events", "buffer", "child_process", "cluster", "console", "dgram",
    "dns", "domain", "net", "querystring", "readline", "repl", "tls",
    "tty", "url", "vm", "zlib", "assert", "perf_hooks", "timers",
    "trace_events", "v8", "wasi", "worker_threads", "async_hooks",
}

# ── Local-root detection ────────────────────────────────────────────


def detect_local_roots(scan_data: dict) -> set[str]:
    """Derive local module roots from file paths in the scan.

    Returns a set of top-level directory names (e.g. 'src', 'app', 'lib')
    that look like package directories.  Files at the root (no '/') are
    ignored — they'd be modules, not packages.
    """
    roots: set[str] = set()
    for entry in scan_data.get("files", []):
        rel = entry.get("path", "")
        # Skip hidden dirs, __pycache__, etc.
        parts = [p for p in rel.split("/") if p and not p.startswith(".")]
        if len(parts) >= 2:
            roots.add(parts[0])
        elif len(parts) == 1 and "." in rel and "/" not in rel:
            # Root-level file like `main.py` — no package root
            pass
    return roots


# ── Classification ──────────────────────────────────────────────────


def _is_relative(module: str, language: str) -> bool:
    """Detect relative imports (checked first, before stdlib/third-party)."""
    if module.startswith("./") or module.startswith("../"):
        return True
    if language in ("python",) and module.startswith("."):
        return True
    return False


def _is_python_stdlib(module: str) -> bool:
    """Check if a module name is in the Python stdlib."""
    if _PYTHON_STDLIB is not None:
        root = module.split(".")[0]
        return root in _PYTHON_STDLIB
    # Fallback for very old Python (< 3.10)
    _FALLBACK_PYTHON_STDLIB = {
        "os", "sys", "json", "pathlib", "re", "datetime", "collections",
        "itertools", "functools", "typing", "io", "math", "subprocess",
        "threading", "multiprocessing", "socket", "http", "urllib",
        "email", "html", "xml", "csv", "logging", "unittest", "doctest",
        "pdb", "profile", "timeit", "argparse", "configparser", "shutil",
        "glob", "fnmatch", "tempfile", "hashlib", "hmac", "secrets",
        "pickle", "shelve", "xmlrpc", "zipfile", "tarfile",
        "gzip", "bz2", "lzma", "zipimport", "importlib", "pkgutil",
        "abc", "dataclasses", "enum", "contextlib", "asyncio", "concurrent",
        "queue", "selectors", "signal", "mmap", "ctypes", "struct",
        "codecs", "unicodedata", "string", "textwrap", "difflib",
        "gettext", "locale", "calendar", "decimal", "fractions", "random",
        "statistics", "array", "binascii", "copy", "pprint", "reprlib",
        "types", "weakref", "warnings", "traceback", "linecache",
        "tokenize", "token", "keyword", "dis", "ast", "compileall",
        "py_compile", "pyclbr", "symtable", "site", "codeop", "code",
        "builtins", "__future__", "gc", "inspect", "operator",
    }
    return module.split(".")[0] in _FALLBACK_PYTHON_STDLIB


def _is_go_stdlib(module: str) -> bool:
    """Check if a Go import path is stdlib."""
    root = module.split("/")[0]
    return root in _GO_STDLIB


def _is_rust_stdlib(module: str) -> bool:
    """Check if a Rust crate is stdlib."""
    root = module.split("::")[0].split("-")[0]
    return root in _RUST_STDLIB


def _is_js_std(module: str) -> bool:
    """Check if a JS/TS module is a Node.js built-in."""
    # node: prefix is always stdlib
    if module.startswith("node:"):
        return True
    root = module.split("/")[0]
    return root in _JS_STD_NODES


def classify_import(module: str, language: str, local_roots: set[str]) -> str:
    """Classify a single import string.

    Resolution order (checked in this order):
      1. empty → unknown
      2. relative (./ or ../ or . prefix) → relative
      3. stdlib (language-specific) → stdlib
      4. local root match → local
      5. unknown if language not recognised for stdlib checks → unknown
      6. third_party (conservative default)
    """
    if not module or not module.strip():
        return "unknown"

    module = module.strip()

    # 1. Relative before anything else
    if _is_relative(module, language):
        return "relative"

    # 2. Stdlib checks by language
    lang = language.lower()
    if lang == "python":
        if _is_python_stdlib(module):
            return "stdlib"
    elif lang in ("go",):
        if _is_go_stdlib(module):
            return "stdlib"
    elif lang == "rust":
        if _is_rust_stdlib(module):
            return "stdlib"
    elif lang in ("javascript", "typescript"):
        if _is_js_std(module):
            return "stdlib"

    # 3. Local root
    root = module.split(".")[0].split("/")[0]
    if root in local_roots:
        return "local"

    # 4. If language has no stdlib map at all → unknown
    if lang not in ("python", "go", "rust", "javascript", "typescript"):
        return "unknown"

    # 5. Third-party (conservative default for recognised languages)
    return "third_party"


# ── Build classified map ────────────────────────────────────────────


def _classify_imports_for_file(
    imports: list[str],
    language: str,
    local_roots: set[str],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Classify all imports for a single file.

    Returns (classified_list, per-category totals).
    """
    classified: list[dict[str, str]] = []
    counts: dict[str, int] = {
        "stdlib": 0,
        "third_party": 0,
        "local": 0,
        "relative": 0,
        "unknown": 0,
    }
    for mod in imports:
        cat = classify_import(mod, language, local_roots)
        classified.append({"module": mod, "classification": cat})
        counts[cat] = counts.get(cat, 0) + 1
    return classified, counts


def build_classified_map(scan_data: dict, imports_data: dict) -> dict[str, Any]:
    """Transform imports.json into classified-imports.json.

    Schema:
        schema_version, source_scan, source_imports, files, totals
    Each file entry contains:
        imports  (list of {module, classification}),
        totals   (per-category counts),
        warnings (preserved if present in input).
    """
    # Guard: detect already-classified input (idempotency check)
    _check_already_classified(imports_data)

    local_roots = detect_local_roots(scan_data)
    scan_files = scan_data.get("files", [])
    # Build language map: path → language
    lang_map: dict[str, str] = {}
    for entry in scan_files:
        lang_map[entry["path"]] = entry.get("language", "unknown")

    global_counts: dict[str, int] = {
        "stdlib": 0,
        "third_party": 0,
        "local": 0,
        "relative": 0,
        "unknown": 0,
    }

    files_out: dict[str, dict] = {}

    for file_path, file_entry in imports_data.get("files", {}).items():
        raw_imports = file_entry.get("imports", [])
        language = lang_map.get(file_path, "unknown")

        classified, file_counts = _classify_imports_for_file(
            raw_imports, language, local_roots,
        )

        file_output: dict[str, Any] = {
            "imports": classified,
            "totals": {
                "stdlib": file_counts["stdlib"],
                "third_party": file_counts["third_party"],
                "local": file_counts["local"],
                "relative": file_counts["relative"],
                "unknown": file_counts["unknown"],
            },
        }

        # Preserve warnings if present
        if "warnings" in file_entry and file_entry["warnings"]:
            file_output["warnings"] = file_entry["warnings"]

        files_out[file_path] = file_output

        for cat in global_counts:
            global_counts[cat] += file_counts.get(cat, 0)

    return {
        "schema_version": "1.0.0",
        "source_scan": imports_data.get("source_scan", {}),
        "source_imports": {
            "schema_version": imports_data.get("schema_version", "unknown"),
            "generated_at": imports_data.get("generated_at", ""),
        },
        "files": files_out,
        "totals": {
            "stdlib": global_counts["stdlib"],
            "third_party": global_counts["third_party"],
            "local": global_counts["local"],
            "relative": global_counts["relative"],
            "unknown": global_counts["unknown"],
        },
    }


def _check_already_classified(imports_data: dict) -> None:
    """Fail if the input already contains classified dicts.

    Scans every import in every file (not just the first import of the first
    file) to avoid missing already-classified input that could cause
    accidental double-wrapping when classify_imports.py is fed its own output.
    """
    for file_path, file_entry in imports_data.get("files", {}).items():
        raw_imports = file_entry.get("imports", [])
        for idx, item in enumerate(raw_imports):
            if isinstance(item, dict) and "classification" in item:
                raise ValueError(
                    f"Input appears already classified "
                    f"(file: {file_path}, import #{idx} has 'classification' key). "
                    f"Pass unclassified imports.json from extract_imports.py instead."
                )


# ── Loading ─────────────────────────────────────────────────────────


def load_scan_and_imports(
    scan_path: str, imports_path: str
) -> tuple[dict, dict]:
    """Load and validate both JSON files.

    Returns (scan_data, imports_data).  Raises on error.
    """
    if not os.path.isfile(scan_path):
        raise FileNotFoundError(f"Scan file not found: {scan_path}")
    if not os.path.isfile(imports_path):
        raise FileNotFoundError(f"Imports file not found: {imports_path}")

    with open(scan_path, "r", encoding="utf-8") as fh:
        scan_data = json.load(fh)

    with open(imports_path, "r", encoding="utf-8") as fh:
        imports_data = json.load(fh)

    if "files" not in scan_data:
        raise ValueError("Invalid scan schema: missing required 'files' key")

    if "files" not in imports_data:
        raise ValueError("Invalid imports schema: missing required 'files' key")

    return scan_data, imports_data


# ── CLI ─────────────────────────────────────────────────────────────


def main() -> int:
    """CLI entry point.  Takes scan.json and imports.json, writes classified JSON."""
    parser = argparse.ArgumentParser(
        description="Classify imports from a scan + imports JSON.",
    )
    parser.add_argument("scan_json", help="Path to scan_project.py output")
    parser.add_argument("imports_json", help="Path to extract_imports.py output")
    args = parser.parse_args()

    try:
        scan_data, imports_data = load_scan_and_imports(
            args.scan_json, args.imports_json,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        result = build_classified_map(scan_data, imports_data)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
