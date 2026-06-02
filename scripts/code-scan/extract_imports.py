#!/usr/bin/env python3
"""extract_imports.py — Phase 2 D1: import/dependency map extractor.

Reads a scan_project.py JSON output, extracts import/dependency statements
per file using language-specific regex patterns, and emits a structured
import map JSON to stdout.

Usage:
    python extract_imports.py <scan_output.json>

Stdlib only — no external dependencies.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Callable, Iterator


# ── Regex patterns ──────────────────────────────────────────────────

# Python: import X, from X import Y
_PYTHON_RE = re.compile(r"^\s*(?:import\s+(\w+)|from\s+(\w+))", re.MULTILINE)

# JS/TS: import ... from 'Y', require('Y'), import('Y')
_JS_TS_RE = re.compile(
    r"(?:import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]"
    r"|require\s*\(\s*['\"]([^'\"]+)['\"]"
    r"|import\s*\(\s*['\"]([^'\"]+)['\"])"
)

# Rust: use X::Y, extern crate X
_RUST_USE_RE = re.compile(r"^\s*use\s+((?:\w+)(?:::\w+)*)", re.MULTILINE)
_RUST_EXTERN_RE = re.compile(r"^\s*extern\s+crate\s+(\w+)", re.MULTILINE)

# Go: import "pkg" and parenthesized blocks
_GO_IMPORT_SINGLE = re.compile(r'import\s+"([^"]+)"')
_GO_IMPORT_GROUP = re.compile(r'^\s+"([^"]+)"', re.MULTILINE)

# Shell: source X, . X, bash X, zsh X, sh X
_SHELL_RE = re.compile(
    r"(?:"
    r"^(?:source|\.)\s+([^\s#]+)"
    r"|^(?:bash|zsh|sh)\s+([^\s#]+)"
    r")",
    re.MULTILINE,
)


# ── Core functions ──────────────────────────────────────────────────

def load_scan_output(path: str) -> dict:
    """Load and validate scan JSON from file path.

    Returns parsed dict.  Raises ``FileNotFoundError`` if the file does
    not exist and ``ValueError`` if the schema is missing required keys.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Scan output not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if "files" not in data:
        raise ValueError(
            "Invalid scan schema: missing required 'files' key"
        )

    return data


def iter_scanned_files(scan_data: dict) -> Iterator[tuple[str, str]]:
    """Yield (relative_path, language) pairs from the scan files array."""
    for entry in scan_data["files"]:
        yield entry["path"], entry["language"]


def extract_python_imports(source: str) -> list[str]:
    """Extract ``import X`` and ``from X import Y`` patterns.

    Returns list of top-level module names (e.g. ``import os.path`` → ``"os"``).
    """
    modules: list[str] = []
    for m in _PYTHON_RE.finditer(source):
        # group 1 = import X, group 2 = from X
        mod = m.group(1) or m.group(2)
        if mod:
            # Take only the first segment before any '.'
            modules.append(mod.split(".")[0])
    return modules


def extract_js_ts_imports(
    source: str,
    *,
    return_warnings: bool = False,
) -> list[str] | tuple[list[str], list[str]]:
    """Extract JS/TS import, require, and dynamic-import statements.

    Returns resolved module names.  When ``return_warnings=True``
    also returns a warnings list (e.g. for dynamic imports).
    """
    imports: list[str] = []
    warnings: list[str] = []
    for m in _JS_TS_RE.finditer(source):
        mod = m.group(1) or m.group(2) or m.group(3)
        if mod:
            imports.append(mod)
            # Detect dynamic imports
            if m.group(3) is not None:
                warnings.append("dynamic import detected")
    if return_warnings:
        return imports, warnings
    return imports


def extract_rust_imports(source: str) -> list[str]:
    """Extract ``use X::Y`` and ``extern crate X`` patterns.

    Returns top-level crate names (first segment before ``::``).
    """
    crates: list[str] = []
    for m in _RUST_USE_RE.finditer(source):
        crate_name = m.group(1).split("::")[0]
        crates.append(crate_name)
    for m in _RUST_EXTERN_RE.finditer(source):
        crates.append(m.group(1))
    return crates


def extract_go_imports(source: str) -> list[str]:
    """Extract ``import "pkg"`` and grouped ``import ( "pkg" )`` blocks.

    Returns package paths.
    """
    pkgs: list[str] = []
    # Single-line: import "pkg"
    for m in _GO_IMPORT_SINGLE.finditer(source):
        pkgs.append(m.group(1))
    # Grouped: lines inside import (…) — only lines that are indented with a string
    # Avoid double-counting single-line imports that also match the group pattern
    # by checking they are preceded by an import ( line (simplified: just dedup)
    for m in _GO_IMPORT_GROUP.finditer(source):
        pkg = m.group(1)
        # The single-line regex already catches `import "pkg"` on its own line,
        # but grouped imports also match the group pattern.  We use a set approach.
        pass
    # Use set-based union to avoid double-counting
    group_pkgs = [m.group(1) for m in _GO_IMPORT_GROUP.finditer(source)]
    seen: set[str] = set()
    result: list[str] = []
    for pkg in pkgs + group_pkgs:
        if pkg not in seen:
            seen.add(pkg)
            result.append(pkg)
    return result


def extract_shell_imports(source: str) -> list[str]:
    """Extract ``source <file>``, ``. <file>``, ``bash|zsh|sh <file>``.

    Returns sourced / invoked file paths.
    """
    paths: list[str] = []
    for m in _SHELL_RE.finditer(source):
        p = m.group(1) or m.group(2)
        if p:
            paths.append(p)
    return paths


# Language → extractor dispatch
_LANG_EXTRACTORS: dict[str, Callable] = {
    "python": extract_python_imports,
    "javascript": extract_js_ts_imports,
    "typescript": extract_js_ts_imports,
    "rust": extract_rust_imports,
    "go": extract_go_imports,
    "shell": extract_shell_imports,
    "bash": extract_shell_imports,
}


def extract_imports_for_file(
    path: str,
    language: str,
    scan_root: str,
) -> tuple[list[str], list[str]]:
    """Dispatch to language-specific extractor.

    Returns ``(imports, warnings)`` for the given file.
    Unsupported languages emit a warning and return empty imports.
    """
    extractor = _LANG_EXTRACTORS.get(language)
    if extractor is None:
        return [], [f"unsupported language: {language}"]

    try:
        full_path = os.path.join(scan_root, path)
        with open(full_path, "r", encoding="utf-8", errors="replace") as fh:
            source = fh.read()
    except OSError as exc:
        return [], [f"could not read file: {exc}"]

    # Some extractors support return_warnings; use kwargs if available
    result = extractor(source)
    if isinstance(result, tuple):
        return result
    return result, []


def build_import_map(scan_data: dict, scan_root: str) -> dict:
    """Orchestrate: iterate files, extract imports, assemble output dict.

    Follows the exact output schema:
        schema_version, source_scan, generated_at, files, totals
    """
    all_modules: set[str] = set()
    files_map: dict[str, dict] = {}
    files_with_imports = 0
    files_without_imports = 0
    total_warnings = 0

    for rel_path, language in iter_scanned_files(scan_data):
        imports, warnings = extract_imports_for_file(
            rel_path, language, scan_root
        )
        files_map[rel_path] = {
            "imports": imports,
            "warnings": warnings,
        }
        total_warnings += len(warnings)
        if imports:
            files_with_imports += 1
            for mod in imports:
                all_modules.add(mod)
        else:
            files_without_imports += 1

    return {
        "schema_version": "1.0.0",
        "source_scan": {
            "project_root": scan_data.get("project_root", scan_root),
            "total_files": scan_data.get("total_files", 0),
        },
        "generated_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "files": files_map,
        "totals": {
            "files_with_imports": files_with_imports,
            "files_without_imports": files_without_imports,
            "unique_modules": len(all_modules),
            "total_warnings": total_warnings,
        },
    }


def main() -> int:
    """CLI entry point. Parse args, read scan JSON, write import map to stdout."""
    parser = argparse.ArgumentParser(
        description="Extract import/dependency map from a code-scan JSON output."
    )
    parser.add_argument(
        "scan_output",
        help="Path to the scan_project.py JSON output file",
    )
    args = parser.parse_args()

    try:
        scan_data = load_scan_output(args.scan_output)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    scan_root = scan_data.get("project_root", os.getcwd())
    import_map = build_import_map(scan_data, scan_root)
    print(json.dumps(import_map, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
