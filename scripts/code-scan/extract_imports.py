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
from pathlib import Path
from typing import Callable, Iterator

# ── Extension inference ──────────────────────────────────────────────

_JS_TS_EXTENSIONS_ORDER = [".js", ".jsx", ".ts", ".tsx"]

_EXTENSION_PRIORITY = {
    "javascript": [".js", ".jsx", ".ts", ".tsx"],
    "typescript": [".ts", ".tsx", ".js", ".jsx"],
    "js": [".js", ".jsx", ".ts", ".tsx"],
    "jsx": [".jsx", ".js", ".ts", ".tsx"],
    "tsx": [".tsx", ".ts", ".jsx", ".js"],
    "ts": [".ts", ".tsx", ".js", ".jsx"],
}

_KNOWN_JS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx"}


def _infer_extensions(source_language: str) -> list[str]:
    """Return ordered list of extensions to try when resolving bare imports.

    Prioritises extensions matching the source language family (JS prefers
    .js/.jsx, TS prefers .ts/.tsx), then falls back to the full set.
    """
    lang = source_language.lower()
    return _EXTENSION_PRIORITY.get(lang, list(_JS_TS_EXTENSIONS_ORDER))


def _is_index_candidate(import_path: str) -> bool:
    """Check whether an import path is a bare directory-style import.

    Returns True for paths like ``./utils`` or ``../helpers`` that lack
    an explicit file extension, indicating the resolver should look for
    an index file inside that directory.
    """
    if not (import_path.startswith("./") or import_path.startswith("../")):
        return False
    ext = Path(import_path).suffix
    return ext not in _KNOWN_JS_EXTENSIONS


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


# ── JS/TS import resolution helpers ─────────────────────────────────────

def _probe_with_extensions(base_path: Path, extensions: list[str]) -> Path | None:
    """Try *base_path* with each extension in order, returning the first hit."""
    for ext in extensions:
        candidate = base_path.with_suffix(base_path.suffix + ext)
        # If base has no suffix yet (e.g., "./utils"), with_suffix adds ext directly
        if candidate.exists():
            return candidate
    return None


def _probe_index(directory: Path, extensions: list[str]) -> Path | None:
    """Look for index.{ext} inside *directory* with given extensions."""
    if not directory.is_dir():
        return None
    for ext in extensions:
        candidate = directory / f"index{ext}"
        if candidate.is_file():
            return candidate
    return None


def resolve_js_relative_import(
    import_path: str,
    source_file_path: str,
    project_root: str | Path,
) -> dict | None:
    """Resolve a relative JS/TS import to an actual file on disk.

    Handles:
    - Extension inference (e.g. ``./Widget`` → ``./Widget.jsx``)
    - Index file lookup (e.g. ``./utils`` → ``./utils/index.ts``)
    - Direct matches (e.g. ``./Widget.tsx`` when that file exists)

    Args:
        import_path: The raw import string (e.g. ``"./components/Widget"``).
        source_file_path: Absolute path of the file containing the import.
        project_root: The project root directory (string or Path).

    Returns:
        ``{"resolved_path": <str>, "strategy": <str>}`` on success,
        ``None`` if the import cannot be resolved.
    """
    if not (import_path.startswith("./") or import_path.startswith("../")):
        return None

    root = Path(project_root).resolve()
    source = Path(source_file_path).resolve()

    # Determine source file's language from its extension
    source_ext = source.suffix.lower()
    source_lang = {
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".ts": "ts",
        ".js": "js",
    }.get(source_ext, "javascript")

    exts = _infer_extensions(source_lang)

    # Resolve the relative import against the source file's directory
    source_dir = source.parent
    candidate = Path(import_path)

    # Normalise: ./foo → foo, ../bar → ../bar
    if import_path.startswith("./"):
        candidate = source_dir / candidate.relative_to("./".strip())
    else:
        # For ../ paths, just join and normalise
        candidate = (source_dir / import_path).resolve()

    # 1. Direct match: if candidate already has an extension, just check existence
    if source_ext in _KNOWN_JS_EXTENSIONS:
        # Import already has a JS/TS extension
        if candidate.suffix.lower() in _KNOWN_JS_EXTENSIONS:
            if candidate.is_file():
                try:
                    rel = candidate.relative_to(root)
                    return {"resolved_path": str(rel), "strategy": "direct"}
                except ValueError:
                    return {"resolved_path": str(candidate), "strategy": "direct"}
            return None

    # 2. Try extension inference on the bare path
    # First, check if the path without any suffix matches a file with an extension
    bare = candidate
    if candidate.suffix:
        bare = Path(str(candidate)[:-len(candidate.suffix)])

    result = _probe_with_extensions(bare, exts)
    if result is not None:
        try:
            rel = result.relative_to(root)
            return {"resolved_path": str(rel), "strategy": "extension_inference"}
        except ValueError:
            return {"resolved_path": str(result), "strategy": "extension_inference"}

    # 3. Try index file resolution
    if _is_index_candidate(import_path):
        if candidate.is_dir():
            index_result = _probe_index(candidate, exts)
            if index_result is not None:
                try:
                    rel = index_result.relative_to(root)
                    return {"resolved_path": str(rel), "strategy": "index_file"}
                except ValueError:
                    return {"resolved_path": str(index_result), "strategy": "index_file"}

    return None


def _resolve_alias_base(import_path: str, aliases: dict[str, str], project_root: Path) -> tuple[Path | None, str]:
    """Resolve an alias-prefixed import to a base path on disk.

    Returns (resolved_base_path, strategy_hint).
    """
    for alias, target in sorted(aliases.items(), key=lambda x: -len(x[0])):
        if alias.endswith("/*"):
            prefix = alias[:-2]
            if import_path.startswith(prefix + "/") or import_path == prefix + "/":
                remainder = import_path[len(prefix) + 1:]
                target_base = target[:-2] if target.endswith("/*") else target
                base = project_root / target_base / remainder
                return base, "alias_wildcard"
        else:
            if import_path == alias or import_path.startswith(alias + "/"):
                remainder = import_path[len(alias):].lstrip("/")
                base = project_root / target
                if remainder:
                    base = base / remainder
                return base, "alias_exact"
    return None, ""


def _resolve_base_with_extensions(base: Path, extensions: list[str]) -> Path | None:
    """Try *base* directly, then with extensions appended, then as a directory with index."""
    # 1. Direct (already has extension)
    if base.suffix.lower() in _KNOWN_JS_EXTENSIONS and base.is_file():
        return base

    # 2. Extension inference
    if base.suffix:
        bare = Path(str(base)[: -len(base.suffix)])
        result = _probe_with_extensions(bare, extensions)
        if result is not None:
            return result
    else:
        result = _probe_with_extensions(base, extensions)
        if result is not None:
            return result

    # 3. Index file
    if base.is_dir():
        idx = _probe_index(base, extensions)
        if idx is not None:
            return idx

    return None


def resolve_alias_import(
    import_path: str,
    aliases: dict[str, str],
    project_root: str | Path,
) -> dict | None:
    """Resolve an aliased JS/TS import (e.g. ``@/lib/api``) to a file path.

    Args:
        import_path: The aliased import string (e.g. ``"@/lib/api"``).
        aliases: Mapping from alias prefix to relative path (e.g. ``{"@": "src"}``).
        project_root: The project root directory.

    Returns:
        ``{"resolved_path": <str>, "strategy": <str>}`` on success,
        ``None`` if unresolvable.
    """
    root = Path(project_root).resolve()
    base, kind = _resolve_alias_base(import_path, aliases, root)
    if base is None:
        return None

    # Try to resolve with extension inference + index lookup
    result = _resolve_base_with_extensions(base, _JS_TS_EXTENSIONS_ORDER)
    if result is not None:
        try:
            rel = result.relative_to(root)
            strategy = "alias_wildcard_index_file" if "index" in (kind or "") else f"alias_with_extension"
            if result.name == "index" + result.suffix:
                strategy = "alias_index_file"
            return {"resolved_path": str(rel), "strategy": strategy}
        except ValueError:
            return {"resolved_path": str(result), "strategy": "alias_with_extension"}
    return None


# ── Config-based alias extraction ─────────────────────────────────────

def _parse_tsconfig_paths(tsconfig_path: str | Path) -> dict[str, str]:
    """Parse ``paths`` from tsconfig.json and return an alias map.

    Only handles statically parseable JSON configs — no JSON5, no comments,
    no extends resolution.  Returns empty dict on any parse error.
    """
    try:
        with open(str(tsconfig_path), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        paths = cfg.get("compilerOptions", {}).get("paths", {})
        result = {}
        for alias, targets in paths.items():
            if targets:
                # Take the first target; strip trailing "/*" from both sides
                target = targets[0]
                if alias.endswith("/*"):
                    if target.endswith("/*"):
                        target = target[:-2]
                result[alias] = target
        return result
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return {}


def _parse_jsconfig_paths(jsconfig_path: str | Path) -> dict[str, str]:
    """Parse ``paths`` from jsconfig.json (same format as tsconfig)."""
    return _parse_tsconfig_paths(jsconfig_path)


def _parse_vite_aliases(vite_config_path: str | Path) -> dict[str, str]:
    """Statically parse ``resolve.alias`` from vite.config.* files.

    Uses regex-based extraction — does NOT execute the config.
    Supports simple object and array-of-{find,replacement} forms.
    """
    try:
        with open(str(vite_config_path), "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return {}

    aliases: dict[str, str] = {}

    # Pattern: alias: { '@': 'src', ... }
    obj_match = re.search(r"alias\s*:\s*\{([^}]+)\}", content, re.DOTALL)
    if obj_match:
        body = obj_match.group(1)
        for m in re.finditer(
            r"""['"]([^'"]+)['"]\s*:\s*['"]([^'"]+)['"]""", body
        ):
            aliases[m.group(1)] = m.group(2)

    # Pattern: alias: [{ find: '@', replacement: 'src' }, ...]
    for m in re.finditer(
        r"find\s*:\s*['\"]([^'\"]+)['\"]\s*,\s*replacement\s*:\s*['\"]([^'\"]+)['\"]",
        content,
    ):
        aliases[m.group(1)] = m.group(2)

    return aliases


def _discover_js_aliases(project_root: str | Path) -> dict[str, str]:
    """Discover statically parseable JS/TS aliases from common config files.

    This is intentionally best-effort and stdlib-only: malformed or dynamic
    configs simply contribute no aliases.
    """
    root = Path(project_root)
    aliases: dict[str, str] = {}
    for config_name, parser in (
        ("tsconfig.json", _parse_tsconfig_paths),
        ("jsconfig.json", _parse_jsconfig_paths),
    ):
        config_path = root / config_name
        if config_path.is_file():
            aliases.update(parser(config_path))

    for pattern in ("vite.config.js", "vite.config.ts", "vite.config.mjs", "vite.config.cjs"):
        config_path = root / pattern
        if config_path.is_file():
            aliases.update(_parse_vite_aliases(config_path))
    return aliases


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
    js_aliases = _discover_js_aliases(scan_root)

    for rel_path, language in iter_scanned_files(scan_data):
        imports, warnings = extract_imports_for_file(
            rel_path, language, scan_root
        )
        files_map[rel_path] = {
            "imports": imports,
            "warnings": warnings,
        }
        # For JS/TS-like files, add resolved dict by wiring existing resolver (only
        # for relative imports that succeed; non-JS/TS and unresolved stay bare).
        lang = language.lower()
        if lang in {"javascript", "typescript", "js", "ts", "jsx", "tsx"}:
            resolved: dict[str, dict] = {}
            for mod in imports:
                full_source = os.path.join(scan_root, rel_path)
                res = resolve_js_relative_import(mod, full_source, scan_root)
                if res is None and js_aliases:
                    res = resolve_alias_import(mod, js_aliases, scan_root)
                if res is not None:
                    resolved[mod] = res
            files_map[rel_path]["resolved"] = resolved
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
