"""Aider-style repository map — extracts key symbols from source files.

Produces a concise overview of a repo's structure (classes, functions,
methods, structs, etc.) to include in the system prompt so the model
understands the codebase layout without reading every file.

Uses tree-sitter-languages when available, falls back to regex parsing.
"""

import hashlib
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IGNORE_DIRS: Set[str] = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".eggs",
    "egg-info", ".nox", "target", "vendor", ".next", ".nuxt",
}

# Extension -> language name mapping
EXT_TO_LANG: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".php": "php",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".r": "r",
    ".R": "r",
    ".dart": "dart",
    ".zig": "zig",
}

# Max file size to parse (skip huge generated files)
MAX_FILE_SIZE = 100_000  # 100KB

# ---------------------------------------------------------------------------
# Tree-sitter support (optional)
# ---------------------------------------------------------------------------

_ts_available: Optional[bool] = None


def _check_tree_sitter() -> bool:
    """Check if tree-sitter-languages is usable."""
    global _ts_available
    if _ts_available is not None:
        return _ts_available
    try:
        from tree_sitter_languages import get_parser
        # Test that it actually works (version compat issues are common)
        p = get_parser("python")
        tree = p.parse(b"def foo(): pass")
        _ts_available = tree.root_node is not None
    except Exception:
        _ts_available = False
    return _ts_available


def _ts_parse_python(source: bytes, lang_name: str) -> List[str]:
    """Extract symbols from Python using tree-sitter."""
    from tree_sitter_languages import get_parser
    parser = get_parser("python")
    tree = parser.parse(source)
    lines: List[str] = []
    _ts_walk_python(tree.root_node, source, lines, indent=0)
    return lines


def _ts_walk_python(node, source: bytes, lines: List[str], indent: int):
    """Recursively walk Python AST nodes."""
    for child in node.children:
        if child.type == "class_definition":
            name_node = child.child_by_field_name("name")
            superclasses = child.child_by_field_name("superclasses")
            name = _node_text(name_node, source) if name_node else "?"
            supers = f"({_node_text(superclasses, source)})" if superclasses else ""
            lines.append(f"{'  ' * (indent + 1)}class {name}{supers}:")
            _ts_walk_python(child, source, lines, indent + 1)
        elif child.type == "function_definition":
            name_node = child.child_by_field_name("name")
            params_node = child.child_by_field_name("parameters")
            ret_node = child.child_by_field_name("return_type")
            name = _node_text(name_node, source) if name_node else "?"
            params = _node_text(params_node, source) if params_node else "()"
            ret = f" -> {_node_text(ret_node, source)}" if ret_node else ""
            lines.append(f"{'  ' * (indent + 1)}def {name}{params}{ret}")
        elif child.type == "decorated_definition":
            _ts_walk_python(child, source, lines, indent)


def _node_text(node, source: bytes) -> str:
    """Get text content of a tree-sitter node."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Regex-based fallback parsers (always available)
# ---------------------------------------------------------------------------

def _regex_parse_python(source: str) -> List[str]:
    """Extract Python symbols using regex."""
    lines: List[str] = []
    current_class_indent = -1

    for line in source.splitlines():
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Track when we leave a class
        if current_class_indent >= 0 and indent <= current_class_indent and stripped:
            current_class_indent = -1

        # Class definition
        m = re.match(r'^(class\s+\w+(?:\([^)]*\))?)\s*:', stripped)
        if m:
            current_class_indent = indent
            level = 1 if indent == 0 else 2
            lines.append(f"{'  ' * level}{m.group(1)}:")
            continue

        # Function/method definition
        m = re.match(r'^((?:async\s+)?def\s+\w+\s*\([^)]*\)(?:\s*->\s*\S+)?)', stripped)
        if m:
            sig = m.group(1)
            # Simplify long signatures
            if len(sig) > 80:
                sig = re.sub(r'\([^)]*\)', '(...)', sig, count=1)
            if current_class_indent >= 0 and indent > current_class_indent:
                lines.append(f"    {sig}")
            else:
                lines.append(f"  {sig}")
            continue

    return lines


def _regex_parse_javascript(source: str) -> List[str]:
    """Extract JavaScript/TypeScript symbols using regex."""
    lines: List[str] = []

    for line in source.splitlines():
        stripped = line.lstrip()

        # Class
        m = re.match(r'^(?:export\s+)?(?:default\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{?', stripped)
        if m:
            lines.append(f"  class {m.group(1)}:")
            continue

        # Function declaration
        m = re.match(r'^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)\s*\(', stripped)
        if m:
            lines.append(f"  function {m.group(1)}()")
            continue

        # Arrow / const function
        m = re.match(r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(?', stripped)
        if m and ('=>' in stripped or 'function' in stripped):
            lines.append(f"  const {m.group(1)} = ...")
            continue

        # Method in class
        m = re.match(r'^(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{', stripped)
        if m and m.group(1) not in ('if', 'for', 'while', 'switch', 'catch'):
            lines.append(f"    {m.group(1)}()")
            continue

        # Interface / type (TypeScript)
        m = re.match(r'^(?:export\s+)?(?:interface|type)\s+(\w+)', stripped)
        if m:
            lines.append(f"  {m.group(0).strip().split('{')[0].strip()}")
            continue

    return lines


def _regex_parse_go(source: str) -> List[str]:
    """Extract Go symbols using regex."""
    lines: List[str] = []

    for line in source.splitlines():
        stripped = line.lstrip()

        # Function
        m = re.match(r'^func\s+(\([^)]+\)\s+)?(\w+)\s*\(', stripped)
        if m:
            receiver = m.group(1) or ""
            name = m.group(2)
            if receiver:
                lines.append(f"  func {receiver.strip()} {name}()")
            else:
                lines.append(f"  func {name}()")
            continue

        # Struct
        m = re.match(r'^type\s+(\w+)\s+struct\s*\{', stripped)
        if m:
            lines.append(f"  type {m.group(1)} struct")
            continue

        # Interface
        m = re.match(r'^type\s+(\w+)\s+interface\s*\{', stripped)
        if m:
            lines.append(f"  type {m.group(1)} interface")
            continue

    return lines


def _regex_parse_rust(source: str) -> List[str]:
    """Extract Rust symbols using regex."""
    lines: List[str] = []

    for line in source.splitlines():
        stripped = line.lstrip()

        # Function
        m = re.match(r'^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)', stripped)
        if m:
            lines.append(f"  fn {m.group(1)}()")
            continue

        # Struct
        m = re.match(r'^(?:pub\s+)?struct\s+(\w+)', stripped)
        if m:
            lines.append(f"  struct {m.group(1)}")
            continue

        # Enum
        m = re.match(r'^(?:pub\s+)?enum\s+(\w+)', stripped)
        if m:
            lines.append(f"  enum {m.group(1)}")
            continue

        # Impl block
        m = re.match(r'^impl(?:<[^>]+>)?\s+(?:(\w+)\s+for\s+)?(\w+)', stripped)
        if m:
            trait = m.group(1)
            target = m.group(2)
            if trait:
                lines.append(f"  impl {trait} for {target}")
            else:
                lines.append(f"  impl {target}")
            continue

        # Trait
        m = re.match(r'^(?:pub\s+)?trait\s+(\w+)', stripped)
        if m:
            lines.append(f"  trait {m.group(1)}")
            continue

    return lines


def _regex_parse_generic(source: str) -> List[str]:
    """Best-effort symbol extraction for unknown languages."""
    lines: List[str] = []

    for line in source.splitlines():
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if indent > 4:
            continue  # skip deeply nested

        # Class-like
        m = re.match(r'^(?:(?:pub|public|export|abstract)\s+)*(?:class|struct|interface|enum|trait|type)\s+(\w+)', stripped)
        if m:
            lines.append(f"  {stripped.split('{')[0].strip()}")
            continue

        # Function-like
        m = re.match(r'^(?:(?:pub|public|export|static|async|private|protected|virtual|override)\s+)*(?:fn|func|def|function|fun|sub)\s+(\w+)', stripped)
        if m:
            lines.append(f"  {stripped.split('{')[0].split(':')[0].strip()}")
            continue

    return lines


_REGEX_PARSERS = {
    "python": _regex_parse_python,
    "javascript": _regex_parse_javascript,
    "typescript": _regex_parse_javascript,
    "go": _regex_parse_go,
    "rust": _regex_parse_rust,
}

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _iter_source_files(root: Path) -> List[Path]:
    """Walk repo and yield parseable source files."""
    files: List[Path] = []
    try:
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            # Prune ignored directories in-place
            dirnames[:] = [
                d for d in dirnames
                if d not in IGNORE_DIRS and not d.startswith(".")
            ]
            for fname in filenames:
                ext = os.path.splitext(fname)[1]
                if ext in EXT_TO_LANG:
                    fpath = Path(dirpath) / fname
                    try:
                        if fpath.stat().st_size <= MAX_FILE_SIZE:
                            files.append(fpath)
                    except OSError:
                        pass
    except OSError:
        pass
    return files


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

_file_cache: Dict[str, Tuple[float, List[str]]] = {}  # path -> (mtime, symbols)


def _parse_file(fpath: Path, lang: str) -> List[str]:
    """Parse a single file and return symbol lines. Uses cache."""
    key = str(fpath)
    try:
        mtime = fpath.stat().st_mtime
    except OSError:
        return []

    cached = _file_cache.get(key)
    if cached and cached[0] == mtime:
        return cached[1]

    try:
        source = fpath.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return []

    # Try tree-sitter first for Python (extend as needed)
    symbols: List[str] = []
    if lang == "python" and _check_tree_sitter():
        try:
            symbols = _ts_parse_python(source.encode("utf-8"), lang)
        except Exception:
            symbols = []

    # Regex fallback
    if not symbols:
        parser = _REGEX_PARSERS.get(lang, _regex_parse_generic)
        symbols = parser(source)

    _file_cache[key] = (mtime, symbols)
    return symbols


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def _get_recently_modified(root: Path, limit: int = 50) -> List[str]:
    """Get recently modified files from git log."""
    try:
        result = subprocess.run(
            ["git", "log", "--all", "--pretty=format:", "--name-only",
             "--diff-filter=ACMR", f"-{limit * 2}"],
            cwd=root, capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return []
        seen = set()
        recent = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                recent.append(line)
                if len(recent) >= limit:
                    break
        return recent
    except Exception:
        return []


def _count_references(files: List[Path], root: Path) -> Dict[str, int]:
    """Count cross-file references (imports / symbol mentions)."""
    # Collect all symbol names across files
    all_symbols: Set[str] = set()
    file_symbols: Dict[str, Set[str]] = {}

    for fpath in files:
        rel = str(fpath.relative_to(root))
        symbols = set()
        lang = EXT_TO_LANG.get(fpath.suffix, "")
        cached = _file_cache.get(str(fpath))
        if cached:
            for line in cached[1]:
                # Extract identifier from symbol lines
                m = re.search(r'(?:class|def|func|fn|function|struct|enum|trait|impl|type|const|interface)\s+(\w+)', line)
                if m:
                    symbols.add(m.group(1))
        file_symbols[rel] = symbols
        all_symbols.update(symbols)

    # Count how many other files reference symbols from each file
    ref_counts: Dict[str, int] = {str(f.relative_to(root)): 0 for f in files}

    for fpath in files:
        rel = str(fpath.relative_to(root))
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for other_rel, syms in file_symbols.items():
            if other_rel == rel:
                continue
            for sym in syms:
                if sym in content:
                    ref_counts[other_rel] = ref_counts.get(other_rel, 0) + 1
                    break  # One reference per file pair is enough

    return ref_counts


def _rank_files(
    files: List[Path],
    root: Path,
    focus_files: Optional[List[str]] = None,
) -> List[Tuple[Path, float]]:
    """Rank files by relevance. Returns [(path, score)] sorted descending."""
    recent = _get_recently_modified(root)
    recent_set = set(recent)
    recent_rank = {f: i for i, f in enumerate(recent)}

    ref_counts = _count_references(files, root)

    focus_set: Set[str] = set()
    if focus_files:
        for f in focus_files:
            focus_set.add(f)
            # Also match basename
            focus_set.add(os.path.basename(f))

    scored: List[Tuple[Path, float]] = []
    for fpath in files:
        try:
            rel = str(fpath.relative_to(root))
        except ValueError:
            continue

        score = 1.0

        # Focus files get highest priority
        if rel in focus_set or fpath.name in focus_set:
            score += 100.0

        # Recently modified files
        if rel in recent_set:
            rank = recent_rank.get(rel, 50)
            score += max(0, 20.0 - rank * 0.4)

        # Cross-references
        refs = ref_counts.get(rel, 0)
        score += min(refs * 2.0, 15.0)

        # Prefer shorter paths (top-level files)
        depth = rel.count("/")
        score -= depth * 0.5

        scored.append((fpath, score))

    scored.sort(key=lambda x: -x[1])
    return scored


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_repo_map(
    root_dir: str,
    max_tokens: int = 4000,
    focus_files: Optional[List[str]] = None,
) -> str:
    """Build a concise repository map showing key symbols per file.

    Parameters
    ----------
    root_dir : str
        Path to the repository root.
    max_tokens : int
        Approximate token budget (uses chars/4 heuristic).
    focus_files : list[str] | None
        Files to prioritize in the map (e.g., currently open files).

    Returns
    -------
    str
        The formatted repo map, or empty string on any failure.
    """
    try:
        return _build_repo_map_impl(root_dir, max_tokens, focus_files)
    except Exception as e:
        logger.debug("Failed to build repo map: %s", e)
        return ""


def _build_repo_map_impl(
    root_dir: str,
    max_tokens: int,
    focus_files: Optional[List[str]],
) -> str:
    root = Path(root_dir).resolve()
    if not root.is_dir():
        return ""

    max_chars = max_tokens * 4

    # Discover source files
    files = _iter_source_files(root)
    if not files:
        return ""

    # Pre-parse all files (fills cache)
    for fpath in files:
        lang = EXT_TO_LANG.get(fpath.suffix, "generic")
        _parse_file(fpath, lang)

    # Rank files
    ranked = _rank_files(files, root, focus_files)

    # Build map within token budget
    map_lines: List[str] = []
    total_chars = 0

    for fpath, score in ranked:
        rel = str(fpath.relative_to(root))
        lang = EXT_TO_LANG.get(fpath.suffix, "generic")
        symbols = _parse_file(fpath, lang)

        if not symbols:
            continue

        # Build file entry
        entry_lines = [f"{rel}:"] + symbols
        entry_text = "\n".join(entry_lines) + "\n"
        entry_chars = len(entry_text)

        if total_chars + entry_chars > max_chars:
            # Try adding just the file header
            header = f"{rel}: ({len(symbols)} symbols)\n"
            if total_chars + len(header) <= max_chars:
                map_lines.append(header)
                total_chars += len(header)
            break

        map_lines.append(entry_text)
        total_chars += entry_chars

    if not map_lines:
        return ""

    return "".join(map_lines).rstrip("\n")


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    result = build_repo_map(root, max_tokens=tokens)
    if result:
        print(result)
    else:
        print("(empty repo map)", file=sys.stderr)
