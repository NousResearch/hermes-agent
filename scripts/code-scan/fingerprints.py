"""Phase 3 D1 — Fingerprint extraction, comparison, and persistence module.

Stdlib only. Extracts per-file fingerprints (content hash, structural metadata),
persists to JSON, and compares fingerprint sets to classify changes as
UNCHANGED / COSMETIC / STRUCTURAL.
"""
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── Regex patterns ──────────────────────────────────────────────────────────

_PY_FUNC_RE = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE)
_PY_CLASS_RE = re.compile(r"^\s*class\s+(\w+)", re.MULTILINE)
_JS_FUNC_RE = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(", re.MULTILINE
)
_JS_CLASS_RE = re.compile(r"^\s*(?:export\s+)?class\s+(\w+)", re.MULTILINE)

# Import extraction patterns
_PY_IMPORT_FROM_RE = re.compile(r"^from\s+([\w.]+)\s+import", re.MULTILINE)
_PY_IMPORT_RE = re.compile(r"^import\s+([\w.]+)", re.MULTILINE)
_JS_IMPORT_RE = re.compile(r"""^import\s+.+?\s+from\s+['"]([^'"]+)['"]""", re.MULTILINE)
_JS_REQUIRE_RE = re.compile(r"""require\(\s*['"]([^'"]+)['"]\s*\)""")
_JS_DYNAMIC_IMPORT_RE = re.compile(r"""(?:import|await\s+import)\(\s*['"]([^'"]+)['"]\s*\)""")

_REQUIRED_FILE_KEYS = [
    "content_hash", "line_count", "size_bytes",
    "functions", "classes", "imports",
]


# ── Structural extraction helpers ───────────────────────────────────────────

def _extract_functions(source: str) -> list[str]:
    """Extract sorted, deduplicated function names from source code."""
    names = _PY_FUNC_RE.findall(source)
    return sorted(set(names))


def _extract_classes(source: str) -> list[str]:
    """Extract sorted, deduplicated class names from source code."""
    names = _PY_CLASS_RE.findall(source)
    return sorted(set(names))


def _extract_content_hash(file_path: str) -> str:
    """Compute SHA-256 hash of raw file bytes. Returns 'sha256:<hex>'."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _extract_imports_from_source(source: str, language: str) -> list[str]:
    """Extract sorted, deduplicated import module names.

    For Python and JS/TS uses regex matching.  For unknown languages returns
    an empty list with a stderr warning.
    """
    lang = language.lower()
    modules: set[str] = set()

    if lang in ("python", "py"):
        for m in _PY_IMPORT_RE.findall(source):
            modules.add(m.split(".")[0])
        for m in _PY_IMPORT_FROM_RE.findall(source):
            modules.add(m.split(".")[0])
    elif lang in ("javascript", "typescript", "js", "ts"):
        for m in _JS_IMPORT_RE.findall(source):
            # Strip leading ./ or ../ and .js/.ts extensions
            mod = m.lstrip("./")
            mod = re.sub(r"\.(js|ts|tsx|jsx)$", "", mod)
            modules.add(mod)
        for m in _JS_REQUIRE_RE.findall(source):
            mod = m.lstrip("./")
            mod = re.sub(r"\.(js|ts|tsx|jsx)$", "", mod)
            modules.add(mod)
        for m in _JS_DYNAMIC_IMPORT_RE.findall(source):
            mod = m.lstrip("./")
            mod = re.sub(r"\.(js|ts|tsx|jsx)$", "", mod)
            modules.add(mod)
    else:
        print(
            f"[fingerprints] warning: no import extraction for language '{language}'",
            file=sys.stderr,
        )

    return sorted(modules)


def _detect_language(file_path: str) -> str:
    """Return a language tag based on filename extension."""
    ext = os.path.splitext(file_path)[1].lower()
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
    }
    return mapping.get(ext, "unknown")


# ── Public API ──────────────────────────────────────────────────────────────

def extract_fingerprint(
    file_path: str,
    scan_root: str,  # noqa: ARG001 — kept for API stability
    line_count: int,
    size_bytes: int,
    imports: Optional[list[str]] = None,
) -> dict:
    """Extract a fingerprint dict for a single file.

    Returns dict with keys: content_hash, line_count, size_bytes, functions,
    classes, imports.
    """
    content_hash = _extract_content_hash(file_path)

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        source = f.read()

    functions = _extract_functions(source)
    classes = _extract_classes(source)

    if imports is not None:
        final_imports = sorted(set(imports))
    else:
        language = _detect_language(file_path)
        final_imports = _extract_imports_from_source(source, language)

    return {
        "content_hash": content_hash,
        "line_count": line_count,
        "size_bytes": size_bytes,
        "functions": functions,
        "classes": classes,
        "imports": final_imports,
    }


def load_fingerprint_file(fingerprints_path: str) -> Optional[dict]:
    """Load a fingerprints JSON file.

    Returns None if the file is missing, corrupt, or fails schema validation.
    Validates schema_version == '1.0.0' and presence of required top-level keys.
    """
    try:
        with open(fingerprints_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    # Schema validation
    if not isinstance(data, dict):
        return None
    if data.get("schema_version") != "1.0.0":
        return None
    for key in ("schema_version", "project_root", "captured_at", "files"):
        if key not in data:
            return None

    return data


def save_fingerprint_file(
    fingerprints_path: str,
    project_root: str,
    files: dict[str, dict],
) -> str:
    """Write fingerprint JSON file, creating parent directories if needed.

    Schema: {"schema_version": "1.0.0", "project_root": <abs>,
             "captured_at": <ISO 8601>, "files": <files>}
    Returns the path written.
    """
    os.makedirs(os.path.dirname(fingerprints_path), exist_ok=True)

    payload = {
        "schema_version": "1.0.0",
        "project_root": os.path.abspath(project_root),
        "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": files,
    }

    with open(fingerprints_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return fingerprints_path


def compare_fingerprints(
    old_fp: dict, new_fp: dict
) -> dict[str, str]:
    """Compare old and new fingerprint sets.

    Returns {relative_path: 'UNCHANGED' | 'COSMETIC' | 'STRUCTURAL'}

    Rules:
    - UNCHANGED: exact content_hash match
    - COSMETIC: hash differs but functions/classes/imports lists are identical
    - STRUCTURAL: structural lists differ or file is new/deleted
    """
    old_files = old_fp.get("files", {})
    new_files = new_fp.get("files", {})
    all_paths = set(old_files.keys()) | set(new_files.keys())

    result: dict[str, str] = {}

    for path in all_paths:
        if path not in old_files:
            result[path] = "STRUCTURAL"  # new file
            continue
        if path not in new_files:
            result[path] = "STRUCTURAL"  # deleted file
            continue

        old_rec = old_files[path]
        new_rec = new_files[path]

        # UNCHANGED: exact hash match
        if old_rec.get("content_hash") == new_rec.get("content_hash"):
            result[path] = "UNCHANGED"
            continue

        # Compare structural lists (sorted for safe equality)
        old_funcs = sorted(old_rec.get("functions", []))
        new_funcs = sorted(new_rec.get("functions", []))
        old_classes = sorted(old_rec.get("classes", []))
        new_classes = sorted(new_rec.get("classes", []))
        old_imports = sorted(old_rec.get("imports", []))
        new_imports = sorted(new_rec.get("imports", []))

        if (old_funcs == new_funcs
                and old_classes == new_classes
                and old_imports == new_imports):
            result[path] = "COSMETIC"
        else:
            result[path] = "STRUCTURAL"

    return result


def build_fingerprint_map(
    scan_data: dict,
    project_root: str,
    import_data: Optional[dict] = None,
) -> dict[str, dict]:
    """Build a complete fingerprint map from scan data.

    Iterates scan_data['files'] and extracts a fingerprint for each file.
    Optionally enriches with pre-extracted imports from import_data.
    Returns {relative_path: fingerprint_dict}.
    """
    result: dict[str, dict] = {}

    for entry in scan_data.get("files", []):
        rel_path = entry["relative_path"]
        abs_path = entry["path"]
        line_count = entry.get("lines", 0)
        size_bytes = entry.get("size_bytes", 0)

        imports: Optional[list[str]] = None
        if import_data:
            # Support real Phase 2 extract_imports.py nested schema:
            #   {"schema_version": "1.0.0", "files": {"src/main.py": {"imports": ["os"]}}}
            # Also retain support for a simple direct map:
            #   {"src/main.py": ["os"]}
            if "files" in import_data and isinstance(import_data.get("files"), dict):
                file_entry = import_data["files"].get(rel_path)
                if isinstance(file_entry, dict) and "imports" in file_entry:
                    imports = file_entry["imports"]
            elif rel_path in import_data:
                imports = import_data[rel_path]

        result[rel_path] = extract_fingerprint(
            file_path=abs_path,
            scan_root=project_root,
            line_count=line_count,
            size_bytes=size_bytes,
            imports=imports,
        )

    return result


def get_fingerprint_path(project_root: str, *, in_repo: bool = True,
                         external_dir: Optional[str] = None) -> str:
    """Return the fingerprints.json path.

    When *in_repo* is True (default), returns the legacy path inside the
    target repo at ``.hermes/code-state/fingerprints.json``.

    When *in_repo* is False, fingerprints are stored outside the target repo.
    If *external_dir* is provided it is used as the parent directory; otherwise
    the current working directory is used with a derived name based on the
    target root.
    """
    if in_repo:
        return os.path.join(project_root, ".hermes", "code-state",
                            "fingerprints.json")

    # External (non-mutating) path
    if external_dir is None:
        external_dir = os.getcwd()

    # Derive a safe directory name from the target root path
    safe_name = os.path.basename(os.path.normpath(project_root))
    cache_dir = os.path.join(external_dir, ".hermes-cache", safe_name)
    return os.path.join(cache_dir, "fingerprints.json")
