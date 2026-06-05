#!/usr/bin/env python3
"""Entrypoint detection for the UA Flywheel code-scan module.

Reads a scan.json artifact and produces structured entrypoint hints.
Does not wire into other scanners. JIT-only, read-only against target repos.
No execution of scanned code. No new dependencies.

Usage:
    python scripts/code-scan/detect_entrypoints.py <scan.json> > <entrypoints.json>

Exit codes: 0 = success, 1 = error.
"""
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SCHEMA_VERSION = "1.0.0"


def _normalize_rel_path(path: str) -> str:
    """Return a stable POSIX-like relative path for ranking heuristics."""
    return path.replace("\\", "/")


def _entrypoint_rank(ep: Dict[str, Any]) -> tuple[int, float, str]:
    """Sort likely app roots ahead of generic standalone entrypoints."""
    path = _normalize_rel_path(str(ep.get("file") or ""))
    lowered = path.lower()
    base = os.path.basename(lowered)

    if re.fullmatch(r"src/main\.(ts|tsx|js|jsx|mjs|cjs)", lowered):
        priority = 0
    elif re.fullmatch(r"src/app\.(ts|tsx|js|jsx)", lowered):
        priority = 1
    elif re.fullmatch(r"vite\.config\.(ts|js|mjs|mts|cts)", lowered):
        priority = 2
    elif re.fullmatch(r"supabase/functions/[^/]+/index\.(ts|js)", lowered):
        priority = 3
    elif path == "package.json":
        priority = 4
    elif base == "__main__.py":
        priority = 5
    elif base.startswith("main."):
        priority = 6
    elif base.startswith("app.") or base.startswith("index."):
        priority = 7
    else:
        priority = 8

    return (priority, -float(ep.get("confidence") or 0), path)


def _read_file_content(abs_path: str) -> Optional[str]:
    """Read file content safely. Returns None on any read error."""
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except (OSError, IOError, ValueError):
        return None


def _resolve_project_root(scan_data: dict) -> str:
    """Get the project root from scan data, or infer from file paths."""
    root = scan_data.get("project_root", "")
    if root and os.path.isdir(root):
        return root
    # Infer from first file path
    for f in scan_data.get("files", []):
        abs_path = f.get("path", "")
        rel_path = f.get("relative_path", "")
        if abs_path and rel_path:
            return abs_path.rpartition("/" + rel_path)[0]
    return ""


# ── Python entrypoint detection ──────────────────────────────────────────

def _detect_python_entrypoints(
    abs_path: str,
    rel_path: str,
    content: str,
) -> List[Dict[str, Any]]:
    """Detect Python entrypoints from file content."""
    entrypoints = []
    basename = os.path.basename(rel_path)
    signals = []

    has_name_main = bool(re.search(
        r'if\s+__name__\s*==\s*["\']__main__["\']', content
    ))
    has_def_main = bool(re.search(
        r'^\s*def\s+main\s*\(', content, re.MULTILINE
    ))
    has_argparse = "argparse" in content
    has_typer = "typer.Typer" in content or (
        "import typer" in content and "Typer()" in content
    )
    has_click = "click.command" in content or (
        "import click" in content and "@click" in content
    )
    has_uvicorn = "uvicorn" in content.lower()
    has_fastapi = "FastAPI()" in content
    is_main_py = basename == "__main__.py"

    # __main__.py is always an entrypoint candidate
    if is_main_py:
        signals.append("__main__.py")
        if has_def_main:
            signals.append("def main")
        if has_name_main:
            signals.append("if __name__ == '__main__'")
        confidence = 0.90
        if has_name_main and has_def_main:
            confidence = 0.95
        entrypoints.append({
            "file": rel_path,
            "language": "python",
            "type": "python_module",
            "signals": signals,
            "confidence": round(confidence, 2),
        })
        return entrypoints

    # Regular Python files: need multiple signals
    if has_def_main:
        signals.append("def main")
    if has_name_main:
        signals.append("if __name__ == '__main__'")
    if has_argparse:
        signals.append("argparse")
    if has_typer:
        signals.append("typer.Typer")
    if has_click:
        signals.append("click.command")
    if has_uvicorn:
        signals.append("uvicorn")
    if has_fastapi:
        signals.append("FastAPI()")

    if not signals:
        return entrypoints

    # Classify entrypoint type based on signals
    if has_name_main and has_def_main:
        if has_typer:
            ep_type = "python_typer"
        elif has_click:
            ep_type = "python_click"
        elif has_fastapi or has_uvicorn:
            ep_type = "python_fastapi"
        elif has_argparse:
            ep_type = "python_cli"
        else:
            ep_type = "python_cli"
    elif has_def_main and (has_typer or has_click or has_fastapi or has_uvicorn):
        if has_typer:
            ep_type = "python_typer"
        elif has_click:
            ep_type = "python_click"
        elif has_fastapi or has_uvicorn:
            ep_type = "python_fastapi"
        else:
            ep_type = "python_cli"
    elif has_name_main:
        ep_type = "python_module"
    elif has_def_main:
        # def main alone without __name__ guard — check it's really a main,
        # not a helper like main_helper or main_util
        # We already confirmed def main via regex above, so it's real
        if has_uvicorn or has_fastapi:
            ep_type = "python_fastapi"
        elif has_typer:
            ep_type = "python_typer"
        elif has_click:
            ep_type = "python_click"
        else:
            ep_type = "python_cli"
    elif has_uvicorn or has_fastapi:
        ep_type = "python_fastapi"
    elif has_typer or has_click:
        ep_type = "python_cli"
    elif has_argparse and has_name_main:
        ep_type = "python_cli"
    else:
        # We have some signals but not a clear entrypoint pattern
        # Give low confidence
        ep_type = "python_cli"

    # Calculate confidence
    confidence = 0.5
    if has_name_main:
        confidence += 0.15
    if has_def_main:
        confidence += 0.15
    if has_argparse or has_typer or has_click or has_fastapi:
        confidence += 0.1
    if has_uvicorn:
        confidence += 0.05
    if is_main_py:
        confidence += 0.1
    confidence = min(round(confidence, 2), 0.98)

    entrypoints.append({
        "file": rel_path,
        "language": "python",
        "type": ep_type,
        "signals": signals,
        "confidence": confidence,
    })
    return entrypoints


# ── JS/TS entrypoint detection ───────────────────────────────────────────

def _detect_js_ts_entrypoints(
    abs_path: str,
    rel_path: str,
    content: str,
    basename: str,
    scan_data: dict,
) -> List[Dict[str, Any]]:
    """Detect JavaScript/TypeScript entrypoints."""
    entrypoints = []
    signals = []
    norm_path = _normalize_rel_path(rel_path)
    lowered_path = norm_path.lower()
    basename_l = basename.lower()

    is_index = basename_l in {"index.js", "index.jsx", "index.ts", "index.tsx"}
    is_main = basename_l in {"main.js", "main.jsx", "main.ts", "main.tsx"}
    is_app = basename_l in {"app.js", "app.jsx", "app.ts", "app.tsx"}
    is_src_main = bool(re.fullmatch(r"src/main\.(ts|tsx|js|jsx|mjs|cjs)", lowered_path))
    is_src_app = bool(re.fullmatch(r"src/app\.(ts|tsx|js|jsx)", lowered_path))
    is_vite_config = bool(re.fullmatch(r"vite\.config\.(ts|js|mjs|mts|cts)", lowered_path))
    is_supabase_function = bool(re.fullmatch(r"supabase/functions/[^/]+/index\.(ts|js)", lowered_path))

    has_app_listen = bool(re.search(r'\.listen\s*\(', content))
    has_app_listen_explicit = "app.listen" in content
    has_react_create_root = "createRoot(" in content or "react-dom/client" in content
    has_vite_define_config = "defineConfig" in content and "vite" in content
    has_deno_serve = "Deno.serve" in content

    # Check if referenced in package.json
    pkg_main = None
    pkg_bin = None
    pkg_start = None
    for f in scan_data.get("files", []):
        if f.get("relative_path") == "package.json":
            pkg_path = f.get("path", "")
            pkg_content = _read_file_content(pkg_path)
            if pkg_content:
                try:
                    pkg = json.loads(pkg_content)
                    pkg_main = pkg.get("main")
                    pkg_bin = pkg.get("bin")
                    scripts = pkg.get("scripts", {})
                    if isinstance(scripts, dict):
                        pkg_start = scripts.get("start")
                except (json.JSONDecodeError, ValueError):
                    pass
            break

    # Check if this file is referenced in package.json
    is_pkg_referenced = False
    if pkg_main and (rel_path == pkg_main or abs_path.endswith(pkg_main)):
        is_pkg_referenced = True
        signals.append("package.json:main")
    if pkg_start and rel_path in pkg_start:
        is_pkg_referenced = True
        signals.append("package.json:scripts.start")

    if has_app_listen_explicit:
        signals.append("app.listen")
    elif has_app_listen:
        signals.append(".listen()")
    if is_src_main:
        signals.append("src/main app root")
        if has_react_create_root:
            signals.append("React createRoot")
    if is_src_app:
        signals.append("src/App app root")
    if is_vite_config:
        signals.append("vite config")
        if has_vite_define_config:
            signals.append("vite defineConfig")
    if is_supabase_function:
        signals.append("supabase function index")
        if has_deno_serve:
            signals.append("Deno.serve")

    # Entry file heuristic: index.js, main.ts, app.js in root/app/src
    if is_index and not signals:
        signals.append("index entry file")
    if is_main and not signals:
        signals.append("main entry file")
    if is_app and not signals:
        signals.append("app entry file")

    if not signals:
        return entrypoints

    # Determine type
    has_ts = rel_path.endswith(".ts") or rel_path.endswith(".tsx")
    if has_ts:
        ep_type = "typescript_app"
    else:
        ep_type = "javascript_app"

    # Confidence calculation
    confidence = 0.4
    if is_index or is_main or is_app:
        confidence += 0.15
    if has_app_listen_explicit:
        confidence += 0.25
    if is_pkg_referenced:
        confidence += 0.15
    if is_src_main:
        confidence += 0.25
    if is_src_app:
        confidence += 0.2
    if is_vite_config or is_supabase_function:
        confidence += 0.25
    confidence = min(round(confidence, 2), 0.98)

    entrypoints.append({
        "file": rel_path,
        "language": "typescript" if has_ts else "javascript",
        "type": ep_type,
        "signals": signals,
        "confidence": confidence,
    })
    return entrypoints


def _detect_package_json_entrypoints(
    rel_path: str,
    content: str,
) -> List[Dict[str, Any]]:
    """Detect package.json scripts and framework roots as entrypoint hints."""
    if _normalize_rel_path(rel_path) != "package.json":
        return []
    try:
        pkg = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(pkg, dict):
        return []

    signals = []
    scripts = pkg.get("scripts")
    if isinstance(scripts, dict):
        for name in ("dev", "start", "build", "preview"):
            if scripts.get(name):
                signals.append(f"package.json:scripts.{name}")
    for key in ("main", "module", "browser", "bin"):
        if pkg.get(key):
            signals.append(f"package.json:{key}")
    deps: dict[str, Any] = {}
    for key in ("dependencies", "devDependencies"):
        value = pkg.get(key)
        if isinstance(value, dict):
            deps.update(value)
    for framework in ("vite", "next", "react", "svelte", "vue", "@supabase/supabase-js"):
        if framework in deps:
            signals.append(f"package.json:dependency:{framework}")

    if not signals:
        return []

    return [{
        "file": rel_path,
        "language": "json",
        "type": "package_scripts",
        "signals": signals,
        "confidence": 0.88,
    }]


# ── Go entrypoint detection ──────────────────────────────────────────────

def _detect_go_entrypoints(
    rel_path: str,
    content: str,
) -> List[Dict[str, Any]]:
    """Detect Go entrypoints (package main + func main)."""
    entrypoints = []
    signals = []

    has_package_main = bool(re.search(
        r'^\s*package\s+main\s*$', content, re.MULTILINE
    ))
    has_func_main = bool(re.search(
        r'^\s*func\s+main\s*\(', content, re.MULTILINE
    ))

    if has_package_main:
        signals.append("package main")
    if has_func_main:
        signals.append("func main")

    if not (has_package_main and has_func_main):
        return entrypoints

    confidence = 0.85
    if has_package_main and has_func_main:
        confidence = 0.95

    entrypoints.append({
        "file": rel_path,
        "language": "go",
        "type": "go_main",
        "signals": signals,
        "confidence": round(confidence, 2),
    })
    return entrypoints


# ── Rust entrypoint detection ────────────────────────────────────────────

def _detect_rust_entrypoints(
    abs_path: str,
    rel_path: str,
    content: str,
    basename: str,
) -> List[Dict[str, Any]]:
    """Detect Rust entrypoints (fn main in src/main.rs or src/bin/*.rs)."""
    entrypoints = []

    has_fn_main = bool(re.search(
        r'^\s*(?:pub\s+)?fn\s+main\s*\(', content, re.MULTILINE
    ))

    is_main_rs = basename == "main.rs"
    is_bin_rs = rel_path.startswith("src/bin/") and basename.endswith(".rs")

    if not has_fn_main:
        return entrypoints

    # Only flag fn main in appropriate locations
    if not (is_main_rs or is_bin_rs):
        return entrypoints

    signals = ["fn main"]
    if is_bin_rs:
        signals.append("src/bin/ binary crate")
    else:
        signals.append("src/main.rs")

    confidence = 0.90
    if is_main_rs:
        confidence = 0.95

    entrypoints.append({
        "file": rel_path,
        "language": "rust",
        "type": "rust_main",
        "signals": signals,
        "confidence": round(confidence, 2),
    })
    return entrypoints


# ── Shell script entrypoint detection ────────────────────────────────────

def _detect_shell_entrypoints(
    abs_path: str,
    rel_path: str,
    content: str,
    basename: str,
) -> List[Dict[str, Any]]:
    """Detect shell script entrypoints by shebang and path heuristic."""
    entrypoints = []

    # Check for shebang
    has_shebang = content.lstrip().startswith("#!")
    if not has_shebang:
        return entrypoints

    shebang_match = re.match(r'^#!\s*(/\S+)', content.lstrip())
    shebang_path = shebang_match.group(1) if shebang_match else "unknown"

    # Only flag scripts in root, bin/, or scripts/ directories
    parts = rel_path.replace("\\", "/").split("/")
    in_allowed_dir = (
        len(parts) == 1  # root level
        or (len(parts) >= 2 and parts[0] in ("bin", "scripts"))
    )

    if not in_allowed_dir:
        return entrypoints

    signals = ["shebang: " + shebang_path]
    if "bin/" in rel_path:
        signals.append("bin/ directory")
    elif "scripts/" in rel_path:
        signals.append("scripts/ directory")

    confidence = 0.65
    if "bin/" in rel_path:
        confidence += 0.1
    confidence = min(round(confidence, 2), 0.95)

    entrypoints.append({
        "file": rel_path,
        "language": "shell",
        "type": "shell_script",
        "signals": signals,
        "confidence": confidence,
    })
    return entrypoints


# ── Main detection orchestration ─────────────────────────────────────────

def detect_entrypoints(scan_json_path: str) -> Dict[str, Any]:
    """Read a scan.json file and detect all entrypoints.

    Returns the structured entrypoints dict matching the required output shape.
    """
    with open(scan_json_path, "r", encoding="utf-8") as f:
        scan_data = json.load(f)

    project_root = _resolve_project_root(scan_data)
    all_entrypoints = []

    for file_record in scan_data.get("files", []):
        rel_path = file_record.get("relative_path", "")
        abs_path = file_record.get("path", "")
        language = file_record.get("language", "").lower()
        basename = os.path.basename(rel_path)

        # Try to read file content
        content = _read_file_content(abs_path)
        if content is None:
            continue

        # Dispatch to language-specific detectors
        if language == "python":
            eps = _detect_python_entrypoints(abs_path, rel_path, content)
            all_entrypoints.extend(eps)

        elif language in ("javascript", "typescript"):
            eps = _detect_js_ts_entrypoints(
                abs_path, rel_path, content, basename, scan_data
            )
            all_entrypoints.extend(eps)

        elif language == "go":
            eps = _detect_go_entrypoints(rel_path, content)
            all_entrypoints.extend(eps)

        elif language == "rust":
            eps = _detect_rust_entrypoints(
                abs_path, rel_path, content, basename
            )
            all_entrypoints.extend(eps)

        elif language == "shell":
            eps = _detect_shell_entrypoints(
                abs_path, rel_path, content, basename
            )
            all_entrypoints.extend(eps)

        elif language == "json" and _normalize_rel_path(rel_path) == "package.json":
            eps = _detect_package_json_entrypoints(rel_path, content)
            all_entrypoints.extend(eps)

    all_entrypoints.sort(key=_entrypoint_rank)

    # Build totals
    by_type: Dict[str, int] = {}
    for ep in all_entrypoints:
        t = ep["type"]
        by_type[t] = by_type.get(t, 0) + 1

    result = {
        "schema_version": SCHEMA_VERSION,
        "entrypoints": all_entrypoints,
        "totals": {
            "entrypoints_found": len(all_entrypoints),
            "by_type": by_type,
        },
    }
    return result


def main() -> int:
    """CLI entry point. Writes JSON to stdout."""
    if len(sys.argv) < 2:
        print("Usage: detect_entrypoints.py <scan.json>", file=sys.stderr)
        return 1

    scan_path = sys.argv[1]

    if not os.path.isfile(scan_path):
        print(f"Error: '{scan_path}' not found", file=sys.stderr)
        return 1

    try:
        result = detect_entrypoints(scan_path)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
