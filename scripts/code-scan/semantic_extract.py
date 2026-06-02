#!/usr/bin/env python3
"""Non-LLM semantic signal extractor for the UA Flywheel code-scan module.

Reads a scan.json report and extracts bounded, deterministic semantic signals
from each source file:
  - Python  : stdlib ast for docstrings, symbols, decorators, base classes,
              annotated assignments.
  - JS/TS   : bounded regex for exports, function/class declarations,
              route/listen patterns.
  - Go      : minimal regex for top-level functions including main.
  - Rust    : minimal regex for pub/private functions including main.
  - Shell   : minimal regex for function definitions and main.

Never executes or imports target files.  Fails soft on parse errors with
warnings.  Caps per-file signals and includes truncation flags.
Stdlib only — no new dependencies.

CLI:
    python scripts/code-scan/semantic_extract.py <scan.json> \\
        [--scan-root <project-root>] [--max-signals-per-file 50] \\
        > <semantic-signals.json>
"""

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


# ── Language helpers ─────────────────────────────────────────────

LANG_BY_EXT = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
}


def detect_language(rel_path: str) -> str:
    """Infer language from file extension."""
    ext = Path(rel_path).suffix.lower()
    return LANG_BY_EXT.get(ext, "unknown")


# ── Python AST extraction ───────────────────────────────────────


class _PythonSignalVisitor(ast.NodeVisitor):
    """Walk a Python AST and collect semantic signals."""

    def __init__(self) -> None:
        self.symbols: list[dict[str, Any]] = []
        self.docstrings: list[dict[str, str]] = []
        self.decorators: set[str] = set()
        self._module_docstring: str | None = None

    # ── entry ────────────────────────────────────────────────────

    def visit(self, node: ast.AST) -> None:
        # Capture module docstring
        if self._module_docstring is None:
            if (
                isinstance(node, ast.Module)
                and node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, (ast.Constant, ast.Str))
            ):
                val = node.body[0].value
                if isinstance(val, ast.Constant):
                    self._module_docstring = str(val.value)
                elif isinstance(val, ast.Str):
                    self._module_docstring = val.s

            self.docstrings.append(
                {"owner": "__module__", "summary": self._module_docstring or ""}
            )
        super().visit(node)

    # ── functions ────────────────────────────────────────────────

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add_function(node)
        self.generic_visit(node)

    def _add_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.symbols.append({"kind": "function", "name": node.name})
        # decorator names
        for deco in node.decorator_list:
            self._extract_decorator_name(deco)
        # docstring
        self._extract_docstring(node)

    # ── classes ──────────────────────────────────────────────────

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = [self._name_or_none(b) for b in node.bases]
        bases = [b for b in bases if b]
        sym: dict[str, Any] = {"kind": "class", "name": node.name}
        if bases:
            sym["bases"] = bases
        self.symbols.append(sym)
        # decorators
        for deco in node.decorator_list:
            self._extract_decorator_name(deco)
        # docstring
        self._extract_docstring(node)
        self.generic_visit(node)

    # ── annotated assignments (module-level) ─────────────────────

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            self.symbols.append(
                {"kind": "assignment", "name": node.target.id}
            )
        self.generic_visit(node)

    # ── helpers ──────────────────────────────────────────────────

    def _extract_docstring(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> None:
        ds = ast.get_docstring(node)
        if ds:
            self.docstrings.append({"owner": node.name, "summary": ds})

    def _extract_decorator_name(self, deco: ast.expr) -> None:
        name = self._name_or_none(deco)
        if name:
            self.decorators.add(name)

    @staticmethod
    def _name_or_none(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts: list[str] = []
            cur: ast.expr | None = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            return ".".join(reversed(parts))
        if isinstance(node, ast.Call):
            return _PythonSignalVisitor._name_or_none(node.func)
        return None


def extract_python(source: str) -> dict[str, Any]:
    """Extract signals from Python source via ast."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {
            "symbols": [],
            "docstrings": [{"owner": "__module__", "summary": ""}],
            "decorators": [],
            "warnings": [f"SyntaxError: {exc}"],
        }

    visitor = _PythonSignalVisitor()
    visitor.visit(tree)
    return {
        "symbols": visitor.symbols,
        "docstrings": visitor.docstrings,
        "decorators": sorted(visitor.decorators),
        "warnings": [],
    }


# ── JS/TS regex extraction ──────────────────────────────────────

# Patterns are intentionally bounded and non-recursive.

_JS_FUNC = re.compile(
    r"(?:export\s+)?(?:async\s+)?function\s+(\w+)", re.MULTILINE
)
_JS_CLASS = re.compile(
    r"(?:export\s+)?class\s+(\w+)", re.MULTILINE
)
_JS_EXPORT_DEFAULT = re.compile(
    r"export\s+default\s+(\w+)", re.MULTILINE
)
_JS_ROUTE = re.compile(
    r"(\w+)\.(listen|get|post|put|delete|patch|head|options|use)\s*\(",
    re.MULTILINE,
)


def extract_js_ts(source: str) -> dict[str, Any]:
    """Extract signals from JavaScript/TypeScript via bounded regex."""
    symbols: list[dict[str, Any]] = []
    warnings: list[str] = []

    for m in _JS_FUNC.finditer(source):
        symbols.append({"kind": "function", "name": m.group(1)})

    for m in _JS_CLASS.finditer(source):
        symbols.append({"kind": "class", "name": m.group(1)})

    for m in _JS_EXPORT_DEFAULT.finditer(source):
        symbols.append(
            {"kind": "variable", "name": m.group(1), "export": "default"}
        )

    for m in _JS_ROUTE.finditer(source):
        obj, method = m.group(1), m.group(2)
        symbols.append(
            {"kind": "route_hint", "name": f"{obj}.{method}"}
        )

    return {
        "symbols": symbols,
        "docstrings": [],
        "decorators": [],
        "warnings": warnings,
    }


# ── Go regex extraction ─────────────────────────────────────────

_GO_FUNC = re.compile(
    r"^func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(", re.MULTILINE
)


def extract_go(source: str) -> dict[str, Any]:
    """Extract signals from Go source via minimal regex."""
    symbols: list[dict[str, Any]] = []

    for m in _GO_FUNC.finditer(source):
        symbols.append({"kind": "function", "name": m.group(1)})

    return {
        "symbols": symbols,
        "docstrings": [],
        "decorators": [],
        "warnings": [],
    }


# ── Rust regex extraction ───────────────────────────────────────

_RUST_FUNC = re.compile(
    r"^(?:pub\s+)?fn\s+(\w+)\s*[\(<]", re.MULTILINE
)


def extract_rust(source: str) -> dict[str, Any]:
    """Extract signals from Rust source via minimal regex."""
    symbols: list[dict[str, Any]] = []

    for m in _RUST_FUNC.finditer(source):
        symbols.append({"kind": "function", "name": m.group(1)})

    return {
        "symbols": symbols,
        "docstrings": [],
        "decorators": [],
        "warnings": [],
    }


# ── Shell regex extraction ──────────────────────────────────────

_SHELL_FUNC = re.compile(
    r"^(\w+)\s*\(\s*\)\s*\{", re.MULTILINE
)


def extract_shell(source: str) -> dict[str, Any]:
    """Extract signals from Shell source via minimal regex."""
    symbols: list[dict[str, Any]] = []

    for m in _SHELL_FUNC.finditer(source):
        symbols.append({"kind": "function", "name": m.group(1)})

    return {
        "symbols": symbols,
        "docstrings": [],
        "decorators": [],
        "warnings": [],
    }


# ── Dispatch table ──────────────────────────────────────────────

EXTRACTORS = {
    "python": extract_python,
    "javascript": extract_js_ts,
    "typescript": extract_js_ts,
    "go": extract_go,
    "rust": extract_rust,
    "shell": extract_shell,
}


def extract_signals(source: str, language: str) -> dict[str, Any]:
    """Route to the correct extractor for the given language."""
    extractor = EXTRACTORS.get(language)
    if extractor is None:
        return {
            "symbols": [],
            "docstrings": [],
            "decorators": [],
            "warnings": [f"Unsupported language: {language}"],
        }
    return extractor(source)


# ── Main processing ─────────────────────────────────────────────


def process_scan(
    scan_path: Path,
    scan_root: str | None,
    max_signals: int,
) -> dict[str, Any]:
    """Process a scan.json and return semantic signal output."""
    with open(scan_path, "r", encoding="utf-8") as fh:
        scan = json.load(fh)

    project_root = scan_root or scan.get("project_root", os.getcwd())
    files_spec = scan.get("files", [])

    output_files: dict[str, Any] = {}
    total_symbols = 0
    total_warnings = 0

    for file_entry in files_spec:
        rel_path = file_entry["relative_path"]
        lang = file_entry.get("language") or detect_language(rel_path)
        abs_path = Path(project_root) / rel_path

        try:
            source = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            output_files[rel_path] = {
                "language": lang,
                "symbols": [],
                "docstrings": [],
                "decorators": [],
                "warnings": [f"Read error: {exc}"],
                "truncated": False,
            }
            total_warnings += 1
            continue

        result = extract_signals(source, lang)
        symbols = result["symbols"]

        # Apply per-file cap
        truncated = len(symbols) > max_signals
        if truncated:
            symbols = symbols[:max_signals]

        file_out = {
            "language": lang,
            "symbols": symbols,
            "docstrings": result.get("docstrings", []),
            "decorators": result.get("decorators", []),
            "warnings": result.get("warnings", []),
            "truncated": truncated,
        }
        output_files[rel_path] = file_out
        total_symbols += len(symbols)
        total_warnings += len(file_out["warnings"])

    return {
        "schema_version": "1.0.0",
        "files": dict(sorted(output_files.items())),
        "totals": {
            "files_processed": len(output_files),
            "symbols": total_symbols,
            "warnings": total_warnings,
        },
    }


# ── CLI ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract semantic signals from a code-scan report."
    )
    parser.add_argument(
        "scan_json",
        type=Path,
        help="Path to the scan.json report.",
    )
    parser.add_argument(
        "--scan-root",
        type=str,
        default=None,
        help="Project root. Defaults to the root in scan.json.",
    )
    parser.add_argument(
        "--max-signals-per-file",
        type=int,
        default=50,
        help="Maximum signals per file before truncation (default: 50).",
    )
    args = parser.parse_args()

    if not args.scan_json.is_file():
        print(f"Error: scan.json not found: {args.scan_json}", file=sys.stderr)
        sys.exit(1)

    result = process_scan(args.scan_json, args.scan_root, args.max_signals_per_file)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
