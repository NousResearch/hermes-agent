"""No production module may redact a structure by serializing it.

#123688 fixed four call sites of ``json.loads(redact_sensitive_text(json.dumps(obj)))``
by adding ``redact_sensitive_json``. This AST guard keeps the fifth from appearing: the
pattern corrupts JSON whenever a string leaf ends in an ENV-style secret, because the
text patterns are not JSON-aware.

It is a class guard, not a test of the four known sites — the point is that the next
``json.dumps`` -> ``redact_sensitive_text`` -> ``json.loads`` round trip fails here.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Directories where the pattern is legitimate: tests (they may assert on it), and the
# redaction module itself (it owns the text-level entry point).
EXEMPT_TOP_LEVEL = {"tests", "evals", "scripts", "website", "node_modules", ".venv"}


def _is_redact_sensitive_text(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = getattr(func, "id", None) or getattr(func, "attr", None)
    return name == "redact_sensitive_text"


def _inner_most(node: ast.AST) -> ast.AST:
    """Strip json.dumps(...) / json.loads(...) wrappers down to the redact call."""
    while isinstance(node, ast.Call):
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name in {"dumps", "loads"}:
            node = node.args[0] if node.args else node
            continue
        break
    return node


def _round_trip_sites(tree: ast.AST) -> list[int]:
    """Lines where redact_sensitive_text() is wrapped by json.dumps() or json.loads()."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # json.loads(redact_sensitive_text(json.dumps(x))) — walk down the loads argument.
        candidates = []
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name in {"loads", "dumps"} and node.args:
            candidates.append((node.args[0], node.lineno))
        for arg, lineno in candidates:
            inner = _inner_most(arg)
            if _is_redact_sensitive_text(inner):
                lines.append(lineno)
    return sorted(set(lines))


def _offending() -> list[str]:
    found: list[str] = []
    for path in sorted(REPO.rglob("*.py")):
        rel = path.relative_to(REPO)
        if rel.parts[0] in EXEMPT_TOP_LEVEL:
            continue
        if rel.parts[:2] == ("agent", "redact.py") or rel.name == "redact.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for lineno in _round_trip_sites(tree):
            found.append(f"  {rel.as_posix()}:{lineno}")
    return found


def test_no_production_module_redacts_by_serialize_then_parse():
    sites = _offending()
    assert not sites, (
        "redact a structure with redact_sensitive_json(...), not "
        "json.loads(redact_sensitive_text(json.dumps(...))) — the text patterns are not "
        "JSON-aware and corrupt the structure (#123688):\n" + "\n".join(sites)
    )


def test_the_guard_detects_the_pattern_it_is_looking_for():
    """Negative control: if the matcher regressed, the guard above would pass vacuously."""
    tree = ast.parse(
        "import json\n"
        "from agent.redact import redact_sensitive_text\n"
        "def bad(obj):\n"
        "    return json.loads(redact_sensitive_text(json.dumps(obj)))\n"
        "def also_bad(payload):\n"
        "    return json.dumps(redact_sensitive_text(json.dumps(payload)))\n"
        "def good(obj):\n"
        "    return redact_sensitive_json(obj)\n"
        "def texty(s):\n"
        "    return redact_sensitive_text(s)\n"
    )
    sites = _round_trip_sites(tree)

    def body_line(fn_name: str) -> int:
        """First body line of *fn_name* — the guard reports the call site, not the ``def``."""
        node = next(n for n in tree.body if getattr(n, "name", None) == fn_name)
        return node.body[0].lineno

    assert body_line("bad") in sites, "the guard missed the serialize-then-parse pattern"
    assert body_line("also_bad") in sites, "the guard missed the nested dumps form"
    assert body_line("good") not in sites, "redact_sensitive_json must not be flagged"
    assert body_line("texty") not in sites, "plain redact_sensitive_text must not be flagged"