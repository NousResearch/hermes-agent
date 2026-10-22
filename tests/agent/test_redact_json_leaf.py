"""Per-leaf JSON redaction.

``redact_sensitive_text(json.dumps(obj))`` is not JSON-safe: _ENV_ASSIGN_RE's
unquoted value group ``(\\S+)`` swallows the closing ``"``/``}`` of the string it
sits in, so ``{"x": "DB_PASSWORD=abc"}`` came back as ``{"x": "DB_PASSWORD=***``.
``redact_sensitive_json`` redacts each string on its own. The AST guard at the
bottom keeps the serialize-then-redact shape from coming back.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from agent.redact import redact_sensitive_json, redact_sensitive_text

REPO = Path(__file__).resolve().parents[2]

# Leaves that corrupted the serialized form on main.
_CORRUPTING_LEAVES = [
    ("DB_PASSWORD=abcdef12", "abcdef12"),
    ("PASS=174", "174"),
    ("db_password=abcdef12", "abcdef12"),
    ("spring.datasource.password=hunter2x", "hunter2x"),
    ("export API_KEY=abcdefghijklmnopqrstuv", "abcdefghijklmnopqrstuv"),
]
_SHAPES = [
    lambda s: {"x": s},
    lambda s: {"x": [s], "y": 1},
    lambda s: {"x": s, "y": "z"},
    lambda s: {"outer": {"inner": s}},
]


@pytest.mark.parametrize("leaf,secret", _CORRUPTING_LEAVES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_masked_metadata_round_trips_and_stays_masked(leaf, secret, shape):
    obj = shape(leaf)
    out = redact_sensitive_json(obj, force=True)
    text = json.dumps(out)
    assert json.loads(text) == out
    assert secret not in text


def test_serialized_path_is_the_bug():
    """Documents why the helper exists: the text path breaks JSON."""
    s = redact_sensitive_text(json.dumps({"x": "DB_PASSWORD=abcdef12"}), force=True)
    with pytest.raises(json.JSONDecodeError):
        json.loads(s)


def test_key_rule_masks_secret_named_fields():
    out = redact_sensitive_json(
        {"password": "hunter2", "apiKey": "abcdefghijklmnopqrstuvwxyz",
         "nested": {"Token": "t0k"}, "notes": "fine"},
        force=True,
    )
    assert "hunter2" not in json.dumps(out)
    assert "abcdefghijklmnopqrstuvwxyz" not in json.dumps(out)
    assert out["nested"]["Token"] != "t0k"
    assert out["notes"] == "fine"


def test_non_string_and_env_lookup_values_pass_through():
    obj = {"password": 1234, "n": None, "b": True, "secret": "os.getenv('X')",
           "list": [1, 2.5, "plain"]}
    assert redact_sensitive_json(obj, force=True) == obj


def test_secret_in_key_is_masked():
    out = redact_sensitive_json({"PASS=abcdef12": 1}, force=True)
    assert "abcdef12" not in json.dumps(out)


def test_disabled_redaction_is_a_no_op(monkeypatch):
    import agent.redact as r
    monkeypatch.setattr(r, "_REDACT_ENABLED", False)
    obj = {"password": "hunter2", "x": "DB_PASSWORD=abcdef12"}
    assert redact_sensitive_json(obj) == obj


def test_per_leaf_masks_everything_the_serialized_path_masked():
    """No regression in coverage vs the old serialize-then-redact path."""
    body = "Q7" * 16  # 32 chars, built at runtime so no literal secret sits in source
    leaves = [
        "TOKEN=" + body,
        "sk-" + body,
        "Authorization: Bearer " + body,
        "x-api-key: " + body,
        "postgres://user:" + "pw" + body[:6] + "@host/db",
        "password: " + "hunter" + "22",
    ]
    assert all(len(l) > 12 for l in leaves)
    for leaf in leaves:
        plain = json.dumps({"x": leaf})
        old = redact_sensitive_text(plain, force=True)
        new = json.dumps(redact_sensitive_json({"x": leaf}, force=True))
        if old != plain:
            assert new != plain, leaf


# ---------------------------------------------------------------------------
# Class guard: no serialize-then-redact-as-text call sites.
# ---------------------------------------------------------------------------

def _is_call_to(node, name):
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    return (isinstance(f, ast.Name) and f.id == name) or (
        isinstance(f, ast.Attribute) and f.attr == name
    )


def _is_json_call(node, attr):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "json"
    )


def _violations(tree):
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            continue
        dumped = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and _is_json_call(node.value, "dumps"):
                dumped.update(t.id for t in node.targets if isinstance(t, ast.Name))
        for node in ast.walk(fn):
            if _is_call_to(node, "redact_sensitive_text") and node.args:
                a = node.args[0]
                if _is_json_call(a, "dumps") or (isinstance(a, ast.Name) and a.id in dumped):
                    out.append(node.lineno)
            if _is_json_call(node, "loads") and node.args and _is_call_to(
                node.args[0], "redact_sensitive_text"
            ):
                out.append(node.lineno)
    return sorted(set(out))


def test_guard_detects_the_shapes():
    src = (
        "import json\n"
        "def a(m):\n    return redact_sensitive_text(json.dumps(m))\n"
        "def b(m):\n    s = json.dumps(m)\n    s = redact_sensitive_text(s)\n    return s\n"
        "def c(s):\n    return json.loads(redact_sensitive_text(s))\n"
        "def ok(m):\n    return json.dumps(redact_sensitive_json(m))\n"
    )
    assert _violations(ast.parse(src)) == [3, 6, 9]


def test_no_serialize_then_redact_text_call_sites():
    offenders = []
    for path in REPO.rglob("*.py"):
        rel = path.relative_to(REPO)
        if rel.parts[0] in {"tests", "venv", ".venv", "node_modules"} or ".worktrees" in rel.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        offenders += [f"{rel}:{ln}" for ln in _violations(tree)]
    assert not offenders, (
        "redact serialized JSON per leaf with agent.redact.redact_sensitive_json, "
        f"not redact_sensitive_text(json.dumps(...)): {offenders}"
    )
