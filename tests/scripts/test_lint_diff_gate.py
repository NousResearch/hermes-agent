"""The --fail-on-new gate in scripts/lint_diff.py blocks only diagnostics the
head introduces vs base, scoped to the listed ty rule classes.

Pinned here with ty-gitlab-shaped fixtures, no real ty run.
"""
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "lint_diff.py"


def _load():
    spec = importlib.util.spec_from_file_location("lint_diff", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ty_entry(rule, path="a.py", line=1, message="m"):
    return {
        "check_name": rule,
        "location": {"path": path, "positions": {"begin": {"line": line}}},
        "description": message,
    }


def _run(monkeypatch, tmp_path, base_entries, head_entries, *extra):
    (tmp_path / "ruff.json").write_text("[]", encoding="utf-8")
    base_ty = tmp_path / "base-ty.json"
    if base_entries is not None:
        base_ty.write_text(json.dumps(base_entries), encoding="utf-8")
    head_ty = tmp_path / "head-ty.json"
    head_ty.write_text(json.dumps(head_entries), encoding="utf-8")
    argv = [
        "lint_diff.py",
        "--base-ruff", str(tmp_path / "ruff.json"),
        "--head-ruff", str(tmp_path / "ruff.json"),
        "--base-ty", str(base_ty),
        "--head-ty", str(head_ty),
        "--output", str(tmp_path / "summary.md"),
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return _load().main()


def test_new_gated_rule_fails(monkeypatch, tmp_path):
    entry = _ty_entry("invalid-method-override", path="p.py", message="bad override")
    assert _run(monkeypatch, tmp_path, [], [entry],
                "--fail-on-new", "invalid-method-override") == 1


def test_preexisting_gated_and_new_ungated_pass(monkeypatch, tmp_path):
    # A pre-existing override error carried over from base and a NEW error of a
    # non-gated class must both stay advisory.
    old = _ty_entry("invalid-method-override", path="p.py", message="bad override")
    other = _ty_entry("unresolved-import", path="p.py", message="no module")
    assert _run(monkeypatch, tmp_path, [old], [old, other],
                "--fail-on-new", "invalid-method-override") == 0
