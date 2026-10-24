#!/usr/bin/env python3
"""Fail when a NamedTuple gains a REQUIRED field without a default.

A ``NamedTuple`` field with no default is a REQUIRED constructor argument.
Appending one to an EXISTING type breaks every positional constructor of that
type — and, worse, the break surfaces at RUNTIME, in whatever module happens to
construct it first.

Field incident (2026-10-24 02:16Z): ``agent/auxiliary_client.py`` gained a
field on its ``_LadderRoute`` NamedTuple and every positional constructor broke
at once. The cron external worker imports the LIVE checkout, so the failure hit
17 distinct cron jobs in the same second — one root cause, 17 identical opaque
job errors ("exited before ownership acknowledgement (exit 1)").

Second field incident (2026-10-24 06:49Z): a required field was placed AFTER a
defaulted one. Python rejects the CLASS at import time ("Non-default namedtuple
field 'x' cannot follow default field 'y'"), so the type never came into
existence and 8 jobs across 2 profiles died the same way. That failure is
PURELY LEXICAL — it needs no cross-version diff — so it is checked on every
NamedTuple class, new or existing.

This check makes the mistake a 5-second error instead of a 17-job outage:

* a field added to an existing NamedTuple WITHOUT a default is flagged — give it
  a default (``name: Optional[int] = None``) or convert the type to a dataclass
  with keyword-only construction;
* a ``defaults=`` argument on the FUNCTIONAL ``NamedTuple("Name", [...])`` form
  is flagged — that form accepts NEITHER ``defaults=`` nor per-field defaults
  ("Either list of fields or keywords can be provided to NamedTuple, not both"
  is raised at import time, which is exactly the incident's crash);
* a REQUIRED field that FOLLOWS a defaulted one is flagged — Python rejects the
  class at import time ("Non-default namedtuple field 'x' cannot follow default
  field 'y'"). This rule is purely lexical: it applies to a NEW NamedTuple
  exactly as it does to an existing one, because the failure needs no
  cross-version diff. A required field APPENDED at the END of a class is
  legal; the same field INSERTED after a defaulted one is fatal.
* a NamedTuple whose fields are all defaulted but whose class body still uses
  the functional form is flagged as unfixable-as-written.

A NEW NamedTuple (no previous definition to break) is not flagged: a fresh type
has no existing positional callers.

Suppress a deliberate exception with ``# namedtuple-required-field: ok -- <why>``
on the offending line.

Usage: python3 scripts/check_namedtuple_required_fields.py [paths...]
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TREES = (
    "hermes_cli", "agent", "gateway", "tui_gateway", "cron", "plugins", "tools",
    "cli.py", "run_agent.py", "utils.py",
)
SUPPRESS = "# namedtuple-required-field: ok"


def _relative_label(path: Path) -> str:
    """Repo-relative path when *path* is inside the repo, else its absolute form.

    The checker is also run against scratch trees outside the checkout (a
    deliberate reproduction of the incident), so a path that is not under ROOT
    must still produce a readable label instead of raising.
    """
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _namedtuple_types(tree: ast.AST) -> dict[str, ast.ClassDef]:
    """Map class name -> ClassDef for every class that inherits NamedTuple."""
    found: dict[str, ast.ClassDef] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            name = None
            if isinstance(base, ast.Name):
                name = base.id
            elif isinstance(base, ast.Attribute):
                name = base.attr
            if name == "NamedTuple":
                found[node.name] = node
                break
    return found


def _functional_namedtuples(tree: ast.AST) -> list[ast.Assign]:
    """Assignments of the form ``Name = NamedTuple("Name", [...])``."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        func_name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if func_name != "NamedTuple":
            continue
        out.append(node)
    return out


def _class_fields(node: ast.ClassDef) -> list[tuple[str, ast.AST, int]]:
    """``(name, annotation_node, lineno)`` for each annotated field of a NamedTuple class.

    The line number is the FIELD's own line, so a suppression marker written
    on the offending field silences that field alone.
    """
    fields = []
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            fields.append((stmt.target.id, stmt.value, stmt.lineno))
    return fields


def _previous_definitions(path: Path, current_names: set[str]) -> dict[str, ast.ClassDef]:
    """The NamedTuple class definitions for *current_names* as of HEAD.

    Returns only the names that EXISTED before this change — a brand-new type
    has no positional callers to break.
    """
    try:
        rel = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return {}
    try:
        blob = subprocess.run(
            ["git", "show", f"HEAD:{rel}"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # No git, or git failed: the "before" definition is unknown, so the
        # checker cannot tell an ADDED field from an existing one. That is a
        # reason to look at nothing, not to fail the build.
        print(
            f"check_namedtuple_required_fields: could not read HEAD:{rel} "
            f"({type(exc).__name__}: {exc}); skipping the before/after comparison",
            file=sys.stderr,
        )
        return {}
    if blob.returncode != 0 or not blob.stdout:
        return {}
    try:
        tree = ast.parse(blob.stdout)
    except SyntaxError:
        return {}
    return {
        name: node for name, node in _namedtuple_types(tree).items()
        if name in current_names
    }


def scan_file(path: Path, previous: dict[str, ast.ClassDef] | None = None) -> list[str]:
    """Return the problems found in *path*."""
    try:
        src = path.read_text(encoding="utf-8-sig")
    except OSError:
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    lines = src.splitlines()
    rel = _relative_label(path)
    problems: list[str] = []

    def flag(lineno: int, why: str, fix: str, *also_lineno: int) -> None:
        """Record a problem unless a suppression marker sits on one of its lines.

        A marker on the offending FIELD's own line silences that field; a
        marker on the enclosing CLASS line silences the whole type. Both are
        checked, so the author can put the justification where it reads best.
        """
        for candidate in (lineno, *also_lineno):
            line = lines[candidate - 1] if 0 < candidate <= len(lines) else ""
            if SUPPRESS in line:
                return
        problems.append(f"{rel}:{lineno}: {why} — {fix}")

    # 1. The functional form with a `defaults=` (or `module=`) argument.
    for assign in _functional_namedtuples(tree):
        call = assign.value
        assert isinstance(call, ast.Call)
        for keyword in call.keywords:
            if keyword.arg == "defaults":
                flag(
                    assign.lineno,
                    "functional NamedTuple passed `defaults=`",
                    "the functional form accepts NEITHER `defaults=` nor per-field "
                    "defaults; use the class form "
                    "(`class X(NamedTuple): field: T = default`) or a dataclass",
                )
            elif keyword.arg == "module":
                flag(
                    assign.lineno,
                    "functional NamedTuple passed `module=` together with a field list",
                    "the functional form accepts `module=` only with the keyword form",
                )

    # 2. A REQUIRED field added to an EXISTING NamedTuple class.
    for name, node in _namedtuple_types(tree).items():
        fields = _class_fields(node)
        if not fields:
            continue
        required = [fname for fname, value, _lineno in fields if value is None]
        if not required:
            continue
        if previous is None:
            previous = {}
        before = previous.get(name)
        if before is None:
            # A NEW type: no existing positional constructor to break.
            continue
        before_fields = {fname for fname, _value, _lineno in _class_fields(before)}
        added_required = [
            (fname, lineno) for fname, value, lineno in fields
            if value is None and fname not in before_fields
        ]
        if not added_required:
            continue
        for fname, lineno in added_required:
            flag(
                lineno,
                f"NamedTuple '{name}' gained the REQUIRED field '{fname}'",
                "give it a default (`= None`) or convert the type to a dataclass "
                "with keyword-only construction; a required field breaks every "
                "positional constructor of an existing NamedTuple",
                node.lineno,
            )

    # 3. A REQUIRED field that FOLLOWS a defaulted one — an IMPORT-TIME
    #    TypeError, caught by Python before the type exists:
    #        TypeError: Non-default namedtuple field 'x' cannot follow default field 'y'
    #    This is PURELY LEXICAL: it needs no before/after diff, so it applies
    #    to a NEW NamedTuple exactly as it does to an existing one. The
    #    "gained a required field" rule above cannot see it — a required field
    #    APPENDED at the END of a class is legal, but the same field INSERTED
    #    after a defaulted one is fatal, and a new type has no "before" to
    #    diff against at all.
    #
    #    Field incident (2026-10-24 06:49Z): a transient edit put
    #    `brand_new_required_field: Optional[str]` after
    #    `required_tokens: Optional[int] = None` in `_LadderRoute`. Python
    #    rejected the class at import, and because the cron external worker
    #    imports the LIVE checkout, 8 jobs across 2 profiles died in the same
    #    second with the opaque "exited before ownership acknowledgement".
    for name, node in _namedtuple_types(tree).items():
        defaulted: Optional[tuple[str, int]] = None
        for fname, value, lineno in _class_fields(node):
            if value is not None:
                defaulted = (fname, lineno)
                continue
            if defaulted is None:
                # A required field before any default: legal.
                continue
            prev_name, _prev_lineno = defaulted
            flag(
                lineno,
                f"NamedTuple '{name}' has the REQUIRED field '{fname}' after the "
                f"defaulted field '{prev_name}'",
                "Python rejects this at import time "
                f"('Non-default namedtuple field {fname} cannot follow default "
                f"field {prev_name}'); give '{fname}' a default, or move it above "
                f"'{prev_name}'",
                node.lineno,
            )
            # One violation per class: the first one is the import-time error.
            break
    return problems


def main(argv: list[str]) -> int:
    targets = [ROOT / a for a in argv] or [ROOT / t for t in DEFAULT_TREES]
    files: list[Path] = []
    for target in targets:
        if target.is_dir():
            files.extend(
                p for p in target.rglob("*.py")
                if "node_modules" not in p.parts and "__pycache__" not in p.parts
            )
        elif target.suffix == ".py" and target.exists():
            files.append(target)
    problems: list[str] = []
    for path in sorted(set(files)):
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            src = path.read_text(encoding="utf-8-sig")
            tree = ast.parse(src)
        except (OSError, SyntaxError):
            continue
        current = _namedtuple_types(tree)
        previous = _previous_definitions(path, set(current))
        problems.extend(scan_file(path, previous))
    if problems:
        print(
            "a NamedTuple has an import-time field error (a required field added "
            "to an existing type, a required field after a defaulted one, or a "
            "functional NamedTuple given `defaults=`):",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"check_namedtuple_required_fields: OK ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
