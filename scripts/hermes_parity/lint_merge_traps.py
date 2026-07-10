"""AST linters for the three deferred merge-trap classes.

Each catches a documented incident class from the 2026-07-10 parity sync
(see docs/sync/2026-07-10-hermes-parity-SPEC.md — "Merge-trap AST lint"):

- dead-code-after-return: statements in a block after an unconditional
  ``return``/``raise`` — the merge concatenated fork + upstream bodies and an
  early return orphaned the fork's block (the ``check_dangerous_command``
  cron-gate incident).
- getattr-lambda-fallback: ``getattr(obj, "name", <callable fallback>)`` where
  ``name`` is not defined anywhere in the repo — the merge kept a call to a
  method the other side deleted, and the fallback silently swallows it (the
  ``should_defer_preflight_to_real_usage`` incident).
- duplicate-function-bodies: two same-named or structurally-identical sibling
  functions in one module — the merge kept both sides' dispatch bodies (the
  ``cron/scheduler.py`` ``_process_one_job``/``run_one_job`` incident).

Like lint_unbound, output is a REVIEW list, not a hard fail (stage passes
with warnings unless --strict). False positives allowed.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrapIssue:
    path: str
    line: int
    column: int
    kind: str
    detail: str


_TERMINAL_TYPES = (ast.Return, ast.Raise, ast.Break, ast.Continue)


def _lint_dead_code_after_return(tree: ast.AST, path: str) -> list[TrapIssue]:
    issues: list[TrapIssue] = []
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for index, stmt in enumerate(body[:-1]):
            if isinstance(stmt, _TERMINAL_TYPES):
                orphan = body[index + 1]
                # A bare string expression (docstring-ish) or ... after a
                # return is noise, not the incident class.
                if isinstance(orphan, ast.Expr) and isinstance(orphan.value, ast.Constant):
                    continue
                # `return` followed by `yield` is the standard way to write
                # an empty (async) generator — idiom, not a merge trap.
                if isinstance(orphan, ast.Expr) and isinstance(
                    orphan.value, (ast.Yield, ast.YieldFrom)
                ):
                    continue
                issues.append(
                    TrapIssue(
                        path=path,
                        line=orphan.lineno,
                        column=orphan.col_offset,
                        kind="dead-code-after-return",
                        detail=(
                            f"unreachable statement after unconditional "
                            f"{type(stmt).__name__.lower()} at line {stmt.lineno} "
                            "(merge may have orphaned one side's block)"
                        ),
                    )
                )
                break  # one report per block
    return issues


def _collect_defined_names(trees: list[ast.AST]) -> set[str]:
    defined: set[str] = set()
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        defined.add(target.attr)
                    elif isinstance(target, ast.Name):
                        defined.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, (ast.Attribute, ast.Name)):
                defined.add(
                    node.target.attr if isinstance(node.target, ast.Attribute) else node.target.id
                )
    return defined


def _is_callable_fallback(node: ast.AST) -> bool:
    return isinstance(node, ast.Lambda) or (
        isinstance(node, ast.Name) and node.id in {"bool", "int", "str", "float", "list", "dict"}
    )


def _lint_getattr_lambda_fallback(
    tree: ast.AST, path: str, defined_names: set[str]
) -> list[TrapIssue]:
    issues: list[TrapIssue] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"):
            continue
        if len(node.args) != 3:
            continue
        name_arg, fallback = node.args[1], node.args[2]
        if not (isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str)):
            continue
        if not _is_callable_fallback(fallback):
            continue
        attr_name = name_arg.value
        if attr_name in defined_names:
            continue
        issues.append(
            TrapIssue(
                path=path,
                line=node.lineno,
                column=node.col_offset,
                kind="getattr-lambda-fallback",
                detail=(
                    f"getattr(..., {attr_name!r}, <callable fallback>) but "
                    f"{attr_name!r} is not defined anywhere scanned — the "
                    "method may have been deleted on one merge side and the "
                    "fallback now silently swallows every call"
                ),
            )
        )
    return issues


def _normalize_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Structural fingerprint of a function body (names kept, positions/docstrings dropped)."""
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


def _lint_duplicate_function_bodies(tree: ast.AST, path: str) -> list[TrapIssue]:
    issues: list[TrapIssue] = []
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        seen: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        for stmt in body:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if len(stmt.body) < 4:  # trivial bodies (getters, pass-throughs) are noise
                continue
            fingerprint = _normalize_body(stmt)
            prior = seen.get(fingerprint)
            if prior is not None:
                issues.append(
                    TrapIssue(
                        path=path,
                        line=stmt.lineno,
                        column=stmt.col_offset,
                        kind="duplicate-function-bodies",
                        detail=(
                            f"{stmt.name!r} (line {stmt.lineno}) has a structurally "
                            f"identical body to {prior.name!r} (line {prior.lineno}) — "
                            "the merge may have kept both sides' dispatch bodies"
                        ),
                    )
                )
            else:
                seen[fingerprint] = stmt
    return issues


def lint_source(
    source: str, *, path: str = "<string>", defined_names: set[str] | None = None
) -> list[TrapIssue]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # the imports gate owns syntax errors
    if defined_names is None:
        defined_names = _collect_defined_names([tree])
    issues: list[TrapIssue] = []
    issues.extend(_lint_dead_code_after_return(tree, path))
    issues.extend(_lint_getattr_lambda_fallback(tree, path, defined_names))
    issues.extend(_lint_duplicate_function_bodies(tree, path))
    return sorted(issues, key=lambda issue: (issue.path, issue.line))


def lint_paths(paths: list[Path], *, repo: Path | None = None) -> list[TrapIssue]:
    trees: list[tuple[str, str, ast.AST]] = []
    for file_path in paths:
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            continue
        rel = str(file_path.relative_to(repo)) if repo else str(file_path)
        trees.append((rel, source, tree))
    # getattr-fallback needs repo-wide defined names: a method referenced in
    # file A is usually defined in file B.
    defined_names = _collect_defined_names([tree for _, _, tree in trees])
    issues: list[TrapIssue] = []
    for rel, _source, tree in trees:
        issues.extend(_lint_dead_code_after_return(tree, rel))
        issues.extend(_lint_getattr_lambda_fallback(tree, rel, defined_names))
        issues.extend(_lint_duplicate_function_bodies(tree, rel))
    return sorted(issues, key=lambda issue: (issue.path, issue.line))
