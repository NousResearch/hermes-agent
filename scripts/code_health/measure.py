"""Measure a set of files in one tree (a revision or the working tree)."""

from __future__ import annotations

import ast
import io
import tokenize
import json
import re
import tempfile
import warnings
from pathlib import Path

from scripts.code_health import py_rules, py_structure
from scripts.code_health.config import RULES, RULES_BY_ID, in_scope, rule_applies
from scripts.code_health.gitio import read_file
from scripts.code_health.model import FileMeasure, Unit
from scripts.code_health.ruff_runner import run_ruff
from scripts.code_health.ts_measure import measure_ts

_PATTERNS_FILE = Path("scripts/ci/profile_scope_patterns.json")


def _regex_rules(repo: Path) -> list[tuple[str, re.Pattern[str], re.Pattern[str] | None]]:
    data = json.loads((repo / _PATTERNS_FILE).read_text(encoding="utf-8-sig"))
    by_id = {p["id"]: p for p in data["patterns"]}
    out = []
    for rule in RULES:
        if rule.source != "regex":
            continue
        pat = by_id[rule.pattern_id]
        path_re = re.compile(pat["path_regex"]) if pat.get("path_regex") else None
        out.append((rule.id, re.compile(pat["pattern_regex"], re.M), path_re))
    return out


def _own_complexity(fm: FileMeasure, cc_by_line: dict[int, int]) -> None:
    """Ruff's C901 for a function includes every nested def; subtract the direct children so
    each function carries only its own branches (nested defs are separate units)."""
    full = {q: cc_by_line[u.line] for q, u in fm.units.items() if u.line in cc_by_line}
    own = dict(full)
    for qual, unit in fm.units.items():
        if unit.parent in own and qual in full:
            own[unit.parent] -= full[qual]
    for qual, value in own.items():
        fm.units[qual].metrics["CC"] = value


def _python_comments(text: str) -> dict[int, str]:
    comments: dict[int, str] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                comments[tok.start[0]] = comments.get(tok.start[0], "") + tok.string
    except (tokenize.TokenError, SyntaxError):
        pass  # unparseable source has no AST findings to waive either
    return comments


# Characters str.splitlines() breaks on; never blanked, so the blanked text keeps its line numbers.
_LINE_BREAKS = frozenset("\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029")


def _string_statements(tree: ast.Module):
    """Docstrings and other bare string statements: prose, never executed."""
    for node in ast.walk(tree):
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            yield node


def _executable_lines(text: str, tree: ast.Module) -> list[str]:
    """``text.splitlines()`` with comments and string statements blanked out.

    The profile regex rules describe operations (`env = os.environ.copy()`), so a comment or a
    docstring that mentions one must not count as doing it. String literals inside code stay:
    `os.getenv("DISCORD_TOKEN")` is matched by its argument.
    """
    rows = [list(line) for line in io.StringIO(text).readlines()]  # ast/tokenize line numbering

    def blank(line: int, start: int, end: int | None = None) -> None:
        row = rows[line - 1]
        for col in range(start, len(row) if end is None else end):
            if row[col] not in _LINE_BREAKS:
                row[col] = " "

    def char_col(line: int, byte_col: int) -> int:  # ast columns are UTF-8 byte offsets
        return len("".join(rows[line - 1]).encode("utf-8")[:byte_col].decode("utf-8", "ignore"))

    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                blank(tok.start[0], tok.start[1], tok.end[1])
    except (tokenize.TokenError, SyntaxError):
        return text.splitlines()  # fail closed: every line stays visible to the rules
    for node in _string_statements(tree):
        last = node.end_lineno or node.lineno
        end = char_col(last, node.end_col_offset or 0)
        if last == node.lineno:
            blank(node.lineno, char_col(node.lineno, node.col_offset), end)
            continue
        blank(node.lineno, char_col(node.lineno, node.col_offset))
        for line in range(node.lineno + 1, last):
            blank(line, 0)
        blank(last, 0, end)
    return "".join("".join(row) for row in rows).splitlines()


class Measurer:
    def __init__(self, repo: Path, ruff: list[str], known_env: set[str]) -> None:
        self.repo = repo
        self.ruff = ruff
        self.ctx = py_rules.Ctx(known_env=known_env)
        self.regex_rules = _regex_rules(repo)

    def measure(self, tree: str | None, paths: list[str]) -> dict[str, FileMeasure]:
        contents = {}
        for path in paths:
            if in_scope(path):
                text = read_file(self.repo, tree, path)
                if text is not None:
                    contents[path] = text
        if tree is None:
            return self._measure_in(self.repo, contents)
        with tempfile.TemporaryDirectory(prefix="code-health-") as tmp:
            root = Path(tmp)
            for path, text in contents.items():
                dest = root / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                # newline="": the blob's own line endings, so ruff's line numbers match the
                # AST's (Windows text mode would turn a CRLF blob into CR CR LF).
                dest.write_text(text, encoding="utf-8", newline="")
            return self._measure_in(root, contents)

    def _measure_in(self, root: Path, contents: dict[str, str]) -> dict[str, FileMeasure]:
        result = {path: FileMeasure(path, lines=text.splitlines()) for path, text in contents.items()}
        py = sorted(p for p in contents if in_scope(p) == "py")
        ts = sorted(p for p in contents if in_scope(p) == "ts")
        ruff_out = run_ruff(self.ruff, root, py) if py else {}
        for path in py:
            self._python(result[path], contents[path], ruff_out.get(path))
        ts_out = measure_ts(self.repo, root, ts) if ts else {}
        for path in ts:
            self._typescript(result[path], ts_out.get(path))
        return result

    def _python(self, fm: FileMeasure, text: str, ruff_file) -> None:
        fm.metrics["FILE_LINES"] = len(fm.lines)
        fm.comments = _python_comments(text)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(text)
                rules_tree = py_rules.canonical_tree(ast.parse(text))
        except SyntaxError as exc:
            fm.error = f"does not parse: {exc.msg} (line {exc.lineno})"
            return
        scopes = py_structure.measure_structure(fm, tree)
        if ruff_file is None or ruff_file.errors:
            fm.error = "ruff could not measure it: " + "; ".join((ruff_file.errors if ruff_file else ["no result"])[:3])
        else:
            _own_complexity(fm, ruff_file.cc_by_line)
            for code, row in ruff_file.hits:
                if code in RULES_BY_ID and rule_applies(RULES_BY_ID[code], fm.path):
                    fm.add_hit(code, scopes.scope(row), row)
        for rule_id, checker in py_rules.CHECKERS.items():
            if rule_applies(RULES_BY_ID[rule_id], fm.path):
                for row in sorted(set(checker(rules_tree, self.ctx))):
                    fm.add_hit(rule_id, scopes.scope(row), row)
        self._regex(fm, scopes, text, tree)

    def _regex(self, fm: FileMeasure, scopes, text: str, tree: ast.Module) -> None:
        code: str | None = None
        for rule_id, pattern, path_re in self.regex_rules:
            if not rule_applies(RULES_BY_ID[rule_id], fm.path):
                continue
            if path_re and not path_re.search(fm.path):
                continue
            if code is None:
                code = "\n".join(_executable_lines(text, tree))
            # Whole-text match, anchored on the line where it starts (as the advisory
            # check_profile_scope_patterns.scan_text does): a call wrapped over two lines is
            # still the operation.
            starts = {code.count("\n", 0, m.start()) + 1 for m in pattern.finditer(code)}
            for index in sorted(starts):
                fm.add_hit(rule_id, scopes.scope(index), index)

    @staticmethod
    def _typescript(fm: FileMeasure, data: dict | None) -> None:
        fm.metrics["FILE_LINES"] = len(fm.lines)
        if data is None or "error" in data:
            fm.error = (data or {}).get("error", "the TypeScript measurer returned nothing for it")
            return
        fm.comments = {line: text for line, text in data.get("comments", [])}
        for unit in data.get("units", []):
            fm.units[unit["q"]] = Unit(
                qualname=unit["q"],
                line=unit["line"],
                metrics={"CC": unit["cc"], "FUNC_LINES": unit["lines"], "NESTING": unit["nesting"]},
                body_hash=unit["hash"],
                end_line=unit.get("end"),
            )
