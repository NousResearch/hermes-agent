#!/usr/bin/env python3
"""Static gate for the RE2 migration: every literal regex of product code must behave on `hermes_re` as on stdlib `re`.

    python scripts/check_re2_patterns.py                      # every tracked non-test .py file
    python scripts/check_re2_patterns.py [--pre-swap] [--verbose] [--json OUT] PATH...

For every call `<alias>.compile/search/match/fullmatch/findall/finditer/split/sub/subn(<string literal>, ...)` where
<alias> is the name the file binds to the regex module, and every call of such a function imported by name
(`from hermes_re import compile [as c]`); the pattern may be positional or the `pattern=` keyword. An argument a
literal `**{...}` names is read from it; one a `*args` or another `**mapping` may supply is not a literal:
  REJECTED   hermes_re refuses the pattern, or its RE2 program cannot be built
  SURROGATE  the pattern names a lone surrogate (U+D800..U+DFFF), as a character or a \\u / \\U escape; RE2 can
             never match one
  INEXACT    the pattern has a `\\b` / `\\B` that RE2 evaluates with ASCII word rules
  DIFFERS    the call's operation, with its own literal count / maxsplit, gives another result than stdlib `re` on
             generated strings (bytes too)
  UNCHECKED  stdlib `re` did not finish the comparison within DEADLINE seconds (catastrophic backtracking on the
             generated strings), so nothing says the swap keeps the results; review the pattern and waive it
  DYNAMIC    not checked, with the reason (listed with --verbose): the pattern is not a string literal, or the
             flags are not a literal (every verdict depends on the flags, so such a call reports only a REJECTED
             or SURROGATE that holds under every combination of ASCII, I, M, S and X), or the count / maxsplit
             of sub/subn/split is not a literal (only DIFFERS depends on it, so the other findings still apply)
Any finding except DYNAMIC makes the exit status 1. One call is waived with a comment on its first or last line
(a real comment token: the marker inside a string literal waives nothing):
    # re2: reviewed -- <reason>
Calls are resolved per scope (scripts/code_health canonical_tree): a function-local `import re` that shadows a
module-level `import hermes_re as re` is stdlib code there and is not checked without --pre-swap.
Without --pre-swap only files that import `hermes_re` are checked (the state after their swap). With --pre-swap,
files that still say `import re` are checked as if already swapped: run this BEFORE swapping a file.
Reads files as text; imports `hermes_re`, which needs the google-re2 package (exit 2 with a message otherwise), and
the code-health AST helpers. The deadline uses SIGALRM, so it applies on POSIX hosts (CI, Linux and macOS commits).
This script is a development tool under scripts/ and uses stdlib `re` as the reference engine.
"""
import ast
import importlib
import json
import random
import re as stdlib_re
from re import _parser  # stdlib's own parse tree: what a pattern names once comments are gone
import signal
import subprocess
import sys
import zlib
from pathlib import Path


def _repo_root():
    """The tree this script belongs to (scripts/ of a checkout or of a `scripts/check` snapshot), else git's toplevel."""
    here = Path(__file__).resolve().parent.parent
    if (here / "hermes_re").is_dir():
        return here
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, encoding="utf-8", check=False, timeout=60,
    ).stdout.strip()
    return Path(out) if out else Path.cwd()


ROOT = _repo_root()
# The call's own result limit: (keyword, positional index); compared as given, so `count=1` is not judged unlimited.
LIMITS = {"split": ("maxsplit", 2), "sub": ("count", 3), "subn": ("count", 3)}
FUNCS = {"compile": 1, "search": 2, "match": 2, "fullmatch": 2, "findall": 2, "finditer": 2, "split": 3, "sub": 4, "subn": 4}
FLAGS = {"I": 2, "IGNORECASE": 2, "M": 8, "MULTILINE": 8, "S": 16, "DOTALL": 16, "A": 256, "ASCII": 256, "U": 32,
         "UNICODE": 32, "X": 64, "VERBOSE": 64, "L": 4, "LOCALE": 4, "DEBUG": 128, "NOFLAG": 0}
WAIVER = "# re2: reviewed --"
# What a non-literal flags argument may hold: ASCII first, so a pattern that builds only with it is found early.
UNKNOWN_FLAG_BITS = (256, 2, 8, 16, 64)  # ASCII, IGNORECASE, MULTILINE, DOTALL, VERBOSE
EXTRA = " \n\t-_.,:;/|@=\"'()aZ09\u00e9\u00df\u0436\u65e5\u017f\u0130"
SAMPLES = 160
DEADLINE = 2.0


class _Deadline(BaseException):
    """Raised by SIGALRM inside a stdlib match; a BaseException so `run()` does not record it as a result."""


def load_facade():
    """(hermes_re, canonical_tree, python_comments) from the tree this script checks."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        hermes_re = importlib.import_module("hermes_re")
        hermes_re.build("a")
    except ImportError as exc:
        print(f"check_re2_patterns: hermes_re / google-re2 is not importable in {sys.executable} ({exc}); run it in the "
              "repository environment (source ./activate, or: just check --only re2-patterns)")
        sys.exit(2)
    rules = importlib.import_module("scripts.code_health.py_rules")
    measure = importlib.import_module("scripts.code_health.measure")
    return hermes_re, rules.canonical_tree, measure._python_comments


def flag_value(node):
    if node is None:
        return 0
    if isinstance(node, ast.Attribute) and node.attr in FLAGS:
        return FLAGS[node.attr]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left, right = flag_value(node.left), flag_value(node.right)
        return None if left is None or right is None else left | right
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def subjects(pattern):
    """Deterministic test strings built from the pattern's own characters plus a fixed mixed alphabet.

    A bytes pattern gets the same strings as UTF-8 bytes (multi-byte sequences included)."""
    if isinstance(pattern, bytes):
        return [s.encode("utf-8") for s in subjects(pattern.decode("latin-1"))]
    rng = random.Random(zlib.crc32(pattern.encode("utf-8", "surrogatepass")))
    letters = [c for c in pattern if c.isalnum() or c in " -_./:@=,"] or ["a"]
    alphabet = letters * 3 + list(EXTRA)
    out = ["", "\n", pattern[:24]]
    for _ in range(SAMPLES):
        out.append("".join(rng.choice(alphabet) for _ in range(rng.randint(0, 18))))
    return [s for s in out if not any("\ud800" <= c <= "\udfff" for c in s)]


def _view(m):
    return None if m is None else (m.span(), m.groups())


def operation(fn, repl, limit=0):
    def marker(p):
        return b"<>" if isinstance(p.pattern, bytes) else "<>"

    table = {
        "compile": lambda p, s: (_view(p.search(s)), p.findall(s), p.sub(marker(p), s), p.split(s)),
        "search": lambda p, s: _view(p.search(s)),
        "match": lambda p, s: _view(p.match(s)),
        "fullmatch": lambda p, s: _view(p.fullmatch(s)),
        "findall": lambda p, s: p.findall(s),
        "finditer": lambda p, s: [_view(m) for m in p.finditer(s)],
        "split": lambda p, s: p.split(s, limit),
        "sub": lambda p, s: p.sub(repl, s, limit),
        "subn": lambda p, s: p.subn(repl, s, limit),
    }
    return table[fn]


def run(call, compiled, text):
    try:
        return call(compiled, text)
    except Exception as exc:  # health: allow BLE001 -- any failure of either engine is itself the compared result
        return "raises " + type(exc).__name__


def _compare(call, old, new, texts):
    for text in texts:
        if run(call, old, text) != run(call, new, text):
            return text
    return None


def _on_deadline(signum, frame):
    raise _Deadline


def _within_deadline(work):
    """``work()``, or raise _Deadline after DEADLINE seconds where SIGALRM exists (stdlib matching checks signals)."""
    alarm = getattr(signal, "SIGALRM", None)
    if alarm is None:
        return work()
    previous = signal.signal(alarm, _on_deadline)
    signal.setitimer(signal.ITIMER_REAL, DEADLINE)
    try:
        return work()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(alarm, previous)


def differs(facade, pattern, flags, fn, repl, limit=0):
    """The first generated subject on which the call's result differs, else None; raises _Deadline when stdlib stalls."""
    try:
        old = stdlib_re.compile(pattern, flags)
    except (stdlib_re.error, ValueError, RecursionError):
        return None
    new = facade.compile(pattern, flags)
    call = operation(fn, repl, limit)
    texts = subjects(pattern)
    return _within_deadline(lambda: _compare(call, old, new, texts))


def _surrogate_code(value) -> bool:
    return isinstance(value, int) and 0xD800 <= value <= 0xDFFF


def _names_surrogate(items) -> bool:
    """Whether stdlib's parse tree has a surrogate literal or range end; comments and verbose text are already gone."""
    for op, av in items:
        if op in (_parser.LITERAL, _parser.NOT_LITERAL) and _surrogate_code(av):
            return True
        if op == _parser.RANGE and any(_surrogate_code(end) for end in av):
            return True
        if op == _parser.IN and _names_surrogate(av):
            return True
        nested = av if isinstance(av, (tuple, list)) else ()
        for part in nested:
            if isinstance(part, _parser.SubPattern) and _names_surrogate(part):
                return True
            if isinstance(part, (tuple, list)) and any(isinstance(x, _parser.SubPattern) and _names_surrogate(x)
                                                      for x in part):
                return True
    return False


def names_surrogate(pattern: str, flags: int) -> bool:
    """A lone surrogate written as a character or an escape: stdlib matches one in a subject, hermes_re never does.
    Read through stdlib's own parser, so text inside a `(?#...)` comment does not count."""
    try:
        return _names_surrogate(_parser.parse(pattern, flags & ~stdlib_re.DEBUG))  # DEBUG would print the tree
    except (stdlib_re.error, ValueError, RecursionError, OverflowError):
        return False


UNKNOWN = ast.Name(id="<unpacked>")  # an argument `*args` or `**mapping` may supply: never a literal


def _mapping_value(mapping, name):
    """What a `**mapping` gives for ``name``: its value, None when a literal dict lacks the key, else UNKNOWN."""
    if not isinstance(mapping, ast.Dict) or not all(isinstance(k, ast.Constant) and isinstance(k.value, str)
                                                    for k in mapping.keys):
        return UNKNOWN  # a name, a call, or a dict with `**inner` / computed keys: any key may be there
    return next((v for k, v in zip(mapping.keys, mapping.values) if k.value == name), None)


def argument(node, name, index):
    """The call's argument for parameter ``name`` at positional ``index``: its node, None when the call does not pass
    it, or UNKNOWN when a `*args` at or before ``index`` or a non-literal `**mapping` may supply it."""
    for k in node.keywords:
        if k.arg == name:
            return k.value
    for position, arg in enumerate(node.args):
        if isinstance(arg, ast.Starred):
            return UNKNOWN
        if position == index:
            return arg
    found = None
    for k in node.keywords:
        if k.arg is None:
            value = _mapping_value(k.value, name)
            found = value if value is not None else found
    return found


def call_limit(node, fn):
    """The call's literal count / maxsplit (0 when absent, as in stdlib), or None when it is not a literal int."""
    if fn not in LIMITS:
        return 0
    limit_node = argument(node, *LIMITS[fn])
    if limit_node is None:
        return 0
    if isinstance(limit_node, ast.Constant) and isinstance(limit_node.value, int):
        return limit_node.value
    if isinstance(limit_node, ast.UnaryOp) and isinstance(limit_node.op, ast.USub) \
            and isinstance(limit_node.operand, ast.Constant) and isinstance(limit_node.operand.value, int):
        return -limit_node.operand.value
    return None


def build_error(facade, pattern, flags):
    """Why hermes_re cannot build the pattern with these flags, or None when it builds."""
    try:
        facade.build(pattern, flags)
    except Exception as exc:  # health: allow BLE001 -- every way a pattern fails to build is reported as a finding
        return f"{type(exc).__name__}: {exc}"[:110]
    return None


def classify_unknown_flags(facade, pattern):
    """A finding that holds under EVERY flag combination the call could pass, else DYNAMIC.

    Every other verdict depends on the flags: ASCII makes `\\b` exact and shrinks `\\w` programs under the memory
    budget, VERBOSE turns text into comments."""
    combos = [sum(bit for i, bit in enumerate(UNKNOWN_FLAG_BITS) if mask >> i & 1)
              for mask in range(1 << len(UNKNOWN_FLAG_BITS))]
    if isinstance(pattern, str) and all(names_surrogate(pattern, flags) for flags in combos):
        return "SURROGATE", "the pattern names a lone surrogate (with any flags)"
    first = build_error(facade, pattern, 0)
    if first and all(build_error(facade, pattern, flags) for flags in combos[1:]):
        return "REJECTED", first + " (with any flags)"
    return "DYNAMIC", "not checked (flags not a literal)"


def classify(facade, node, fn, pattern):
    """(kind, detail) for one call with a literal pattern, or None when it is clean."""
    flags = flag_value(argument(node, "flags", FUNCS[fn]))
    if flags is None:
        return classify_unknown_flags(facade, pattern)
    if isinstance(pattern, str) and names_surrogate(pattern, flags):
        return "SURROGATE", "the pattern names a lone surrogate"
    error = build_error(facade, pattern, flags)
    if error:
        return "REJECTED", error
    if isinstance(pattern, str) and facade.inexact_boundaries(pattern, flags):
        return "INEXACT", "ASCII-only word boundary"
    limit = call_limit(node, fn)
    if limit is None:  # only the comparison depends on the limit; the verdicts above hold for any limit
        return "DYNAMIC", f"not checked ({LIMITS[fn][0]} not a literal)"
    repl_node = argument(node, "repl", 1)
    marker = b"<>" if isinstance(pattern, bytes) else "<>"
    repl = repl_node.value if fn in ("sub", "subn") and isinstance(repl_node, ast.Constant) \
        and isinstance(repl_node.value, type(pattern)) else marker
    try:
        example = differs(facade, pattern, flags, fn, repl, limit)
    except _Deadline:
        return "UNCHECKED", f"stdlib re did not finish {fn}() on the generated strings within {DEADLINE:g}s"
    if example is not None:
        return "DIFFERS", f"{fn}() differs from stdlib on {example!r}"[:110]
    return None


def imports_regex(tree, wanted):
    """The file imports one of the ``wanted`` regex modules (absolute imports only)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name in wanted for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and not node.level and node.module in wanted:
            return True
    return False


def called_function(node, wanted):
    """The regex function a call of the canonical tree invokes, or None when it is not one.

    canonical_tree spells every import-bound name in its own scope (``c`` from ``from hermes_re import compile as
    c`` -> ``hermes_re.compile``), so a local ``import re`` shadowing a module-level ``import hermes_re as re``
    stays ``re`` inside that function only."""
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in FUNCS and isinstance(func.value, ast.Name) \
            and getattr(func.value, "imported", False) and not func.value.relative and func.value.id in wanted:
        return func.attr
    return None


def check_file(tools, path, pre_swap):
    facade, canonical_tree, python_comments = tools
    text = Path(path).read_text(encoding="utf-8-sig")
    tree = ast.parse(text)
    wanted = {"hermes_re"} | ({"re"} if pre_swap else set())
    if not imports_regex(tree, wanted):
        return None
    comments = python_comments(text)
    rows = []
    for node in ast.walk(canonical_tree(tree)):
        fn = called_function(node, wanted) if isinstance(node, ast.Call) else None
        arg = argument(node, "pattern", 0) if fn else None
        if arg is None:
            continue
        if not isinstance(arg, ast.Constant) or not isinstance(arg.value, (str, bytes)):
            rows.append(("DYNAMIC", node.lineno, "", "not checked (pattern not a literal)"))
            continue
        found = classify(facade, node, fn, arg.value)
        waived = any(WAIVER in comments.get(line, "") for line in (node.lineno, node.end_lineno or node.lineno))
        if found and not waived:
            rows.append((found[0], node.lineno, repr(arg.value)[:60], found[1]))
    return rows


def tracked_files():
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        capture_output=True, text=True, encoding="utf-8", check=True, cwd=ROOT, timeout=60,
    ).stdout
    return [f for f in out.split("\0") if f and "tests" not in Path(f).parts]


def expand(paths):
    for root in paths:
        root_path = Path(root)
        if root_path.is_dir():
            yield from sorted(str(p) for p in root_path.rglob("*.py")
                              if "tests" not in p.parts and not any(part.startswith(".") for part in p.parts))
        else:
            yield root


def main(argv):
    pre_swap = "--pre-swap" in argv
    verbose = "--verbose" in argv
    out = argv[argv.index("--json") + 1] if "--json" in argv else None
    paths = [a for a in argv if not a.startswith("--") and a != out]
    tools = load_facade()
    files = list(expand(paths)) if paths else [str(ROOT / f) for f in tracked_files()]
    report, bad, dynamic, checked = {}, 0, 0, 0
    for path in files:
        try:
            rows = check_file(tools, path, pre_swap)
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            rows = [("REJECTED", 0, "", f"cannot read or parse the file: {exc}")]
        if rows is None:
            continue
        checked += 1
        for kind, line, shown, why in rows:
            if kind == "DYNAMIC":
                dynamic += 1
                if verbose:
                    print(f"DYNAMIC   {path}:{line}: {shown} {why}")
            else:
                bad += 1
                print(f"{kind:9} {path}:{line}: {shown} {why}")
        if rows:
            report[path] = rows
    if out:
        Path(out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"checked {checked} files that import the regex module: {bad} findings, {dynamic} dynamic sites not checked")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
