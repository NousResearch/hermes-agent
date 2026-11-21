#!/usr/bin/env python3
"""Static gate for the RE2 migration: every literal regex of product code must behave on `hermes_re` as on stdlib `re`.

    python scripts/check_re2_patterns.py                      # every tracked non-test .py file
    python scripts/check_re2_patterns.py [--pre-swap] [--verbose] [--json OUT] PATH...

For every call `<alias>.compile/search/match/fullmatch/findall/finditer/split/sub/subn(<string literal>, ...)` where
<alias> is the name the file binds to the regex module:
  REJECTED   hermes_re refuses the pattern, or its RE2 program cannot be built
  SURROGATE  the pattern names a lone surrogate (U+D800..U+DFFF); RE2 can never match one
  INEXACT    the pattern has a `\\b` / `\\B` that RE2 evaluates with ASCII word rules
  DIFFERS    the call's operation gives another result than stdlib `re` on generated strings
  DYNAMIC    the first argument is not a string literal: counted, not checked (listed with --verbose)
Any finding except DYNAMIC makes the exit status 1. One call is waived with a comment on its first or last line:
    # re2: reviewed -- <reason>
Without --pre-swap only files that import `hermes_re` are checked (the state after their swap). With --pre-swap,
files that still say `import re` are checked as if already swapped: run this BEFORE swapping a file.
Reads files as text; imports only `hermes_re`, which needs the google-re2 package (exit 2 with a message otherwise).
This script is a development tool under scripts/ and uses stdlib `re` as the reference engine.
"""
import ast
import importlib
import json
import random
import re as stdlib_re
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
FUNCS = {"compile": 1, "search": 2, "match": 2, "fullmatch": 2, "findall": 2, "finditer": 2, "split": 3, "sub": 4, "subn": 4}
FLAGS = {"I": 2, "IGNORECASE": 2, "M": 8, "MULTILINE": 8, "S": 16, "DOTALL": 16, "A": 256, "ASCII": 256, "U": 32,
         "UNICODE": 32, "X": 64, "VERBOSE": 64, "NOFLAG": 0}
WAIVER = "# re2: reviewed --"
EXTRA = " \n\t-_.,:;/|@=\"'()aZ09\u00e9\u00df\u0436\u65e5\u017f\u0130"
SAMPLES = 160


def load_facade():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        hermes_re = importlib.import_module("hermes_re")
        hermes_re.build("a")
    except ImportError as exc:
        print(f"check_re2_patterns: hermes_re / google-re2 is not importable in {sys.executable} ({exc}); run it in the "
              "repository environment (source ./activate, or: just check --only re2-patterns)")
        sys.exit(2)
    return hermes_re


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
    """Deterministic test strings built from the pattern's own characters plus a fixed mixed alphabet."""
    rng = random.Random(zlib.crc32(pattern.encode("utf-8", "surrogatepass")))
    letters = [c for c in pattern if c.isalnum() or c in " -_./:@=,"] or ["a"]
    alphabet = letters * 3 + list(EXTRA)
    out = ["", "\n", pattern[:24]]
    for _ in range(SAMPLES):
        out.append("".join(rng.choice(alphabet) for _ in range(rng.randint(0, 18))))
    return [s for s in out if not any("\ud800" <= c <= "\udfff" for c in s)]


def _view(m):
    return None if m is None else (m.span(), m.groups())


def operation(fn, repl):
    table = {
        "compile": lambda p, s: (_view(p.search(s)), p.findall(s), p.sub("<>", s), p.split(s)),
        "search": lambda p, s: _view(p.search(s)),
        "match": lambda p, s: _view(p.match(s)),
        "fullmatch": lambda p, s: _view(p.fullmatch(s)),
        "findall": lambda p, s: p.findall(s),
        "finditer": lambda p, s: [_view(m) for m in p.finditer(s)],
        "split": lambda p, s: p.split(s),
        "sub": lambda p, s: p.sub(repl, s),
        "subn": lambda p, s: p.subn(repl, s),
    }
    return table[fn]


def run(call, compiled, text):
    try:
        return call(compiled, text)
    except Exception as exc:  # health: allow BLE001 -- any failure of either engine is itself the compared result
        return "raises " + type(exc).__name__


def differs(facade, pattern, flags, fn, repl):
    try:
        old = stdlib_re.compile(pattern, flags)
    except (stdlib_re.error, ValueError, RecursionError):
        return None
    new = facade.compile(pattern, flags)
    call = operation(fn, repl)
    for text in subjects(pattern) if isinstance(pattern, str) else ():
        if run(call, old, text) != run(call, new, text):
            return text
    return None


def classify(facade, node, fn):
    """(kind, detail) for one call with a literal pattern, or None when it is clean."""
    pattern = node.args[0].value
    flag_node = next((k.value for k in node.keywords if k.arg == "flags"), None)
    index = FUNCS[fn]
    if flag_node is None and len(node.args) > index:
        flag_node = node.args[index]
    flags = flag_value(flag_node)
    note = "" if flags is not None else " (flags not a literal: checked with flags=0)"
    flags = flags or 0
    if isinstance(pattern, str) and any("\ud800" <= c <= "\udfff" for c in pattern):
        return "SURROGATE", "lone surrogate in the pattern"
    try:
        facade.build(pattern, flags)
    except Exception as exc:  # health: allow BLE001 -- every way a pattern fails to build is reported as a finding
        return "REJECTED", f"{type(exc).__name__}: {exc}"[:110] + note
    if isinstance(pattern, str) and facade.inexact_boundaries(pattern, flags):
        return "INEXACT", "ASCII-only word boundary" + note
    repl = node.args[1].value if fn in ("sub", "subn") and len(node.args) > 1 and isinstance(node.args[1], ast.Constant) \
        and isinstance(node.args[1].value, str) else "<>"
    example = differs(facade, pattern, flags, fn, repl)
    if example is not None:
        return "DIFFERS", f"{fn}() differs from stdlib on {example!r}"[:110] + note
    return None


def aliases(tree, pre_swap):
    wanted = {"hermes_re"} | ({"re"} if pre_swap else set())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.asname or a.name for a in node.names if a.name in wanted}
    return found


def check_file(facade, path, pre_swap):
    text = Path(path).read_text(encoding="utf-8-sig")
    tree = ast.parse(text)
    names = aliases(tree, pre_swap)
    if not names:
        return None
    lines = text.splitlines()
    rows = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in FUNCS
                and isinstance(node.func.value, ast.Name) and node.func.value.id in names and node.args):
            continue
        arg = node.args[0]
        if not isinstance(arg, ast.Constant) or not isinstance(arg.value, (str, bytes)):
            rows.append(("DYNAMIC", node.lineno, "", ""))
            continue
        found = classify(facade, node, node.func.attr)
        last = lines[(node.end_lineno or node.lineno) - 1]
        if found and WAIVER not in last and WAIVER not in lines[node.lineno - 1]:
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
    facade = load_facade()
    files = list(expand(paths)) if paths else [str(ROOT / f) for f in tracked_files()]
    report, bad, dynamic, checked = {}, 0, 0, 0
    for path in files:
        try:
            rows = check_file(facade, path, pre_swap)
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            rows = [("REJECTED", 0, "", f"cannot read or parse the file: {exc}")]
        if rows is None:
            continue
        checked += 1
        for kind, line, shown, why in rows:
            if kind == "DYNAMIC":
                dynamic += 1
                if verbose:
                    print(f"DYNAMIC   {path}:{line}")
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
