#!/usr/bin/env python3
"""Graphify knowledge graph — token-efficient codebase index for agents.

Builds a compact static index of the checkout so "where is X?" lookups cost
a small JSON read instead of a full-tree grep + multi-file reads:

    python scripts/graphify_knowledge.py build
    python scripts/graphify_knowledge.py query --symbol AIAgent --limit 10
    python scripts/graphify_knowledge.py query --text "memory provider"
    python scripts/graphify_knowledge.py area --name tools --limit 30
    python scripts/graphify_knowledge.py route "where is cron scheduling?"

Output (gitignored cache, regenerable): ``.graphify/`` with
``manifest.json`` (areas, facade families, hub modules), ``areas/<area>.json``
(one compact record per file) and ``symbols.jsonl`` (top-level defs only).

Token contract: ``query``/``area``/``route`` print capped one-line hits
(``path:line kind name — summary``); they never dump file contents.
Rebuild when the tree changes; the manifest carries the git sha + mtime.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / ".graphify"

SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", ".worktrees",
    "dist", "build", ".pytest_cache", ".hermes", ".hermes-sandbox",
    "MagicMock", "mini-swe-agent", "browser-use", "agent-browser",
    # Docs site + test/build outputs: no agent-runtime source worth indexing.
    "website", "test-results", "playwright-report", "testlogs", "logs",
    "wandb", ".direnv", "hermes_agent.egg-info",
}
SKIP_PATH_PARTS = {"static", "snapshots", "__snapshots__"}
SKIP_SUFFIXES = (".pyc", ".pyo", ".so", ".png", ".jpg", ".mp4", ".tgz", ".tar.gz")
PY_RE = re.compile(r"\.py$")
TS_RE = re.compile(r"\.(ts|tsx|js|mjs|cjs)$")
TS_EXPORT_RE = re.compile(
    r"^\s*export\s+(?:default\s+)?(?:async\s+)?(?:class|function|interface|type|const|enum)\s+([A-Za-z_$][\w$]*)"
    r"|^\s*export\s*\{\s*([^}]{1,400}?)\s*\}",
    re.MULTILINE,
)
MAX_SUMMARY = 120
MAX_DEFINES_PER_FILE = 40
MAX_IMPORTS_PER_FILE = 8
MAX_HITS_DEFAULT = 15

_ROUTE_STOPWORDS = frozenset({
    "where", "what", "which", "how", "who", "when", "why", "does", "do",
    "the", "and", "for", "with", "from", "that", "this", "about",
})


def _first_line(text: str, limit: int = MAX_SUMMARY) -> str:
    for line in (text or "").splitlines():
        s = line.strip().strip('"').strip("'").strip()
        if s:
            return s[:limit]
    return ""


def parse_python_source(src: str) -> dict:
    """Parse source text into a compact node (no I/O; pure for tests)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            tree = ast.parse(src)
        except (SyntaxError, ValueError):
            return {"summary": "", "defines": [], "imports": []}
    summary = ""
    try:
        summary = _first_line(ast.get_docstring(tree) or "")
    except Exception:
        summary = ""
    defines: list[str] = []
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef,)):
            defines.append(f"C:{node.name}:{node.lineno}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nargs = len(node.args.args) + len(node.args.kwonlyargs)
            defines.append(f"F:{node.name}:{node.lineno}:{nargs}")
        elif isinstance(node, (ast.Import,)):
            for a in node.names:
                imports.append(a.name.split(".")[0])
        elif isinstance(node, (ast.ImportFrom,)):
            if node.module and node.level == 0:
                imports.append(node.module.split(".")[0])
        if len(defines) >= MAX_DEFINES_PER_FILE:
            break
    return {"summary": summary, "defines": defines, "imports": imports[:MAX_IMPORTS_PER_FILE]}


def ts_exports(src: str) -> list[str]:
    """Best-effort export names from TS/JS source (regex, pure for tests)."""
    names: list[str] = []
    for m in TS_EXPORT_RE.finditer(src[:200_000]):
        if m.group(1):
            names.append(m.group(1)[:80])
        elif m.group(2):
            for part in m.group(2).split(","):
                base = part.strip().split(" as ")[-1].strip().strip('"').strip("'")
                if re.match(r"^[A-Za-z_$][\w$]*$", base):
                    names.append(base[:80])
        if len(names) >= MAX_DEFINES_PER_FILE:
            break
    seen: dict[str, None] = {}
    for n in names:
        seen.setdefault(n, None)
    return [f"J:{n}" for n in seen][:MAX_DEFINES_PER_FILE]


def area_of(rel: str) -> str:
    """Top-level area for a repo-relative path (pure for tests)."""
    rel = rel.replace("\\", "/")
    if "/" not in rel:
        return "root"
    return rel.split("/")[0]


def iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        rel_dir = os.path.relpath(dirpath, root).replace("\\", "/")
        if rel_dir != "." and any(p in SKIP_PATH_PARTS for p in rel_dir.split("/")):
            continue
        for f in filenames:
            if f.endswith(SKIP_SUFFIXES):
                continue
            if PY_RE.search(f) or TS_RE.search(f) or f == "AGENTS.md":
                yield Path(dirpath) / f


# Only the head of a file is parsed for defines/exports; line counts use the
# full text. Keeps huge generated files from stalling the build.
PARSE_HEAD_CHARS = 300_000
READ_HARD_CAP = 2_000_000


def _git_sha(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip()[:12]
    except Exception:
        return ""


def build_index(root: Path = REPO_ROOT) -> dict:
    """Walk the tree and return the full graph dict (areas + symbols)."""
    files: list[dict] = []
    symbols: list[dict] = []
    fan_in: dict[str, int] = {}
    first_party: set[str] = set()
    rels: list[str] = []
    for p in iter_files(root):
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            continue
        rels.append(rel)
    first_party = {r[:-3].replace("/", ".") for r in rels if r.endswith(".py")}
    for rel in sorted(rels):
        p = root / rel
        try:
            st = p.stat()
        except OSError:
            continue
        rec: dict = {"p": rel, "l": 0, "s": "", "d": [], "i": []}
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                src = f.read(READ_HARD_CAP)
        except OSError:
            continue
        rec["l"] = src.count("\n") + 1
        head = src[:PARSE_HEAD_CHARS]
        if rel.endswith(".py"):
            node = parse_python_source(head)
            rec["s"] = node["summary"]
            rec["d"] = node["defines"]
            own = rel[:-3].replace("/", ".")
            clean: list[str] = []
            for imp in node["imports"]:
                top = imp.split(".")[0]
                if top in first_party or top in {r.split("/")[0] for r in rels}:
                    clean.append(top)
            rec["i"] = clean[:MAX_IMPORTS_PER_FILE]
            for d in node["defines"]:
                parts = d.split(":")
                symbols.append({"n": parts[1].lower(), "raw": d, "p": rel})
        elif TS_RE.search(rel):
            # Minified bundles (one giant line) carry no useful export names.
            lines = rec["l"] or 1
            if len(src) / lines > 2000:
                rec["d"] = []
            else:
                rec["d"] = ts_exports(head)
            for d in rec["d"]:
                symbols.append({"n": d[2:].lower(), "raw": d, "p": rel})
        elif rel.endswith("AGENTS.md"):
            rec["s"] = _first_line(src)
        files.append(rec)
    for rec in files:
        for imp in rec.get("i", []):
            fan_in[imp] = fan_in.get(imp, 0) + 1
    areas: dict[str, list[dict]] = {}
    for rec in files:
        areas.setdefault(area_of(rec["p"]), []).append(rec)
    families = _facade_families([r["p"] for r in files])
    hubs = sorted(fan_in.items(), key=lambda kv: -kv[1])[:10]
    return {
        "files": files, "symbols": symbols, "areas": areas,
        "families": families, "hubs": hubs,
    }


def _facade_families(paths: list[str]) -> list[dict]:
    """Group facade + sibling files sharing dir + stem prefix (pure helper)."""
    groups: dict[str, list[str]] = {}
    for rel in paths:
        if not rel.endswith(".py"):
            continue
        d, _, base = rel.rpartition("/")
        stem = base[:-3]
        prefix = stem.split("_")[0] if "_" in stem else stem
        key = f"{d}/{prefix}" if d else prefix
        groups.setdefault(key, []).append(rel)
    fams = [
        {"f": k, "n": len(v), "m": sorted(v)[:6]}
        for k, v in groups.items() if len(v) >= 3
    ]
    fams.sort(key=lambda f: -f["n"])
    return fams[:25]


def _load_graph(out: Path = DEFAULT_OUT, *, need: str = "all") -> dict:
    man_path = out / "manifest.json"
    if not man_path.is_file():
        raise SystemExit(f"no graph at {out} — run `build` first")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    areas: dict[str, list[dict]] = {}
    syms: list[dict] = []
    if need in ("all", "areas"):
        for a in man.get("areas", []):
            fp = out / "areas" / f"{a['name']}.json"
            if fp.is_file():
                areas[a["name"]] = json.loads(fp.read_text(encoding="utf-8"))
    if need in ("all", "symbols"):
        sp = out / "symbols.jsonl"
        if sp.is_file():
            for line in sp.read_text(encoding="utf-8").splitlines():
                try:
                    syms.append(json.loads(line))
                except ValueError:
                    continue
    return {"manifest": man, "areas": areas, "symbols": syms}


def _is_test_path(path: str) -> bool:
    return path.startswith("tests/") or "/tests/" in path or "test_" in path.rsplit("/", 1)[-1]


def _fmt_hit(path: str, raw: str, summary: str = "") -> str:
    tail = f" — {summary}" if summary else ""
    return f"{path} {raw}{tail}"[:220]


def search_symbols(symbols: list[dict], query: str, limit: int) -> list[str]:
    """Case-insensitive substring match over symbol names (pure for tests)."""
    q = query.lower().strip()
    if not q:
        return []
    scored: list[tuple[int, dict]] = []
    for s in symbols:
        name = s["n"]
        if name == q:
            scored.append((0, s))
        elif name.startswith(q):
            scored.append((1, s))
        elif q in name:
            scored.append((2, s))
    # Implementation first: test files mirror names and would otherwise
    # bury the defining module.
    scored.sort(key=lambda t: (_is_test_path(t[1]["p"]), t[0], t[1]["n"], t[1]["p"]))
    return [_fmt_hit(s["p"], s["raw"]) for _, s in scored[:limit]]


def search_text(files: list[dict], query: str, limit: int) -> list[str]:
    """Substring match over paths + one-line summaries (pure for tests)."""
    q = query.lower().strip()
    if not q:
        return []
    hits: list[tuple[int, dict]] = []
    for f in files:
        score = 99
        if q in f["p"].lower():
            score = 0
        elif q in (f.get("s") or "").lower():
            score = 1
        elif any(q in d.lower() for d in f.get("d", [])):
            score = 2
        if score < 99:
            hits.append((score, f))
    hits.sort(key=lambda t: (_is_test_path(t[1]["p"]), t[0], t[1]["p"]))
    out = []
    for _, f in hits[:limit]:
        defs = ",".join(d.split(":")[1] for d in f.get("d", [])[:5])
        tail = f" [{defs}]" if defs else ""
        summ = f" — {f['s']}" if f.get("s") else ""
        out.append(f"{f['p']}:{f['l']}l{tail}{summ}"[:220])
    return out


def cmd_build(args: argparse.Namespace) -> int:
    out = Path(args.out or DEFAULT_OUT)
    t0 = time.perf_counter()
    graph = build_index(REPO_ROOT)
    (out / "areas").mkdir(parents=True, exist_ok=True)
    area_rows = []
    for name in sorted(graph["areas"]):
        recs = graph["areas"][name]
        fp = out / "areas" / f"{name}.json"
        fp.write_text(json.dumps(recs, separators=(",", ":")), encoding="utf-8")
        area_rows.append({"name": name, "files": len(recs),
                          "bytes": fp.stat().st_size})
    sp = out / "symbols.jsonl"
    with open(sp, "w", encoding="utf-8") as f:
        for s in graph["symbols"]:
            f.write(json.dumps(s, separators=(",", ":")) + "\n")
    manifest = {
        "version": 1, "sha": _git_sha(REPO_ROOT),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_s": round(time.perf_counter() - t0, 1),
        "files": len(graph["files"]), "symbols": len(graph["symbols"]),
        "areas": area_rows, "families": graph["families"], "hubs": graph["hubs"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    total = sum(a["bytes"] for a in area_rows) + sp.stat().st_size
    print(f"graphify: {manifest['files']} files, {manifest['symbols']} symbols, "
          f"{total / 1024:.0f} KB in {manifest['build_s']}s -> {out}")
    for a in sorted(area_rows, key=lambda a: -a["files"])[:12]:
        print(f"  {a['name']}: {a['files']} files")
    return 0


def _all_files(graph: dict) -> list[dict]:
    files: list[dict] = []
    for recs in graph["areas"].values():
        files.extend(recs)
    by_path = {f["p"]: f for f in files}
    return list(by_path.values())


def cmd_query(args: argparse.Namespace) -> int:
    want_symbols = bool(args.symbol)
    want_text = bool(args.text)
    if not want_symbols and not want_text:
        raise SystemExit("query needs --symbol and/or --text")
    limit = args.limit or MAX_HITS_DEFAULT
    out = Path(args.out or DEFAULT_OUT)
    if want_symbols:
        graph = _load_graph(out, need="symbols")
        for h in search_symbols(graph["symbols"], args.symbol, limit):
            print(h)
    if want_text:
        graph = _load_graph(out, need="areas")
        for h in search_text(_all_files(graph), args.text, limit):
            print(h)
    return 0


def cmd_area(args: argparse.Namespace) -> int:
    out = Path(args.out or DEFAULT_OUT)
    man_path = out / "manifest.json"
    if not man_path.is_file():
        raise SystemExit(f"no graph at {out} — run `build` first")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    names = [a["name"] for a in man.get("areas", [])]
    if args.name not in names:
        raise SystemExit(f"unknown area '{args.name}' — areas: {', '.join(sorted(names))}")
    fp = out / "areas" / f"{args.name}.json"
    recs = json.loads(fp.read_text(encoding="utf-8"))
    for f in sorted(recs, key=lambda r: r["p"])[: args.limit or 50]:
        defs = ",".join(d.split(":")[1] for d in f.get("d", [])[:6])
        tail = f" [{defs}]" if defs else ""
        summ = f" — {f['s']}" if f.get("s") else ""
        print(f"{f['p']}:{f['l']}l{tail}{summ}"[:220])
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    """Keyword route: rank areas + families for a natural-language question."""
    graph = _load_graph(Path(args.out or DEFAULT_OUT), need="areas")
    man = graph["manifest"]
    q = " ".join(args.words).lower()
    tokens = [t for t in re.split(r"[^a-z0-9_]+", q)
              if len(t) > 2 and t not in _ROUTE_STOPWORDS]
    files = _all_files(graph)
    # OR-match per token so "where is cron scheduling" still finds cron/*.
    seen: dict[str, str] = {}
    for tok in tokens[:6]:
        for h in search_text(files, tok, 5):
            seen.setdefault(h.split(" ")[0], h)
    hits = list(seen.values())[:8]
    area_score: dict[str, int] = {}
    for h in hits:
        name = area_of(h.split(" ")[0])
        area_score[name] = area_score.get(name, 0) + 1
    for tok in tokens:
        for a in man.get("areas", []):
            if tok in a["name"].lower():
                area_score[a["name"]] = area_score.get(a["name"], 0) + 2
    print(f"question: {' '.join(args.words)}"[:200])
    if area_score:
        top = sorted(area_score.items(), key=lambda kv: -kv[1])[:3]
        print("areas: " + ", ".join(f"{n} ({c})" for n, c in top))
    fams = [f for f in man.get("families", [])
            if any(t in f["f"].lower() for t in tokens)]
    for f in fams[:3]:
        print(f"family: {f['f']} x{f['n']} e.g. {f['m'][0]}")
    for h in hits[:8]:
        print(f"hit: {h}")
    print("next: query --symbol <Name> or area --name <area> for the slice")
    return 0


_CMDS = {"build": cmd_build, "query": cmd_query, "area": cmd_area, "route": cmd_route}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Token-efficient codebase knowledge graph")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="walk tree and write .graphify/")
    b.set_defaults(fn="build")
    q = sub.add_parser("query", help="capped one-line hits, no file dumps")
    q.add_argument("--symbol", default="")
    q.add_argument("--text", default="")
    q.add_argument("--limit", type=int, default=MAX_HITS_DEFAULT)
    q.set_defaults(fn="query")
    a = sub.add_parser("area", help="list one area slice")
    a.add_argument("--name", required=True)
    a.add_argument("--limit", type=int, default=50)
    a.set_defaults(fn="area")
    r = sub.add_parser("route", help="rank areas for a question")
    r.add_argument("words", nargs="+")
    r.set_defaults(fn="route")
    args = ap.parse_args(argv)
    return _CMDS[args.fn](args)


if __name__ == "__main__":
    sys.exit(main())
