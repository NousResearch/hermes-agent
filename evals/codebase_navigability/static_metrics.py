"""Static code metrics for one tree. Usage: python static_metrics.py <tree> <label> -> writes <label>.static.json

First-party Python only (excludes tests/, node_modules, apps/, website, build, .venv, skills md).
"""
import ast, collections, io, json, os, shutil, subprocess, sys, time, tokenize

TREE, LABEL = sys.argv[1], sys.argv[2]
SKIP = {".git", "node_modules", "apps", "website", "build", ".venv", "venv", "MagicMock", "__pycache__", ".worktrees", "dist", "evals", "skills", "optional-skills", "docs"}

def py_files(root, include_tests):
    for dp, dns, fns in os.walk(root):
        rel = os.path.relpath(dp, root)
        top = rel.split(os.sep)[0]
        if top in SKIP: dns[:] = []; continue
        if (top == "tests") != include_tests and rel != ".": dns[:] = []; continue
        if rel == "." and not include_tests: pass
        for f in fns:
            if f.endswith(".py"):
                p = os.path.join(dp, f)
                if include_tests and not os.path.relpath(p, root).startswith("tests"): continue
                if not include_tests and os.path.relpath(p, root).startswith("tests"): continue
                yield p

def code_lines(src):
    """Non-blank, non-comment, non-docstring logical source lines (pygount-style 'code')."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except Exception:
        return sum(1 for l in src.splitlines() if l.strip() and not l.strip().startswith("#")), 0, 0
    code_rows, comment_rows, doc_rows = set(), set(), set()
    # docstrings: STRING tokens that are the first statement of a module/def/class → approximate via ast
    try:
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant) and isinstance(n.body[0].value.value, str):
                for r in range(n.body[0].lineno, n.body[0].end_lineno + 1): doc_rows.add(r)
    except Exception:
        pass
    for t in toks:
        if t.type == tokenize.COMMENT: comment_rows.add(t.start[0])
        elif t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING):
            for r in range(t.start[0], t.end[0] + 1):
                if r not in doc_rows: code_rows.add(r)
    return len(code_rows - comment_rows), len(comment_rows - code_rows), len(doc_rows)

def analyse(files):
    m = {}; per_file = []; funcs = []; if_chains = []; nest = []
    mods = {}; edges = collections.defaultdict(set)
    t0 = time.perf_counter()
    for p in files:
        try: src = open(p, encoding="utf-8", errors="replace").read()
        except Exception: continue
        lines = src.count("\n") + (0 if src.endswith("\n") else 1)
        code, comm, doc = code_lines(src)
        m["files"] = m.get("files", 0) + 1; m["lines"] = m.get("lines", 0) + lines; m["code"] = m.get("code", 0) + code; m["comment"] = m.get("comment", 0) + comm; m["docstring"] = m.get("docstring", 0) + doc; m["bytes"] = m.get("bytes", 0) + len(src.encode())
        per_file.append((lines, p))
        try: tree = ast.parse(src)
        except Exception: m["unparsable"] = m.get("unparsable", 0) + 1; continue
        rel = os.path.relpath(p, TREE)[:-3].replace(os.sep, ".")
        if rel.endswith(".__init__"): rel = rel[:-9]
        mods[rel] = p
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                L = n.end_lineno - n.lineno + 1; funcs.append(L)
                m["functions"] = m.get("functions", 0) + 1
                # nesting depth
                def depth(node, d=0):
                    best = d
                    for c in ast.iter_child_nodes(node):
                        if isinstance(c, (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.AsyncFor, ast.AsyncWith, ast.Match)):
                            best = max(best, depth(c, d + 1))
                        else:
                            best = max(best, depth(c, d))
                    return best
                nest.append(depth(n))
            elif isinstance(n, ast.ClassDef): m["classes"] = m.get("classes", 0) + 1
            elif isinstance(n, ast.If):
                # count elif chain length (If whose orelse is a single If, recursively)
                k, cur = 1, n
                while len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
                    k += 1; cur = cur.orelse[0]
                if k >= 2: if_chains.append(k)
            if isinstance(n, ast.Import):
                for a in n.names: edges[rel].add(a.name)
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                edges[rel].add(n.module)
    m["parse_all_s"] = round(time.perf_counter() - t0, 2)
    funcs.sort(); per_file.sort(reverse=True)
    def pct(a, q): return a[int(len(a) * q) - 1] if a else 0
    m.update({
        "files_gt_1000": sum(1 for L, _ in per_file if L > 1000), "files_gt_2000": sum(1 for L, _ in per_file if L > 2000), "files_gt_5000": sum(1 for L, _ in per_file if L > 5000),
        "largest_files": [(L, os.path.relpath(p, TREE)) for L, p in per_file[:10]],
        "median_file_lines": sorted(L for L, _ in per_file)[len(per_file) // 2] if per_file else 0,
        "funcs_gt_100": sum(1 for L in funcs if L > 100), "funcs_gt_300": sum(1 for L in funcs if L > 300), "func_len_p50": pct(funcs, .5), "func_len_p95": pct(funcs, .95), "func_len_max": funcs[-1] if funcs else 0,
        "elif_chains_ge4": sum(1 for k in if_chains if k >= 4), "elif_chains_ge8": sum(1 for k in if_chains if k >= 8), "longest_elif_chain": max(if_chains) if if_chains else 0,
        "nesting_ge5": sum(1 for d in nest if d >= 5), "nesting_max": max(nest) if nest else 0,
    })
    # first-party import graph
    fp = set(mods)
    def resolve(name):
        while name and name not in fp: name = name.rpartition(".")[0]
        return name
    E = {a: {resolve(b) for b in bs} - {"", a} for a, bs in edges.items() if a in fp}
    E = {a: {b for b in bs if b in fp} for a, bs in E.items()}
    fan_out = [len(v) for v in E.values()]; fan_in = collections.Counter(b for bs in E.values() for b in bs)
    # SCCs (Tarjan) for cycles
    index = {}; low = {}; onstack = set(); stack = []; sccs = []; counter = [0]
    sys.setrecursionlimit(20000)
    def strong(v):
        index[v] = low[v] = counter[0]; counter[0] += 1; stack.append(v); onstack.add(v)
        for w in E.get(v, ()):
            if w not in index: strong(w); low[v] = min(low[v], low[w])
            elif w in onstack: low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop(); onstack.discard(w); comp.append(w)
                if w == v: break
            sccs.append(comp)
    for v in list(E):
        if v not in index: strong(v)
    cyc = [c for c in sccs if len(c) > 1]
    m.update({"modules": len(fp), "import_edges": sum(fan_out), "fan_out_avg": round(sum(fan_out) / max(1, len(fan_out)), 2), "fan_out_max": max(fan_out) if fan_out else 0,
              "fan_in_max": max(fan_in.values()) if fan_in else 0, "fan_in_top": fan_in.most_common(5), "import_cycles": len(cyc), "largest_cycle": max((len(c) for c in cyc), default=0), "modules_in_cycles": sum(len(c) for c in cyc)})
    return dict(m)

def radon(files):
    """radon cc + mi over the file list (JSON), aggregated."""
    R = shutil.which("radon") or os.path.join(os.path.dirname(sys.executable), "radon")
    import tempfile
    out = {}
    lst = "\n".join(files)
    # radon can't take a list file; run per-directory roots instead
    roots = sorted({os.path.relpath(f, TREE).split(os.sep)[0] if os.sep in os.path.relpath(f, TREE) else os.path.relpath(f, TREE) for f in files})
    roots = [r for r in roots if r != "tests"]
    cc = subprocess.run([R, "cc", "-j", "-s", "-e", "tests/*,tests/**", *roots], cwd=TREE, capture_output=True, text=True).stdout
    mi = subprocess.run([R, "mi", "-j", "-e", "tests/*,tests/**", *roots], cwd=TREE, capture_output=True, text=True).stdout
    try:
        ccj = json.loads(cc); blocks = [b for v in ccj.values() if isinstance(v, list) for b in v if isinstance(b, dict) and "complexity" in b]
        cs = sorted(b["complexity"] for b in blocks)
        out.update({"cc_blocks": len(cs), "cc_avg": round(sum(cs) / max(1, len(cs)), 2), "cc_p95": cs[int(len(cs) * .95) - 1] if cs else 0, "cc_max": cs[-1] if cs else 0,
                    "cc_gt_15": sum(1 for c in cs if c > 15), "cc_gt_30": sum(1 for c in cs if c > 30), "cc_gt_50": sum(1 for c in cs if c > 50),
                    "cc_worst": sorted(((b["complexity"], f, b["name"]) for f, v in ccj.items() if isinstance(v, list) for b in v if isinstance(b, dict) and "complexity" in b), reverse=True)[:8]})
    except Exception as e: out["cc_error"] = repr(e)[:200]
    try:
        mij = json.loads(mi); vals = [v["mi"] for v in mij.values() if isinstance(v, dict) and "mi" in v]
        out.update({"mi_files": len(vals), "mi_avg": round(sum(vals) / max(1, len(vals)), 2), "mi_lt_20": sum(1 for v in vals if v < 20), "mi_lt_10": sum(1 for v in vals if v < 10)})
    except Exception as e: out["mi_error"] = repr(e)[:200]
    return out

src_files = sorted(py_files(TREE, False)); test_files = sorted(py_files(TREE, True))
res = {"label": LABEL, "tree": TREE, "source": analyse(src_files), "tests": analyse(test_files)}
res["source"].update({"radon_" + k: v for k, v in radon(src_files).items()})
OUT = os.environ.get("NAV_OUT", "."); os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, f"{LABEL}.static.json"), "w", encoding="utf-8"), indent=1, default=str)
s = res["source"]; t = res["tests"]
print(f"{LABEL}: src files={s['files']} lines={s['lines']} code={s['code']} funcs={s['functions']} >300={s['funcs_gt_300']} files>2k={s['files_gt_2000']} >5k={s['files_gt_5000']} cycles={s['import_cycles']} cc_avg={s.get('radon_cc_avg')} mi_avg={s.get('radon_mi_avg')} | tests files={t['files']} lines={t['lines']}")
