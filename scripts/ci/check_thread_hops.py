#!/usr/bin/env python3
"""Ratchet unbound thread handoffs by lexical call site, not by file counts.

A real comment ``thread-hop: ok — <why>`` on the call line or a standalone preceding comment line
exempts the first handoff visited there; wrappers do not exempt calls automatically. Matches
Thread and Timer with an explicit target/function, or any */** argument, through syntactically
imported names (even if later rebound), any .submit
and .run_in_executor, and .map on a directly constructed ThreadPoolExecutor or
name assigned one anywhere in the file. These file-wide name heuristics can yield
false positives; use a reasoned comment. Dynamic aliases, subclasses, unknown
executor receivers and indirect dispatch are unsupported and can escape detection.
For example, `from concurrent import futures` is not a recognized pool import.
Sites are keyed by their pre-move path (hermes_agent/compat/moves.py), so a pure module
move keeps its baseline entries; renaming the enclosing function or class does not.
The baseline captures existing hits, not a claim that every hit is unsafe.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from scripts.ci.production_scan import comment_lines, in_scope, is_production, original_paths, production_files
from scripts.code_health import gitio
from scripts.code_health.py_ast import _dotted as dotted

BASELINE = ROOT / 'scripts/ci/thread_hops_baseline.json'
MARKER = 'thread-hop: ok'
DUMP_OPTIONS = {"show_empty": True} if "show_empty" in inspect.signature(ast.dump).parameters else {}
HANDOFF_WORDS = ('Thread', 'Timer', 'submit', 'run_in_executor')


def passes_callable(call, pos, keyword):
    if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
        return True
    return any(k.arg == keyword for k in call.keywords) or len(call.args) > pos


class Visitor(ast.NodeVisitor):
    def __init__(self, path, text, tree, identity_path=None):
        self.path, self.identity_path, self.lines = path, identity_path or path, text.splitlines()
        self.comments = comment_lines(text, MARKER)
        self.scope = []
        self.counts = Counter()
        self.marker_lines = set()
        self.hits = {}
        self.threads, self.timers, self.pools = set(), set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == 'threading':
                        self.threads.add((alias.asname or 'threading') + '.Thread')
                        self.timers.add((alias.asname or 'threading') + '.Timer')
                    elif alias.name == 'concurrent.futures':
                        self.pools.add((alias.asname or 'concurrent.futures') + '.ThreadPoolExecutor')
            elif isinstance(node, ast.ImportFrom) and not node.level:
                for alias in node.names:
                    name = alias.asname or alias.name
                    if node.module == 'threading' and alias.name == 'Thread':
                        self.threads.add(name)
                    elif node.module == 'threading' and alias.name == 'Timer':
                        self.timers.add(name)
                    elif node.module == 'concurrent.futures' and alias.name == 'ThreadPoolExecutor':
                        self.pools.add(name)
        self.instances = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if self.pool_call(node.value):
                    self.instances.update(t.id for t in targets if isinstance(t, ast.Name))
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                self.instances.update(item.optional_vars.id for item in node.items
                                      if isinstance(item.optional_vars, ast.Name) and self.pool_call(item.context_expr))

    def first_marked_handoff(self, lineno):
        if lineno in self.marker_lines:
            return False
        above = lineno - 1 in self.comments and self.lines[lineno - 2].lstrip().startswith('#')
        return lineno in self.comments or above

    def pool_call(self, node):
        return isinstance(node, ast.Call) and dotted(node.func) in self.pools

    def visit_FunctionDef(self, node):
        for child in node.decorator_list + node.args.defaults + [v for v in node.args.kw_defaults if v]:
            self.visit(child)
        for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs + [a for a in (node.args.vararg, node.args.kwarg) if a]:
            if a.annotation:
                self.visit(a.annotation)
        if node.returns:
            self.visit(node.returns)
        for param in getattr(node, 'type_params', []):
            self.visit(param)
        self.scope.append(node.name)
        for child in node.body:
            self.visit(child)
        self.scope.pop()
    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        for child in node.decorator_list + node.bases + node.keywords + list(getattr(node, 'type_params', [])):
            self.visit(child)
        self.scope.append(node.name)
        for child in node.body:
            self.visit(child)
        self.scope.pop()

    def visit_Call(self, node):
        fn = node.func
        name = dotted(fn)
        hit = ((name in self.threads and passes_callable(node, 1, 'target')) or
               (name in self.timers and passes_callable(node, 1, 'function')))
        if isinstance(fn, ast.Attribute):
            hit |= fn.attr in ('submit', 'run_in_executor')
            hit |= fn.attr == 'map' and (self.pool_call(fn.value) or
                                           isinstance(fn.value, ast.Name) and fn.value.id in self.instances)
        if hit:
            identity = json.dumps([self.identity_path, '.'.join(self.scope), ast.dump(node, include_attributes=False, **DUMP_OPTIONS)], ensure_ascii=True, separators=(',', ':'))
            self.counts[identity] += 1
            if self.first_marked_handoff(node.lineno):
                self.marker_lines.add(node.lineno)
            else:
                fingerprint = hashlib.sha256((identity + '#' + str(self.counts[identity])).encode()).hexdigest()
                self.hits[fingerprint] = f'{self.path}:{node.lineno}: {self.lines[node.lineno-1].strip()[:160]}'
        self.generic_visit(node)


def sites(path, text, identity_path=None):
    if not any(word in text for word in HANDOFF_WORDS):
        normalized = unicodedata.normalize('NFKC', text)
        if not any(word in normalized for word in HANDOFF_WORDS):
            return {}
    tree = ast.parse(text, filename=path)
    visitor = Visitor(path, text, tree, identity_path)
    visitor.visit(tree)
    return visitor.hits


def source_texts(root, tree, paths):
    if tree is None:
        for rel in paths:
            yield rel, (root / rel).read_text(encoding='utf-8-sig')
        return
    if not paths:
        return
    # Ask for blobs by object id: path requests need `cat-file -z` (Git 2.38+) to carry any file name.
    listing = gitio.git(root, 'ls-tree', '-r', '-z', '--full-tree', tree).split('\0')
    oids = {rel: meta.split()[2] for meta, _, rel in (entry.partition('\t') for entry in listing if entry)}
    if missing := [rel for rel in paths if rel not in oids]:
        raise ValueError(f'cannot read {missing[0]} from judged tree')
    requests = ''.join(oids[rel] + '\n' for rel in paths).encode()
    with tempfile.TemporaryFile() as spool:
        subprocess.run(['git', 'cat-file', '--batch'], cwd=root, input=requests,
                       stdout=spool, stderr=subprocess.PIPE, timeout=60, check=True)
        spool.seek(0)
        for rel in paths:
            header = spool.readline().split()
            if len(header) != 3 or header[1] != b'blob':
                raise ValueError(f'cannot read {rel} from judged tree')
            size = int(header[2])
            data = spool.read(size)
            if len(data) != size or spool.read(1) != b'\n':
                raise ValueError(f'truncated blob for {rel}')
            yield rel, data.decode('utf-8-sig')
        if spool.read(1):
            raise ValueError('unexpected trailing tree data')


def scan(paths=None, root=ROOT, tree=None):
    if paths is None:
        files = production_files(root, tree, untracked=True)
    else:
        original, generated = original_paths(root, tree), gitio.generated_files(root, tree)
        files = {}
        for path in map(Path, paths):
            resolved = (path if path.is_absolute() else root / path).resolve()
            if not resolved.is_relative_to(root.resolve()):
                raise ValueError(f'{path} is outside {root}')
            rel = resolved.relative_to(root.resolve()).as_posix()
            if path.suffix == '.py' and in_scope(rel) and rel not in generated:
                files[rel] = original(rel)
    if clashes := sorted(old for old, n in Counter(files.values()).items() if n > 1):
        raise ValueError(f'{clashes[0]} is both a moved file and an undeclared old-path copy; '
                         'mark the copy generated in .gitattributes or delete it')
    hits = {}
    for rel, text in source_texts(root, tree, sorted(files)):
        hits.update(sites(rel, text, files[rel]))
    return hits


def parse_baseline(text):
    data = json.loads(text)
    if not isinstance(data, list) or any(not isinstance(x, str) or len(x) != 64 or any(c not in '0123456789abcdef' for c in x) for x in data) or len(set(data)) != len(data):
        raise ValueError('baseline must be a list of unique SHA-256 fingerprints')
    return set(data)


def terminal_safe(text):
    return "".join(c if c.isprintable() else c.encode("unicode_escape").decode("ascii") for c in text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    ap.add_argument('paths', nargs='*')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--update', action='store_true')
    ap.add_argument('--base', help='revision to enforce baseline non-increase')
    ap.add_argument('--head', help='judged commit or staged tree')
    args = ap.parse_args(argv)
    if args.update and (args.paths or args.all or args.head):
        ap.error('--update cannot be combined with paths, --all or --head')
    try:
        tree = gitio.resolve_tree(ROOT, args.head) if args.head else None
        hits = scan([Path(p) for p in args.paths] if args.paths else None, root=ROOT, tree=tree)
        if not (args.paths or args.all):
            baseline_text = gitio.read_file(ROOT, tree, 'scripts/ci/thread_hops_baseline.json')
            if baseline_text is None:
                raise ValueError('judged tree has no baseline')
            baseline = parse_baseline(baseline_text)
            tip = gitio.commit_or_none(ROOT, args.head) if args.head else None
            base = gitio.resolve_rev(ROOT, args.base or gitio.default_base(ROOT, tip or 'HEAD'))
            old_text = gitio.read_file(ROOT, base, 'scripts/ci/thread_hops_baseline.json')
            if old_text is not None:
                old = parse_baseline(old_text)
            else:
                original = original_paths(ROOT, tree)
                changed = gitio.changed_files(ROOT, base, tree)
                if any(is_production(original(path)) for c in changed for path in (c.old, c.new) if path):
                    raise ValueError('initial baseline requires unchanged production sources')
                if baseline != hits.keys():
                    raise ValueError('initial baseline must exactly match current production sites')
                old = baseline
            if args.update:
                baseline &= old & hits.keys()
                BASELINE.write_text(json.dumps(sorted(baseline), indent=2) + '\n', encoding='utf-8')
            if baseline - old:
                raise ValueError(f'baseline adds {len(baseline - old)} site(s) relative to {base}')
        else:
            baseline = set()
        stale = baseline - hits.keys()
        if stale:
            raise ValueError(f'baseline contains {len(stale)} stale site(s); run --update to prune')
        problems = hits.keys() - baseline
        for fingerprint in sorted(problems):
            print(terminal_safe(hits[fingerprint]))
        if problems:
            print(f'{len(problems)} new thread handoff site(s); assess context propagation and document exceptions with {MARKER} — <why>')
            return 1
        print(f'thread-hops: no new sites ({len(hits)} present, {len(baseline)} baselined)')
        return 0
    except (OSError, UnicodeError, SyntaxError, ValueError, subprocess.SubprocessError, RuntimeError) as exc:
        print(f'thread-hops: {terminal_safe(str(exc))}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
