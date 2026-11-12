#!/usr/bin/env python3
"""Ratchet unbound thread handoffs by lexical call site, not by file counts.

A reasoned ``thread-hop: ok — <why>`` on the call line or preceding line exempts
intentional process-scoped work (and non-executor submit calls). Direct ctx_bound,
propagate_context_to_thread, _context_thread_target and copy_context().run are
recognized. This is a static guard, not a proof of context propagation: aliases,
dynamic dispatch, executor parameters and Thread subclasses can escape detection.
Executor map is detected only for locally constructed ThreadPoolExecutor receivers.
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci.production_scan import SCAN_ROOT_FILES, SCAN_TREES, in_scope, marked, tracked_files

BASELINE = ROOT / 'scripts/ci/thread_hops_baseline.json'
MARKER = 'thread-hop: ok'
DUMP_OPTIONS = {"show_empty": True} if "show_empty" in inspect.signature(ast.dump).parameters else {}
WRAPPERS = {'ctx_bound', 'propagate_context_to_thread', '_context_thread_target'}


def name(node):
    return getattr(node, 'id', None) or getattr(node, 'attr', None)


def carries(node):
    if isinstance(node, ast.Attribute) and node.attr == 'run':
        return isinstance(node.value, ast.Call) and name(node.value.func) == 'copy_context'
    if isinstance(node, ast.Call):
        return name(node.func) in WRAPPERS or (name(node.func) == 'partial' and bool(node.args) and carries(node.args[0]))
    return isinstance(node, ast.Lambda) and isinstance(node.body, ast.Call) and carries(node.body.func)


def arg(call, pos, keyword):
    return next((k.value for k in call.keywords if k.arg == keyword), call.args[pos] if len(call.args) > pos else None)


class Visitor(ast.NodeVisitor):
    def __init__(self, path, text):
        self.path, self.lines = path, text.splitlines()
        self.scope = []
        self.bindings = [{}]
        self.counts = Counter()
        self.hits = {}

    def lookup(self, key):
        return next((frame[key] for frame in reversed(self.bindings) if key in frame), None)

    def resolve(self, node):
        if isinstance(node, ast.Name):
            return self.lookup(node.id)
        if isinstance(node, ast.Attribute):
            prefix = self.resolve(node.value)
            return f'{prefix}.{node.attr}' if prefix else None
        return None

    def visit_Import(self, node):
        for a in node.names:
            self.bindings[-1][a.asname or a.name.split('.')[0]] = a.name if a.asname else a.name.split('.')[0]

    def visit_ImportFrom(self, node):
        for a in node.names:
            self.bindings[-1][a.asname or a.name] = f'{node.module}.{a.name}'

    def visit_Assign(self, node):
        self.generic_visit(node)
        value = node.value
        kind = 'executor-instance' if isinstance(value, ast.Call) and self.resolve(value.func) == 'concurrent.futures.ThreadPoolExecutor' else None
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.bindings[-1][target.id] = kind

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            value = node.value
            self.bindings[-1][node.target.id] = 'executor-instance' if isinstance(value, ast.Call) and self.resolve(value.func) == 'concurrent.futures.ThreadPoolExecutor' else None

    def visit_With(self, node):
        for item in node.items:
            self.visit(item.context_expr)
            if isinstance(item.optional_vars, ast.Name):
                v = item.context_expr
                self.bindings[-1][item.optional_vars.id] = 'executor-instance' if isinstance(v, ast.Call) and self.resolve(v.func) == 'concurrent.futures.ThreadPoolExecutor' else None
        for child in node.body:
            self.visit(child)
    visit_AsyncWith = visit_With

    def _scope(self, node):
        self.scope.append(node.name)
        self.bindings.append({a.arg: None for a in (node.args.posonlyargs + node.args.args + node.args.kwonlyargs)} if hasattr(node, 'args') else {})
        for child in node.body:
            self.visit(child)
        self.bindings.pop()
        self.scope.pop()
    visit_FunctionDef = _scope
    visit_AsyncFunctionDef = _scope
    visit_ClassDef = _scope

    def visit_Call(self, node):
        fn = node.func
        resolved = self.resolve(fn)
        hit = False
        if resolved in ('threading.Thread', 'threading.Timer'):
            target = arg(node, 1, 'function' if resolved.endswith('Timer') else 'target')
            hit = target is not None and not carries(target)
        elif isinstance(fn, ast.Attribute):
            if fn.attr == 'submit':
                hit = not carries(arg(node, 0, 'fn'))
            elif fn.attr == 'run_in_executor':
                hit = not carries(arg(node, 1, 'func'))
            elif fn.attr == 'map' and (self.resolve(fn.value) == 'executor-instance' or (
                isinstance(fn.value, ast.Call) and self.resolve(fn.value.func) == 'concurrent.futures.ThreadPoolExecutor'
            )):
                hit = True
        if hit and not marked(self.lines, node.lineno, MARKER):
            identity = json.dumps([self.path, '.'.join(self.scope), ast.dump(node, include_attributes=False, **DUMP_OPTIONS)], ensure_ascii=True, separators=(',', ':'))
            self.counts[identity] += 1
            fingerprint = hashlib.sha256((identity + '#' + str(self.counts[identity])).encode()).hexdigest()
            self.hits[fingerprint] = f'{self.path}:{node.lineno}: {self.lines[node.lineno-1].strip()[:160]}'
        self.generic_visit(node)


def sites(path, text):
    visitor = Visitor(path, text)
    visitor.visit(ast.parse(text, filename=path))
    return visitor.hits


def scan(paths=None, root=ROOT):
    hits = {}
    for path in paths if paths is not None else tracked_files(root):
        path = Path(path).resolve()
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = str(path)
        if path.suffix == '.py' and in_scope(rel):
            hits.update(sites(rel, path.read_text(encoding='utf-8-sig')))
    return hits


def parse_baseline(text):
    data = json.loads(text)
    if not isinstance(data, list) or any(not isinstance(x, str) or len(x) != 64 or any(c not in '0123456789abcdef' for c in x) for x in data) or len(set(data)) != len(data):
        raise ValueError('baseline must be a list of unique SHA-256 fingerprints')
    return set(data)


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL, timeout=60).stdout


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('paths', nargs='*')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--update', action='store_true')
    ap.add_argument('--base', help='revision to enforce baseline non-increase (default merge-base with origin/main)')
    args = ap.parse_args(argv)
    if args.update and (args.paths or args.all):
        ap.error('--update cannot be combined with paths or --all')
    try:
        hits = scan([Path(p) for p in args.paths] if args.paths else None)
        baseline = parse_baseline(BASELINE.read_text(encoding='utf-8')) if not (args.paths or args.all) else set()
        if not (args.paths or args.all):
            base = args.base or git('merge-base', 'HEAD', 'origin/main').strip()
            try:
                old = parse_baseline(git('show', f'{base}:scripts/ci/thread_hops_baseline.json'))
            except subprocess.CalledProcessError:
                # Initial introduction only, with no production-source changes from base.
                git('diff', '--quiet', base, '--', *SCAN_TREES, *SCAN_ROOT_FILES)
                if baseline != hits.keys():
                    raise ValueError('initial baseline must exactly match current production sites')
                old = baseline
            if baseline - old:
                raise ValueError(f'baseline adds {len(baseline - old)} site(s) relative to {base}')
        if args.update:
            committed = parse_baseline(git('show', 'HEAD:scripts/ci/thread_hops_baseline.json'))
            baseline &= committed & hits.keys()
            BASELINE.write_text(json.dumps(sorted(baseline), indent=2) + '\n', encoding='utf-8')
        problems = hits.keys() - baseline
        for fingerprint in sorted(problems):
            print(hits[fingerprint])
        if problems:
            print(f'{len(problems)} new unbound thread handoff(s); use a context wrapper or {MARKER} — <why>')
            return 1
        print(f'thread-hops: no new sites ({len(hits)} present, {len(baseline)} baselined)')
        return 0
    except (OSError, UnicodeError, SyntaxError, ValueError, subprocess.SubprocessError, RuntimeError) as exc:
        print(f'thread-hops: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
