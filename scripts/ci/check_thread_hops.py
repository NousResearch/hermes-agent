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
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from scripts.ci.production_scan import SCAN_ROOT_FILES, SCAN_TREES, in_scope, marked, tracked_files

BASELINE = ROOT / 'scripts/ci/thread_hops_baseline.json'
MARKER = 'thread-hop: ok'
DUMP_OPTIONS = {"show_empty": True} if "show_empty" in inspect.signature(ast.dump).parameters else {}
WRAPPERS = {'agent.memory_provider.ctx_bound', 'tools.thread_context.propagate_context_to_thread',
            'agent.chat_completion_helpers._context_thread_target'}


def arg(call, pos, keyword):
    return next((k.value for k in call.keywords if k.arg == keyword), call.args[pos] if len(call.args) > pos else None)


class Visitor(ast.NodeVisitor):
    def __init__(self, path, text):
        self.path, self.lines = path, text.splitlines()
        self.scope = []
        self.bindings = [{}]
        self.scope_kinds = ["module"]
        self.counts = Counter()
        self.hits = {}

    def lookup(self, key):
        skip_classes = False
        for frame, kind in zip(reversed(self.bindings), reversed(self.scope_kinds)):
            if kind == 'class' and skip_classes:
                continue
            if key in frame:
                return frame[key]
            if kind == 'function':
                skip_classes = True
        return None

    def resolve(self, node):
        if isinstance(node, ast.Name):
            return self.lookup(node.id)
        if isinstance(node, ast.Attribute):
            prefix = self.resolve(node.value)
            return f'{prefix}.{node.attr}' if prefix else None
        return None

    def carries(self, node):
        if isinstance(node, ast.Attribute) and node.attr == 'run':
            return (isinstance(node.value, ast.Call) and
                    self.resolve(node.value.func) == 'contextvars.copy_context')
        if isinstance(node, ast.Call):
            resolved = self.resolve(node.func)
            return resolved in WRAPPERS or (resolved == 'functools.partial' and bool(node.args) and self.carries(node.args[0]))
        if isinstance(node, ast.Lambda) and isinstance(node.body, ast.Call):
            self.bindings.append(self.parameters(node.args))
            self.scope_kinds.append('function')
            try:
                return self.carries(node.body.func)
            finally:
                self.scope_kinds.pop()
                self.bindings.pop()
        return False

    def visit_Import(self, node):
        for a in node.names:
            self.bindings[-1][a.asname or a.name.split('.')[0]] = a.name if a.asname else a.name.split('.')[0]

    def visit_ImportFrom(self, node):
        for a in node.names:
            self.bindings[-1][a.asname or a.name] = f'{"." * node.level}{node.module}.{a.name}'

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

    @staticmethod
    def parameters(args):
        params = args.posonlyargs + args.args + args.kwonlyargs
        return {a.arg: None for a in params + [a for a in (args.vararg, args.kwarg) if a]}

    def function_expressions(self, node):
        for decorator in node.decorator_list:
            self.visit(decorator)
        for value in node.args.defaults + [v for v in node.args.kw_defaults if v]:
            self.visit(value)
        for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs + [a for a in (node.args.vararg, node.args.kwarg) if a]:
            if a.annotation:
                self.visit(a.annotation)
        if node.returns:
            self.visit(node.returns)
        for param in getattr(node, 'type_params', []):
            self.visit(param)

    def visit_FunctionDef(self, node):
        self.function_expressions(node)
        self.scope.append(node.name)
        self.bindings.append(self.parameters(node.args))
        self.scope_kinds.append('function')
        for child in node.body:
            self.visit(child)
        self.scope_kinds.pop()
        self.bindings.pop()
        self.scope.pop()
        self.bindings[-1][node.name] = None
    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        for child in node.decorator_list + node.bases + node.keywords + list(getattr(node, 'type_params', [])):
            self.visit(child)
        self.scope.append(node.name)
        self.bindings.append({})
        self.scope_kinds.append('class')
        for child in node.body:
            self.visit(child)
        self.scope_kinds.pop()
        self.bindings.pop()
        self.scope.pop()
        self.bindings[-1][node.name] = None

    def visit_Lambda(self, node):
        for default in node.args.defaults + [v for v in node.args.kw_defaults if v]:
            self.visit(default)
        self.bindings.append(self.parameters(node.args))
        self.scope_kinds.append('function')
        self.visit(node.body)
        self.scope_kinds.pop()
        self.bindings.pop()

    def visit_comprehension_scope(self, node):
        generators = node.generators
        self.visit(generators[0].iter)
        self.bindings.append({})
        self.scope_kinds.append('function')
        for i, gen in enumerate(generators):
            if i:
                self.visit(gen.iter)
            for child in ast.walk(gen.target):
                if isinstance(child, ast.Name):
                    self.bindings[-1][child.id] = None
            for condition in gen.ifs:
                self.visit(condition)
        for value in ([node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]):
            self.visit(value)
        self.scope_kinds.pop()
        self.bindings.pop()
    visit_ListComp = visit_SetComp = visit_GeneratorExp = visit_DictComp = visit_comprehension_scope

    def visit_Call(self, node):
        fn = node.func
        resolved = self.resolve(fn)
        hit = False
        if resolved in ('threading.Thread', 'threading.Timer'):
            target = arg(node, 1, 'function' if resolved.endswith('Timer') else 'target')
            hit = target is not None and not self.carries(target)
        elif isinstance(fn, ast.Attribute):
            if fn.attr == 'submit':
                hit = not self.carries(arg(node, 0, 'fn'))
            elif fn.attr == 'run_in_executor':
                hit = not self.carries(arg(node, 1, 'func'))
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
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True, encoding='utf-8', errors='replace', check=True, stdin=subprocess.DEVNULL, timeout=60).stdout


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('paths', nargs='*')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--update', action='store_true')
    ap.add_argument('--base', help='revision to enforce baseline non-increase')
    ap.add_argument('--head', help='judged commit or staged tree')
    args = ap.parse_args(argv)
    if args.update and (args.paths or args.all):
        ap.error('--update cannot be combined with paths or --all')
    try:
        hits = scan([Path(p) for p in args.paths] if args.paths else None)
        baseline = parse_baseline(BASELINE.read_text(encoding='utf-8-sig')) if not (args.paths or args.all) else set()
        if not (args.paths or args.all):
            from scripts.code_health import gitio
            tip = gitio.commit_or_none(ROOT, args.head) if args.head else None
            base = args.base or gitio.default_base(ROOT, tip or 'HEAD')
            git('rev-parse', '--verify', f'{base}^{{commit}}')
            exists = git('ls-tree', '--name-only', base, '--', 'scripts/ci/thread_hops_baseline.json').strip()
            if exists:
                old = parse_baseline(git('show', f'{base}:scripts/ci/thread_hops_baseline.json'))
            else:
                # Initial introduction only, with no production-source changes from base.
                git('diff', '--quiet', base, *([args.head] if args.head else []), '--', *SCAN_TREES, *SCAN_ROOT_FILES)
                if baseline != hits.keys():
                    raise ValueError('initial baseline must exactly match current production sites')
                old = baseline
            if baseline - old:
                raise ValueError(f'baseline adds {len(baseline - old)} site(s) relative to {base}')
        if args.update:
            baseline &= old & hits.keys()
            BASELINE.write_text(json.dumps(sorted(baseline), indent=2) + '\n', encoding='utf-8')
        stale = baseline - hits.keys()
        if stale:
            raise ValueError(f'baseline contains {len(stale)} stale site(s); run --update to prune')
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
