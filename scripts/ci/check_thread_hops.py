#!/usr/bin/env python3
"""Ratchet unbound thread handoffs by lexical call site, not by file counts.

A real comment ``thread-hop: ok — <why>`` on the call line or a standalone preceding comment line
exempts a site; wrappers do not exempt calls automatically. Matches Thread and
Timer with an explicit target/function through syntactically imported names
(even if later rebound), any .submit
and .run_in_executor, and .map on a directly constructed ThreadPoolExecutor or
name assigned one anywhere in the file. These file-wide name heuristics can yield
false positives; use a reasoned comment. Dynamic aliases, subclasses, unknown
executor receivers and indirect dispatch are unsupported and can escape detection.
For example, `from concurrent import futures` is not a recognized pool import.
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from scripts.ci.production_scan import SCAN_ROOT_FILES, SCAN_TREES, in_scope, comment_lines, marked
from scripts.code_health import gitio

BASELINE = ROOT / 'scripts/ci/thread_hops_baseline.json'
MARKER = 'thread-hop: ok'
DUMP_OPTIONS = {"show_empty": True} if "show_empty" in inspect.signature(ast.dump).parameters else {}
def arg(call, pos, keyword):
    return next((k.value for k in call.keywords if k.arg == keyword), call.args[pos] if len(call.args) > pos else None)


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else None
    return None


class Visitor(ast.NodeVisitor):
    def __init__(self, path, text, tree):
        self.path, self.lines = path, text.splitlines()
        self.comments = comment_lines(text, MARKER)
        self.scope = []
        self.counts = Counter()
        self.marker_uses = Counter()
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
        hit = ((name in self.threads and arg(node, 1, 'target') is not None) or
               (name in self.timers and arg(node, 1, 'function') is not None))
        if isinstance(fn, ast.Attribute):
            hit |= fn.attr in ('submit', 'run_in_executor')
            hit |= fn.attr == 'map' and (self.pool_call(fn.value) or
                                           isinstance(fn.value, ast.Name) and fn.value.id in self.instances)
        if hit and marked(self.lines, node.lineno, MARKER, comments=self.comments):
            self.marker_uses[node.lineno] += 1
            hit = self.marker_uses[node.lineno] > 1
        if hit:
            identity = json.dumps([self.path, '.'.join(self.scope), ast.dump(node, include_attributes=False, **DUMP_OPTIONS)], ensure_ascii=True, separators=(',', ':'))
            self.counts[identity] += 1
            fingerprint = hashlib.sha256((identity + '#' + str(self.counts[identity])).encode()).hexdigest()
            self.hits[fingerprint] = f'{self.path}:{node.lineno}: {self.lines[node.lineno-1].strip()[:160]}'
        self.generic_visit(node)


def sites(path, text):
    tree = ast.parse(text, filename=path)
    visitor = Visitor(path, text, tree)
    visitor.visit(tree)
    return visitor.hits


def source_texts(root, tree, paths):
    """Read one exact tree with one Git process and a disk-backed bounded spool."""
    if tree is None:
        for rel in paths:
            yield rel, (root / rel).read_text(encoding='utf-8-sig')
        return
    if not paths:
        return
    requests = b''.join(f'{tree}:{rel}'.encode() + b'\0' for rel in paths)
    # cat-file -z NUL-delimits input only; blob output is size-framed, not line-delimited.
    with tempfile.TemporaryFile() as spool:
        subprocess.run(['git', 'cat-file', '--batch', '-z'], cwd=root, input=requests,
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
    hits = {}
    if paths is not None:
        files = paths
    elif tree is not None:
        files = gitio.tracked_files(root, tree)
    else:
        # Include new nonignored files, but do not mistake an unstaged removal for unreadable source.
        files = set(gitio.git(root, 'ls-files', '--cached', '--others', '--exclude-standard', '-z').split('\0'))
        files -= set(gitio.git(root, 'ls-files', '--deleted', '-z').split('\0'))
    selected = []
    for path in files:
        path = Path(path)
        if path.is_absolute():
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                rel = str(path)
        else:
            rel = path.as_posix()
        if path.suffix == '.py' and in_scope(rel) and (paths is not None or rel.split('/')[0] in (*SCAN_TREES, *SCAN_ROOT_FILES)):
            selected.append(rel)
    for rel, text in source_texts(root, tree, sorted(set(selected))):
        hits.update(sites(rel, text))
    return hits


def parse_baseline(text):
    data = json.loads(text)
    if not isinstance(data, list) or any(not isinstance(x, str) or len(x) != 64 or any(c not in '0123456789abcdef' for c in x) for x in data) or len(set(data)) != len(data):
        raise ValueError('baseline must be a list of unique SHA-256 fingerprints')
    return set(data)


def terminal_safe(text):
    return "".join(c if c.isprintable() else c.encode("unicode_escape").decode("ascii") for c in text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
                changed = gitio.changed_files(ROOT, base, tree)
                if any((c.old and (c.old.split('/')[0] in (*SCAN_TREES, *SCAN_ROOT_FILES))) or
                       (c.new and (c.new.split('/')[0] in (*SCAN_TREES, *SCAN_ROOT_FILES))) for c in changed):
                    raise ValueError('initial baseline requires unchanged production sources')
                if baseline != hits.keys():
                    raise ValueError('initial baseline must exactly match current production sites')
                old = baseline
            if baseline - old:
                raise ValueError(f'baseline adds {len(baseline - old)} site(s) relative to {base}')
        else:
            baseline = set()
        if args.update:
            baseline &= old & hits.keys()
            BASELINE.write_text(json.dumps(sorted(baseline), indent=2) + '\n', encoding='utf-8')
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
