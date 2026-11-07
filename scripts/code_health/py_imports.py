"""Module-level import graph: the #14182 dependency DAG and module-level acyclicity.

A layer edge and an import cycle are properties of two modules, not of one function, so they
are judged by a whole-tree pass instead of the per-unit ratchet in ``compare``. The semantics
are nevertheless the ratchet's own: the cap is the count measured on the **base revision**, so
the debt of today is frozen and may only go down, and a *new* violation fails the run.

    LAYER_EDGE    a module-level edge in a direction GRAPH_FORBIDDEN forbids
    IMPORT_CYCLE  a strongly connected component that did not exist on the base revision

Only direct children of ``ast.Module`` are edges: the documented convention is a late import
inside a function ("siblings late-import the facade inside functions; never a module-level
cycle"), and a call-time import is not a cycle. The same reading keeps the count honest --
counting call-time imports would report 325 ``hermes_cli.config`` sites instead of the 28 that
are actually module-level.

Scope is product Python (``GRAPH_EXCLUDE``): tests, scripts, plugins, the desktop app and the
docs site are out, because a plugin legitimately imports anything and a test may cross any line.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from scripts.code_health.config import GRAPH_FORBIDDEN, GRAPH_LAYERS
from scripts.code_health.model import Finding


@dataclass(frozen=True)
class Site:
    """One occurrence of an edge: where the import statement is, for the report."""

    path: str
    line: int


@dataclass
class Graph:
    modules: dict[str, str] = field(default_factory=dict)  # module name -> path
    edges: dict[str, dict[str, list[Site]]] = field(default_factory=dict)  # src -> dst -> sites

    def forbidden_pairs(self) -> dict[tuple[str, str], list[tuple[str, str]]]:
        """Forbidden band pairs -> the module edges that produce them."""
        out: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for src, targets in self.edges.items():
            for dst in targets:
                key = (layer(src), layer(dst))
                if key in FORBIDDEN:
                    out.setdefault(key, []).append((src, dst))
        return out


FORBIDDEN = frozenset(GRAPH_FORBIDDEN)


def layer(module: str) -> str:
    """Band of a module: its top-level package mapped through GRAPH_LAYERS, else a leaf."""
    return GRAPH_LAYERS.get(module.split(".", 1)[0], "leaf")


def module_name(path: str) -> str:
    parts = path[:-3].split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve(modules: dict[str, str], base: list[str], name: str) -> str | None:
    """The module a name points at: the longest prefix present in the tree."""
    candidate = ".".join([*base, name]) if name else ".".join(base)
    while candidate:
        if candidate in modules:
            return candidate
        candidate = candidate.rsplit(".", 1)[0] if "." in candidate else ""
    return None


def _targets(node: ast.stmt, pkg: list[str], modules: dict[str, str]) -> list[str]:
    """Every module a module-level import statement names (absolute, relative, from-import)."""
    if isinstance(node, ast.Import):
        return [m for m in (_resolve(modules, [], alias.name) for alias in node.names) if m]
    if not isinstance(node, ast.ImportFrom):
        return []
    base = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else (pkg if node.level else [])
    names: list[str] = []
    if node.module:
        names.append(node.module)
        # `from pkg import submodule` may name a module of its own: try it as well.
        names.extend(".".join([node.module, alias.name]) for alias in node.names)
    else:
        names.extend(alias.name for alias in node.names)
    return [m for m in (_resolve(modules, base, name) for name in names) if m]


def _edges_of(path: str, text: str, modules: dict[str, str]) -> dict[str, list[Site]]:
    """Module-level edges of one file, resolved against the tree's module set."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    src = module_name(path)
    pkg = src.split(".")[:-1]
    edges: dict[str, list[Site]] = {}
    for node in tree.body:  # module level only: a call-time import is not an edge
        for dst in _targets(node, pkg, modules):
            if dst == src:
                continue
            edges.setdefault(dst, []).append(Site(path, node.lineno))
    return edges


def scan(contents: dict[str, str]) -> Graph:
    """Build the module-level graph from path -> source. An unparseable file is skipped: the
    per-file MEASURE finding already fails the run for it."""
    graph = Graph(modules={module_name(path): path for path in contents})
    for path, text in contents.items():
        edges = _edges_of(path, text, graph.modules)
        if edges:
            graph.edges[module_name(path)] = edges
    return graph


def derive_base(head: Graph, changed_at_head: list[str],
                base_contents: dict[str, str]) -> Graph:
    """The base graph without a second whole-tree scan.

    A file whose content did not change has the same edges, so the base graph is the head graph
    minus the outgoing edges of the files this change touched, plus the same files' base edges
    (``base_contents``: base-side text of every changed path, including deleted ones).
    """
    base = Graph(modules=dict(head.modules), edges=dict(head.edges))
    for path in changed_at_head:
        base.modules.pop(module_name(path), None)
        base.edges.pop(module_name(path), None)
    base.modules.update({module_name(path): path for path in base_contents})
    for path, text in base_contents.items():
        edges = _edges_of(path, text, base.modules)
        if edges:
            base.edges[module_name(path)] = edges
    return base


def sccs(graph: Graph) -> list[tuple[str, ...]]:
    """Strongly connected components with more than one module -- Kosaraju, iterative: the
    graph is ~2.5k nodes and a recursive walk would sit on the recursion limit."""
    nodes = set(graph.edges) | {dst for targets in graph.edges.values() for dst in targets}
    reverse: dict[str, list[str]] = {node: [] for node in nodes}
    for src, targets in graph.edges.items():
        for dst in targets:
            reverse[dst].append(src)
    order: list[str] = []
    seen: set[str] = set()
    for start in sorted(nodes):
        if start in seen:
            continue
        stack: list[tuple[str, bool]] = [(start, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if node in seen:
                continue
            seen.add(node)
            stack.append((node, True))
            for nxt in graph.edges.get(node, ()):
                if nxt not in seen:
                    stack.append((nxt, False))
    components: list[tuple[str, ...]] = []
    seen.clear()
    for start in reversed(order):
        if start in seen:
            continue
        group = [start]
        seen.add(start)
        stack = [start]
        while stack:
            for nxt in reverse.get(stack.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    group.append(nxt)
                    stack.append(nxt)
        if len(group) > 1:
            components.append(tuple(sorted(group)))
    return sorted(components)


def _site(graph: Graph, src: str, dst: str) -> Site:
    return graph.edges[src][dst][0]


def _anchor(graph: Graph, component: tuple[str, ...], base: Graph | None) -> Site:
    """Where to point a new cycle: an edge inside it that is new, else its first edge."""
    members = set(component)
    for src in component:
        for dst in graph.edges.get(src, {}):
            if dst in members and (base is None or dst not in base.edges.get(src, {})):
                return _site(graph, src, dst)
    for src in component:
        for dst in graph.edges.get(src, {}):
            if dst in members:
                return _site(graph, src, dst)
    raise AssertionError(f"component without an internal edge: {component}")


def summary(graph: Graph) -> str:
    """Burn-down view of the whole tree, the same numbers the gate compares against."""
    edges = sum(len(targets) for targets in graph.edges.values())
    lines = [f"Import graph (module-level, #14182 DAG): {len(graph.modules)} modules, "
             f"{edges} edges"]
    pairs = graph.forbidden_pairs()
    lines.append("  forbidden directions (the cap is what main measures):")
    for pair in sorted(FORBIDDEN):
        found = pairs.get(pair)
        value = len(found) if found else 0
        where = f"  worst target: {_worst_target(graph, found)}" if found else ""
        lines.append(f"    {pair[0]:<8} -> {pair[1]:<9} {value:>5}{where}")
    components = sccs(graph)
    modules = sum(len(component) for component in components)
    lines.append(f"  module-level cycles: {len(components)} ({modules} modules)")
    for component in components:
        lines.append(f"    [{len(component)}] {', '.join(component[:4])}"
                     + (" ..." if len(component) > 4 else ""))
    return "\n".join(lines)


def _worst_target(graph: Graph, edges: list[tuple[str, str]]) -> str:
    counts: dict[str, int] = {}
    for _, dst in edges:
        counts[dst] = counts.get(dst, 0) + 1
    dst, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    return f"{dst} ({count})"


def gate(base: Graph | None, head: Graph) -> list[Finding]:
    """New violations only: the base measurement is the cap, exactly like TARGETS."""
    findings: list[Finding] = []
    base_pairs = base.forbidden_pairs() if base is not None else {}
    for pair, edges in sorted(head.forbidden_pairs().items()):
        cap = len(base_pairs.get(pair, ()))
        if len(edges) <= cap:
            continue
        frozen_edges = set(base_pairs.get(pair, ()))
        # One statement may read as two edges (`from pkg import submodule` names both): one
        # finding per import site, not per reading of it.
        sites: dict[Site, tuple[str, list[str]]] = {}
        for src, dst in edges:
            if (src, dst) not in frozen_edges:
                site, targets = sites.setdefault(_site(head, src, dst), (src, []))
                targets.append(dst)
        for site, (src, targets) in sites.items():
            findings.append(Finding(
                site.path, "LAYER_EDGE", f"{src} -> {', '.join(targets)}", site.line,
                f"{pair[0]} -> {pair[1]}: {len(edges)} module-level edges, cap {cap} on main"
                f" (+{len(edges) - cap} new); the #14182 DAG does not allow this direction"))
    frozen = {frozenset(component) for component in sccs(base)} if base is not None else set()
    for component in sccs(head):
        if frozenset(component) in frozen:
            continue
        site = _anchor(head, component, base)
        members = ", ".join(component[:4]) + (" ..." if len(component) > 4 else "")
        findings.append(Finding(
            site.path, "IMPORT_CYCLE", site.path.rsplit("/", 1)[0], site.line,
            f"module-level cycle ({len(component)} modules: {members}) that does not exist on"
            f" main; a sibling cycle is what AGENTS.md forbids"))
    return findings
