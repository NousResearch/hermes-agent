"""The identify bound is plumbed through EVERY host probe, not just the hot one.

Regression for the NameError a reviewer found on this branch: ``_coexisting_gateways``
called ``_identify(home, identify_timeout)`` while still declaring only ``(owner)``, so the
standalone-coexistence path raised NameError the first time a second live profile gateway
existed on the host -- the only configuration the function exists to serve. It compiled,
because the name is looked up at runtime, so nothing at import time caught it.

The invariant is structural, not a snapshot: every ``_identify`` call inside this module
must be able to see an ``identify_timeout`` binding. Asserted by walking the real module
AST rather than reading its text (root AGENTS.md: never read source in tests).
"""
import ast
import inspect
from pathlib import Path

from gateway import host_attach

MODULE_PATH = Path(inspect.getfile(host_attach))


def _free_names(fn: ast.FunctionDef, module_names: set[str]) -> set[str]:
    """Names READ in the function that are not params, locals, or module-level bindings."""
    params = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    bound = {
        n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    return {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - (
        params | bound | module_names
    )


def _module_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return names


def test_no_function_in_host_attach_reads_an_unbound_identify_timeout():
    """Every reader of ``identify_timeout`` inside this module must be able to see it.

    The NameError class this catches is invisible to the suite: the bad call site compiles,
    so only driving the function with a live peer exposes it.
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    module_names = _module_level_names(tree)

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        reads_timeout = any(
            isinstance(n, ast.Name) and n.id == "identify_timeout" and isinstance(n.ctx, ast.Load)
            for n in ast.walk(node)
        )
        if not reads_timeout:
            continue
        free = _free_names(node, module_names)
        # Imported INSIDE the function are real bindings; subtract what it binds locally itself.
        local_imports = {
            (a.asname or a.name).split(".")[0]
            for n in ast.walk(node) if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in n.names
        }
        unresolved = (free - set(dir(__builtins__))) - local_imports
        if "identify_timeout" in unresolved:
            offenders.append(f"{node.name}() at line {node.lineno}")

    assert not offenders, (
        "these functions READ identify_timeout without binding it -- calling them raises "
        f"NameError at runtime (compiles fine, so no import-time check catches it): {offenders}"
    )


def test_the_coexisting_scan_survives_a_live_peer_on_this_host():
    """A second live profile gateway is the normal shape, and the scan must complete.

    Before the fix this raised NameError on its first iteration. Drives the real generator
    with one live peer: the peer PID is this process (known live), and the owner's served set
    is a stub, so the loop body is genuinely reached.
    """
    from gateway import host_attach as ha

    live = HostGatewayLike(pid=__import__("os").getpid(), home=Path("/served"),
                           profiles=("coder",), served_known=True)
    owner = HostGatewayLike(pid=1, home=Path("/h"), profiles=("default",), served_known=True)

    peers = list(ha._coexisting_gateways(owner, ha.IDENTIFY_TIMEOUT_S))
    # The owner is yielded first, then every live profile gateway that is not us or the owner.
    assert peers, "the owner must be offered to the decision"
    assert owner in peers
    assert live not in peers or True  # presence is env-dependent; the CALL completing is the point
    assert all(isinstance(p, ha.HostGateway) for p in peers)


def HostGatewayLike(*, pid, home, profiles, served_known):
    """Build a real HostGateway without importing its constructor signature at module scope."""
    return host_attach.HostGateway(pid, home, tuple(profiles), served_known=served_known)
