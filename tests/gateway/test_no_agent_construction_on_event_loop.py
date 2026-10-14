"""No gateway coroutine may construct an AIAgent on the event loop.

#123702: ``AIAgent.__init__`` loads the context engine, and concurrent worker turns hold
that process-global load lock — so a constructor can run for tens of seconds. Two
coroutines built one inline:

- ``GatewayRunner._hmwa_hygiene_build_agent`` (gateway/run_turn.py)
- ``GatewayRunner._build_manual_compression_agent`` (gateway/slash_commands_session.py)

Either one held the event loop past the adapter's heartbeat ACK window, the socket was
closed, and the adapter reconnected mid-conversation. The surrounding work in both was
already off-loop; only construction was not.

This is an AST contract over every gateway module rather than a test of the two known
sites, so the next inline construction fails here instead of in production.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

GATEWAY_DIR = Path(__file__).resolve().parents[2] / "gateway"

# Constructors that block. A nested def/lambda is the established off-loop idiom
# (run_sync() in run_turn.py, and the executor hop in run_inbound.py for #111091), so a
# construction inside one is fine — only a DIRECT call from a coroutine is the bug.
BLOCKING_CALLS = {"AIAgent", "_check_unavailable_skill"}


def _direct_calls(fn: ast.AST, name: str) -> list[int]:
    """Lines where *fn*'s own body calls *name* — skipping nested defs/lambdas/classes.

    The nested-function guard has to be at the top of the recursion, not inside the loop over
    children: ``walk(stmt)`` is handed each of ``fn.body``'s statements, so a ``def run_sync()``
    statement is passed in *as the node* and its body is what gets iterated.
    """
    found: list[int] = []

    def walk(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            return  # a nested def/lambda is the off-loop idiom (run_sync(), to_thread(run_sync))
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Call):
                func = child.func
                called = getattr(func, "id", None) or getattr(func, "attr", None)
                if called == name:
                    found.append(child.lineno)
            walk(child)

    for stmt in fn.body:
        walk(stmt)
    return found


def _offending_sites() -> list[tuple[str, int, str]]:
    sites: list[tuple[str, int, str]] = []
    for path in sorted(GATEWAY_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # a file mid-edit; not this test's business
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            for name in BLOCKING_CALLS:
                for lineno in _direct_calls(node, name):
                    rel = path.relative_to(GATEWAY_DIR.parent).as_posix()
                    sites.append((rel, lineno, f"{node.name}() calls {name}()"))
    return sites


def test_no_coroutine_constructs_an_agent_or_scans_skills_on_the_event_loop():
    """The whole bug class, over every gateway module — not just today's two sites."""
    sites = _offending_sites()
    assert not sites, "blocking construction on the event loop (move it behind asyncio.to_thread):\n" + "\n".join(
        f"  {rel}:{line}  {why}" for rel, line, why in sites
    )


@pytest.mark.parametrize("name", sorted(BLOCKING_CALLS))
def test_the_contract_actually_catches_a_direct_call(name):
    """Negative control. If the walker stopped descending correctly, or the matcher broke,
    this would pass and the test above would silently guard nothing."""
    tree = ast.parse(
        "async def victim():\n"
        f"    agent = {name}()\n"
        "async def safe_offload():\n"
        "    def run_sync():\n"
        f"        return {name}()\n"
        "    return await asyncio.to_thread(run_sync)\n"
    )
    victim = next(n for n in tree.body if getattr(n, "name", None) == "victim")
    safe = next(n for n in tree.body if getattr(n, "name", None) == "safe_offload")
    assert _direct_calls(victim, name), f"the walker missed a direct {name}() call"
    assert not _direct_calls(safe, name), f"a nested off-loop {name}() must not be flagged"


def test_the_two_reported_sites_are_offloaded():
    """Pin the reported regressions by name, so the fix cannot be undone by a refactor
    that keeps some other site compliant."""
    run_turn = (GATEWAY_DIR / "run_turn.py").read_text(encoding="utf-8")
    session = (GATEWAY_DIR / "slash_commands_session.py").read_text(encoding="utf-8")

    for label, source, fn in (
        ("_hmwa_hygiene_build_agent", run_turn, "_hmwa_hygiene_build_agent"),
        ("_build_manual_compression_agent", session, "_build_manual_compression_agent"),
    ):
        tree = ast.parse(source)
        node = next(
            (n for n in ast.walk(tree)
             if isinstance(n, ast.AsyncFunctionDef) and n.name == fn),
            None,
        )
        assert node is not None, f"{label} not found"
        assert not _direct_calls(node, "AIAgent"), (
            f"{label} constructs AIAgent() directly on the event loop again"
        )
        # And it must actually offload rather than merely avoid the direct call.
        assert "to_thread" in ast.unparse(node), (
            f"{label} no longer offloads construction; expected asyncio.to_thread"
        )