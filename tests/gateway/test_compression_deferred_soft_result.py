"""Gateway must treat ``compression_deferred`` as a soft result (#49874).

A lock-contended compression defer means a CONCURRENT compressor is actively
shrinking the session — the opposite of ``compression_exhausted`` (session
permanently too large). The gateway's auto-reset (#9893/#35809) must never
fire for a deferred turn: the session stays intact and the next message
retries normally.

AST invariants on ``gateway/run.py`` (mirrors
``test_35809_auto_reset_clean_context.py``'s load-bearing pin style):

* the ``compression_deferred`` branch guards the auto-reset block — a
  deferred result can never reach ``reset_session``;
* the deferred branch itself performs NO session mutation (no
  ``reset_session``, no ``_evict_cached_agent``, no
  ``_clear_conversation_scope``).
"""

from __future__ import annotations

import ast
import inspect

from gateway import run as gateway_run


def _calls(node: ast.AST) -> set[str]:
    return {
        n.func.attr
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }


def _find_deferred_guarded_reset_chain() -> ast.If:
    """Return the ``if agent_result.get('compression_deferred') ... elif
    agent_result.get('compression_exhausted') ... reset_session`` chain."""
    tree = ast.parse(inspect.getsource(gateway_run))

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_consts = [
            n.value
            for n in ast.walk(node.test)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        ]
        if "compression_deferred" not in test_consts:
            continue
        # The reset must live in the orelse (elif compression_exhausted ...),
        # never in the deferred body.
        orelse_calls = set()
        for sub in node.orelse:
            orelse_calls |= _calls(sub)
        if "reset_session" in orelse_calls:
            return node
    raise AssertionError(
        "Could not locate the compression_deferred guard in front of the "
        "compression-exhausted auto-reset block in gateway/run.py. The "
        "soft-defer contract (#49874: lock-contended defer must never "
        "auto-reset the session) is no longer structurally guaranteed."
    )


class TestCompressionDeferredIsSoft:
    def test_deferred_branch_guards_the_auto_reset(self):
        """The auto-reset (``reset_session``) must be unreachable when
        ``compression_deferred`` is set: the deferred check comes FIRST and
        the reset lives only in its elif chain."""
        node = _find_deferred_guarded_reset_chain()
        # The exhaustion reset is in the orelse — verified by the finder.
        # The deferred body must not mutate the session in any way.
        body_calls = set()
        for sub in node.body:
            body_calls |= _calls(sub)
        forbidden = {
            "reset_session",
            "_evict_cached_agent",
            "_clear_conversation_scope",
        }
        assert not (body_calls & forbidden), (
            f"The compression_deferred branch in gateway/run.py performs "
            f"session mutation ({body_calls & forbidden}). A lock-contended "
            f"defer is transient — the session must stay intact so the next "
            f"message retries against the freshly compressed context "
            f"(#49874, #69870)."
        )

    def test_deferred_result_key_is_passed_through_run_agent_inner(self):
        """``_run_agent_inner``'s result dicts must carry the
        ``compression_deferred`` key so the persistence block can see it —
        the exact gap that made the exhaustion misclassification possible
        (the flag existed but nothing consumed it)."""
        src = inspect.getsource(gateway_run)
        assert src.count('"compression_deferred"') >= 3, (
            "gateway/run.py must read AND pass through compression_deferred "
            "(persistence-block guard + both _run_agent_inner result dicts)."
        )
