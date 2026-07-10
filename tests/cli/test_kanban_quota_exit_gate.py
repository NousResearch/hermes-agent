"""BUILD-343: the kanban worker exit-code gate in ``cli.py`` must recognize
every quota-class FailoverReason value, not just ``rate_limit``/``billing``.

``upstream_rate_limit`` (aggregator 429 — e.g. OpenRouter throttling an
upstream model) is treated as the same quota-wall bucket everywhere else in
the codebase (agent/chat_completion_helpers.py's cooldown logic,
conversation_loop.py's `is_rate_limited` eager-fallback gate). A kanban
worker whose recorded ``failure_reason`` is ``upstream_rate_limit`` should
also exit with ``KANBAN_RATE_LIMIT_EXIT_CODE`` (75) so the dispatcher
requeues it without tripping the circuit breaker, exactly like ``rate_limit``
/ ``billing``.

Source-inspection style (matches ``test_credential_pool_exhaustion_
parking.py`` and ``test_nous_oauth_401_guidance.py`` for this codebase's
convention of pinning control-flow shape in giant, hard-to-unit-execute
functions): ``cli.py``'s ``main()`` is not decomposed into a testable
function for this gate, so we assert on the source text instead of driving
a full CLI invocation.
"""

from __future__ import annotations

import inspect

import cli


def _gate_source() -> str:
    source = inspect.getsource(cli)
    start = source.index("KANBAN_RATE_LIMIT_EXIT_CODE as _RL_CODE")
    # The gate's `in (...)` tuple sits a few lines above the import.
    window_start = source.rindex("if os.environ.get(\"HERMES_KANBAN_TASK\")", 0, start)
    return source[window_start:start]


def test_kanban_exit_gate_includes_all_quota_class_reasons():
    gate_source = _gate_source()
    assert "rate_limit" in gate_source
    assert "billing" in gate_source
    assert "upstream_rate_limit" in gate_source, (
        "upstream_rate_limit is a quota-class FailoverReason (aggregator "
        "429) and belongs in the same exit-75 bucket as rate_limit/billing "
        "— see agent/chat_completion_helpers.py's cooldown grouping."
    )
