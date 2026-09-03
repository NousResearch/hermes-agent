"""Tests for the run_agent.main() ``max_iterations`` parameter rename (#38113).

The CLI entry point ``run_agent.main()`` previously named its
iteration-budget parameter ``max_turns``, which conflicts with the
``AIAgent`` constructor's canonical name ``max_iterations`` and is
ambiguous with the higher-level *turn* concept used elsewhere in the
agent. These tests pin:

1. ``max_iterations`` is the new canonical parameter name and is
   forwarded to ``AIAgent(max_iterations=...)``.
2. ``max_turns`` is still accepted as a deprecated alias and emits a
   ``DeprecationWarning`` while continuing to forward correctly.
3. Passing both at once raises ``TypeError`` rather than silently
   preferring one over the other (no hidden precedence bugs).
4. The default value remains ``10`` so the public contract doesn't
   shift for callers that pass neither.
"""

from __future__ import annotations

import inspect
import warnings
from unittest.mock import MagicMock, patch

import pytest

import run_agent


# ---------------------------------------------------------------------------
# Signature-level pins (don't need the agent to actually run)
# ---------------------------------------------------------------------------


def test_main_signature_exposes_max_iterations() -> None:
    """``max_iterations`` must be a real parameter, not just a kwarg pass-through."""
    sig = inspect.signature(run_agent.main)
    assert "max_iterations" in sig.parameters, (
        "main() must accept max_iterations as a named parameter (#38113)"
    )


def test_main_max_iterations_default_is_none_sentinel() -> None:
    """Default is a sentinel (None); the effective value 10 is applied at runtime."""
    sig = inspect.signature(run_agent.main)
    assert sig.parameters["max_iterations"].default is None


def test_main_signature_still_accepts_max_turns_alias() -> None:
    """``max_turns`` stays in the signature as a deprecated alias."""
    sig = inspect.signature(run_agent.main)
    assert "max_turns" in sig.parameters, (
        "max_turns must remain accepted as a deprecated alias for back-compat"
    )
    # Sentinel default — anything other than 10 means "user didn't pass it".
    assert sig.parameters["max_turns"].default is None


# ---------------------------------------------------------------------------
# Behavioural pins — mock AIAgent so we can assert the forwarded kwarg.
# ---------------------------------------------------------------------------


def _make_agent_mock() -> MagicMock:
    agent = MagicMock()
    agent.chat.return_value = "ok"
    return agent


def test_main_forwards_max_iterations_to_agent() -> None:
    """Passing ``max_iterations=42`` reaches ``AIAgent(max_iterations=42)``."""
    with patch.object(run_agent, "AIAgent") as agent_cls:
        agent_cls.return_value = _make_agent_mock()
        run_agent.main(query="hi", max_iterations=42)

    assert agent_cls.called, "AIAgent should have been constructed"
    kwargs = agent_cls.call_args.kwargs
    assert kwargs.get("max_iterations") == 42


def test_main_max_turns_alias_still_works_and_warns() -> None:
    """``max_turns=7`` keeps working but emits a ``DeprecationWarning``."""
    with patch.object(run_agent, "AIAgent") as agent_cls:
        agent_cls.return_value = _make_agent_mock()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_agent.main(query="hi", max_turns=7)

    assert agent_cls.call_args.kwargs.get("max_iterations") == 7
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, (
        "Using the legacy max_turns alias must emit DeprecationWarning"
    )
    assert any("max_iterations" in str(w.message) for w in deprecations), (
        "Deprecation message should point users at max_iterations"
    )


def test_main_rejects_both_max_iterations_and_max_turns() -> None:
    """Specifying both at once is ambiguous — raise instead of guessing."""
    with patch.object(run_agent, "AIAgent") as agent_cls:
        agent_cls.return_value = _make_agent_mock()
        with pytest.raises(TypeError, match="max_iterations"):
            run_agent.main(query="hi", max_iterations=25, max_turns=20)


def test_main_default_forwards_ten() -> None:
    """No budget argument → AIAgent receives max_iterations=10 (unchanged contract)."""
    with patch.object(run_agent, "AIAgent") as agent_cls:
        agent_cls.return_value = _make_agent_mock()
        run_agent.main(query="hi")

    assert agent_cls.call_args.kwargs.get("max_iterations") == 10
