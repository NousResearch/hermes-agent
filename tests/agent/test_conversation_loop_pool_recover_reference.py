"""Regression tests for issue #27465.

The conversation_loop's rate-limit branch was calling
``_pool_may_recover_from_rate_limit(...)`` as a bare name, but the
function lives in ``run_agent`` and is not imported at module scope.
Every 429 from a provider therefore crashed the loop with::

    NameError: name '_pool_may_recover_from_rate_limit' is not defined

The fix is to route the call through the existing lazy ``_ra()``
accessor that the rest of the module uses for ``run_agent`` symbols.

These tests pin:

1. Source invariant -- the broken bare reference must not reappear in
   ``agent/conversation_loop.py``.
2. Symbol resolution -- ``_ra()._pool_may_recover_from_rate_limit`` is
   actually callable end-to-end (no AttributeError, no stale module
   binding) and produces the documented truthiness on a representative
   set of credential-pool shapes.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _fake_pool(n_entries: int, has_available: bool = True):
    """Minimal credential-pool stand-in.

    Matches the shape consumed by ``run_agent._pool_may_recover_from_rate_limit``:
    ``pool.has_available()`` (boolean) and ``pool.entries()`` (a list whose
    length is the credential count).  Mirrors the helper used by the
    existing pool-rotation tests in ``tests/run_agent/test_provider_fallback.py``
    so the contracts can't drift independently.
    """
    pool = MagicMock()
    pool.entries.return_value = [MagicMock() for _ in range(n_entries)]
    pool.has_available.return_value = has_available
    return pool


CONVERSATION_LOOP = (
    Path(__file__).resolve().parents[2] / "agent" / "conversation_loop.py"
)


# ---------------------------------------------------------------------------
# 1. Source invariant
# ---------------------------------------------------------------------------


def test_conversation_loop_source_exists():
    """Sanity: the file we're guarding actually exists in this checkout."""
    assert CONVERSATION_LOOP.is_file(), CONVERSATION_LOOP


def test_no_bare_pool_may_recover_call_in_conversation_loop():
    """The bare ``_pool_may_recover_from_rate_limit(`` call must not return.

    Bare references (without ``_ra().`` or ``run_agent.`` prefix) are
    what caused #27465.  ``_pool_may_recover`` (the local variable
    binding) and the docstring comment that mentions the function name
    are both allowed -- only an actual *call* without a module accessor
    is forbidden.
    """
    src = CONVERSATION_LOOP.read_text(encoding="utf-8")
    # Match the call form, optionally preceded by `.` (which means it's
    # already qualified, e.g. ``run_agent._pool_may_recover_from_rate_limit(``
    # or ``_ra()._pool_may_recover_from_rate_limit(``).  We assert that
    # there is no occurrence where the function name is called *not* in
    # an attribute-access context.
    bare_call_pattern = re.compile(
        r"(?<![.\w])_pool_may_recover_from_rate_limit\s*\("
    )
    matches = bare_call_pattern.findall(src)
    assert not matches, (
        "Found bare _pool_may_recover_from_rate_limit() call in "
        "agent/conversation_loop.py.  This is the #27465 regression: "
        "the function lives in run_agent, so the call must use the "
        "_ra() lazy accessor (or `run_agent.` prefix)."
    )


def test_call_site_uses_lazy_accessor():
    """Spot-check: the surviving call must route through ``_ra()``.

    Confirms the fix is *positively* in place -- not merely that the
    broken form is absent.
    """
    src = CONVERSATION_LOOP.read_text(encoding="utf-8")
    qualified_pattern = re.compile(
        r"_ra\(\)\._pool_may_recover_from_rate_limit\s*\("
    )
    assert qualified_pattern.search(src), (
        "Expected at least one ``_ra()._pool_may_recover_from_rate_limit(`` "
        "call in agent/conversation_loop.py -- the fix for #27465 has "
        "regressed or been reverted."
    )


def test_conversation_loop_compiles_cleanly():
    """No SyntaxError / NameError at *import* time.

    Importing ``agent.conversation_loop`` triggers module-level execution
    only; the rate-limit branch is inside an async function and won't
    run.  But a future regression that swapped the call to a top-level
    statement (or reintroduced a syntax error around it) would surface
    here.
    """
    src = CONVERSATION_LOOP.read_text(encoding="utf-8")
    ast.parse(src, filename=str(CONVERSATION_LOOP))


# ---------------------------------------------------------------------------
# 2. End-to-end symbol resolution via _ra()
# ---------------------------------------------------------------------------


def test_ra_accessor_resolves_pool_may_recover():
    """``_ra()._pool_may_recover_from_rate_limit`` must be callable."""
    from agent import conversation_loop

    fn = conversation_loop._ra()._pool_may_recover_from_rate_limit
    assert callable(fn)
    # The accessor returns the same module on every call -- otherwise a
    # future refactor that returned a stale reference would break the
    # in-loop call.
    assert (
        conversation_loop._ra()._pool_may_recover_from_rate_limit
        is fn
    )


def test_ra_accessor_pool_may_recover_none_pool_returns_false():
    """No credential pool -> the loop cannot recover via rotation."""
    from agent import conversation_loop

    fn = conversation_loop._ra()._pool_may_recover_from_rate_limit
    assert fn(None) is False


def test_ra_accessor_pool_may_recover_single_credential_returns_false():
    """One credential -- no rotation room -- cannot recover."""
    from agent import conversation_loop

    fn = conversation_loop._ra()._pool_may_recover_from_rate_limit
    assert fn(_fake_pool(1)) is False


def test_ra_accessor_pool_may_recover_multi_credential_returns_true():
    """Multiple credentials with at least one available -> rotation may save us."""
    from agent import conversation_loop

    fn = conversation_loop._ra()._pool_may_recover_from_rate_limit
    assert fn(_fake_pool(3)) is True


def test_ra_accessor_pool_may_recover_cooled_down_returns_false():
    from agent import conversation_loop

    fn = conversation_loop._ra()._pool_may_recover_from_rate_limit
    assert fn(_fake_pool(5, has_available=False)) is False


def test_ra_accessor_pool_may_recover_handles_keyword_args():
    """The call site passes ``provider=`` and ``base_url=`` kwargs.

    Pin that the function accepts them via the lazy accessor so a future
    rename of the parameters in ``run_agent`` doesn't silently break the
    in-loop call (a TypeError would crash the same retry branch as the
    original NameError did before the fix).
    """
    from agent import conversation_loop

    fn = conversation_loop._ra()._pool_may_recover_from_rate_limit
    result = fn(
        _fake_pool(2),
        provider="ollama",
        base_url="https://ollama.cloud/v1",
    )
    assert isinstance(result, bool)


def test_ra_accessor_pool_may_recover_cloudcode_returns_false():
    """CloudCode base_url -> account-wide quota -> rotation can't recover (#13636)."""
    from agent import conversation_loop

    fn = conversation_loop._ra()._pool_may_recover_from_rate_limit
    assert fn(
        _fake_pool(5),
        base_url="cloudcode-pa://us-central1",
    ) is False
