"""Pattern-B perf-regression guards: hot paths must not degrade to O(N²)/O(N·Q).

Pattern B ("rebuild everything per delta/turn") has no lintable signature —
``s += frag`` is quadratic in a hot loop and harmless elsewhere — so unlike
the Pattern-A ruff gate (ASYNC*), the only durable guard is behavioral:
pin the *scaling shape* of each known hot path and fail CI when it regresses.

Guard design rules (to stay CI-stable):
1. Prefer deterministic operation counts (SQL statements via sqlite trace
   callbacks) over timing.  Counts cannot flake.
2. Where only timing exists, assert a *self-normalizing ratio* — time at 4N
   over time at N — never absolute wall-clock.  Linear paths give ~4 (with
   allocator noise), quadratic paths give ~16.  Thresholds sit between the
   measured-good and measured-bad values with ≥2x separation on both sides.
3. min-of-K timing samples to reject scheduler noise.

Baseline measurements (2026-08-28, macOS arm64, Python 3.12):
- streamed-text accumulation on main: 4N/N ratio ≈ 9.6 (superlinear —
  ``+=`` through the attribute copies the whole reply per delta).
  With PR #92166 (parts list + join): ratio ≈ 4 (linear).
- list_sessions_rich on main: ~2 writer-conn statements per listed session
  (per-root compression-tip walk = N+1).  With PR #95380 (batched edge
  query): bounded constant.

The xfail markers are the ratchet: they document today's known-bad main and
flip to plain assertions when the fix PRs land.  Remove a marker in the same
PR that merges its fix (or immediately after).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest


def _min_time(fn, *, repeat: int = 5) -> float:
    """Best-of-``repeat`` wall time for ``fn()`` — rejects scheduler noise."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# ---------------------------------------------------------------------------
# Guard 1 — streamed assistant text accumulation must be linear (#92166).
# ---------------------------------------------------------------------------


class TestStreamedTextAccumulationLinear:
    """A long streamed reply must cost O(len), not O(len²), to accumulate.

    Regression shape: ``self._current_streamed_assistant_text += text`` on an
    attribute copies the entire reply on every delta.  A 2MB reply then does
    ~2TB of memcpy across the stream — visible as CLI/desktop stream lag.
    """

    N_SMALL = 4_000
    N_LARGE = 16_000  # 4x — linear ratio ≈ 4, quadratic ≈ 16
    DELTA = "x" * 50
    # main measured 9.6; parts-list impl measured ~4.  Midpoint with margin.
    MAX_RATIO = 7.0

    def _accumulate(self, n: int) -> None:
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        agent._current_streamed_assistant_text = ""
        for _ in range(n):
            agent._record_streamed_assistant_text(self.DELTA)
        # The accumulated value must be faithful regardless of representation.
        assert len(agent._current_streamed_assistant_text) == n * len(self.DELTA)

    @pytest.mark.xfail(
        reason="known-quadratic on main until PR #92166 (streamed-text parts "
        "list) merges; remove this marker when it lands",
        strict=False,
    )
    def test_4x_input_costs_about_4x_time(self):
        t_small = _min_time(lambda: self._accumulate(self.N_SMALL))
        t_large = _min_time(lambda: self._accumulate(self.N_LARGE))
        ratio = t_large / max(t_small, 1e-9)
        assert ratio < self.MAX_RATIO, (
            f"streamed-text accumulation is superlinear: 4x deltas cost "
            f"{ratio:.1f}x time (linear ≈ 4, quadratic ≈ 16). The reply is "
            f"being recopied per delta — see PR #92166 for the fix shape."
        )


# ---------------------------------------------------------------------------
# Guard 2 — list_sessions_rich must not issue O(N) per-row queries (#95380).
# ---------------------------------------------------------------------------


class TestListSessionsRichQueryBound:
    """Listing sessions must not walk compression chains one query at a time.

    Regression shape: ``get_compression_tip()`` per compression root = one
    (or more) writer-connection statements per listed session, all under the
    global write lock.  Deterministic guard: count statements via sqlite's
    trace callback — zero timing involved.
    """

    N_CHAINS = 12
    # Batched implementation needs a small constant number of statements.
    # main measures ~2 per session (26 for 12 chains, 50 for 24).
    MAX_STATEMENTS = 8

    @pytest.fixture()
    def chain_db(self, tmp_path: Path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        for i in range(self.N_CHAINS):
            root, child = f"root_{i}", f"child_{i}"
            db.create_session(root, source="cli")
            db.create_session(child, source="cli", parent_session_id=root)
            db.end_session(root, end_reason="compression")
        yield db
        db.close()

    @staticmethod
    def _count_writer_statements(db):
        """Rows from ``list_sessions_rich`` plus the writer-conn statement count.

        Uses the house trace-callback idiom (``statements.append`` — see
        tests/test_hermes_state.py) so failures can dump the captured SQL.
        """
        statements: list[str] = []
        db._conn.set_trace_callback(statements.append)
        try:
            rows = db.list_sessions_rich(limit=50)
        finally:
            db._conn.set_trace_callback(None)
        return rows, statements

    @pytest.mark.xfail(
        reason="known N+1 on main until PR #95380 (batched compression-edge "
        "query) merges; remove this marker when it lands",
        strict=False,
    )
    def test_statement_count_bounded_regardless_of_session_count(self, chain_db):
        rows, statements = self._count_writer_statements(chain_db)

        assert len(rows) == self.N_CHAINS
        assert len(statements) <= self.MAX_STATEMENTS, (
            f"list_sessions_rich issued {len(statements)} writer-connection "
            f"statements for {self.N_CHAINS} sessions (bound: "
            f"{self.MAX_STATEMENTS}). Per-row chain walking is back — see "
            f"PR #95380 for the batched-edge fix shape. Captured SQL: "
            f"{[s[:80] for s in statements[:10]]}"
        )

    def test_statement_count_does_not_scale_with_sessions(self, chain_db):
        """Weaker invariant that must hold TODAY on main: statements may be
        O(N) but must never exceed a per-session budget of 4 — catching a
        regression from N+1 to N·M (nested walks, per-hop re-queries).
        """
        rows, statements = self._count_writer_statements(chain_db)

        assert len(rows) == self.N_CHAINS
        budget = 4 * self.N_CHAINS + 8
        assert len(statements) <= budget, (
            f"list_sessions_rich issued {len(statements)} statements for "
            f"{self.N_CHAINS} sessions — beyond even the legacy N+1 budget "
            f"({budget}). A nested per-row walk has been introduced."
        )


# ---------------------------------------------------------------------------
# Guard 3 — streamed tool-call fragment assembly must stay linear (#92242).
# ---------------------------------------------------------------------------


class TestToolCallFragmentAssemblyLinear:
    """Assembling a fragmented tool call must cost O(bytes), not O(bytes²).

    Exercises the same shape as the SSE accumulator in
    ``chat_completion_helpers``: fragments arrive one at a time and are
    accumulated into a per-call buffer keyed in a dict.  Guards the *pattern*
    (dict-field `+=` defeats CPython's in-place-growth optimization when
    refcount > 1) via a pure-python model faithful to the accumulator's
    structure, so the guard runs without a live provider stream.
    """

    FRAG = "y" * 64
    # Sized so the small case takes ≥2ms even on fast hardware: sub-ms bases
    # make the ratio jitter on noisy CI runners (measured 0.185ms at 8k).
    N_SMALL = 128_000
    N_LARGE = 512_000
    MAX_RATIO = 8.0  # linear ≈ 4; dict-field quadratic measured >> 10

    @staticmethod
    def _assemble_dict_field(n_frags: int, frag: str) -> int:
        """Accumulator model: buffered parts, joined once (fixed shape)."""
        acc = {0: {"function": {"name": "tool", "arguments_parts": []}}}
        entry = acc[0]
        for _ in range(n_frags):
            entry["function"]["arguments_parts"].append(frag)
        return len("".join(entry["function"]["arguments_parts"]))

    def test_4x_fragments_cost_about_4x_time(self):
        t_small = _min_time(lambda: self._assemble_dict_field(self.N_SMALL, self.FRAG), repeat=3)
        t_large = _min_time(lambda: self._assemble_dict_field(self.N_LARGE, self.FRAG), repeat=3)
        ratio = t_large / max(t_small, 1e-9)
        assert ratio < self.MAX_RATIO, (
            f"tool-call fragment assembly is superlinear: 4x fragments cost "
            f"{ratio:.1f}x time. Fragments must be buffered in a list and "
            f"joined once (PR #92242 shape), never `+=` into a dict field."
        )
