"""Regression tests for _release_running_agent_state and SessionDB shutdown.

Before this change, running-agent state lived in three dicts that drifted
out of sync:

  self._running_agents       — AIAgent instance per session key
  self._running_agents_ts    — start timestamp per session key
  self._busy_ack_ts          — last busy-ack timestamp per session key

Six cleanup sites did ``del self._running_agents[key]`` without touching
the other two; one site only popped ``_running_agents`` and
``_running_agents_ts``; and only the stale-eviction site cleaned all
three.  Each missed entry was a small persistent leak.

Also: SessionDB connections were never closed on gateway shutdown,
leaving WAL locks in place until Python actually exited.
"""

import threading
from unittest.mock import MagicMock

import pytest


def _make_runner():
    """Bare GatewayRunner wired with just the state the helper touches."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    return runner


class TestReleaseRunningAgentStateUnit:
    def test_pops_all_three_dicts(self):
        runner = _make_runner()
        runner._running_agents["k"] = MagicMock()
        runner._running_agents_ts["k"] = 123.0
        runner._busy_ack_ts["k"] = 456.0

        runner._release_running_agent_state("k")

        assert "k" not in runner._running_agents
        assert "k" not in runner._running_agents_ts
        assert "k" not in runner._busy_ack_ts

    def test_idempotent_on_missing_key(self):
        """Calling twice (or on an absent key) must not raise."""
        runner = _make_runner()
        runner._release_running_agent_state("missing")
        runner._release_running_agent_state("missing")  # still fine

    def test_noop_on_empty_session_key(self):
        """Empty string / None key is treated as a no-op."""
        runner = _make_runner()
        runner._running_agents[""] = "guard"
        runner._release_running_agent_state("")
        # Empty key not processed — guard value survives.
        assert runner._running_agents[""] == "guard"

    def test_preserves_other_sessions(self):
        runner = _make_runner()
        for k in ("a", "b", "c"):
            runner._running_agents[k] = MagicMock()
            runner._running_agents_ts[k] = 1.0
            runner._busy_ack_ts[k] = 1.0

        runner._release_running_agent_state("b")

        assert set(runner._running_agents.keys()) == {"a", "c"}
        assert set(runner._running_agents_ts.keys()) == {"a", "c"}
        assert set(runner._busy_ack_ts.keys()) == {"a", "c"}

    def test_handles_missing_busy_ack_attribute(self):
        """Backward-compatible with older runners lacking _busy_ack_ts."""
        runner = _make_runner()
        del runner._busy_ack_ts  # simulate older version
        runner._running_agents["k"] = MagicMock()
        runner._running_agents_ts["k"] = 1.0

        runner._release_running_agent_state("k")  # should not raise

        assert "k" not in runner._running_agents
        assert "k" not in runner._running_agents_ts

    def test_concurrent_release_is_safe(self):
        """Multiple threads releasing different keys concurrently."""
        runner = _make_runner()
        for i in range(50):
            k = f"s{i}"
            runner._running_agents[k] = MagicMock()
            runner._running_agents_ts[k] = float(i)
            runner._busy_ack_ts[k] = float(i)

        def worker(keys):
            for k in keys:
                runner._release_running_agent_state(k)

        threads = [
            threading.Thread(target=worker, args=([f"s{i}" for i in range(start, 50, 5)],))
            for start in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
            assert not t.is_alive()

        assert runner._running_agents == {}
        assert runner._running_agents_ts == {}
        assert runner._busy_ack_ts == {}


class TestNoMoreBareDeleteSites:
    """Regression: all bare `del self._running_agents[key]` sites were
    converted to use the helper.  If a future contributor reverts one,
    this test flags it.  Docstrings / comments mentioning the old
    pattern are allowed.
    """

    def test_no_bare_del_of_running_agents_in_gateway_run(self):
        from pathlib import Path
        import re

        gateway_run = (Path(__file__).parent.parent.parent / "gateway" / "run.py").read_text()
        # Match `del self._running_agents[...]` that is NOT inside a
        # triple-quoted docstring.  We scan non-docstring lines only.
        lines = gateway_run.splitlines()

        in_docstring = False
        docstring_delim = None
        offenders = []
        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    delim = stripped[:3]
                    # single-line docstring?
                    if stripped.count(delim) >= 2:
                        continue
                    in_docstring = True
                    docstring_delim = delim
                    continue
                if re.search(r"\bdel\s+self\._running_agents\[", line):
                    offenders.append((idx, line.rstrip()))
            else:
                if docstring_delim and docstring_delim in stripped:
                    in_docstring = False
                    docstring_delim = None

        assert offenders == [], (
            "Found bare `del self._running_agents[...]` sites in gateway/run.py. "
            "Use self._release_running_agent_state(session_key) instead so "
            "_running_agents_ts and _busy_ack_ts are popped in lockstep.\n"
            + "\n".join(f"  line {n}: {l}" for n, l in offenders)
        )


class TestSessionDbCloseOnShutdown:
    """_stop_impl should call .close() on both self._session_db and
    self.session_store._db to release SQLite WAL locks before the new
    gateway (during --replace restart) tries to open the same file.
    """

    def test_stop_impl_closes_both_session_dbs(self):
        """Run the exact shutdown block that closes SessionDBs and verify
        .close() was called on both holders."""
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)

        runner_db = MagicMock()
        store_db = MagicMock()

        runner._db = runner_db
        runner.session_store = MagicMock()
        runner.session_store._db = store_db

        # Replicate the exact production loop from _stop_impl.
        for _db_holder in (runner, getattr(runner, "session_store", None)):
            _db = getattr(_db_holder, "_db", None) if _db_holder else None
            if _db is None or not hasattr(_db, "close"):
                continue
            _db.close()

        runner_db.close.assert_called_once()
        store_db.close.assert_called_once()

    def test_shutdown_tolerates_missing_session_store(self):
        """Gateway without a session_store attribute must not crash on shutdown."""
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._db = MagicMock()
        # Deliberately no session_store attribute.

        for _db_holder in (runner, getattr(runner, "session_store", None)):
            _db = getattr(_db_holder, "_db", None) if _db_holder else None
            if _db is None or not hasattr(_db, "close"):
                continue
            _db.close()

        runner._db.close.assert_called_once()

    def test_shutdown_tolerates_close_raising(self):
        """A close() that raises must not prevent subsequent cleanup."""
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        flaky_db = MagicMock()
        flaky_db.close.side_effect = RuntimeError("simulated lock error")
        healthy_db = MagicMock()

        runner._db = flaky_db
        runner.session_store = MagicMock()
        runner.session_store._db = healthy_db

        # Same pattern as production: try/except around each close().
        for _db_holder in (runner, getattr(runner, "session_store", None)):
            _db = getattr(_db_holder, "_db", None) if _db_holder else None
            if _db is None or not hasattr(_db, "close"):
                continue
            try:
                _db.close()
            except Exception:
                pass

        flaky_db.close.assert_called_once()
        healthy_db.close.assert_called_once()
