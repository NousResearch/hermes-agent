#!/usr/bin/env python3
"""Engine contract test — pins FakeCtx semantics to the REAL resumable-script engine.

test_loop.py proves the flow logic against a FakeCtx; nothing there would catch FakeCtx
drifting from the engine's actual step-memoization / replay / suspend behavior. This suite
runs relentless_flow under the real engine (host-side; the engine is stdlib-only) with the
same scripted fakes and asserts:
  (a) the completed result equals the FakeCtx run's result,
  (b) re-invocation replays from the journal without re-executing any phase helper,
  (c) a --gate GUARD-HALT fork suspends (exit 10, pending key c0/fork) and resumes to
      completion with the answer folded in.

Skips (not fails) when the engine is not on disk. Run: python3 tests/test_engine_contract.py
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, _HERE)

import relentless  # noqa: E402
import test_loop as tl  # noqa: E402 — reuse the scripted fakes + input builder

_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
_ENGINE_DIR = os.environ.get("RESUMABLE_ENGINE_DIR") or os.path.join(
    _HOME, "skills", "resumable-script", "scripts")
_HAVE_ENGINE = os.path.exists(os.path.join(_ENGINE_DIR, "engine.py"))
if _HAVE_ENGINE and _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)


def _run_engine(flow_obj, engine, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = engine.run_cli(flow_obj, argv=argv)
    lines = [ln for ln in buf.getvalue().strip().splitlines() if ln.strip()]
    return rc, json.loads(lines[-1])


@unittest.skipUnless(_HAVE_ENGINE, f"resumable-script engine not found in {_ENGINE_DIR!r}")
class EngineContract(tl.LoopBase):
    def setUp(self):
        super().setUp()
        import engine  # noqa: E402
        self.engine = engine
        self.flow_obj = engine.flow(id="relentless-engine-contract", version=1)(
            relentless.relentless_flow)
        self.state_dir = tempfile.mkdtemp(prefix="rls-engine-contract-")

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.state_dir, ignore_errors=True)

    def _wire_two_cycle(self):
        self.wire([tl.clar([tl.ts("q1", "a1")]),
                   tl.clar([], stop="converged (no question above floor)")],
                  [tl.pl(tl.tk("t1", "alfa")), tl.pl(tl.tk("t1", "bravo"))],
                  [{"t1": "failed"}, {}])

    def test_completed_result_matches_fakectx(self):
        self._wire_two_cycle()
        fake_out = relentless.relentless_flow(tl.FakeCtx(), tl.inp())

        self.setUp2 = None  # keep tearDown simple; re-wire fresh fakes for the engine run
        self._orig_restore = None
        self._wire_two_cycle()
        rc, payload = _run_engine(self.flow_obj, self.engine,
                                  ["run", "--input", json.dumps(tl.inp()),
                                   "--state-dir", self.state_dir, "--auto"])
        self.assertEqual(rc, 0)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["result"], fake_out)

    def test_replay_executes_nothing(self):
        self._wire_two_cycle()
        rc1, p1 = _run_engine(self.flow_obj, self.engine,
                              ["run", "--input", json.dumps(tl.inp()),
                               "--state-dir", self.state_dir, "--auto"])
        self.assertEqual(rc1, 0)

        def boom(*a, **kw):
            raise AssertionError("replay must not re-execute any phase helper")
        for n in self.PATCHED:
            setattr(relentless, n, boom)
        rc2, p2 = _run_engine(self.flow_obj, self.engine,
                              ["run", "--input", json.dumps(tl.inp()),
                               "--state-dir", self.state_dir, "--auto"])
        self.assertEqual(rc2, 0)
        self.assertEqual(p1["result"], p2["result"])

    def test_gate_suspends_then_resumes(self):
        fork = [tl.pl(disposition="needs_decision",
                      question="Which branch should be preferred?")]
        self.wire([tl.clar([]), tl.clar([], stop="converged (no question above floor)")],
                  fork + [tl.pl(tl.tk("t1"))])
        inp = tl.inp(gate=True)
        rc, payload = _run_engine(self.flow_obj, self.engine,
                                  ["run", "--input", json.dumps(inp),
                                   "--state-dir", self.state_dir])
        self.assertEqual(rc, 10)
        self.assertEqual(payload["status"], "suspended")
        self.assertEqual(payload["pending"]["key"], "c0/fork")

        # Answer the fork. KEY ENGINE SEMANTIC (the drift this test exists to catch):
        # resume REPLAYS cycle 0 from the journal — the fakes are only consumed by the
        # live tail (cycle 1) — so script the tail, not the full sequence. (FakeCtx's
        # pre-answered run executes everything live from index 0 instead.)
        self.wire([tl.clar([], stop="converged (no question above floor)")],
                  [tl.pl(tl.tk("t1"))])
        rc2, p2 = _run_engine(self.flow_obj, self.engine,
                              ["resume", "--answer", "prefer source D",
                               "--state-dir", self.state_dir])
        self.assertEqual(rc2, 0)
        self.assertEqual(p2["status"], "completed")
        self.assertEqual(p2["result"]["outcome"], "success")
        facts = [r for r in self.reported["ledger"] if r["kind"] == "fact"]
        self.assertTrue(any("prefer source D" in r["text"] for r in facts))


if __name__ == "__main__":
    unittest.main(verbosity=2)
