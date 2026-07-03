#!/usr/bin/env python3
"""Invocation-plane contract — pin oneshot.py's API from the consumer side.

oneshot.py (the bare `hermes -z` dispatch primitive) lives alongside the
resumable-script engine for deployment convenience, but it is TRANSPORT, not
durability, and it has two independent consumers: relentless.py (task/plan/gate
oneshots) and method-explorer's drive.py (the tick loop). Until this file, that
seam was unpinned — either consumer could be broken by an innocent-looking
signature or semantics change. See skills/ARCHITECTURE.md § Invocation plane.

Pinned here:
  - resolution: relentless._oneshot() loads the module the resumable-script
    skill ships (same dir as the engine, RESUMABLE_ENGINE_DIR override);
  - signatures: run_direct / run_docker_exec keyword surface both consumers use;
  - tolerant-timeout semantics: a TimeoutExpired becomes returncode==124 with
    partial stdout, never a raised exception ("artifacts beat stdout" — the
    caller's on-disk artifact is the source of truth);
  - dispatch shape: `-z` single-turn, docker mode's shell-level `timeout` as
    the primary bound, env injection via `docker exec -e`.

House pattern: skip (not fail) when the counterpart skill is not on disk.
Run: python3 tests/test_oneshot_contract.py
"""

import inspect
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import relentless  # noqa: E402

_HAVE_ONESHOT = os.path.exists(os.path.join(relentless._ENGINE_DIR, "oneshot.py"))


@unittest.skipUnless(_HAVE_ONESHOT,
                     f"oneshot.py not found in {relentless._ENGINE_DIR!r}")
class OneshotContract(unittest.TestCase):

    def setUp(self):
        self.mod = relentless._oneshot()

    def test_resolution_matches_the_shipped_module(self):
        self.assertEqual(
            os.path.realpath(self.mod.__file__),
            os.path.realpath(os.path.join(relentless._ENGINE_DIR, "oneshot.py")),
            "relentless must load the oneshot.py the resumable-script skill ships")

    def test_run_direct_signature(self):
        sig = inspect.signature(self.mod.run_direct)
        params = list(sig.parameters)
        self.assertEqual(params[:2], ["prompt", "timeout"],
                         "both consumers call run_direct(prompt, timeout, ...)")
        for kw in ("hermes_bin", "pad", "model", "provider", "toolsets"):
            self.assertIn(kw, params, f"run_direct must keep the {kw}= kwarg")
        self.assertEqual(sig.parameters["pad"].default, 0,
                         "drive.py relies on pad defaulting to 0 (its own timeout "
                         "IS the only bound)")

    def test_run_docker_exec_signature(self):
        sig = inspect.signature(self.mod.run_docker_exec)
        params = list(sig.parameters)
        self.assertEqual(params[:3], ["prompt", "timeout", "container"])
        for kw in ("hermes_bin", "pad", "model", "provider", "toolsets", "env"):
            self.assertIn(kw, params, f"run_docker_exec must keep the {kw}= kwarg")

    def test_timeout_becomes_rc_124_not_an_exception(self):
        # The load-bearing semantics: a timed-out process returns rc 124 with
        # whatever partial stdout existed — callers distinguish timeout from
        # completion by returncode, and NEVER need their own try/except.
        r = self.mod._tolerant_run(
            [sys.executable, "-c",
             "import sys,time; print('partial'); sys.stdout.flush(); time.sleep(5)"],
            timeout=1)
        self.assertEqual(r.returncode, 124)
        self.assertIn("partial", r.stdout)

    def test_run_direct_dispatches_bare_single_turn(self):
        # `-z` single-turn with the prompt as one argv element (never shell-
        # joined, never `chat`), flags appended only when set — pin the cmd
        # shape by source and the flag tail by behavior.
        src = inspect.getsource(self.mod.run_direct)
        self.assertIn('"-z", prompt', src)
        self.assertEqual(self.mod._hermes_flags("m1", "p1", "t1"),
                         ["-m", "m1", "--provider", "p1", "-t", "t1"])
        self.assertEqual(self.mod._hermes_flags(None, None, None), [])

    def test_docker_mode_uses_shell_timeout_and_env_injection(self):
        # Pin the documented docker cmd shape without running docker: the
        # primary bound is the in-container `timeout <secs>`, and env comes in
        # as `-e K=V` pairs before the container name.
        src = inspect.getsource(self.mod.run_docker_exec)
        self.assertIn('"docker", "exec"', src)
        self.assertIn('"timeout", str(timeout)', src)
        self.assertIn('"-e"', src)

    def test_default_bin_is_the_container_path(self):
        self.assertEqual(self.mod.DEFAULT_HERMES_BIN, "/opt/hermes/bin/hermes")


if __name__ == "__main__":
    unittest.main(verbosity=2)
