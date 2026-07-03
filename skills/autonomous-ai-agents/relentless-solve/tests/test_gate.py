#!/usr/bin/env python3
"""Unit tests for the solve gate — classify, slug derivation, verdict persistence.

No container, no network: the oneshot model call is injected/monkeypatched (the same
DI-by-module-attribute style as test_loop.py). Run:
    python3 tests/test_gate.py
"""

import json
import os
import shutil
import sys
import tempfile
import types
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import relentless  # noqa: E402


def solve_args(**kw):
    base = dict(prompt="probe task", prompt_file=None, budget=1800, risk="act",
                gate=False, slug=None, route=None, gate_only=False, answer_cwd=None,
                state_dir=None, accept_flow_change=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


class ExtractJson(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(relentless.extract_json_object('{"route":"trivial","why":"w"}'),
                         {"route": "trivial", "why": "w"})

    def test_prose_and_fence_wrapped(self):
        out = 'Sure!\n```json\n{"route":"full","why":"multi-method"}\n```\nHope that helps.'
        self.assertEqual(relentless.extract_json_object(out)["route"], "full")

    def test_nested_braces(self):
        out = 'x {"route":"full","why":"has {braces} inside"} y'
        self.assertEqual(relentless.extract_json_object(out)["why"], "has {braces} inside")

    def test_garbage_is_none(self):
        self.assertIsNone(relentless.extract_json_object("no json here { broken"))
        self.assertIsNone(relentless.extract_json_object(""))
        self.assertIsNone(relentless.extract_json_object(None))


class Classify(unittest.TestCase):
    def test_each_route_parses(self):
        for route in relentless.GATE_ROUTES:
            v = relentless.classify("t", "act",
                                    oneshot=lambda p, r=route: f'{{"route":"{r}","why":"x"}}')
            self.assertEqual((v["route"], v["source"]), (route, "model"))

    def test_garbage_defaults_full(self):
        v = relentless.classify("t", "act", oneshot=lambda p: "I think it's easy?")
        self.assertEqual((v["route"], v["source"]), ("full", "default"))

    def test_unknown_route_defaults_full(self):
        v = relentless.classify("t", "act", oneshot=lambda p: '{"route":"medium","why":"?"}')
        self.assertEqual(v["route"], "full")

    def test_oneshot_error_defaults_full(self):
        def boom(p):
            raise RuntimeError("backend down")
        v = relentless.classify("t", "act", oneshot=boom)
        self.assertEqual((v["route"], v["source"]), ("full", "default"))
        self.assertIn("RuntimeError", v["why"])


class DeriveSlug(unittest.TestCase):
    def test_deterministic_and_kebab(self):
        a = relentless.derive_slug("Migrate the billing DB to Postgres")
        self.assertEqual(a, relentless.derive_slug("Migrate the billing DB to Postgres"))
        self.assertEqual(a, "migrate-billing-db-postgres")

    def test_stopwords_and_cap(self):
        self.assertEqual(relentless.derive_slug(
            "Please use the api to fetch all of the user records and validate them"),
            "api-fetch-all-user")

    def test_distinct_intents_distinct_slugs(self):
        self.assertNotEqual(relentless.derive_slug("fix the login bug"),
                            relentless.derive_slug("write the login docs"))

    def test_empty_falls_back(self):
        self.assertEqual(relentless.derive_slug("the of to"), "task")


class GateJsonLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gate-test-")
        self._home = relentless._HOME
        relentless._HOME = self.tmp
        self.calls = []
        self._oneshot = relentless.run_oneshot
        relentless.run_oneshot = lambda p, timeout=0: (
            self.calls.append(p) or '{"route":"trivial","why":"one-liner"}')

    def tearDown(self):
        relentless._HOME = self._home
        relentless.run_oneshot = self._oneshot
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _gate_path(self, slug):
        return os.path.join(self.tmp, "relentless", slug, "gate.json")

    def test_reused_verdict_classifies_exactly_once(self):
        args = solve_args(gate_only=True)
        self.assertEqual(relentless.cmd_solve(args, engine_run=None), 0)
        self.assertEqual(relentless.cmd_solve(args, engine_run=None), 0)
        self.assertEqual(len(self.calls), 1, "second invocation must reuse gate.json")
        with open(self._gate_path(relentless.derive_slug("probe task"))) as fh:
            v = json.load(fh)
        self.assertEqual(v["route"], "trivial")
        for field in ("why", "slug", "risk", "budget"):
            self.assertIn(field, v, f"receipt field {field} missing from gate.json")

    def test_route_flag_skips_classify(self):
        args = solve_args(route="full", gate_only=True)
        self.assertEqual(relentless.cmd_solve(args, engine_run=None), 0)
        self.assertEqual(self.calls, [], "--route must not spend a model call")
        with open(self._gate_path(relentless.derive_slug("probe task"))) as fh:
            self.assertEqual(json.load(fh)["source"], "flag")

    def test_gate_only_runs_no_route_handler(self):
        ran = []
        st, ss = relentless.solve_trivial, relentless.solve_single
        relentless.solve_trivial = lambda *a, **k: ran.append("trivial") or 0
        relentless.solve_single = lambda *a, **k: ran.append("single") or 0
        try:
            self.assertEqual(relentless.cmd_solve(solve_args(gate_only=True),
                                                  engine_run=lambda *a: ran.append("full")), 0)
        finally:
            relentless.solve_trivial, relentless.solve_single = st, ss
        self.assertEqual(ran, [], "--gate-only must exit before any route handler")


if __name__ == "__main__":
    unittest.main(verbosity=2)
