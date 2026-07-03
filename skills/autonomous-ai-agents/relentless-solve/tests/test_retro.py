#!/usr/bin/env python3
"""Unit tests for the post-success hindsight seam — run_hindsight (the oneshot wrapper
with its citation-validation retry channel), the flow's retro/judge gating, and
write_report's journey splice. No container, no engine; FakeCtx + module-attribute
monkeypatching, same style as test_loop.py. Run: python3 tests/test_retro.py

The one invariant everything here defends: HINDSIGHT CAN NEVER UN-SUCCEED A RUN —
every failure mode of the judge collapses to a {"skipped": reason} sentinel spliced
into the journey's advisory slot, while outcome/report stand untouched."""

import json
import os
import shutil
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, _HERE)

import journey  # noqa: E402
import relentless  # noqa: E402
import test_journey as tj  # noqa: E402 — the canned journey fixture
import test_loop as tl  # noqa: E402 — FakeCtx + the scripted flow fakes


def valid_hindsight(j):
    return {"schema": 1, "optimality": "acceptable", "hindsight_path": [],
            "avoidable_branches": [
                {"node": "S0", "option": "patch-loader",
                 "enabling_evidence_fp": j["nodes"][0]["evidence"][0]["fp"],
                 "why": "the clarify fact already justified it"}],
            "unavoidable_branches": [], "promoted_learnings": ["L1"]}


class RunHindsight(unittest.TestCase):
    """run_hindsight in isolation: dual-channel read, violation-echo retry, and the
    skip-sentinel posture on every failure mode."""

    def setUp(self):
        self.j, _ = tj.fixture()
        self.tmp = tempfile.mkdtemp(prefix="retro-test-")
        self._orig = relentless.invoke_hermes
        self.prompts = []

    def tearDown(self):
        relentless.invoke_hermes = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _wire(self, replies):
        """Script invoke_hermes; each reply is a string (stdout) or a callable given
        the out_path (to simulate the artifact channel)."""
        state = {"n": 0}

        def fake(prompt, timeout):
            self.prompts.append(prompt)
            r = replies[min(state["n"], len(replies) - 1)]
            state["n"] += 1
            if callable(r):
                return r(os.path.join(self.tmp, "retro.json"))
            return r
        relentless.invoke_hermes = fake

    def test_valid_echo_is_accepted_and_tier_stamped(self):
        self._wire([json.dumps(valid_hindsight(self.j))])
        out = relentless.run_hindsight(self.j, self.tmp, 60)
        self.assertEqual(out["optimality"], "acceptable")
        self.assertEqual(out["avoidable_branches"][0]["tier"], "genuinely-avoidable")
        self.assertIn("HINDSIGHT JUDGE", self.prompts[0])
        # the judge consumes the COMPACT render — the citation skeleton must be there
        self.assertIn(self.j["nodes"][0]["evidence"][0]["fp"], self.prompts[0])

    def test_artifact_beats_stdout(self):
        def write_artifact(out_path):
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(valid_hindsight(self.j), fh)
            return "prose that is not JSON at all"
        self._wire([write_artifact])
        out = relentless.run_hindsight(self.j, self.tmp, 60)
        self.assertEqual(out["optimality"], "acceptable")

    def test_bad_citation_gets_one_violation_echo_retry_then_succeeds(self):
        bad = dict(valid_hindsight(self.j))
        bad["avoidable_branches"] = [{"node": "S99", "option": "x",
                                      "enabling_evidence_fp": "nope", "why": "w"}]
        self._wire([json.dumps(bad), json.dumps(valid_hindsight(self.j))])
        out = relentless.run_hindsight(self.j, self.tmp, 60)
        self.assertEqual(out["optimality"], "acceptable")
        self.assertEqual(len(self.prompts), 2)
        self.assertIn("FAILED validation", self.prompts[1])
        self.assertIn("S99", self.prompts[1])

    def test_persistently_invalid_becomes_a_skip_sentinel(self):
        self._wire(["not json in any way"])
        out = relentless.run_hindsight(self.j, self.tmp, 60)
        self.assertIn("skipped", out)
        self.assertEqual(len(self.prompts), relentless.RETRO_ATTEMPTS)

    def test_oneshot_exception_becomes_a_skip_sentinel(self):
        def boom(prompt, timeout):
            raise RuntimeError("network down")
        relentless.invoke_hermes = boom
        out = relentless.run_hindsight(self.j, self.tmp, 60)
        self.assertIn("RuntimeError", out["skipped"])

    def test_prior_artifact_is_quarantined_before_attempts(self):
        prior = (json.dumps(valid_hindsight(self.j), sort_keys=True) + "\n").encode()
        with open(os.path.join(self.tmp, "retro.json"), "wb") as fh:
            fh.write(prior)
        self._wire(["not json in any way"])
        out = relentless.run_hindsight(self.j, self.tmp, 60)
        self.assertIn("skipped", out)
        with open(os.path.join(self.tmp, "retro.json.prior"), "rb") as fh:
            self.assertEqual(fh.read(), prior)

    def test_non_object_claim_retries_then_accepts_valid_reply(self):
        bad = dict(valid_hindsight(self.j))
        bad["avoidable_branches"] = ["not an object"]
        self._wire([json.dumps(bad), json.dumps(valid_hindsight(self.j))])
        out = relentless.run_hindsight(self.j, self.tmp, 60)
        self.assertEqual(out["optimality"], "acceptable")
        self.assertEqual(len(self.prompts), 2)
        self.assertIn("FAILED validation", self.prompts[1])

    def test_prompt_requests_shorter_path_promoted_learning(self):
        self._wire([json.dumps(valid_hindsight(self.j))])
        relentless.run_hindsight(self.j, self.tmp, 60)
        self.assertIn("restate it as one self-contained", self.prompts[0])


class FlowGating(tl.LoopBase):
    """retro/judge fires ONLY on success + cascade + leftover budget, and its result —
    judged or skipped — never touches the run's outcome."""

    def _wire_success(self):
        self.wire([tl.clar([tl.ts("q1", "a1")])], [tl.pl(tl.tk("t1"))])

    def test_judge_fires_on_success_with_cascade_and_budget(self):
        self._wire_success()
        judged = []
        relentless.run_hindsight = (
            lambda j, d, to: judged.append(to) or {"optimality": "near-optimal"})
        ctx = tl.FakeCtx(completed={"t0": 1000.0, "c0/clock": 1000.0,
                                    "retro/clock": 1010.0})
        out = relentless.relentless_flow(ctx, tl.inp(cascade=True, wallclock=2000))
        self.assertEqual(out["outcome"], "success")
        self.assertIn("retro/judge", ctx.keys)
        self.assertEqual(self.reported["hindsight"], {"optimality": "near-optimal"})
        # timeout clamped into [HINDSIGHT_TO_FLOOR, HINDSIGHT_TO_CAP]
        self.assertEqual(judged, [relentless.HINDSIGHT_TO_CAP])

    def test_budget_starved_success_skips_with_a_receipt(self):
        self._wire_success()
        ctx = tl.FakeCtx(completed={"t0": 1000.0, "c0/clock": 1000.0,
                                    "retro/clock": 1000.0 + 1990.0})
        out = relentless.relentless_flow(ctx, tl.inp(cascade=True, wallclock=2000))
        self.assertEqual(out["outcome"], "success")
        self.assertNotIn("retro/judge", ctx.keys)
        self.assertIn("no leftover budget", self.reported["hindsight"]["skipped"])

    def test_no_cascade_never_reads_the_retro_clock(self):
        self._wire_success()
        ctx = tl.FakeCtx()
        out = relentless.relentless_flow(ctx, tl.inp())
        self.assertEqual(out["outcome"], "success")
        self.assertNotIn("retro/clock", ctx.keys)
        self.assertNotIn("retro/judge", ctx.keys)

    def test_failure_never_judges_even_under_cascade(self):
        self.wire([tl.clar([tl.ts("q1", "a1")]),
                   tl.clar([], stop="converged (no question above floor)")],
                  [tl.pl(tl.tk("t1", "alfa"))], [{"t1": "failed"}])
        ctx = tl.FakeCtx(completed={"t0": 1000.0, "c0/clock": 1000.0,
                                    "c1/clock": 1001.0})
        out = relentless.relentless_flow(ctx, tl.inp(cascade=True, wallclock=10 ** 6,
                                                     max_cycles=2))
        self.assertNotEqual(out["outcome"], "success")
        self.assertNotIn("retro/judge", ctx.keys)

    def test_a_skipped_judge_never_flips_success(self):
        self._wire_success()
        relentless.run_hindsight = lambda j, d, to: {"skipped": "judge exploded"}
        ctx = tl.FakeCtx(completed={"t0": 1000.0, "c0/clock": 1000.0,
                                    "retro/clock": 1010.0})
        out = relentless.relentless_flow(ctx, tl.inp(cascade=True, wallclock=2000))
        self.assertEqual(out["outcome"], "success")
        self.assertEqual(self.reported["hindsight"], {"skipped": "judge exploded"})


class ReportSplice(unittest.TestCase):
    """write_report with a journey: the hindsight splices into journey.json, report.md
    becomes the FULL render, and promoted learnings ride into knowledge promotion as
    records WITHOUT mutating the in-memory ledger."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="retro-report-")
        self.j, _ = tj.fixture()
        self._promote, self.promoted = relentless.knowledge.promote, []
        relentless.knowledge.promote = (
            lambda ledger, slug, project, **kw: self.promoted.extend(ledger))
        relentless.set_knowledge_ctx(True, "proj", "probe")

    def tearDown(self):
        relentless.knowledge.promote = self._promote
        relentless.set_knowledge_ctx(True, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_splice_render_and_promotion(self):
        ledger = [{"cycle": 0, "source": "clarify", "kind": "fact", "text": "F1",
                   "fp": journey.fp("F1"), "meta": {}}]
        hs = {"optimality": "near-optimal", "avoidable_branches": [],
              "unavoidable_branches": [], "hindsight_path": [],
              "promoted_learnings": ["the loader re-reads config on SIGHUP"]}
        path = relentless.write_report(self.tmp, "success", ledger, 1, "d",
                                       journey_obj=self.j, hindsight=hs)
        with open(os.path.join(self.tmp, "journey.json")) as fh:
            spliced = json.load(fh)
        self.assertEqual(spliced["hindsight"]["optimality"], "near-optimal")
        report = open(path).read()
        self.assertTrue(report.startswith("# journey: probe — SUCCESS"),
                        "report.md must be the journey's FULL render")
        self.assertIn("optimality: near-optimal", report)
        self.assertIn("## Established facts", report)  # the legacy appendix survives
        # learnings promoted as records; the in-memory ledger stays untouched
        self.assertTrue(any(r["text"] == "the loader re-reads config on SIGHUP"
                            and r["kind"] == "fact" for r in self.promoted))
        self.assertEqual(len(ledger), 1)

    def test_skip_sentinel_splices_as_the_unavailable_note(self):
        relentless.write_report(self.tmp, "success", [], 1, "d",
                                journey_obj=self.j, hindsight={"skipped": "no budget"})
        with open(os.path.join(self.tmp, "journey.json")) as fh:
            spliced = json.load(fh)
        self.assertEqual(spliced["hindsight"], {"skipped": "no budget"})
        report = open(os.path.join(self.tmp, "report.md")).read()
        self.assertIn("hindsight unavailable — no budget", report)
        self.assertEqual(self.promoted, [])  # no learnings, nothing promoted...

    def test_hindsight_path_synthesizes_and_promotes_learning(self):
        hs = {"optimality": "acceptable", "avoidable_branches": [],
              "unavoidable_branches": [],
              "hindsight_path": [
                  {"method": "inspect loader", "why_available_earlier": "logs named it"},
                  {"method": "patch loader", "why_available_earlier": ""}],
              "promoted_learnings": []}
        relentless.write_report(self.tmp, "success", [], 1, "d",
                                journey_obj=self.j, hindsight=hs)
        with open(os.path.join(self.tmp, "journey.json")) as fh:
            spliced = json.load(fh)
        learnings = spliced["hindsight"]["promoted_learnings"]
        self.assertEqual(len(learnings), 1)
        self.assertIn("inspect loader → patch loader", learnings[0])
        self.assertTrue(any(r["text"] == learnings[0] for r in self.promoted))

    def test_existing_path_learning_is_not_duplicated(self):
        hs = {"optimality": "acceptable", "avoidable_branches": [],
              "unavoidable_branches": [],
              "hindsight_path": [
                  {"method": "inspect loader", "why_available_earlier": "logs named it"},
                  {"method": "patch loader", "why_available_earlier": ""}],
              "promoted_learnings": ["Future runs should inspect loader first."]}
        relentless.write_report(self.tmp, "success", [], 1, "d",
                                journey_obj=self.j, hindsight=hs)
        self.assertEqual(hs["promoted_learnings"],
                         ["Future runs should inspect loader first."])

    def test_legacy_shape_without_a_journey_is_unchanged(self):
        path = relentless.write_report(self.tmp, "success", [], 2, "detail text")
        report = open(path).read()
        self.assertTrue(report.startswith("# relentless-solve report"))
        self.assertIn("OUTCOME: success   CYCLES: 2", report)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "journey.json")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
