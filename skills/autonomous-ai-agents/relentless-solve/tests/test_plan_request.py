#!/usr/bin/env python3
"""Unit tests for the plan-request / task-execution seams — scripted invoke_hermes,
tempdir state, no container, no network. Run:
    python3 tests/test_plan_request.py
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import relentless  # noqa: E402


def plan_obj(**over):
    base = {"schema": 2, "slug": "s", "cycle": 0, "disposition": "tasks",
            "rationale": "r", "question": None,
            "tasks": [{"id": "t1", "method": "alfa", "description": "do alfa",
                       "success_criterion": "alfa observably done",
                       "intent_link": "alfa is the best next step toward the intent",
                       "depends_on": [], "status": "pending"}]}
    base.update(over)
    return base


class SeamBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rls-plan-")
        self.slug_dir = os.path.join(self.tmp, "relentless", "s")
        self.cycle_dir = os.path.join(self.slug_dir, "c0")
        self.prompts = []
        self._invoke = relentless.invoke_hermes

    def tearDown(self):
        relentless.invoke_hermes = self._invoke
        shutil.rmtree(self.tmp, ignore_errors=True)

    def script(self, responses):
        """responses: callables(prompt, cycle_dir) -> stdout (may write artifacts);
        the last one repeats."""
        state = {"i": 0}

        def fake(prompt, timeout):
            self.prompts.append(prompt)
            fn = responses[min(state["i"], len(responses) - 1)]
            state["i"] += 1
            return fn(prompt, self.cycle_dir)
        relentless.invoke_hermes = fake


class RequestPlan(SeamBase):
    def test_artifact_preferred_over_stdout(self):
        def write_both(prompt, cdir):
            with open(os.path.join(cdir, "plan.json"), "w", encoding="utf-8") as fh:
                json.dump(plan_obj(rationale="from artifact"), fh)
            return json.dumps(plan_obj(rationale="stdout decoy"))
        self.script([write_both])
        plan = relentless.request_plan(self.slug_dir, "s", 0, "BODY", 60)
        self.assertEqual(plan["rationale"], "from artifact")

    def test_stdout_fallback_rewrites_the_artifact(self):
        self.script([lambda p, c: json.dumps(plan_obj())])
        plan = relentless.request_plan(self.slug_dir, "s", 0, "BODY", 60)
        self.assertEqual(plan["tasks"][0]["id"], "t1")
        with open(os.path.join(self.cycle_dir, "plan.json"), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["tasks"][0]["id"], "t1")

    def test_malformed_then_valid_on_retry(self):
        bad = plan_obj()
        bad["tasks"][0]["id"] = "BAD ID"

        def write_bad(prompt, cdir):
            with open(os.path.join(cdir, "plan.json"), "w", encoding="utf-8") as fh:
                json.dump(bad, fh)
            return ""
        self.script([write_bad, lambda p, c: json.dumps(plan_obj())])
        plan = relentless.request_plan(self.slug_dir, "s", 0, "BODY", 60)
        self.assertEqual(len(self.prompts), 2, "exactly one retry")
        self.assertIn("FAILED validation", self.prompts[1])
        self.assertIn("BAD ID", self.prompts[1])  # violations echoed into the retry
        self.assertEqual(plan["tasks"][0]["id"], "t1")
        # the rejected attempt was archived, not clobbered
        self.assertTrue(os.path.exists(os.path.join(self.cycle_dir, "plan.json.rej0")))

    def test_three_strikes_raise(self):
        self.script([lambda p, c: "no json here at all"])
        with self.assertRaises(RuntimeError):
            relentless.request_plan(self.slug_dir, "s", 0, "BODY", 60)
        self.assertEqual(len(self.prompts), relentless.PLAN_ATTEMPTS)

    def test_identity_is_stamped_over_model_echo(self):
        self.script([lambda p, c: json.dumps(plan_obj(slug="stale-slug", cycle=99))])
        plan = relentless.request_plan(self.slug_dir, "s", 0, "BODY", 60)
        self.assertEqual((plan["slug"], plan["cycle"]), ("s", 0))

    def test_prompt_carries_body_and_artifact_path(self):
        self.script([lambda p, c: json.dumps(plan_obj())])
        relentless.request_plan(self.slug_dir, "s", 0, "THE-INTENT-BODY", 60)
        self.assertIn("THE-INTENT-BODY", self.prompts[0])
        self.assertIn(os.path.join(self.cycle_dir, "plan.json"), self.prompts[0])


def _tail_task(tid, **over):
    base = {"id": tid, "method": "delta", "description": "do delta",
            "success_criterion": "delta done", "intent_link": "l", "depends_on": [],
            "status": "pending"}
    base.update(over)
    return base


class PartialReplan(SeamBase):
    """request_partial_replan / _attempt_partial_replan — LEVEL 1's mid-cycle plan seam."""

    def test_own_artifact_never_overwrites_plan_json(self):
        def write_replan(prompt, cdir):
            path = os.path.join(cdir, "replan-1.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(plan_obj(tasks=[_tail_task("t2-new")]), fh)
            return ""
        self.script([write_replan])
        replan = relentless.request_partial_replan(self.slug_dir, "s", 0, 1, "BODY",
                                                    {"t1"}, 60)
        self.assertEqual(replan["tasks"][0]["id"], "t2-new")
        self.assertFalse(os.path.exists(os.path.join(self.cycle_dir, "plan.json")))
        self.assertTrue(os.path.exists(os.path.join(self.cycle_dir, "replan-1.json")))

    def test_id_collision_with_done_ids_is_rejected_and_retried(self):
        colliding = plan_obj(tasks=[_tail_task("t1")])  # "t1" is in done_ids
        good = plan_obj(tasks=[_tail_task("t2-new")])
        self.script([lambda p, c: json.dumps(colliding), lambda p, c: json.dumps(good)])
        replan = relentless.request_partial_replan(self.slug_dir, "s", 0, 1, "BODY",
                                                    {"t1"}, 60)
        self.assertEqual(len(self.prompts), 2, "exactly one retry")
        self.assertIn("collides", self.prompts[1])
        self.assertEqual(replan["tasks"][0]["id"], "t2-new")

    def test_exhaustion_raises_like_request_plan(self):
        self.script([lambda p, c: "no json here at all"])
        with self.assertRaises(RuntimeError):
            relentless.request_partial_replan(self.slug_dir, "s", 0, 1, "BODY", {"t1"}, 60)
        self.assertEqual(len(self.prompts), relentless.PLAN_ATTEMPTS)

    def test_prompt_carries_forbidden_ids_and_body(self):
        self.script([lambda p, c: json.dumps(plan_obj(tasks=[_tail_task("t3-new")]))])
        relentless.request_partial_replan(self.slug_dir, "s", 0, 1, "PARTIAL-BODY",
                                          {"t1", "t2"}, 60)
        self.assertIn("PARTIAL-BODY", self.prompts[0])
        self.assertIn("t1", self.prompts[0])
        self.assertIn("t2", self.prompts[0])

    def test_attempt_partial_replan_wraps_exception_non_fatally(self):
        self.script([lambda p, c: "not json"])  # exhausts PLAN_ATTEMPTS -> RuntimeError
        result = relentless._attempt_partial_replan(self.slug_dir, "s", 0, 1, "BODY",
                                                     {"t1"}, 60)
        self.assertEqual(result["disposition"], "replan-failed")
        self.assertIn("RuntimeError", result["error"])

    def test_attempt_partial_replan_passes_through_a_valid_result(self):
        self.script([lambda p, c: json.dumps(plan_obj(tasks=[_tail_task("t2-new")]))])
        result = relentless._attempt_partial_replan(self.slug_dir, "s", 0, 1, "BODY",
                                                     {"t1"}, 60)
        self.assertEqual(result["disposition"], "tasks")
        self.assertEqual(result["tasks"][0]["id"], "t2-new")


class RunTask(SeamBase):
    TASK = {"id": "t1", "method": "alfa", "description": "do alfa",
            "success_criterion": "alfa observably done", "depends_on": [],
            "status": "pending"}

    def setUp(self):
        super().setUp()
        os.makedirs(self.cycle_dir, exist_ok=True)

    def _result_writer(self, payload):
        def fn(prompt, cdir):
            with open(os.path.join(cdir, "result-t1.json"), "w", encoding="utf-8") as fh:
                fh.write(payload if isinstance(payload, str) else json.dumps(payload))
            return "done"
        return fn

    def test_needs_split_verdict_carries_the_split(self):
        self.script([self._result_writer({"verdict": "needs_split", "evidence": "2-in-1",
                                          "split": ["half a", "half b"]})])
        r = relentless.run_task(self.TASK, self.cycle_dir, 60)
        self.assertEqual(r["verdict"], "needs_split")
        self.assertEqual(r["split"], ["half a", "half b"])

    def test_needs_split_without_split_list_is_failed(self):
        self.script([self._result_writer({"verdict": "needs_split", "evidence": "2-in-1"})])
        r = relentless.run_task(self.TASK, self.cycle_dir, 60)
        self.assertEqual(r["verdict"], "failed")
        self.assertIn("without a split list", r["evidence"])

    def test_verdict_from_artifact(self):
        self.script([self._result_writer({"verdict": "worked", "evidence": "saw it"})])
        r = relentless.run_task(self.TASK, self.cycle_dir, 60)
        self.assertEqual(r, {"id": "t1", "method": "alfa", "verdict": "worked",
                             "evidence": "saw it", "learnings": [], "split": []})

    def test_stdout_echo_is_a_fallback_when_artifact_missing(self):
        # no artifact written — only the echoed JSON in stdout, dual-channel per task_prompt
        self.script([lambda p, c: json.dumps({"verdict": "worked", "evidence": "echoed"})])
        r = relentless.run_task(self.TASK, self.cycle_dir, 60)
        self.assertEqual(r["verdict"], "worked")
        self.assertEqual(r["evidence"], "echoed")

    def test_artifact_still_preferred_over_stdout_echo(self):
        def write_and_echo(prompt, cdir):
            with open(os.path.join(cdir, "result-t1.json"), "w", encoding="utf-8") as fh:
                json.dump({"verdict": "worked", "evidence": "from artifact"}, fh)
            return json.dumps({"verdict": "failed", "evidence": "stdout decoy"})
        self.script([write_and_echo])
        r = relentless.run_task(self.TASK, self.cycle_dir, 60)
        self.assertEqual(r["evidence"], "from artifact")

    def test_prompt_carries_the_echo_instruction(self):
        self.script([self._result_writer({"verdict": "worked", "evidence": "e"})])
        relentless.run_task(self.TASK, self.cycle_dir, 60)
        self.assertIn("AND echo the same JSON as your final message", self.prompts[0])

    def test_learnings_parsed_capped_and_truncated(self):
        rich = ("systems: the billing API's /v2/charges endpoint. expected a "
                "synchronous 200 per the docs; " + "x" * 600)  # forces truncation
        payload = {"verdict": "worked", "evidence": "e",
                  "learnings": [rich, "second learning", "", "   ", 42,
                                *[f"extra-{i}" for i in range(6)]]}
        self.script([self._result_writer(payload)])
        r = relentless.run_task(self.TASK, self.cycle_dir, 60)
        self.assertEqual(len(r["learnings"]), relentless.LEARNINGS_MAX_COUNT)
        self.assertLessEqual(len(r["learnings"][0]), relentless.LEARNINGS_MAX_CHARS)
        self.assertEqual(r["learnings"][1], "second learning")
        self.assertNotIn("", r["learnings"])  # blank/whitespace-only entries dropped

    def test_missing_learnings_defaults_to_empty_list(self):
        self.script([self._result_writer({"verdict": "failed", "evidence": "e"})])
        r = relentless.run_task(self.TASK, self.cycle_dir, 60)
        self.assertEqual(r["learnings"], [])

    def test_malformed_learnings_field_ignored_not_fatal(self):
        self.script([self._result_writer(
            {"verdict": "worked", "evidence": "e", "learnings": "not a list"})])
        r = relentless.run_task(self.TASK, self.cycle_dir, 60)
        self.assertEqual(r["learnings"], [])

    def test_missing_artifact_is_failure(self):
        self.script([lambda p, c: "I did it! (but wrote no file)"])
        r = relentless.run_task(self.TASK, self.cycle_dir, 60)
        self.assertEqual(r["verdict"], "failed")
        self.assertIn("no verdict artifact", r["evidence"])

    def test_malformed_artifact_is_failure(self):
        self.script([self._result_writer("{not json")])
        self.assertEqual(relentless.run_task(self.TASK, self.cycle_dir, 60)["verdict"],
                         "failed")

    def test_unknown_verdict_value_is_failure(self):
        self.script([self._result_writer({"verdict": "probably", "evidence": "?"})])
        r = relentless.run_task(self.TASK, self.cycle_dir, 60)
        self.assertEqual(r["verdict"], "failed")
        self.assertIn("malformed verdict artifact", r["evidence"])

    def test_prompt_carries_task_criterion_and_result_path(self):
        self.script([self._result_writer({"verdict": "worked", "evidence": "e"})])
        relentless.run_task(self.TASK, self.cycle_dir, 60)
        p = self.prompts[0]
        self.assertIn("do alfa", p)
        self.assertIn("alfa observably done", p)
        self.assertIn(os.path.join(self.cycle_dir, "result-t1.json"), p)

    def test_task_prompt_adds_read_only_constraint_for_read_capability(self):
        prompt = relentless.task_prompt(self.TASK, self.cycle_dir, capability="read")
        self.assertIn("HARD CONSTRAINT: read-only", prompt)

    def test_task_prompt_default_matches_explicit_none(self):
        default = relentless.task_prompt(self.TASK, self.cycle_dir)
        explicit = relentless.task_prompt(self.TASK, self.cycle_dir, capability=None)
        self.assertEqual(default, explicit)
        self.assertNotIn("HARD CONSTRAINT", default)
        self.assertNotIn("HARD CONSTRAINT", explicit)

    def test_run_task_threads_read_capability_into_prompt(self):
        self.script([self._result_writer({"verdict": "worked", "evidence": "e"})])
        relentless.run_task(self.TASK, self.cycle_dir, 60, capability="read")
        self.assertIn("HARD CONSTRAINT: read-only", self.prompts[0])


class PlanReceipt(SeamBase):
    def test_statuses_written_back(self):
        os.makedirs(self.cycle_dir, exist_ok=True)
        plan = plan_obj()
        plan["tasks"].append({"id": "t2", "method": "beta", "description": "do beta",
                              "success_criterion": "c", "depends_on": ["t1"],
                              "status": "pending"})
        results = [{"id": "t1", "method": "alfa", "verdict": "failed", "evidence": "e"},
                   {"id": "t2", "method": "beta", "verdict": "skipped",
                    "evidence": "dependency failed"}]
        path = relentless.write_plan_receipt(self.cycle_dir, plan, results)
        with open(path, encoding="utf-8") as fh:
            receipt = json.load(fh)
        self.assertEqual([t["status"] for t in receipt["tasks"]], ["failed", "skipped"])
        # the in-memory plan is untouched (the loop folds from results, not the receipt)
        self.assertEqual([t["status"] for t in plan["tasks"]], ["pending", "pending"])


class DodBindings(SeamBase):
    """The two validation-time bindings (coverage / dead-method) ride request_plan's
    existing retry-echo channel — a violating plan is rejected, archived, and the
    violation text reaches the next attempt's prompt."""

    def _writer(self, obj):
        def write(prompt, cdir):
            with open(os.path.join(cdir, "plan.json"), "w", encoding="utf-8") as fh:
                json.dump(obj, fh)
            return ""
        return write

    def test_uncovered_requirement_retries_with_echo(self):
        uncovered = plan_obj()
        uncovered["tasks"][0]["serves"] = ["R1.1"]
        covered = plan_obj(rationale="covered")
        covered["tasks"][0]["serves"] = ["R1.1", "R1.2"]
        self.script([self._writer(uncovered), self._writer(covered)])
        dodctx = {"unmet": ["R1.1", "R1.2"], "known": {"R1", "R1.1", "R1.2"}}
        plan = relentless.request_plan(self.slug_dir, "s", 0, "BODY", 60, dodctx=dodctx)
        self.assertEqual(plan["rationale"], "covered")
        self.assertIn("DEFINITION OF DONE", self.prompts[0])
        self.assertIn("R1.2", self.prompts[1])  # the coverage miss, echoed back
        self.assertIn("FAILED validation", self.prompts[1])

    def test_dead_method_retries_with_echo(self):
        planfile, _ = relentless._decomposer()
        dead = plan_obj()  # method "alfa" is already a dead end
        alive = plan_obj(rationale="fresh method")
        alive["tasks"][0]["method"] = "bravo"
        self.script([self._writer(dead), self._writer(alive)])
        plan = relentless.request_plan(self.slug_dir, "s", 0, "BODY", 60,
                                       dead_fps={planfile.fp("Alfa")})
        self.assertEqual(plan["rationale"], "fresh method")
        self.assertIn("never re-propose", self.prompts[1])

    def test_no_dod_no_dead_is_byte_identical_prompting(self):
        self.script([self._writer(plan_obj())])
        relentless.request_plan(self.slug_dir, "s", 0, "BODY", 60)
        self.assertNotIn("DEFINITION OF DONE", self.prompts[0])
        self.assertNotIn('"serves"', self.prompts[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
