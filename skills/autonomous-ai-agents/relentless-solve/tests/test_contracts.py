#!/usr/bin/env python3
"""Cross-skill contract tests — pin the drift surfaces between relentless-solve and its
building blocks WITHOUT runtime coupling. Each class skips (not fails) when the
counterpart skill is not on disk, so this suite stays runnable standalone.

Surfaces pinned:
  - PlanContract: the plan-as-data seam — relentless._decomposer() resolves the SAME
    planfile.py/envelope.py the task-decomposer skill ships; the golden plan fixture
    validates; the planning prompt names the exact plan.json path request_plan reads.
  - EnvelopeContract: the solve `single_method` seam — solve_single builds its prompt
    via _envelope().real_prompt (method-explorer, fka resilient-planner, owns its
    invocation contract; old env var / dir name stay accepted).
  - InvestigatorContract: the clarify seam — run_clarify's by-name programmatic imports
    (apply_capability, iterate(seed_evidence=...)) and the converged stop_reason
    vocabulary the information-dry rule keys on.

Run: python3 tests/test_contracts.py
"""

import importlib.util
import inspect
import json
import os
import shutil
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import relentless  # noqa: E402

_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))


def _resolve(env_var, skill):
    sibling = os.path.abspath(os.path.join(_HERE, "..", "..", skill, "scripts"))
    return (os.environ.get(env_var) or (sibling if os.path.isdir(sibling) else None)
            or os.path.join(_HOME, "skills", skill, "scripts"))


def _load(path, alias):
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_explorer():
    """method-explorer scripts dir — new env/dir names first, fka fallbacks after."""
    for env in ("METHOD_EXPLORER_DRIVE", "RESILIENT_DRIVE"):
        if os.environ.get(env):
            return os.path.dirname(os.environ[env])
    for env in ("METHOD_EXPLORER_DIR", "RESILIENT_ENVELOPE_DIR"):
        if os.environ.get(env):
            return os.environ[env]
    for skill in ("method-explorer", "resilient-planner"):
        sibling = os.path.abspath(os.path.join(_HERE, "..", "..", skill, "scripts"))
        if os.path.isdir(sibling):
            return sibling
    for skill in ("method-explorer", "resilient-planner"):
        deployed = os.path.join(_HOME, "skills", skill, "scripts")
        if os.path.isdir(deployed):
            return deployed
    return os.path.join(_HOME, "skills", "method-explorer", "scripts")


_EXPLORER_SCRIPTS = _resolve_explorer()
_ENVELOPE = _load(os.path.join(_EXPLORER_SCRIPTS, "envelope.py"), "me_envelope")

_TD_SCRIPTS = _resolve("TASK_DECOMPOSER_DIR", "task-decomposer")
_PLANFILE = _load(os.path.join(_TD_SCRIPTS, "planfile.py"), "td_planfile")
_GOLDEN = os.path.join(_TD_SCRIPTS, "..", "tests", "fixtures", "plan-golden.json")


@unittest.skipUnless(_PLANFILE, f"task-decomposer planfile.py not found in {_TD_SCRIPTS!r}")
class PlanContract(unittest.TestCase):
    def test_planner_resolution_matches_the_shipped_skill(self):
        pf, env = relentless._decomposer()
        self.assertEqual(os.path.realpath(pf.__file__),
                         os.path.realpath(os.path.join(_TD_SCRIPTS, "planfile.py")),
                         "relentless must load the planfile the task-decomposer skill ships")
        self.assertEqual(os.path.dirname(os.path.realpath(env.__file__)),
                         os.path.dirname(os.path.realpath(pf.__file__)),
                         "schema and prompt must come from the SAME directory")
        for name in ("plan_prompt", "retry_suffix"):
            self.assertTrue(callable(getattr(env, name, None)),
                            f"task-decomposer envelope must expose {name}()")

    def test_golden_fixture_validates(self):
        with open(_GOLDEN, encoding="utf-8") as fh:
            self.assertEqual(_PLANFILE.validate(json.load(fh)), [])

    def test_schema_version_is_2_and_golden_fixture_carries_intent_link(self):
        # LEVEL 1's staleness gate (relentless.stale_tail) reads task["intent_link"] —
        # if the task-decomposer skill's schema drifts back to 1, this must fail loudly
        # rather than silently degrade the gate's vocabulary-bleed trigger.
        self.assertEqual(_PLANFILE.SCHEMA_VERSION, 2)
        with open(_GOLDEN, encoding="utf-8") as fh:
            golden = json.load(fh)
        self.assertEqual(golden["schema"], 2)
        self.assertTrue(all(t.get("intent_link") for t in golden["tasks"]))

    def test_validator_rejects_the_driver_hazards(self):
        def base(**over):
            d = {"schema": 1, "slug": "s", "cycle": 0, "disposition": "tasks",
                 "rationale": "r", "question": None,
                 "tasks": [{"id": "t1", "method": "m", "description": "d",
                            "success_criterion": "c", "depends_on": [],
                            "status": "pending"}]}
            d.update(over)
            return d
        dup = base()
        dup["tasks"].append(dict(dup["tasks"][0]))
        self.assertTrue(_PLANFILE.validate(dup), "duplicate ids are step-key collisions")
        many = base(tasks=[{**base()["tasks"][0], "id": f"t{i}"}
                           for i in range(_PLANFILE.MAX_TASKS + 1)])
        self.assertTrue(_PLANFILE.validate(many))
        self.assertTrue(_PLANFILE.validate(base(disposition="needs_decision", tasks=[])),
                        "needs_decision without a question must be rejected")
        judged = base()
        judged["tasks"][0]["status"] = "worked"
        self.assertTrue(_PLANFILE.validate(judged), "a planner never pre-judges")

    def test_partial_replan_prompt_resolvable_and_names_forbidden_ids(self):
        # LEVEL 1's mid-cycle seam (relentless.request_partial_replan) — pin that the
        # envelope module relentless resolves also exposes partial_replan_prompt, and
        # that its output actually surfaces the forbidden ids (drift tripwire, same
        # style as test_plan_prompt_names_the_artifact_request_plan_reads below).
        _, env = relentless._decomposer()
        self.assertTrue(callable(getattr(env, "partial_replan_prompt", None)),
                        "task-decomposer envelope must expose partial_replan_prompt()")
        p = env.partial_replan_prompt("BODY", "/x/replan-1.json", {"t1", "t2"})
        self.assertIn("PARTIAL REPLAN", p)
        self.assertIn("t1", p)
        self.assertIn("t2", p)

    def test_alternatives_requested_but_never_binding(self):
        # The journey fold (journey.py) reads plan["alternatives"] as the run's
        # prospective decision record — pin that the resolved envelope asks for it in
        # BOTH prompts and that the resolved validator never rejects a plan over it
        # (advisory capture; control flow untouched).
        _, env = relentless._decomposer()
        for p in (env.plan_prompt("BODY", "/x/plan.json"),
                  env.partial_replan_prompt("BODY", "/x/replan-1.json", set())):
            self.assertIn('"alternatives"', p)
            self.assertIn("why_not_now", p)
        with open(_GOLDEN, encoding="utf-8") as fh:
            golden = json.load(fh)
        golden["alternatives"] = [{"method": "another way", "why_not_now": "slower"}]
        self.assertEqual(_PLANFILE.validate(golden), [])
        golden["alternatives"] = "garbage, not a list"
        self.assertEqual(_PLANFILE.validate(golden), [],
                         "a malformed advisory field must never reject a plan")

    def test_plan_prompt_names_the_artifact_request_plan_reads(self):
        """The drift tripwire: the path the envelope tells the model to write must be
        the path request_plan reads back."""
        tmp = tempfile.mkdtemp(prefix="rls-plancontract-")
        prompts = []
        orig = relentless.invoke_hermes
        pf, _ = relentless._decomposer()
        with open(_GOLDEN, encoding="utf-8") as fh:
            golden = json.load(fh)
        relentless.invoke_hermes = lambda p, t: prompts.append(p) or json.dumps(golden)
        try:
            slug_dir = os.path.join(tmp, "relentless", "s")
            plan = relentless.request_plan(slug_dir, "s", 3, "BODY", 60)
            expected = pf.plan_path(os.path.join(slug_dir, "c3"))
            self.assertIn(expected, prompts[0])
            self.assertTrue(os.path.exists(expected))
            self.assertEqual(plan["cycle"], 3)
        finally:
            relentless.invoke_hermes = orig
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(_ENVELOPE, f"method-explorer envelope.py not found in {_EXPLORER_SCRIPTS!r}")
class EnvelopeContract(unittest.TestCase):
    """The solve `single_method` seam: solve_single calls _envelope().real_prompt
    directly — pin the resolution and the names/behavior it relies on."""

    def test_envelope_resolves_to_canonical_module(self):
        self.assertEqual(os.path.realpath(relentless._envelope().__file__),
                         os.path.realpath(os.path.join(_EXPLORER_SCRIPTS, "envelope.py")))

    def test_real_prompt_supports_solve_single(self):
        sig = inspect.signature(_ENVELOPE.real_prompt)
        self.assertIn("extra", sig.parameters,
                      "solve_single pins risk=read via the extra= kwarg")
        p = _ENVELOPE.real_prompt("I", "slug-single", relentless.PLANS_DIR,
                                  extra="HARD CONSTRAINT: read-only")
        self.assertIn("method-explorer skill", p)
        self.assertIn("skill_view", p)
        self.assertIn("slug-single/plan-tree.md", p)
        self.assertIn("read-only", p)


_INV_SCRIPTS = os.path.join(relentless._AA, "investigator", "scripts")


def _load_iterate():
    """Mirror relentless._investigator()'s resolution (sibling under _AA) so the pinned
    module is the one the loop would import. iterate.py's infogain/ask imports are
    graceful at import time (DESIGN 'Isolation posture'), so a bare import is safe."""
    path = os.path.join(_INV_SCRIPTS, "iterate.py")
    if not os.path.exists(path):
        return None
    os.environ.setdefault("INFOGAIN_SCRIPTS_DIR",
                          os.path.join(relentless._AA, "information-gain", "scripts"))
    if _INV_SCRIPTS not in sys.path:
        sys.path.insert(0, _INV_SCRIPTS)
    return _load(path, "investigator_iterate")


_ITERATE = _load_iterate()


@unittest.skipUnless(_ITERATE, f"investigator iterate.py not found in {_INV_SCRIPTS!r}")
class InvestigatorContract(unittest.TestCase):
    """The clarify seam is a by-name programmatic import (unlike the oneshot subprocess
    seams) — pin the names relentless.run_clarify/stop_is_converged actually rely on."""

    def test_api_signatures(self):
        self.assertTrue(callable(getattr(_ITERATE, "apply_capability", None)),
                        "iterate.py must expose apply_capability(cfg, capability)")
        sig = inspect.signature(_ITERATE.iterate)
        self.assertIn("seed_evidence", sig.parameters,
                      "iterate() lost the seed_evidence kwarg — run_clarify feeds the "
                      "ledger through it every cycle")

    def test_stop_reason_vocabulary_is_pinned(self):
        const = getattr(_ITERATE, "STOP_CONVERGED", None)
        self.assertIsNotNone(
            const,
            "iterate.py must export STOP_CONVERGED (the converged stop_reason wording): "
            "relentless's information-dry rule keys on it via stop_is_converged(). Add "
            "e.g. STOP_CONVERGED = 'converged' and build converged stop_reasons from it.")
        self.assertIn("converged", const,
                      "STOP_CONVERGED must contain 'converged' — replays never import the "
                      "investigator and fall back to that substring; live and replay "
                      "decisions must agree or the engine flags NonDeterminism")

    def test_run_dir_cfg_key(self):
        # run_clarify passes the per-cycle journal dir through cfg["run_dir"]; the key
        # must exist in DEFAULTS (default None = in-memory) or the durability seam drifted.
        self.assertIn("run_dir", _ITERATE.DEFAULTS)
        self.assertIsNone(_ITERATE.DEFAULTS["run_dir"])

    def test_tombstone_shape_is_the_fold_clarify_seam(self):
        # fold_clarify reads question/status/evidence and fps on the question — pin the
        # exact record shape _tombstone emits (previously unpinned).
        t = _ITERATE._tombstone({"question": "Q?"}, True, "A")
        gap = _ITERATE._tombstone({"question": "Q?"}, False, "gap")
        self.assertLessEqual({"question", "status", "fact", "evidence"}, set(t))
        self.assertEqual({t["status"], gap["status"]}, {"ANSWERED", "NOT_FOUND"})
        self.assertEqual(t["via"], "research")
        self.assertEqual(gap["via"], "research")
        allowed_via = frozenset({"research", "derived", "assumed"})
        for via in ("derived", "assumed"):
            tomb = _ITERATE._tombstone({"question": "Q?"}, True, "A", via=via)
            self.assertEqual(tomb["via"], via)
            self.assertIn(tomb["via"], allowed_via)
        self.assertFalse(_ITERATE.DEFAULTS.get("triage"))

    def test_apply_capability_sets_the_answer_keys(self):
        cfg = _ITERATE.apply_capability({}, "read")
        self.assertIn("answer_toolsets", cfg)
        self.assertIn("answer_directive", cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
