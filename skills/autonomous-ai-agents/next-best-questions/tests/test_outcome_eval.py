"""Tests for the objective-outcome eval (evals/outcome_eval.py + outcome_bank.py) — all mocked."""

import os
import sys
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "evals"))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

try:
    import outcome_bank
    import outcome_eval
    import pipeline
    _OK = True
except Exception:  # pragma: no cover
    _OK = False


@unittest.skipUnless(_OK, "skill scripts not importable")
class TestOutcomeBank(unittest.TestCase):
    def test_schema_and_documented_ambiguity(self):
        seen = set()
        for t in outcome_bank.TASKS:
            for key in ("id", "category", "ambiguous_prompt", "hidden_spec", "func",
                        "tests", "ambiguity"):
                self.assertIn(key, t, t.get("id"))
            self.assertNotIn(t["id"], seen)
            seen.add(t["id"])
            self.assertGreaterEqual(len(t["tests"]), 2, t["id"])
            # the ambiguity must be real: >= 2 plausible readings documented
            self.assertGreaterEqual(len(t["ambiguity"]), 2, t["id"])
            # the hidden detail is HIDDEN: the discriminating spec never leaks verbatim
            self.assertNotIn(t["hidden_spec"], t["ambiguous_prompt"], t["id"])
            for test in t["tests"]:
                self.assertIn(t["func"], test + t["ambiguous_prompt"], t["id"])

    def test_tests_are_import_free_expressions(self):
        for t in outcome_bank.TASKS:
            for test in t["tests"]:
                self.assertNotIn("import", test, t["id"])
                compile(test, "<test>", "eval")   # must be a pure expression


@unittest.skipUnless(_OK, "skill scripts not importable")
class TestRunner(unittest.TestCase):
    def test_per_test_scoring_and_crash_isolation(self):
        frac, per = outcome_eval.run_tests("def f(x): return x + 1",
                                           ["f(1) == 2", "f(2) == 4"])
        self.assertEqual((frac, per), (0.5, [True, False]))
        frac, per = outcome_eval.run_tests("raise RuntimeError('boom')", ["1 == 1"])
        self.assertEqual(frac, 0.0)         # a crashing solution fails everything

    def test_timeout_kills_infinite_solution(self):
        frac, _ = outcome_eval.run_tests("while True: pass", ["1 == 1"], timeout=2)
        self.assertEqual(frac, 0.0)

    def test_extract_code_block_or_raw(self):
        self.assertEqual(outcome_eval.extract_code("```python\ndef g(): pass\n```"),
                         "def g(): pass\n")
        self.assertEqual(outcome_eval.extract_code("def h(): pass"), "def h(): pass")
        # truncated-at-token-limit reply: opening fence, no closer (live smoke caught this —
        # the raw fence reached the interpreter and every check failed on exit_code)
        self.assertEqual(outcome_eval.extract_code("```python\ndef t(): pass"),
                         "def t(): pass")


@unittest.skipUnless(_OK, "skill scripts not importable")
class TestSimulator(unittest.TestCase):
    def test_strict_refusal_is_not_a_reveal(self):
        with mock.patch.object(pipeline, "raw_chat",
                               return_value={"content": outcome_eval.NO_ANSWER, "error": None}):
            got = outcome_eval.simulate_user("spec", "Generic fishing question?", "m")
        self.assertFalse(got["revealed"])
        with mock.patch.object(pipeline, "raw_chat",
                               return_value={"content": "Round half up.", "error": None}):
            got = outcome_eval.simulate_user("spec", "How to round?", "m")
        self.assertTrue(got["revealed"])
        # empty reply (model error) must not count as a reveal either
        with mock.patch.object(pipeline, "raw_chat", return_value={"content": "", "error": "x"}):
            self.assertFalse(outcome_eval.simulate_user("s", "q", "m")["revealed"])

    def test_simulator_prompt_carries_spec_and_rule(self):
        p = outcome_eval.simulator_prompt("HALF UP", "how round?")
        self.assertIn("HALF UP", p)
        self.assertIn(outcome_eval.NO_ANSWER, p)
        self.assertIn("Never invent", p)


@unittest.skipUnless(_OK, "skill scripts not importable")
class TestArms(unittest.TestCase):
    def _models(self):
        return {"skill": "m", "solver": "m", "sim": "m"}

    def test_baseline_asks_nothing(self):
        task = outcome_bank.TASKS[0]
        with mock.patch.object(outcome_eval, "solve_and_score",
                               return_value={"code": "", "frac": 1.0, "per_test": []}) as s:
            row = outcome_eval.run_cell(task, "baseline", 3, self._models())
        self.assertEqual(row["questions"], [])
        self.assertEqual(row["qa"], [])
        self.assertEqual(s.call_args[0][1], [])   # solver saw no Q&A

    def test_nbq_arm_uses_bucket_topk_and_same_solver(self):
        task = outcome_bank.TASKS[0]
        fake = {"bucket": [{"question": f"q{i}", "value": 0.9 - i / 10} for i in range(5)],
                "derived": [], "usage": {}}
        with mock.patch.object(outcome_eval.infogain, "run", return_value=fake) as runm, \
             mock.patch.object(outcome_eval, "simulate_user",
                               side_effect=lambda spec, q, m: {"question": q, "answer": "A",
                                                               "revealed": True}), \
             mock.patch.object(outcome_eval, "solve_and_score",
                               return_value={"code": "", "frac": 1.0, "per_test": []}) as s:
            row = outcome_eval.run_cell(task, "nbq", 3, self._models())
        self.assertEqual(row["questions"], ["q0", "q1", "q2"])       # top-K by rank
        self.assertEqual(len(s.call_args[0][1]), 3)                  # solver saw K Q&As
        cfg = runm.call_args[0][1]
        self.assertNotIn("auto_derive", cfg)                          # plain nbq: derive off
        self.assertEqual(row["meta"]["q_values"], [0.9, 0.8, 0.7])

    def test_nbq_derive_arm_folds_tombstones(self):
        task = outcome_bank.TASKS[0]
        fake = {"bucket": [{"question": "q0", "value": 0.9}],
                "derived": [{"question": "dq", "answer": "da", "derivable_prob": 0.9,
                             "round": 1}], "usage": {}}
        with mock.patch.object(outcome_eval.infogain, "run", return_value=fake) as runm, \
             mock.patch.object(outcome_eval, "simulate_user",
                               side_effect=lambda spec, q, m: {"question": q, "answer": "A",
                                                               "revealed": True}), \
             mock.patch.object(outcome_eval, "solve_and_score",
                               return_value={"code": "", "frac": 1.0, "per_test": []}) as s:
            row = outcome_eval.run_cell(task, "nbq-derive", 2, self._models())
        self.assertEqual(runm.call_args[0][1].get("auto_derive"), "on")
        qa = s.call_args[0][1]
        self.assertEqual(qa[-1]["question"], "dq")                    # tombstone reaches solver
        self.assertIn("derived", qa[-1]["answer"])

    def test_zeroshot_arm_one_call_numbered_parse_k_cap(self):
        task = outcome_bank.TASKS[0]
        calls = []

        def fake_chat(model, prompt, timeout=0, num_predict=0, **kw):
            calls.append(prompt)
            return {"content": "1. Case sensitivity?\n2) Sort order?\n3. Locale?\n4. Extra?",
                    "error": None}

        with mock.patch.object(pipeline, "raw_chat", side_effect=fake_chat):
            qs, meta = outcome_eval.questions_zeroshot(task, 3, "m")
        self.assertEqual(len(calls), 1)                       # ONE naive call — the P4 control
        self.assertIn(task["ambiguous_prompt"], calls[0])
        self.assertIn("3 best clarifying questions", calls[0])
        self.assertEqual(qs, ["Case sensitivity?", "Sort order?", "Locale?"])  # K cap applied
        # dry-run shows the same prompt (the once-dead branch, now wired)
        self.assertEqual(calls[0], outcome_eval.zeroshot_prompt(task, 3))

    def test_prompt_evsi_arm_is_one_call_with_framework(self):
        task = outcome_bank.TASKS[0]
        calls = []

        def fake_chat(model, prompt, timeout=0, num_predict=0, **kw):
            calls.append(prompt)
            return {"content": "1. What order?\n2. Case sensitivity?", "error": None}

        with mock.patch.object(pipeline, "raw_chat", side_effect=fake_chat):
            qs, meta = outcome_eval.questions_prompt_evsi(task, 2, "m")
        self.assertEqual(len(calls), 1)
        self.assertIn("Expected Value of Sample Information", calls[0])
        self.assertIn("DERIVE-don't-ask", calls[0])
        self.assertEqual(qs, ["What order?", "Case sensitivity?"])


_AGENTIC_REF = {
    "csv-report": """import csv, os
os.makedirs('out', exist_ok=True)
rows = []
for f in sorted(os.listdir('data')):
    if not f.endswith('.csv'):
        continue
    with open(os.path.join('data', f)) as fh:
        r = list(csv.reader(fh))
    n = len(r[0])
    rows.append((f, sum(1 for x in r[1:] if len(x) == n)))
with open('out/report.csv', 'w') as fh:
    fh.write('file,rows\\n')
    for f, c in rows:
        fh.write(f'{f},{c}\\n')
""",
    "log-clean": """import os, time
logs = [os.path.join('logs', f) for f in os.listdir('logs') if f.endswith('.log')]
newest = max(logs, key=os.path.getmtime)
n = 0
for p in logs:
    if p != newest and time.time() - os.path.getmtime(p) > 7 * 86400:
        os.remove(p)
        n += 1
print(n)
""",
    "db-config": """import configparser, os, sys
cp = configparser.ConfigParser()
cp.read('config.ini')
url = cp.get('db', 'url', fallback=None) or os.environ.get('DB_URL')
if not url:
    print('no database url configured', file=sys.stderr)
    sys.exit(2)
print(url)
""",
    "dupe-finder": """import hashlib, os
groups = {}
for f in sorted(os.listdir('data')):
    p = os.path.join('data', f)
    if os.path.isfile(p):
        h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        groups.setdefault(h, []).append(f)
for h, fs in sorted(groups.items(), key=lambda kv: kv[1][0]):
    if len(fs) > 1:
        print(' '.join(sorted(fs)))
""",
    "json-migrate": """import json, shutil
shutil.copy('records.json', 'records.json.bak')
recs = json.load(open('records.json'))
for r in recs:
    r['username'] = r.pop('user')
    r['version'] = 2
json.dump(recs, open('records.json', 'w'))
""",
    "photo-rename": """import os
jpgs = [f for f in os.listdir('photos') if f.endswith('.jpg')]
jpgs.sort(key=lambda f: os.path.getmtime(os.path.join('photos', f)))
for i, f in enumerate(jpgs, 1):
    os.rename(os.path.join('photos', f), os.path.join('photos', f'img_{i:03d}.jpg'))
""",
    "todo-report": """import os, sys
found = 0
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d != 'vendor']
    for f in files:
        if f.endswith('.py') and f != 'check_runner.py' and f != 'solution.py':
            p = os.path.join(root, f)[2:]
            for i, line in enumerate(open(os.path.join(root, f)), 1):
                if 'TODO' in line:
                    print(f'{p}:{i}: {line.strip()}')
                    found += 1
sys.exit(1 if found else 0)
""",
    "retry-wrapper": """import time
def fetch_with_retry(fn):
    for attempt in range(3):
        try:
            return fn()
        except ConnectionError:
            if attempt == 2:
                raise
            time.sleep(0.01 * (2 ** attempt))
""",
}

# one plausible MISREADING per task — the ambiguity must be real (checks must fail)
_AGENTIC_MISREAD = {
    "log-clean": """import os, time
n = 0
for f in os.listdir('logs'):
    p = os.path.join('logs', f)
    if f.endswith('.log') and time.time() - os.path.getmtime(p) > 7 * 86400:
        os.remove(p)   # deletes the newest too
        n += 1
print(n)
""",
    "json-migrate": """import json
recs = json.load(open('records.json'))
for r in recs:
    r['username'] = r.pop('user')
    r['version'] = 2
json.dump(recs, open('records.json', 'w'))   # no backup
""",
    "retry-wrapper": """def fetch_with_retry(fn):
    for attempt in range(3):
        try:
            return fn()
        except Exception:   # retries ValueError too
            if attempt == 2:
                raise
""",
}


@unittest.skipUnless(_OK, "skill scripts not importable")
class TestAgenticTier(unittest.TestCase):
    def test_schema(self):
        for t in outcome_bank.AGENTIC:
            self.assertEqual(t["kind"], "script", t["id"])
            for key in ("id", "ambiguous_prompt", "hidden_spec", "checks", "ambiguity"):
                self.assertIn(key, t, t["id"])
            self.assertGreaterEqual(len(t["ambiguity"]), 2, t["id"])
            self.assertGreaterEqual(len(t["checks"]), 2, t["id"])
        self.assertEqual(len(outcome_bank.AGENTIC), len(_AGENTIC_REF))

    def test_reference_solutions_pass_all_checks(self):
        for t in outcome_bank.AGENTIC:
            frac, per = outcome_eval.run_script_task(t, _AGENTIC_REF[t["id"]])
            self.assertEqual(frac, 1.0, f"{t['id']}: {per}")

    def test_misreadings_fail_at_least_one_check(self):
        for tid, code in _AGENTIC_MISREAD.items():
            t = outcome_bank.BY_ID[tid]
            frac, per = outcome_eval.run_script_task(t, code)
            self.assertLess(frac, 1.0, f"{tid}: misreading passed everything — ambiguity not real")

    def test_fixture_mtimes_and_sandbox_isolation(self):
        t = outcome_bank.BY_ID["log-clean"]
        code = ("import os, time\n"
                "print(int((time.time() - os.path.getmtime('logs/old1.log')) / 86400))")
        frac, per = outcome_eval.run_script_task(dict(t, checks=["stdout.strip() == '30'"]), code)
        self.assertEqual(frac, 1.0)

    def test_run_cell_dispatch_and_behavior_arms(self):
        task = outcome_bank.BY_ID["json-migrate"]
        fake = {"bucket": [{"question": "q0", "value": 0.9}], "derived": [], "usage": {}}
        with mock.patch.object(outcome_eval.infogain, "run", return_value=fake) as runm, \
             mock.patch.object(outcome_eval, "simulate_user",
                               side_effect=lambda s, q, m: {"question": q, "answer": "A",
                                                            "revealed": True}), \
             mock.patch.object(outcome_eval, "solve_and_score",
                               return_value={"code": "", "frac": 1.0, "per_test": []}):
            outcome_eval.run_cell(task, "nbq-behavior", 2, {"skill": "m", "solver": "m",
                                                            "sim": "m"})
            self.assertEqual(runm.call_args[0][1].get("value_judge_mode"), "behavior")
            outcome_eval.run_cell(task, "nbq-derive-behavior", 2, {"skill": "m", "solver": "m",
                                                                   "sim": "m"})
            cfg = runm.call_args[0][1]
            self.assertEqual(cfg.get("value_judge_mode"), "behavior")
            self.assertEqual(cfg.get("auto_derive"), "on")


@unittest.skipUnless(_OK, "skill scripts not importable")
class TestAnalysis(unittest.TestCase):
    def test_paired_deltas_and_p6_anchor(self):
        rows = []
        for i in range(6):
            base = 0.4
            rows.append({"task": f"t{i}", "arm": "baseline", "k": 3, "questions": [],
                         "qa": [], "revealed": 0, "unanswerable": 0, "frac": base,
                         "per_test": [], "code": "", "meta": {}, "elapsed_s": 1})
            rows.append({"task": f"t{i}", "arm": "nbq", "k": 3, "questions": ["q"],
                         "qa": [{"revealed": True}], "revealed": 1, "unanswerable": 0,
                         "frac": base + 0.1 * i, "per_test": [], "code": "",
                         "meta": {"q_values": [0.1 * i]}, "elapsed_s": 1})
        stats = outcome_eval.analyze(rows)
        arm = stats["arms"]["nbq"]
        self.assertEqual(arm["n"], 6)
        self.assertEqual(arm["losses"], 0)
        self.assertEqual(arm["wins"], 5)                  # t0 is a tie
        # q_value rises exactly with delta -> perfect P6 anchor
        self.assertEqual(stats["p6_qvalue_vs_delta_rho"], 1.0)

    def test_sign_test_exact(self):
        self.assertAlmostEqual(outcome_eval._sign_test_p(5, 0), 0.0625, places=4)
        self.assertEqual(outcome_eval._sign_test_p(0, 0), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
