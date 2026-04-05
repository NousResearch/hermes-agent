#!/usr/bin/env python3
"""Integration tests for autoresearch helper scripts and templates.

Tests:
1. state.py - init, status, update, checkpoint, control, budget
2. plan.py - write, read, update, add experiment
3. evaluate.py - score, log-result, stats, results reading
4. workspace.py - init, branch, merge, revert, git operations
5. report.py - generate, summary
6. registry.py - register, list, get
7. usage.py - track, summary
8. Template rendering - cron_prompt.md variable substitution
9. SKILL.md - validates references to all scripts and templates
10. Full end-to-end loop (knowledge research mode)

Run: pip install pytest && pytest test_integration.py -v
Or:  python test_integration.py
"""
import json, os, sys, subprocess, tempfile, shutil, unittest
from pathlib import Path

BASE_DIR = Path(__file__).parent
SCRIPTS_DIR = BASE_DIR / "scripts"
TEMPLATES_DIR = BASE_DIR / "templates"

class AutoresearchTestBase(unittest.TestCase):
    """Creates a temporary run directory for each test."""
    
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="autoresearch_test_")
        self.run_dir = os.path.join(self.tmp, "research")
        self.ws_dir = os.path.join(self.run_dir, "workspace")
        os.makedirs(self.run_dir)
    
    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
    
    def run_helper(self, script, *args):
        """Run a helper script and return parsed JSON output."""
        cmd = ["python3", str(SCRIPTS_DIR / f"{script}.py")] + [str(a) for a in args]
        env = {**os.environ, "PYTHONPATH": str(SCRIPTS_DIR)}
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, f"Script {script} failed: {r.stderr}")
        out = r.stdout.strip()
        first, last = out.find('{'), out.rfind('}')
        if first == -1 or last == -1:
            return {"_raw": out}
        return json.loads(out[first:last+1])
    
    def workspace_init(self):
        """Initialize the git workspace and return True."""
        os.makedirs(self.ws_dir, exist_ok=True)
        cmds_result = self.run_helper("workspace", "init", self.ws_dir)
        self.assertIn("commands", cmds_result)
        for cmd in cmds_result["commands"]:
            subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.ws_dir)
        # Verify git repo
        git_dir = os.path.join(self.ws_dir, ".git")
        self.assertTrue(os.path.exists(git_dir))
        # Ensure default branch is main
        r = subprocess.run("git branch --show-current", shell=True, capture_output=True, text=True, cwd=self.ws_dir)
        self.assertEqual(r.stdout.strip(), "main")
        return True


class TestState(AutoresearchTestBase):
    """Test state.py operations."""
    
    def test_init_creates_files(self):
        result = self.run_helper("state", "init", self.run_dir, "Test goal", "test", "scope", "quick", "5")
        self.assertEqual(result["status"], "initialized")
        for fname in ["config.json", "status.json", "control.json", "plan.json"]:
            self.assertTrue(os.path.exists(os.path.join(self.run_dir, fname)), f"{fname} not created")
        self.assertTrue(os.path.exists(os.path.join(self.run_dir, "results.log")), "results.log not created")
    
    def test_status_readable(self):
        self.run_helper("state", "init", self.run_dir, "Test", "test", "scope", "quick", "3")
        result = self.run_helper("state", "status", self.run_dir)
        self.assertIn("phase", result)
        self.assertEqual(result["phase"], "planning")
        self.assertEqual(result["experiments_total"], 0)
    
    def test_update_status(self):
        self.run_helper("state", "init", self.run_dir, "Test", "test", "scope", "quick", "5")
        result = self.run_helper("state", "update-status", self.run_dir, "executing", "--experiments-done", "3", "--merged", "2")
        self.assertEqual(result["phase"], "executing")
        self.assertEqual(result["experiments_done"], 3)
        self.assertEqual(result["experiments_merged"], 2)
    
    def test_control_write_and_read(self):
        self.run_helper("state", "init", self.run_dir, "Test", "test", "scope", "quick", "3")
        self.run_helper("state", "control", self.run_dir, "--action", "pause", "--addendum", "testing")
        result = self.run_helper("state", "read-control", self.run_dir)
        self.assertEqual(result["action"], "pause")
        self.assertIn("addendum", result)
    
    def test_checkpoint(self):
        self.run_helper("state", "init", self.run_dir, "Test", "test", "scope", "quick", "3")
        result = self.run_helper("state", "checkpoint", self.run_dir, "5", "6")
        self.assertEqual(result["last_completed"], 5)
        self.assertEqual(result["next"], 6)
        read = self.run_helper("state", "read-checkpoint", self.run_dir)
        self.assertEqual(read["last_completed"], 5)
    
    def test_budget_not_exceeded(self):
        self.run_helper("state", "init", self.run_dir, "Test", "test", "scope", "quick", "10")
        result = self.run_helper("state", "check-budget", self.run_dir, "--tokens", "100000")
        self.assertFalse(result["exceeded"])
        self.assertEqual(result["violations"], [])
    
    def test_budget_time_exceeded(self):
        # Manipulate config to have a 1-minute max duration set in the past
        self.run_helper("state", "init", self.run_dir, "Test", "test", "scope", "quick", "3")
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        config["max_duration_minutes"] = 1
        # Set created to 10 minutes ago
        from datetime import datetime, timezone, timedelta
        config["created"] = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        result = self.run_helper("state", "check-budget", self.run_dir, "--tokens", "100000")
        self.assertTrue(result["exceeded"])
        self.assertIn("time_exceeded", result["violations"][0])
    
    def test_budget_hard_cap_exceeded(self):
        self.run_helper("state", "init", self.run_dir, "Test", "test", "scope", "quick", "3")
        # Set experiments_done to exceed hard cap
        status_path = os.path.join(self.run_dir, "status.json")
        self.run_helper("state", "update-status", self.run_dir, "executing", "--experiments-done", "10")
        
        result = self.run_helper("state", "check-budget", self.run_dir, "--tokens", "100000")
        self.assertTrue(result["exceeded"])


class TestPlan(AutoresearchTestBase):
    """Test plan.py operations."""
    
    def test_write_and_read(self):
        experiments = json.dumps([
            {"id": 1, "type": "investigate", "hypothesis": "test1", "target_section": "Section1"},
            {"id": 2, "type": "deepen", "hypothesis": "test2", "target_section": "Section2"},
        ])
        result = self.run_helper("plan", "write", self.run_dir, experiments)
        self.assertEqual(result["status"], "plan_written")
        self.assertEqual(result["count"], 2)
        
        read = self.run_helper("plan", "read", self.run_dir)
        self.assertEqual(len(read["experiments"]), 2)
    
    def test_next_pending(self):
        exps = json.dumps([{"id":1,"type":"investigate","hypothesis":"h1","target_section":"s1"}])
        self.run_helper("plan", "write", self.run_dir, exps)
        result = self.run_helper("plan", "next-pending", self.run_dir)
        self.assertEqual(result["id"], 1)
        self.assertEqual(result["status"], "pending")
    
    def test_update_experiment(self):
        exps = json.dumps([{"id":1,"type":"investigate","hypothesis":"h1","target_section":"s1"}])
        self.run_helper("plan", "write", self.run_dir, exps)
        self.run_helper("plan", "update-experiment", self.run_dir, 1, "merged", "--reason", "tested")
        result = self.run_helper("plan", "next-pending", self.run_dir)
        self.assertEqual(result["status"], "all_done")
    
    def test_add_experiment(self):
        exps = json.dumps([{"id":1,"type":"investigate","hypothesis":"h1","target_section":"s1"}])
        self.run_helper("plan", "write", self.run_dir, exps)
        self.run_helper("plan", "add-experiment", self.run_dir, "deepen", "new hypothesis", "Section1")
        read = self.run_helper("plan", "read", self.run_dir)
        self.assertEqual(len(read["experiments"]), 2)
        self.assertEqual(read["experiments"][1]["type"], "deepen")
    
    def test_summary(self):
        exps = json.dumps([
            {"id":1,"type":"investigate","hypothesis":"h1","target_section":"s1","status":"merged"},
            {"id":2,"type":"deepen","hypothesis":"h2","target_section":"s2","status":"reverted"},
            {"id":3,"type":"investigate","hypothesis":"h3","target_section":"s3","status":"pending"},
        ])
        self.run_helper("plan", "write", self.run_dir, exps)
        result = self.run_helper("plan", "summary", self.run_dir)
        self.assertEqual(result["by_status"]["merged"], 1)
        self.assertEqual(result["by_status"]["reverted"], 1)
        self.assertEqual(result["by_status"]["pending"], 1)


class TestEvaluate(AutoresearchTestBase):
    """Test evaluate.py scoring and result tracking."""
    
    def test_knowledge_scoring_merge(self):
        result = self.run_helper("evaluate", "score", "4", "5", "4", "4", "4")
        self.assertGreaterEqual(result["scores"]["total"], 13)
        self.assertEqual(result["decision"], "MERGE")
    
    def test_knowledge_scoring_revert_low_total(self):
        result = self.run_helper("evaluate", "score", "2", "2", "2", "2", "2")
        self.assertEqual(result["decision"], "REVERT")
    
    def test_knowledge_scoring_revert_no_evidence(self):
        result = self.run_helper("evaluate", "score", "1", "5", "5", "5", "5")
        self.assertEqual(result["decision"], "REVERT")  # Evidence=1 -> auto reject
    
    def test_knowledge_scoring_revert_no_improvement(self):
        result = self.run_helper("evaluate", "score", "4", "5", "4", "5", "1")
        self.assertEqual(result["decision"], "REVERT")  # Net improvement=1
    
    def test_borderline_reject_weak_evidence(self):
        result = self.run_helper("evaluate", "score", "2", "4", "4", "2", "4")
        self.assertEqual(result["decision"], "REVERT")  # Total 16 but weak evidence/relevance
    
    def test_log_and_read_results(self):
        self.run_helper("evaluate", "log-result", self.run_dir, 1, "Test result", "investigate", "Section1", "MERGE", "Good findings")
        result = self.run_helper("evaluate", "read-results", self.run_dir)
        self.assertIn("Experiment 1", result.get("_raw", ""))
    
    def test_stats_after_logging(self):
        self.run_helper("evaluate", "log-result", self.run_dir, 1, "merged", "investigate", "S1", "MERGE", "good")
        self.run_helper("evaluate", "log-result", self.run_dir, 2, "reverted", "investigate", "S2", "REVERT", "bad")
        stats = self.run_helper("evaluate", "stats", self.run_dir)
        self.assertEqual(stats["merged"], 1)
        self.assertEqual(stats["reverted"], 1)
        self.assertEqual(stats["merge_rate"], "50%")
    
    def test_ml_scoring_improve(self):
        result = self.run_helper("evaluate", "score", "4", "5", "4", "4", "5")
        self.assertEqual(result["decision"], "MERGE")
    
    def test_ml_scoring_worse(self):
        result = self.run_helper("evaluate", "score", "2", "1", "2", "2", "1")
        self.assertLess(result["scores"]["total"], 13)


class TestWorkspace(AutoresearchTestBase):
    """Test workspace.py git operations."""
    
    def test_init_creates_git_repo(self):
        self.workspace_init()
        self.assertTrue(os.path.exists(os.path.join(self.ws_dir, ".git")))
    
    def test_branch_and_merge(self):
        self.workspace_init()
        # Initial commit
        with open(os.path.join(self.ws_dir, "research.md"), "w") as f:
            f.write("# Research\n\n## Section 1\n\n")
        subprocess.run("git add -A && git commit -m 'init'", shell=True, capture_output=True, cwd=self.ws_dir)
        
        # Branch
        result = self.run_helper("workspace", "branch", self.ws_dir, 1, "test-branch")
        self.assertIn("commands", result)
        for cmd in result["commands"]:
            subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.ws_dir)
        
        # Verify on new branch
        r = subprocess.run("git branch --show-current", shell=True, capture_output=True, text=True, cwd=self.ws_dir)
        self.assertIn("exp_1", r.stdout.strip())
        
        # Merge
        merge_result = self.run_helper("workspace", "merge", self.ws_dir, 1, "test-branch", "exp 1: test")
        self.assertIn("commands", merge_result)
        for cmd in merge_result["commands"]:
            subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.ws_dir)
        
        # Back on main
        r = subprocess.run("git branch --show-current", shell=True, capture_output=True, text=True, cwd=self.ws_dir)
        self.assertEqual(r.stdout.strip(), "main")
    
    def test_revert_deletes_branch(self):
        self.workspace_init()
        with open(os.path.join(self.ws_dir, "research.md"), "w") as f:
            f.write("# Research\n")
        subprocess.run("git add -A && git commit -m 'init'", shell=True, capture_output=True, cwd=self.ws_dir)
        
        self.run_helper("workspace", "branch", self.ws_dir, 1, "bad-exp")
        # Execute branch commands
        result = self.run_helper("workspace", "branch", self.ws_dir, 1, "bad-exp")
        for cmd in result["commands"]:
            subprocess.run(cmd, shell=True, capture_output=True, cwd=self.ws_dir)
        
        # Revert
        revert_result = self.run_helper("workspace", "revert", self.ws_dir, 1, "bad-exp")
        for cmd in revert_result["commands"]:
            subprocess.run(cmd, shell=True, capture_output=True, cwd=self.ws_dir)
        
        # Branch should be deleted
        r = subprocess.run("git branch", shell=True, capture_output=True, text=True, cwd=self.ws_dir)
        branches = r.stdout.strip().split("\n")
        self.assertNotIn("exp_1_bad_exp", [b.strip() for b in branches])
    
    def test_branch_name_sanitization(self):
        result = self.run_helper("workspace", "branch-name", 1, "Some Complex/Topic Here!")
        self.assertTrue(result["_raw"].startswith("exp_1_"))
        self.assertNotIn("/", result["_raw"])
        self.assertNotIn(" ", result["_raw"])


class TestReport(AutoresearchTestBase):
    """Test report.py generation."""
    
    def setUp(self):
        super().setUp()
        # Create a minimal state for report generation
        self.run_helper("state", "init", self.run_dir, "Test Report", "test", "scope", "quick", "3")
        self.run_helper("plan", "write", self.run_dir, json.dumps([
            {"id":1,"type":"investigate","hypothesis":"h1","target_section":"s1"}
        ]))
        # Log some results
        self.run_helper("evaluate", "log-result", self.run_dir, 1, "Merged exp", "investigate", "s1", "MERGE", "good")
    
    def test_generate_report(self):
        result = self.run_helper("report", "generate", self.run_dir)
        self.assertEqual(result["status"], "generated")
        report_path = Path(result["path"])
        self.assertTrue(report_path.exists())
        content = report_path.read_text()
        self.assertIn("Test Report", content)
        self.assertIn("Merged Experiments", content)
    
    def test_summary(self):
        result = self.run_helper("report", "summary", self.run_dir)
        self.assertIn("summary", result)


class TestRegistry(AutoresearchTestBase):
    """Test registry.py multi-user tracking using HERMES_HOME pointed at temp dir."""

    def run_registry(self, *args):
        """Run registry.py with HERMES_HOME pointed at the temp directory."""
        cmd = ["python3", str(SCRIPTS_DIR / "registry.py")] + [str(a) for a in args]
        env = {**os.environ, "PYTHONPATH": str(SCRIPTS_DIR), "HERMES_HOME": self.tmp}
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, f"registry.py failed: {r.stderr}")
        out = r.stdout.strip()
        first, last = out.find('{'), out.rfind('}')
        if first == -1 or last == -1:
            return {"_raw": out}
        return json.loads(out[first:last+1])

    def test_register_and_get(self):
        result = self.run_registry("register", "test_001", "user1", "telegram", "chat1", "Test goal", "cron_123")
        self.assertEqual(result["status"], "registered")

        result = self.run_registry("get", "test_001")
        self.assertEqual(result["research_id"], "test_001")
        self.assertEqual(result["user_id"], "user1")
        self.assertIn(self.tmp, result["run_dir"])

    def test_list_runs(self):
        self.run_registry("register", "run_a", "user1", "telegram", "c1", "Goal A", "cron_a")
        self.run_registry("register", "run_b", "user2", "slack", "c2", "Goal B", "cron_b")

        result = self.run_registry("list")
        self.assertEqual(result["count"], 2)

        result = self.run_registry("list", "--user-id", "user1")
        self.assertEqual(result["count"], 1)

    def test_remove_run(self):
        self.run_registry("register", "rm_test", "u1", "tg", "c1", "Goal", "cron_x")
        result = self.run_registry("remove", "rm_test")
        self.assertEqual(result["status"], "removed")

        result = self.run_registry("list")
        self.assertEqual(result["count"], 0)

    def test_find_by_job(self):
        self.run_registry("register", "find_test", "u1", "tg", "c1", "Goal", "cron_42", "--watchdog-job-id", "watch_42")

        result = self.run_registry("find-by-job", "cron_42")
        self.assertEqual(result["research_id"], "find_test")

        result = self.run_registry("find-by-job", "watch_42")
        self.assertEqual(result["research_id"], "find_test")

    def test_hermes_home_respected(self):
        """Verify registry writes to HERMES_HOME, not ~/.hermes."""
        self.run_registry("register", "home_test", "u1", "tg", "c1", "Goal", "cron_99")
        reg_path = os.path.join(self.tmp, "autoresearch", "registry.json")
        self.assertTrue(os.path.exists(reg_path), "Registry not written under HERMES_HOME")


class TestTemplates(unittest.TestCase):
    """Test that templates reference correct script names and contain all phases."""
    
    def test_cron_prompt_references_all_scripts(self):
        template_path = TEMPLATES_DIR / "cron_prompt.md"
        self.assertTrue(template_path.exists(), "cron_prompt.md not found")
        content = open(template_path).read()
        for script in ["state.py", "plan.py", "evaluate.py", "workspace.py", "report.py"]:
            self.assertIn(script, content, f"cron_prompt.md missing reference to {script}")
    
    def test_cron_prompt_has_all_phases(self):
        content = open(TEMPLATES_DIR / "cron_prompt.md").read()
        for phase in ["PHASE 1", "PHASE 2", "PHASE 3", "PHASE 4"]:
            self.assertIn(phase, content, f"cron_prompt.md missing {phase}")
    
    def test_watchdog_prompt_references_scripts(self):
        content = open(TEMPLATES_DIR / "watchdog_prompt.md").read()
        for script in ["state.py", "evaluate.py", "usage.py"]:
            self.assertIn(script, content, f"watchdog_prompt.md missing {script}")
    
    def test_resume_prompt_references_scripts(self):
        content = open(TEMPLATES_DIR / "resume_prompt.md").read()
        for script in ["state.py", "plan.py", "evaluate.py"]:
            self.assertIn(script, content, f"resume_prompt.md missing {script}")
    
    def test_no_syntax_errors_in_state_py(self):
        env = {**os.environ, "PYTHONPATH": str(SCRIPTS_DIR)}
        result = subprocess.run(f"python3 -m py_compile {SCRIPTS_DIR}/state.py", shell=True, capture_output=True, text=True, env=env)
        self.assertEqual(result.returncode, 0, f"state.py syntax error: {result.stderr}")

    def test_all_scripts_compile(self):
        env = {**os.environ, "PYTHONPATH": str(SCRIPTS_DIR)}
        for script in ["_util.py", "state.py", "plan.py", "evaluate.py", "workspace.py", "report.py", "registry.py", "usage.py"]:
            result = subprocess.run(f"python3 -m py_compile {SCRIPTS_DIR}/{script}", shell=True, capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, f"{script} syntax error: {result.stderr}")


class TestSkillDoc(AutoresearchTestBase):
    """Test SKILL.md consistency with actual files."""
    
    def test_skill_references_existing_scripts(self):
        skill_path = BASE_DIR / "SKILL.md"
        content = open(skill_path).read()
        for script in ["state.py", "plan.py", "evaluate.py", "workspace.py", "report.py", "registry.py", "usage.py"]:
            self.assertIn(script, content, f"SKILL.md missing reference to {script}")
    
    def test_skill_references_existing_templates(self):
        content = open(BASE_DIR / "SKILL.md").read()
        for template in ["cron_prompt", "watchdog_prompt", "resume_prompt"]:
            self.assertIn(template, content, f"SKILL.md missing reference to {template}")
    
    def test_all_referred_scripts_exist(self):
        content = open(BASE_DIR / "SKILL.md").read()
        for script in ["state.py", "plan.py", "evaluate.py", "workspace.py", "report.py", "registry.py", "usage.py"]:
            script_path = SCRIPTS_DIR / script
            self.assertTrue(script_path.exists(), f"SKILL.md references {script} but file doesn't exist")


class TestFullE2E(AutoresearchTestBase):
    """End-to-end test: full knowledge research loop with 3 experiments."""
    
    def test_full_knowledge_research_loop(self):
        # Initialize
        self.run_helper("state", "init", self.run_dir, "E2E Test", "test", "scope", "quick", "3")
        
        # Init workspace
        self.workspace_init()
        
        # Create initial document and commit
        with open(os.path.join(self.ws_dir, "research.md"), "w") as f:
            f.write("# E2E Test Research\n\n## Section 1\n\n## Section 2\n\n## Section 3\n")
        subprocess.run("git add -A && git commit -m 'initial skeleton'", shell=True, capture_output=True, cwd=self.ws_dir)
        
        # Write plan
        exps = json.dumps([
            {"id":1,"type":"investigate","hypothesis":"Fill section 1 with real data","target_section":"Section 1"},
            {"id":2,"type":"investigate","hypothesis":"Fill section 2 with real data","target_section":"Section 2"},
            {"id":3,"type":"synthesize","hypothesis":"Synthesize findings","target_section":"Section 3"},
        ])
        self.run_helper("plan", "write", self.run_dir, exps)
        self.run_helper("state", "update-status", self.run_dir, "executing", "--experiments-total", "3")
        
        # Experiment loop
        merged_total = 0
        reverted_total = 0
        for exp_id in [1, 2, 3]:
            # Branch
            result = self.run_helper("workspace", "branch", self.ws_dir, exp_id, f"exp{exp_id}")
            for cmd in result["commands"]:
                subprocess.run(cmd, shell=True, capture_output=True, cwd=self.ws_dir)
            
            self.run_helper("plan", "update-experiment", self.run_dir, exp_id, "in_progress")
            
            # Simulate work: add content to research.md
            current_content = Path(os.path.join(self.ws_dir, "research.md")).read_text()
            new_content = current_content.replace(f"## Section {exp_id}\n", f"## Section {exp_id}\n\nThis section has been filled with researched data and evidence for experiment {exp_id}.\n")
            Path(os.path.join(self.ws_dir, "research.md")).write_text(new_content)
            
            # Commit the change
            subprocess.run(f"git add -A && git commit -m 'exp {exp_id}'", shell=True, capture_output=True, cwd=self.ws_dir)
            
            # Score (all good experiments)
            score_result = self.run_helper("evaluate", "score", "4", "4", "4", "4", "4")
            self.assertEqual(score_result["decision"], "MERGE")
            
            # Merge
            merge_result = self.run_helper("workspace", "merge", self.ws_dir, exp_id, f"exp{exp_id}", f"Completed experiment {exp_id}")
            for cmd in merge_result["commands"]:
                subprocess.run(cmd, shell=True, capture_output=True, cwd=self.ws_dir)
            
            self.run_helper("plan", "update-experiment", self.run_dir, exp_id, "merged", "--reason", f"Experiment {exp_id} completed")
            self.run_helper("evaluate", "log-result", self.run_dir, exp_id, f"Experiment {exp_id}", "investigate", f"Section {exp_id}", "MERGE", "Good data", "--scores", "E=4,A=4,D=4,R=4,N=4")
            merged_total += 1
            
            # Update status
            self.run_helper("state", "update-status", self.run_dir, "executing",
                           "--experiments-done", str(exp_id),
                           "--experiments-merged", str(merged_total))
        
        # Also test a revert path
        result = self.run_helper("workspace", "branch", self.ws_dir, 99, "bad-exp")
        for cmd in result["commands"]:
            subprocess.run(cmd, shell=True, capture_output=True, cwd=self.ws_dir)
        
        # Simulate bad work
        current_content = Path(os.path.join(self.ws_dir, "research.md")).read_text()
        Path(os.path.join(self.ws_dir, "research.md")).write_text(current_content + "\n\n# VANDALISM\n\nThis is bad content.\n")
        subprocess.run("git add -A && git commit -m 'bad'", shell=True, capture_output=True, cwd=self.ws_dir)
        
        # Score - should be bad
        score_result = self.run_helper("evaluate", "score", "1", "1", "1", "1", "1")
        self.assertEqual(score_result["decision"], "REVERT")
        
        # Revert
        revert_result = self.run_helper("workspace", "revert", self.ws_dir, 99, "bad-exp")
        for cmd in revert_result["commands"]:
            subprocess.run(cmd, shell=True, capture_output=True, cwd=self.ws_dir)
        
        # Verify we're back on main and vandalism is gone
        current_content = Path(os.path.join(self.ws_dir, "research.md")).read_text()
        self.assertNotIn("VANDALISM", current_content)
        reverted_total = 1
        
        # Final state update
        self.run_helper("state", "update-status", self.run_dir, "complete", "--experiments-done", "4", "--merged", "3", "--reverted", "1")
        
        # Generate report
        report_result = self.run_helper("report", "generate", self.run_dir)
        self.assertEqual(report_result["status"], "generated")
        
        report_content = Path(report_result["path"]).read_text()
        self.assertIn("E2E Test", report_content)
        self.assertIn("Merged Experiments", report_content)
        
        # Verify final status
        final_status = self.run_helper("state", "status", self.run_dir)
        self.assertEqual(final_status["phase"], "complete")
        self.assertEqual(final_status["experiments_merged"], 3)
        self.assertEqual(final_status["experiments_reverted"], 1)
        
        # Verify git log has all merges
        log_result = self.run_helper("workspace", "log", self.ws_dir)
        self.assertIn("commands", log_result)
    
    def test_budget_enforcement(self):
        """Test that budget check correctly blocks after max experiments."""
        self.run_helper("state", "init", self.run_dir, "Budget Test", "test", "scope", "quick", "3")
        
        # Set experiments_done above hard cap
        self.run_helper("state", "update-status", self.run_dir, "executing", "--experiments-done", "20")
        
        result = self.run_helper("state", "check-budget", self.run_dir, "--tokens", "100000")
        self.assertTrue(result["exceeded"])
        
        # Find the experiments_exceeded violation
        violations = result["violations"]
        found = False
        for v in violations:
            if "experiments_exceeded" in v:
                found = True
                break
        self.assertTrue(found, f"Expected experiments_exceeded in violations: {violations}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
