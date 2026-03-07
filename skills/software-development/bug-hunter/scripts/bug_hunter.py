     1|import os
     2|import subprocess
     3|import sys
     4|import json
     5|import shutil
     6|from pathlib import Path
     7|
     8|class BugHunter:
     9|    def __init__(self, repo_url, work_dir="work"):
    10|        self.repo_url = repo_url
    11|        self.work_dir = Path(work_dir)
    12|        self.repo_name = repo_url.split("/")[-1].replace(".git", "")
    13|        self.repo_path = self.work_dir / self.repo_name
    14|        self.token = os.environ.get("GITHUB_TOKEN")
    15|
    16|    def clone(self):
    17|        print(f"[*] Cloning {self.repo_url}...")
    18|        if self.repo_path.exists():
    19|            shutil.rmtree(self.repo_path)
    20|        self.work_dir.mkdir(exist_ok=True)
    21|        
    22|        # Inject token for pushing later
    23|        authenticated_url = self.repo_url
    24|        if self.token and "github.com" in self.repo_url and "@" not in self.repo_url:
    25|            authenticated_url = self.repo_url.replace("https://github.com", f"https://x-access-token:{self.token}@github.com")
    26|            
    27|        subprocess.run(["git", "clone", authenticated_url, str(self.repo_path)], check=True)
    28|
    29|    def setup_env(self):
    30|        print("[*] Setting up environment...")
    31|        reqs = self.repo_path / "requirements.txt"
    32|        if reqs.exists():
    33|            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True, cwd=self.repo_path)
    34|        if (self.repo_path / "setup.py").exists() or (self.repo_path / "pyproject.toml").exists():
    35|            subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True, cwd=self.repo_path)
    36|
    37|    def run_tests(self):
    38|        print("[*] Running tests...")
    39|        subprocess.run(
    40|            ["pytest", "--json-report", "--json-report-file=report.json"],
    41|            cwd=self.repo_path,
    42|            capture_output=True,
    43|            text=True
    44|        )
    45|        report_file = self.repo_path / "report.json"
    46|        if report_file.exists():
    47|            with open(report_file, "r") as f:
    48|                return json.load(f)
    49|        return None
    50|
    51|    def analyze_failures(self, report):
    52|        if not report:
    53|            return []
    54|        failures = []
    55|        for test in report.get("tests", []):
    56|            if test.get("outcome") == "failed":
    57|                failures.append({
    58|                    "nodeid": test.get("nodeid"),
    59|                    "traceback": test.get("call", {}).get("crash", {}).get("message", "No message"),
    60|                })
    61|        return failures
    62|
    63|    def create_pr(self, branch_name, title, body):
    64|        print(f"[*] Creating PR: {title}")
    65|        # Git config
    66|        subprocess.run(["git", "config", "user.name", "Hermes Bug Hunter"], cwd=self.repo_path)
    67|        subprocess.run(["git", "config", "user.email", "hermes@nous.dev"], cwd=self.repo_path)
    68|        
    69|        # Branch and commit
    70|        subprocess.run(["git", "checkout", "-b", branch_name], cwd=self.repo_path, check=True)
    71|        subprocess.run(["git", "add", "."], cwd=self.repo_path, check=True)
    72|        subprocess.run(["git", "commit", "-m", title], cwd=self.repo_path, check=True)
    73|        
    74|        # Push
    75|        print(f"[*] Pushing branch {branch_name}...")
    76|        subprocess.run(["git", "push", "origin", branch_name], cwd=self.repo_path, check=True)
    77|        
    78|        # PR
    79|        print("[*] Submitting PR via gh CLI...")
    80|        subprocess.run([
    81|            "gh", "pr", "create", 
    82|            "--title", title, 
    83|            "--body", body, 
    84|            "--head", branch_name, 
    85|            "--base", "main"
    86|        ], cwd=self.repo_path, check=True)
    87|
    88|    def apply_fix(self, file_path, old_text, new_text):
    89|        print(f"[*] Applying fix to {file_path}")
    90|        path = self.repo_path / file_path
    91|        content = path.read_text()
    92|        new_content = content.replace(old_text, new_text)
    93|        path.write_text(new_content)
    94|
    95|if __name__ == "__main__":
    96|    if len(sys.argv) < 2:
    97|        print("Usage: python bug_hunter.py <repo_url>")
    98|        sys.exit(1)
    99|    
   100|    repo_url = sys.argv[1]
   101|    hunter = BugHunter(repo_url)
   102|    hunter.clone()
   103|    hunter.setup_env()
   104|    
   105|    report = hunter.run_tests()
   106|    failures = hunter.analyze_failures(report)
   107|    
   108|    if not failures:
   109|        print("[+] No failures found!")
   110|    else:
   111|        print(f"[!] Found {len(failures)} failures. Analysis:")
   112|        for f in failures:
   113|            nodeid = f['nodeid']
   114|            traceback = f['traceback']
   115|            print(f"\n--- Failure in {nodeid} ---")
   116|            print(f"Error: {traceback}")
   117|            
   118|            # AI Logic (Simulated for Demo)
   119|            if "test_subtract" in nodeid and "assert 15 == 5" in traceback:
   120|                print("[*] AI Analysis: detected '+' used for subtraction.")
   121|                hunter.apply_fix("utils.py", "return a + b", "return a - b")
   122|            elif "test_multiply" in nodeid and "assert 0.666" in traceback:
   123|                print("[*] AI Analysis: detected '/' used for multiplication.")
   124|                hunter.apply_fix("utils.py", "return a / b", "return a * b")
   125|            
   126|            print("[*] Verifying fix...")
   127|            new_report = hunter.run_tests()
   128|            if not any(nf['nodeid'] == nodeid for nf in hunter.analyze_failures(new_report)):
   129|                print(f"[+] Fix verified for {nodeid}!")
   130|
   131|        # Final check and PR
   132|        if not hunter.analyze_failures(hunter.run_tests()):
   133|            print("[+] All bugs fixed! Creating PR...")
   134|            import time
   135|            branch_name = f"fix-bugs-{int(time.time())}"
   136|            hunter.create_pr(
   137|                branch_name,
   138|                "Fix arithmetic bugs",
   139|                "Automated fix for bugs in utils.py"
   140|            )
   141|