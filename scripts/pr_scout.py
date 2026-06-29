#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Adjust Python path to load hermes modules if running directly
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid UnicodeEncodeError on Windows CP1252 consoles when printing emojis
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from hermes_constants import get_hermes_home
except ImportError:
    def get_hermes_home():
        val = os.environ.get("HERMES_HOME", "").strip()
        if val:
            return Path(val)
        if sys.platform == "win32":
            local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
            base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
            return base / "hermes"
        return Path.home() / ".hermes"

def run_gh_cmd(args):
    """Run a gh CLI command and return parsed JSON or empty dict/list."""
    try:
        proc = subprocess.run(
            ["gh", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True
        )
        return json.loads(proc.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running 'gh {' '.join(args)}': {e.stderr}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Unexpected error running 'gh {' '.join(args)}': {e}", file=sys.stderr)
        return None

def main():
    repo = "NousResearch/hermes-agent"
    author = "aniruddhaadak80"
    
    # 1. Check if gh is authenticated
    try:
        subprocess.run(["gh", "auth", "status"], check=True, capture_output=True)
    except Exception:
        print("ERROR: gh CLI is not authenticated or not installed. Please authenticate using 'gh auth login'.")
        sys.exit(1)

    # 2. Get list of open PRs
    prs = run_gh_cmd([
        "pr", "list", 
        "--author", author, 
        "-R", repo, 
        "--json", "number,title,state,headRefName"
    ])
    if not prs:
        print("No open PRs found or failed to list PRs.")
        sys.exit(0)

    # 3. Load state
    state_file = get_hermes_home() / "pr_scout_state.json"
    state = {}
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load state file: {e}", file=sys.stderr)

    new_activity_detected = False
    markdown_report = []
    
    markdown_report.append("# Pull Request Scout Report")
    markdown_report.append(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for pr in prs:
        pr_num = str(pr["number"])
        title = pr["title"]
        head_branch = pr["headRefName"]
        
        # Initialize state entry for this PR if missing
        if pr_num not in state:
            state[pr_num] = {
                "reviews": [],
                "general_comments": [],
                "inline_comments": [],
                "mergeable": "UNKNOWN"
            }
            
        pr_state = state[pr_num]
        
        # Fetch detailed PR info
        pr_details = run_gh_cmd(["api", f"repos/{repo}/pulls/{pr_num}"])
        mergeable = "UNKNOWN"
        base_branch = "main"
        if pr_details:
            mergeable = pr_details.get("mergeable_state", "UNKNOWN")
            base_branch = pr_details.get("base", {}).get("ref", "main")
        
        # Fetch reviews
        reviews = run_gh_cmd(["api", f"repos/{repo}/pulls/{pr_num}/reviews"]) or []
        # Fetch general comments
        general_comments = run_gh_cmd(["api", f"repos/{repo}/issues/{pr_num}/comments"]) or []
        # Fetch inline comments
        inline_comments = run_gh_cmd(["api", f"repos/{repo}/pulls/{pr_num}/comments"]) or []
        
        pr_report = []
        pr_report.append(f"## PR #{pr_num}: {title}")
        pr_report.append(f"- **Head Branch**: `{head_branch}`")
        pr_report.append(f"- **Base Branch**: `{base_branch}`")
        pr_report.append(f"- **Mergeable State**: `{mergeable}`")
        pr_report.append(f"- **URL**: https://github.com/{repo}/pull/{pr_num}")
        
        pr_new_activity = False
        
        # Process Reviews
        review_lines = []
        for r in reviews:
            r_id = str(r["id"])
            r_user = r.get("user", {}).get("login", "unknown")
            r_state = r.get("state", "COMMENTED")
            r_body = r.get("body", "").strip()
            r_time = r.get("submitted_at", "")
            
            # Skip pending or author's own reviews
            if r_state == "PENDING" or r_user == author:
                continue
                
            is_new = r_id not in pr_state["reviews"]
            if is_new:
                pr_new_activity = True
                new_activity_detected = True
                pr_state["reviews"].append(r_id)
                tag = "🔴 [NEW REVIEW]"
            else:
                tag = "[REVIEW]"
                
            review_lines.append(f"  - **{tag} @{r_user}** ({r_state} at {r_time}):")
            if r_body:
                # Indent body lines
                indented_body = "\n".join(f"    {line}" for line in r_body.splitlines())
                review_lines.append(indented_body)
            else:
                review_lines.append("    *(No description body)*")
                
        # Process General Comments
        gen_comment_lines = []
        for c in general_comments:
            c_id = str(c["id"])
            c_user = c.get("user", {}).get("login", "unknown")
            c_body = c.get("body", "").strip()
            c_time = c.get("created_at", "")
            
            # Skip author's own comments
            if c_user == author:
                continue
                
            is_new = c_id not in pr_state["general_comments"]
            if is_new:
                pr_new_activity = True
                new_activity_detected = True
                pr_state["general_comments"].append(c_id)
                tag = "💬 [NEW COMMENT]"
            else:
                tag = "[COMMENT]"
                
            gen_comment_lines.append(f"  - **{tag} @{c_user}** ({c_time}):")
            if c_body:
                indented_body = "\n".join(f"    {line}" for line in c_body.splitlines())
                gen_comment_lines.append(indented_body)
                
        # Process Inline Comments
        inline_comment_lines = []
        for c in inline_comments:
            c_id = str(c["id"])
            c_user = c.get("user", {}).get("login", "unknown")
            c_path = c.get("path", "")
            c_line = c.get("line", c.get("original_line", "?"))
            c_body = c.get("body", "").strip()
            c_time = c.get("created_at", "")
            
            # Skip author's own comments
            if c_user == author:
                continue
                
            is_new = c_id not in pr_state["inline_comments"]
            if is_new:
                pr_new_activity = True
                new_activity_detected = True
                pr_state["inline_comments"].append(c_id)
                tag = "📝 [NEW INLINE COMMENT]"
            else:
                tag = "[INLINE COMMENT]"
                
            inline_comment_lines.append(f"  - **{tag} @{c_user}** on `{c_path}` line {c_line} ({c_time}):")
            if c_body:
                indented_body = "\n".join(f"    {line}" for line in c_body.splitlines())
                inline_comment_lines.append(indented_body)

        # Check for changes in mergeable state
        if mergeable != pr_state["mergeable"]:
            pr_state["mergeable"] = mergeable
            if mergeable == "conflicting":
                pr_new_activity = True
                new_activity_detected = True
                pr_report.append("⚠️ **[NEW] Merge conflict detected on this PR!**")
                
        if review_lines:
            pr_report.append("### Reviews")
            pr_report.extend(review_lines)
        if gen_comment_lines:
            pr_report.append("### General Comments")
            pr_report.extend(gen_comment_lines)
        if inline_comment_lines:
            pr_report.append("### Inline Comments")
            pr_report.extend(inline_comment_lines)
            
        if not review_lines and not gen_comment_lines and not inline_comment_lines:
            pr_report.append("*No reviews or comments from other users yet.*")
            
        if pr_new_activity:
            markdown_report.extend(pr_report)
            markdown_report.append("\n" + "="*40 + "\n")
            
    if new_activity_detected:
        # Save updated state
        try:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save state file: {e}", file=sys.stderr)
            
        # Print the markdown report
        print("\n".join(markdown_report))
    else:
        # No new activity
        print("NO_NEW_ACTIVITY")

if __name__ == "__main__":
    main()
