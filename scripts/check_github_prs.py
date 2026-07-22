import json
import subprocess
import os
import sys
import shutil
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STATE_FILE = Path.home() / ".hermes" / "pr_monitor_state.json"
PRIMARY_REPO = "NousResearch/hermes-agent"

def get_gh_path():
    gh = shutil.which("gh")
    return gh if gh else "gh"

def run_gh_cmd(args):
    gh = get_gh_path()
    cmd = [gh] + args
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except Exception:
        return None

def load_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"prs": {}}

def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def derive_recommendation(pr_number, mergeable, ci_failures, review_decision, external_comments):
    suggestions = []

    if mergeable == "CONFLICTING":
        suggestions.append("Rebase onto latest origin/main and resolve merge conflicts.")

    if ci_failures:
        suggestions.append(f"Fix failing CI checks ({', '.join(ci_failures)}).")

    if review_decision == "CHANGES_REQUESTED":
        suggestions.append("Address requested changes from reviewer.")

    for comment in external_comments:
        body = comment.get("body", "")
        author = comment.get("author", {}).get("login", "")
        
        if "Duplicate of #" in body:
            dup_ref = body.split("Duplicate of #")[1].split()[0].rstrip(".:,")
            suggestions.append(f"Evaluate closing or commenting on PR #{pr_number} regarding duplicate #{dup_ref}.")
        elif "Related to #" in body:
            rel_ref = body.split("Related to #")[1].split()[0].rstrip(".:,")
            suggestions.append(f"Review overlap/complementary relationship with #{rel_ref} and clarify in comment.")

    if not suggestions and external_comments:
        latest_author = external_comments[-1].get("author", {}).get("login", "reviewer")
        suggestions.append(f"Reply to latest comment from @{latest_author}.")

    if not suggestions:
        suggestions.append("No immediate action required — awaiting maintainer review.")

    return suggestions

def main():
    prs = run_gh_cmd([
        "pr", "list",
        "--repo", PRIMARY_REPO,
        "--author", "@me",
        "--json", "number,title,state,reviewDecision,statusCheckRollup,url,updatedAt,isDraft,mergeable,mergeStateStatus"
    ])

    if prs is None:
        prs = []

    state = load_state()
    prev_prs_state = state.get("prs", {})
    new_prs_state = {}

    action_items = []
    status_summary = []
    new_updates_count = 0

    for pr in prs:
        number = str(pr["number"])
        title = pr["title"]
        url = pr["url"]
        
        details = run_gh_cmd([
            "pr", "view", number,
            "--repo", PRIMARY_REPO,
            "--json", "number,title,state,reviewDecision,mergeable,mergeStateStatus,statusCheckRollup,reviews,comments,reviewRequests,url,updatedAt,headRefName,baseRefName"
        ])

        if not details:
            details = pr

        comments = details.get("comments", [])
        reviews = details.get("reviews", [])
        status_checks = details.get("statusCheckRollup", [])
        mergeable = details.get("mergeable", "UNKNOWN")
        merge_state = details.get("mergeStateStatus", "UNKNOWN")
        review_decision = details.get("reviewDecision", "")

        external_comments = [c for c in comments if not c.get("viewerDidAuthor", False)]
        latest_ext_comment = external_comments[-1] if external_comments else None

        ci_failures = []
        if isinstance(status_checks, list):
            for check in status_checks:
                if isinstance(check, dict):
                    status = check.get("status") or check.get("state") or check.get("conclusion")
                    conclusion = check.get("conclusion") or check.get("state")
                    name = check.get("name") or check.get("context") or "Check"
                    if conclusion in ["FAILURE", "FAILED", "ERROR", "CANCELLED"] or status in ["FAILURE", "FAILED", "ERROR"]:
                        ci_failures.append(name)

        prev = prev_prs_state.get(number, {})
        prev_comment_count = prev.get("external_comment_count", 0)
        prev_latest_comment_id = prev.get("latest_comment_id")
        prev_ci_failed = prev.get("ci_failed", False)
        prev_mergeable = prev.get("mergeable", "MERGEABLE")

        has_new_comment = len(external_comments) > prev_comment_count or (
            latest_ext_comment and latest_ext_comment.get("id") != prev_latest_comment_id
        )
        has_new_ci_failure = (len(ci_failures) > 0) and not prev_ci_failed
        has_new_conflict = (mergeable == "CONFLICTING") and (prev_mergeable != "CONFLICTING")

        if has_new_comment or has_new_ci_failure or has_new_conflict:
            new_updates_count += 1

        new_prs_state[number] = {
            "title": title,
            "url": url,
            "external_comment_count": len(external_comments),
            "latest_comment_id": latest_ext_comment["id"] if latest_ext_comment else None,
            "ci_failed": len(ci_failures) > 0,
            "ci_failures": ci_failures,
            "mergeable": mergeable,
            "merge_state": merge_state,
            "review_decision": review_decision,
            "updated_at": details.get("updatedAt")
        }

        pr_actions = []

        if mergeable == "CONFLICTING":
            pr_actions.append("⚡ Merge conflicts detected — rebase/merge required.")

        if ci_failures:
            pr_actions.append(f"❌ CI Failing ({len(ci_failures)} checks): {', '.join(ci_failures)}")

        if review_decision == "CHANGES_REQUESTED":
            pr_actions.append("⚠️ Changes Requested by reviewer.")

        if external_comments:
            for comment in external_comments:
                author = comment.get("author", {}).get("login", "unknown")
                body = comment.get("body", "").strip()
                body_snippet = body[:250] + ("..." if len(body) > 250 else "")
                is_new = (latest_ext_comment and comment.get("id") == latest_ext_comment.get("id") and has_new_comment)
                prefix = "🆕 " if is_new else "💬 "
                pr_actions.append(f"{prefix}Comment from @{author}:\n     \"{body_snippet}\"")

        recommendations = derive_recommendation(number, mergeable, ci_failures, review_decision, external_comments)

        status_line = f"- PR #{number}: {title} (Mergeable: {mergeable}, Decision: {review_decision or 'None'})"
        status_summary.append(status_line)

        action_items.append({
            "number": number,
            "title": title,
            "url": url,
            "actions": pr_actions,
            "recommendations": recommendations,
            "has_new": has_new_comment or has_new_ci_failure or has_new_conflict
        })

    save_state({"prs": new_prs_state})

    print("=== GITHUB PR MONITOR REPORT ===")
    print(f"Repository: {PRIMARY_REPO}")
    print(f"Total Open PRs: {len(prs)}")
    print(f"New Updates Since Last Run: {new_updates_count}")

    print("\n--- Current Open PRs ---")
    if not status_summary:
        print("No open PRs found.")
    else:
        for line in status_summary:
            print(line)

    print("\n--- Action Items & Recommendations ---")
    if not action_items:
        print("✅ All open PRs are clean and require no immediate action.")
    else:
        for item in action_items:
            flag = " 🔥 [NEW ACTIVITY]" if item["has_new"] else ""
            print(f"\n📌 PR #{item['number']}: {item['title']}{flag}")
            print(f"   URL: {item['url']}")
            if item['actions']:
                print("   Status/Comments:")
                for act in item['actions']:
                    print(f"     - {act}")
            print("   Suggested Next Steps:")
            for rec in item['recommendations']:
                print(f"     👉 {rec}")

if __name__ == "__main__":
    main()
