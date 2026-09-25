"""Opt-in, operator-owned GitHub Actions policies for private repositories.

Never inferred from task metadata or a green check name. Unconfigured repositories
continue through the repository-required-check collector. The approved workflow
blob is the trust anchor for what its checkout/test jobs actually do.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from urllib.parse import quote

_SHA = re.compile(r"[0-9a-f]{40}")
_CHECKOUT = re.compile(
    r"(?m)^\d{4}-\d\d-\d\dT\S+ \[command\]/[^\r\n]+/git log -1 --format=%H\r?\n"
    r"\d{4}-\d\d-\d\dT\S+ ([0-9a-f]{40})\r?$"
)
_KEYS = {"workflow_id", "workflow_path", "workflow_blob_sha", "app_id", "required_checks", "checkout_checks"}


def policy_for(repo):
    from hermes_cli.config_effective import load_user_config_effective

    from yaml import YAMLError
    try:
        config = load_user_config_effective(fail_closed=True)
    except YAMLError:
        raise ValueError("Named-check configuration is unreadable") from None
    kanban = config.get("kanban", {})
    if not isinstance(kanban, dict):
        raise ValueError("Invalid kanban configuration")
    policies = kanban.get("pr_acceptance_policies", {})
    if not isinstance(policies, dict):
        raise ValueError("Invalid named-check policy map")
    if repo not in policies:
        return None
    policy = policies[repo]
    if not isinstance(policy, dict) or set(policy) != _KEYS:
        raise ValueError("Invalid named-check policy keys")
    for key in ("workflow_id", "app_id"):
        if type(policy[key]) is not int or policy[key] <= 0:
            raise ValueError("Policy requires positive workflow and app IDs")
    if not isinstance(policy["workflow_path"], str) or not re.fullmatch(r"\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml", policy["workflow_path"]):
        raise ValueError("Policy requires an exact workflow path")
    if not isinstance(policy["workflow_blob_sha"], str) or not _SHA.fullmatch(policy["workflow_blob_sha"]):
        raise ValueError("Policy requires an approved workflow blob SHA")
    for key in ("required_checks", "checkout_checks"):
        names = policy[key]
        if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n.strip() for n in names) or len(set(names)) != len(names):
            raise ValueError("Policy requires nonempty unique check names")
    if not set(policy["checkout_checks"]) <= set(policy["required_checks"]):
        raise ValueError("Checkout checks must be required checks")
    return policy


def _items(pages, key):
    if not isinstance(pages, list) or not pages:
        raise ValueError("Missing paginated evidence")
    items = [item for page in pages for item in page[key]]
    count = pages[0]["total_count"]
    if any(page["total_count"] != count for page in pages) or len(items) != count or len({i["id"] for i in items}) != count:
        raise ValueError("Incomplete or inconsistent pagination")
    return items


def _snapshot(pr):
    return (pr["head"]["sha"], pr["base"]["sha"], pr["base"]["ref"],
            pr["head"]["repo"]["full_name"], pr["base"]["repo"]["full_name"],
            pr["state"], pr.get("merged", False), pr.get("merge_commit_sha"))


def _log(repo, job_id):
    response = subprocess.run(
        ["gh", "api", f"repos/{repo}/actions/jobs/{job_id}/logs", "--hostname", "github.com"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=True,
    )
    if len(response.stdout) > 20_000_000:
        raise ValueError("Checkout log exceeds evidence budget")
    return response.stdout


def collect_named(repo, number, policy, receipt, api):
    """Mutates the ordinary acceptance receipt; caller preserves lifecycle fencing."""
    receipt.update(policy_source="explicit_named_checks", policy_repository=repo,
                   policy_sha256=hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
                   workflow_id=policy["workflow_id"], workflow_path=policy["workflow_path"],
                   workflow_blob_sha=policy["workflow_blob_sha"], app_id=policy["app_id"])
    prefix = f"repos/{repo}"
    pr = api(f"{prefix}/pulls/{number}")
    snapshot = _snapshot(pr)
    head = pr["head"]["sha"]
    receipt["head_sha"] = head
    if not _SHA.fullmatch(head) or pr["number"] != number or pr["head"]["repo"]["full_name"] != repo or pr["base"]["repo"]["full_name"] != repo:
        raise ValueError("Named policy requires a same-repository PR")
    if pr["state"] != "open" and not (pr["state"] == "closed" and pr.get("merged")):
        raise ValueError("PR is closed without merge")

    def latest():
        runs = _items(api(f"{prefix}/actions/workflows/{policy['workflow_id']}/runs?head_sha={head}&event=pull_request&per_page=100", paginate=True), "workflow_runs")
        if not runs:
            raise ValueError("No workflow run for the current head")
        # Do not filter successes: a new red/pending attempt supersedes old green.
        return max(runs, key=lambda r: (r["id"], r["run_attempt"]))

    run = latest()
    run_id, attempt = run["id"], run["run_attempt"]
    if type(run_id) is not int or type(attempt) is not int or run_id <= 0 or attempt <= 0:
        raise ValueError("Invalid run identity")
    receipt.update(workflow_run_id=run_id, run_attempt=attempt,
                   workflow_run_url=f"https://github.com/{repo}/actions/runs/{run_id}/attempts/{attempt}")
    if (run["head_sha"] != head or run["event"] != "pull_request" or
        run["repository"]["full_name"] != repo or run["head_repository"]["full_name"] != repo or
        run["workflow_id"] != policy["workflow_id"] or run["path"] != policy["workflow_path"]):
        raise ValueError("Untrusted workflow run provenance")
    bindings = [p for p in run["pull_requests"] if p["number"] == number and p["url"] == f"https://api.github.com/repos/{repo}/pulls/{number}"]
    if len(bindings) != 1 or bindings[0]["head"]["sha"] != head or bindings[0]["base"]["sha"] != pr["base"]["sha"]:
        raise ValueError("Run is not bound to the current PR head/base")
    if run["status"] != "completed" or run["conclusion"] != "success":
        receipt.update(classification="pending" if run["status"] != "completed" else "failure", detail="Latest workflow run is not successful.")
        return receipt
    jobs_endpoint = f"{prefix}/actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100"
    jobs = _items(api(jobs_endpoint, paginate=True), "jobs")
    tested = set()
    for name in policy["required_checks"]:
        matching = [j for j in jobs if j["name"] == name]
        if len(matching) != 1:
            receipt.update(classification="missing", detail="A required named check is missing or ambiguous.")
            return receipt
        job = matching[0]
        if type(job["id"]) is not int or job["id"] <= 0:
            raise ValueError("Invalid job identity")
        check = api(f"{prefix}/check-runs/{job['id']}")
        if (job["run_id"] != run_id or job["run_attempt"] != attempt or job["head_sha"] != head or
            check["id"] != job["id"] or check["name"] != name or check["head_sha"] != head or
            check["app"]["id"] != policy["app_id"] or check["check_suite"]["id"] != run["check_suite_id"]):
            raise ValueError("Untrusted check provenance")
        success = all(item["status"] == "completed" and item["conclusion"] == "success" for item in (job, check))
        receipt["checks"].append({"name": name, "id": job["id"], "head_sha": head,
                                  "app_id": policy["app_id"], "run_id": run_id, "run_attempt": attempt,
                                  "classification": "success" if success else "failure",
                                  "conclusion": check["conclusion"],
                                  "url": f"https://github.com/{repo}/actions/runs/{run_id}/job/{job['id']}"})
        if not success:
            receipt.update(classification="failure", detail="A required named check is not successful.")
            return receipt
        if name in policy["checkout_checks"]:
            log = _log(repo, job["id"])
            shas = _CHECKOUT.findall(log)
            if len(shas) != 1:
                raise ValueError("Missing or ambiguous tested checkout SHA")
            tested.add(shas[0])
            receipt["checks"][-1].update(tested_sha=shas[0], log_sha256=hashlib.sha256(log.encode()).hexdigest())
    if len(tested) != 1:
        raise ValueError("Checkout jobs tested different commits")
    tested_sha = tested.pop()
    if tested_sha not in {head, pr.get("merge_commit_sha")}:
        raise ValueError("Tested checkout is neither current head nor current PR merge")
    head_commit = api(f"{prefix}/git/commits/{head}")
    tested_commit = api(f"{prefix}/git/commits/{tested_sha}")
    tree = head_commit["tree"]["sha"]
    if head_commit["sha"] != head or tested_commit["sha"] != tested_sha or not _SHA.fullmatch(tree) or tested_commit["tree"]["sha"] != tree:
        raise ValueError("Tested checkout tree differs from current head")
    if tested_sha != head and {p["sha"] for p in tested_commit["parents"]} != {head, pr["base"]["sha"]}:
        raise ValueError("Tested merge does not bind current head/base")
    # pull_request executes the merge-ref workflow even if it checks out head.
    # Bind that execution revision separately: head-only pinning could trust an
    # unapproved workflow introduced by base, running tests against approved head.
    execution_sha = pr.get("merge_commit_sha")
    if not isinstance(execution_sha, str) or not _SHA.fullmatch(execution_sha):
        raise ValueError("Workflow execution merge is unavailable")
    execution_commit = api(f"{prefix}/git/commits/{execution_sha}")
    if execution_commit["sha"] != execution_sha or {p["sha"] for p in execution_commit["parents"]} != {head, pr["base"]["sha"]}:
        raise ValueError("Workflow execution merge does not bind current head/base")
    for sha in {head, tested_sha, execution_sha}:
        blob = api(f"{prefix}/contents/{quote(policy['workflow_path'], safe='/')}?ref={sha}")
        if blob["sha"] != policy["workflow_blob_sha"]:
            raise ValueError("Workflow differs from operator-approved blob")
    receipt.update(tested_sha=tested_sha, tree_sha=tree, workflow_execution_sha=execution_sha)
    current_run = api(f"{prefix}/actions/runs/{run_id}")
    current_latest = latest()
    if (current_run != run or current_latest != run or
        _items(api(jobs_endpoint, paginate=True), "jobs") != jobs or
        _snapshot(api(f"{prefix}/pulls/{number}")) != snapshot):
        receipt.update(classification="stale", detail="PR or workflow evidence changed during collection; retry.")
        return receipt
    receipt.update(ok=True, classification="success")
    return receipt
