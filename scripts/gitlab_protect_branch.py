#!/usr/bin/env python3
"""Lock a GitLab project's production branch so only mergers can merge.

This is the authoritative, server-side enforcement: GitLab refuses any merge to
the protected branch from anyone below the allowed role. A local git hook can
warn earlier, but only this cannot be bypassed by the client.

Policy for the team:
    - production branch (default: master) is protected
    - direct push to it is blocked for everyone (merge via MR only)
    - merge is allowed at Maintainer level
    - mergers (nat, namton, nam) must be Maintainers; everyone else Developer

This script sets the protected-branch rule via the GitLab API. Member roles
must be granted to the right people on GitLab (the script can list current
members so you can verify, but assigning roles needs each person's GitLab
username and is confirmed by the owner).

Auth: export GITLAB_TOKEN with a token that has api scope and Owner/Maintainer
on the project. Nothing is hardcoded.

Dry-run by default. Pass --apply to write the protected-branch rule.

    export GITLAB_TOKEN=...               # owner-provided, api scope
    python3 scripts/gitlab_protect_branch.py \\
        --gitlab https://gitlab.dev.jigsawgroups.work \\
        --project Nat-Rattanasak/lotto-reward \\
        --branch master --apply
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request

# GitLab access levels.
NO_ACCESS = 0
DEVELOPER = 30
MAINTAINER = 40

MERGERS = ("nat", "namton", "nam")
DEVELOPERS = ("peter", "ing", "poppap")


def api(token: str, base: str, path: str, method: str = "GET", body: dict | None = None) -> tuple[int, object]:
    url = base.rstrip("/") + "/api/v4/" + path.lstrip("/")
    data = None
    headers = {"PRIVATE-TOKEN": token}
    if body is not None:
        data = urllib.parse.urlencode(body, doseq=True).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            payload = resp.read().decode()
            return resp.status, (json.loads(payload) if payload else None)
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode()
        try:
            return exc.code, json.loads(payload)
        except json.JSONDecodeError:
            return exc.code, payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Protect a GitLab production branch.")
    parser.add_argument("--gitlab", required=True, help="GitLab base URL.")
    parser.add_argument("--project", required=True, help="Project path, e.g. group/repo.")
    parser.add_argument("--branch", default="master", help="Branch to protect (default master).")
    parser.add_argument("--apply", action="store_true", help="Write the rule (default dry-run).")
    args = parser.parse_args()

    token = os.getenv("GITLAB_TOKEN", "").strip()
    plan = {
        "gitlab": args.gitlab,
        "project": args.project,
        "branch": args.branch,
        "rule": {
            "push_access_level": NO_ACCESS,
            "merge_access_level": MAINTAINER,
            "allow_force_push": False,
        },
        "roles": {
            "maintainer_mergers": list(MERGERS),
            "developer_only": list(DEVELOPERS),
        },
    }

    if not token:
        plan["error"] = "missing GITLAB_TOKEN env (owner must provide api-scope token)"
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 2

    pid = urllib.parse.quote(args.project, safe="")

    status, members = api(token, args.gitlab, f"projects/{pid}/members/all?per_page=100")
    if isinstance(members, list):
        plan["current_members"] = [
            {"username": m.get("username"), "access_level": m.get("access_level")}
            for m in members
        ]
    else:
        plan["members_lookup"] = {"status": status, "body": members}

    if not args.apply:
        plan["mode"] = "dry-run"
        plan["note"] = "no changes; pass --apply to write the protected-branch rule"
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    # Unprotect first (ignore failure if not protected), then protect with policy.
    branch_q = urllib.parse.quote(args.branch, safe="")
    api(token, args.gitlab, f"projects/{pid}/protected_branches/{branch_q}", method="DELETE")
    status, body = api(
        token,
        args.gitlab,
        f"projects/{pid}/protected_branches",
        method="POST",
        body={
            "name": args.branch,
            "push_access_level": NO_ACCESS,
            "merge_access_level": MAINTAINER,
            "allow_force_push": "false",
        },
    )
    plan["mode"] = "apply"
    plan["apply_result"] = {"status": status, "body": body}
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0 if status in (200, 201) else 2


if __name__ == "__main__":
    raise SystemExit(main())
