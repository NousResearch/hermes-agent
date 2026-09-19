---
name: self-report-pr
description: "Draft fork issues/PRs only; never merge origin."
version: 1.0.0
author: Kevin Rajan (kvnloo) + Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [github, pull-requests, issues, fork, draft, oss-factory]
    category: software-development
    related_skills: [github]
---

# Self-report / self-PR (optional)

Draft GitHub issues and pull requests on **this fork** (`kvnloo/hermes-agent`) so a human can review. This is not merge authority.

## Do

1. Search duplicates on the fork, then draft **one** issue or **one** PR.
2. Target `kvnloo/hermes-agent` `main`. Keep the PR a **draft**.
3. Attach a Verified OSS Loop-shaped receipt when it fits. Cluster similar titles with the local fixture only:

```bash
python3 .verified-oss-loop/scripts/cluster-similar-issues.py tests/fixtures/issues-tiny.json
```

## Refuse

- `gh pr merge` (any repo, any flag). Workers never merge.
- Any write to origin `NousResearch/hermes-agent` (`github_writes=0`).
- Opening a PR whose `--repo` or base remote is `NousResearch/hermes-agent`.
- Live tracker scrapes beyond the tiny local fixture (cap 64).

If asked to merge or to PR origin, stop and say no.
