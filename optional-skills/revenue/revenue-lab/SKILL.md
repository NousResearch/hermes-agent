---
name: revenue-lab
description: Operate Hermes Revenue Lab with guarded local workflows.
version: 0.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Revenue, Experiments, Governance, Local]
    category: revenue
---

# Hermes Revenue Lab

Use this skill when the user asks Hermes to inspect, repair, or reason about the external Hermes Revenue Lab checkout.

Hermes Revenue Lab is an external local-first project, not Hermes core. Resolve its checkout root from `integrations.revenue_lab.root` in `config.yaml`; if that setting is absent, use `~/HermesRevenueLab`. Do not introduce a user-facing `HERMES_*` environment variable for this non-secret setting.

Keep these boundaries:

- Confirm the HRL root, branch, HEAD, dirty state, and origin before changing HRL behavior.
- Read HRL `AGENTS.md`, `README.md`, and the relevant `docs/runbooks/hrl-*.md` before changing HRL behavior.
- Treat HRL evidence as diagnostic-only unless the runbook says otherwise and the current command output proves it.
- Do not publish, spend, contact customers, create accounts, enter contracts, or mutate external services from this skill.
- Do not copy HRL source into Hermes core. Use the HRL checkout and its scripts as the authority.
- Do not weaken the resource governor, approval policy, compliance registry, provenance checks, corpus checksum, or model identity checks to pass tests.

Useful local checks from the HRL checkout:

```bash
git status --short --branch
python3 scripts/revenue_guard.py --workload guard_check
```

For provider benchmark work, follow the HRL provider benchmark runbook in the HRL checkout. Provider, effort, and concurrency receipts are diagnostic-only.
