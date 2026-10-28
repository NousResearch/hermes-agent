---
name: iac-change
description: Terraform/Pulumi/k8s/Helm drift, blast radius, staged apply.
version: 0.1.0
author: hope (rlaope), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [terraform, opentofu, pulumi, cloudformation, kubernetes, kustomize, helm, infrastructure-as-code, drift, rollback]
    category: devops
    related_skills: [release-cut, docker-management]
---

# IaC Change Skill

Change declared infrastructure (Terraform/OpenTofu, Pulumi, CloudFormation, Kubernetes manifests,
Helm) by reading drift and blast radius from a saved plan, estimating cost, and applying in stages
that each have a health gate and a rollback. It does not cover shipping a new application version
(use `release-cut`) or commanding a live outage.

## When to Use

- "Change/add/remove this resource", a module upgrade, a provider bump, a Helm chart upgrade.
- Reviewing a `terraform plan`, `pulumi preview`, change set, or `kubectl diff` before apply.
- The plan shows changes nobody made in code (drift).
- Promoting an infra change dev → staging → prod.

Don't use for: building container images (`docker-management`), app release/canary decisions
(`release-cut`).

## Prerequisites

- The tool's CLI installed and authenticated by the user (cloud credentials stay in the user's
  environment; never read, print or copy them).
- Which environment/workspace/stack/context each command targets. Confirm with
  `terraform workspace show`, `pulumi stack`, `kubectl config current-context` before any plan.

## How to Run

Run read-only commands (drift, plan, diff, template, rollout status) through `terminal` freely.
**Apply, destroy, `rollout undo`, `helm rollback` and state edits run only after the user approves
that specific saved plan.** Use `background=true` + `process_manage` for long applies. Write saved
plans and rendered manifests into the project or a scratch dir, never commit them (they can
contain secrets).

## Quick Reference

| Tool | Drift | Saved plan / diff | Health gate | Rollback |
| --- | --- | --- | --- | --- |
| Terraform / OpenTofu | `terraform plan -refresh-only` | `terraform plan -out=tfplan` then `terraform show tfplan` | a follow-up `plan` reports no changes, plus the service's own signal | revert the commit and apply the prior config; a destroyed resource does not come back |
| Pulumi | `pulumi refresh --preview-only` | `pulumi preview --diff` (`--save-plan` + `pulumi up --plan` to pin it) | `pulumi preview` shows no changes, plus the service's own signal | revert and `pulumi up` the prior program |
| CloudFormation | `aws cloudformation detect-stack-drift` | a change set (`create-change-set`, `describe-change-set`) reviewed before `execute-change-set` | stack reaches `UPDATE_COMPLETE`, plus the service's own signal | automatic rollback on failure, or a change set to the prior template |
| Kubernetes / kustomize | `kubectl diff -f <manifests>` | `kubectl diff -f` or `kustomize build` output | `kubectl rollout status <kind>/<name> --timeout=<t>` | `kubectl rollout undo` for a workload; re-apply the prior manifest otherwise |
| Helm | `helm get manifest <release> \| kubectl diff -n <ns> -f -` (saved release manifest vs live objects) | `helm diff upgrade` or `helm template` | `helm upgrade --wait --timeout <t>`, then the release's probes | `helm rollback <release> <revision>` |

Replacement markers: Terraform `-/+` or `+/-` / "must be replaced"; CloudFormation change set
`Replacement: True`; Pulumi `replace`; Kubernetes immutable-field errors (a forced delete+create).

## Procedure

1. **Pin the target.** State tool, version, environment/stack/context for every command. Done when
   each command you plan to run names its target explicitly.
2. **Resolve drift first.** Run the drift command. For each drifted resource choose one: import,
   `moved` block / `state mv`, re-declare in code, or revert the manual change. Record which. Done
   when a drift check reports clean or every remaining drift is an accepted, named exception. An
   apply over unresolved drift silently reverts someone's hotfix or recreates something
   deliberately removed.
3. **Save the plan and read the blast radius from it, not from the code diff.** Count creates,
   in-place updates, replacements and destroys. For every replace/destroy, list what depends on it
   (DNS, security groups, IAM bindings, PVCs, other stacks' outputs). A renamed resource with no
   `moved` block or import is a destroy + create. Done when the counts and the dependents list are
   written down.
4. **Gate stateful replacements.** A replace or destroy of a database, volume, bucket, load
   balancer address, or anything with an identity others reference is a blocker requiring the
   user's explicit approval, not a line item. Prefer `lifecycle { prevent_destroy = true }`,
   `create_before_destroy`, or a `moved` block to avoid it.
5. **Cost delta.** Estimate from the saved plan (Infracost or the provider's calculator) and record
   the monthly delta and the resources driving it. If no estimate ran, write `unestimated` and name
   those resources; a missing estimate is not zero.
6. **Staged apply.** Lowest environment first. Re-plan per environment, apply that environment's
   saved plan, observe its health gate, then promote. Stop on any gate failure (roll that stage
   back) or on a plan that differs from the reviewed one. A clean follow-up plan or a completed
   stack update proves the change applied, not that the service works: every gate includes a
   service check (health endpoint, synthetic request, error rate) read after the apply. Production additionally needs every
   earlier gate green and explicit approval for any stateful replacement. A stage with no health
   gate or no rollback is not a stage.

## Pitfalls

- **Apply only the saved plan that was reviewed.** A fresh `terraform apply` re-plans and can
  differ from what was read.
- **Never hand-edit a state file or a live object to hide drift.** Import, move or re-declare.
- `kubectl apply` of a changed immutable field (selector, `volumeClaimTemplates`) fails or forces a
  delete+recreate; check the diff for it.
- Helm `--atomic` (renamed `--rollback-on-failure` in Helm 4) rolls back on a failed `--wait`; it
  does not undo data a hook already migrated.
- A `-target` apply leaves the rest of the plan pending and the state partially converged; use it
  for recovery, not routine changes.
- Provider/module version bumps can change many resources at once; read the plan, not the
  changelog.

## Verification

- A follow-up drift/plan/diff in each applied environment shows no unexpected changes.
- Each stage's health gate output was observed before promoting.
- Every replacement/destroy of a stateful resource has a recorded user approval.
- The cost delta is a number or explicitly `unestimated` with its drivers named.

Ported from rlaope/oh-my-hermes (MIT), `skills/omh-iac-change`.
