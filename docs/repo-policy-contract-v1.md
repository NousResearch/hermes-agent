# Repo Policy Contract v1

Repo policy v1 is the repo-local authority contract for branch/release/live-apply safety. It exists to make repo-specific rules discoverable without relying on Yuuka memory, Discord context, or per-turn reminders.

The authority file is:

```text
.hermes/repo-policy.yaml
```

`AGENTS.md` may point to that file, but must not duplicate or expand authority. If the policy is missing, malformed, unsupported, stale, or mismatched with observable repo state, the checker must report `Policy drift detected` and fail closed to local-only work.

## Files in this slice

- `schemas/repo-policy-v1.schema.json` — schema contract.
- `examples/repo-policy/product-develop.yaml` — product repo policy with `develop` landing and `main` release target.
- `examples/repo-policy/product-develop-master.yaml` — product repo policy with `develop` landing and `master` release target.
- `examples/repo-policy/runtime-tooling-hermes-agent.yaml` — runtime/tooling policy with live-apply queue and bounded gateway restart model.
- `examples/repo-policy/invalid-*.yaml` — negative fixtures for the checker lane.
- `tests/repo_policy/test_repo_policy_schema_examples.py` — schema/example contract tests.

## Required policy shape

```yaml
version: 1
repo:
  name: example
  class: product # product | runtime_tooling | other
authority:
  policy_source: .hermes/repo-policy.yaml
roles:
  work: task_branch
  landing: develop
  release: release_gate_to_release_target # or release_pr_to_main for repos whose release target is main
  live_apply: deploy_gate
branches:
  landing: develop
  release_base: main # repo-local release target; may be master
gates:
  green_allowed:
    - code_changes
    - tests
    - cleanup
  yellow_queue:
    - release_pr_prepare
    - live_apply_pending
  red_requires_explicit_approval:
    - merge
    - deploy
    - prod_mutation
    - env_secret_change
    - billing_pricing_change
runtime:
  restart_policy: not_applicable
  live_apply_queue: not_applicable
closeout:
  required_sections:
    - Policy check
    - Green 완료
    - Yellow 대기
    - Red 필요
    - 검증
    - Git 상태
    - Live 상태
```

## Product repo v1 rule

Product repos standardize their ordinary work landing on `develop`, while the production release target remains repo-local and explicit:

```text
work branch/worktree -> develop -> release PR/gate -> <repo release target>/prod deploy gate
```

Examples:

- DailyChingu: `develop -> main`.
- WhyStarve migration target: `develop -> master` after explicit branch/push approval.

Schema-level requirements:

- `repo.class: product`
- `roles.landing: develop`
- `branches.landing: develop`
- `roles.release: release_pr_to_main` or `release_gate_to_release_target`
- `workflow.release_source: develop`
- `workflow.release_target: main` or `master`
- `branches.release_base` matching `workflow.release_target`
- `roles.live_apply: deploy_gate`

Checker-level freshness requirements for BO-101:

- observe `develop` locally or remotely before external authority expands;
- fail closed if `develop` is absent or remote tracking is ambiguous;
- never create `develop` automatically from policy validation.

## Runtime/tooling repo v1 rule

Runtime/tooling repos, including `hermes-agent`, use the same Green/Yellow/Red closeout contract but not product release topology:

```text
work branch/worktree -> verified landing -> live-apply queue -> gateway/runtime proof
```

Schema-level requirements:

- `repo.class: runtime_tooling`
- `roles.live_apply: gateway_runtime_queue`
- `runtime.restart_policy: queue_apply_one_restart` or `manual_explicit_only`
- `runtime.live_apply_queue: required`

For `hermes-agent`, gateway restart/reload is a live mutation. Ordinary closeout may only add `gateway_restart_needed` / `live_apply_pending` to Yellow. A current-turn queue-apply command such as `대기열 적용해` or `restart 필요한 것 모아서 적용해` authorizes only one bounded batch, at most one restart/reload, and post-apply proof for the listed entries.

## Closeout sections

The canonical template lives in `docs/repo-policy-closeout-template.md` and `agent/repo_policy_closeout.py`.

Every repo task closeout must include these sections:

```text
Policy check
Green 완료
Yellow 대기
Red 필요
검증
Git 상태
Live 상태
```

Runtime/tooling closeouts should also include:

```text
Gateway restart 필요
Live runtime 반영됨
대기열 포함됨
```

## Fail-closed meaning

When policy is missing, malformed, unsupported, stale, or mismatched:

Allowed:

- read files;
- inspect git state and repo instructions;
- run non-mutating checks;
- edit local files and run tests when the current task scope allows;
- report `Policy drift detected` with exact reasons.

Blocked:

- push;
- upstream PR/comment/update/merge;
- merge/release/deploy;
- gateway restart/reload/live runtime apply;
- env/secret/config authority mutation;
- billing/pricing/customer/prod mutation;
- destructive cleanup;
- live automation enablement.

## AGENTS.md pointer text

Use this shape when a later rollout card adds the pointer to a real repo:

```md
## Repo policy

Before repo work, read `.hermes/repo-policy.yaml`.
Before external side effects, run/perform the repo policy check.
If policy is missing, malformed, or mismatched with observable repo state, fail closed to local-only.
Every closeout must include `Policy check` plus Green/Yellow/Red/Git/Live sections.

`AGENTS.md` is a pointer only; `.hermes/repo-policy.yaml` is the repo authority.
```

## BO-101 checker expectations

This BO-100 slice does not implement the checker CLI. It gives BO-101 deterministic inputs:

- valid product policy passes schema shape;
- valid runtime/tooling policy passes schema shape;
- product landing other than `develop` fails schema shape;
- runtime/tooling policy without a live-apply queue fails schema shape;
- unsupported version fails schema shape;
- the future checker adds observable freshness checks such as product `develop` branch presence and AGENTS.md pointer conflicts.
