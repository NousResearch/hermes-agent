# Deployment engine: CLI, state machine, plan safety, rollback, tests

## Target CLI
`nova init | preflight | plan | deploy | verify | status | logs | rollback`
Internals the operator never types: Terraform docker commands, SSM, ECR, S3, Docker, IAM, Bedrock probes.
Mapping from today's manual steps:

| manual today | future command |
|---|---|
| docker-wrapped terraform plan/apply | `nova plan`, `nova infrastructure apply` |
| SSM send-command | `nova runtime exec` |
| Bedrock converse probe | `nova model verify` |
| bundle tar + sha + S3 + presign | `nova bundle deploy` |
| manual checks | `nova verify` |

## State machine
Happy path:
DISCOVER → PREFLIGHT → PLAN → PLAN_REVIEW → INFRASTRUCTURE_APPLY → IMAGE_DEPLOY → BUNDLE_DEPLOY →
RUNTIME_BOOT → PROFILE_APPLY → MODEL_PREFLIGHT → WORKER_PREFLIGHT → END_TO_END_TEST → VERIFY → READY

Failure states: FAILED_PREFLIGHT, FAILED_PLAN, FAILED_INFRASTRUCTURE, FAILED_IMAGE, FAILED_BUNDLE,
FAILED_MODEL, FAILED_WORKER, FAILED_VERIFY.

Each state defines: inputs, outputs, validation, logs, failure reason, safe retry behaviour.
`scripts/deploy_state.py` is the reference implementation of transitions + journal; the real CLI
should reuse its transition table (and its tests). Retry rule: a failed state may be retried only
from the state that produced its inputs; PLAN_REVIEW must be re-entered after any new plan.

Customer journey: connect account → discover → validate → choose region/model → configure agents →
generate plan → customer approves → provision → images → bundle → Bedrock validation → workers →
verification task → dashboard READY. Raw Terraform is only shown in advanced diagnostics.

## Plan safety in the engine
Parse the JSON plan (see `scripts/plan_guard.py`); count add/change/destroy/**replace**; refuse
unexpected destroy/replace unless an approval names the exact resource address. `0/1/0` ≠ `2/1/2`.

## Rollback — designed before production
| kind | mechanism | touches persistent state? |
|---|---|---|
| application | restart previous container version | no |
| image | redeploy previous digest from journal | no |
| bundle | re-apply previous bundle sha (no `--prune` unless verified) | no |
| configuration | restore previous config version | no |
| Terraform | apply a reviewed plan of previous config — never `terraform destroy` | must be checked |

## Tests the engine needs
Terraform generation, plan safety (fixtures for create/update/delete/replace/protected),
IAM policy generation (no `*`, profile + model ARNs), Bedrock config parsing & error classification,
bundle validation & hashing, state transitions (legal/illegal), failure handling, rollback selection,
CLI behaviour, security boundaries (no secrets in outputs). Don't delete a test because it fails; fix the code.

## Documentation
Durable knowledge lives in `docs/deployment/`, `docs/runbooks/`, `docs/architecture/`,
`docs/troubleshooting/`, `docs/lessons/`. A lesson in chat only is a lesson lost.
