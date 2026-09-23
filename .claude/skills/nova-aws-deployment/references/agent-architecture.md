# NOVA Deployment Agent — architecture, install, delegation protocol

## Contents
1. Diagram → files  2. Install in a repo (Claude Code)  3. Delegation protocol
4. Memory model  5. Safety layers  6. Typical flows  7. Extending the team

## 1. Diagram → files

```
                 ┌────────────────────────┐
                 │  NOVA DEPLOYMENT AGENT  │  main session + SKILL.md (orchestrator)
                 └───────────┬────────────┘
       ┌─────────────────────┼──────────────────────┐
   Skills/Knowledge      Memory/State          Decision/Safety
   references/*.md       .nova/state.json      scripts/plan_guard.py      (mechanical, per plan)
                         docs/lessons/*.md     hooks/command_guard.py     (mechanical, per tool call)
                                               agents/nova-safety-reviewer (judgement, per change)
                                               human approval             (authority)
                             │
                      Tool Orchestrator  = main session delegating via the Task/Agent tool
       ┌────────────┬────────┼────────┬─────────────┬──────────────┐
   nova-terraform  nova-aws-  nova-ssm-  nova-docker-  nova-bedrock-
                   inspector  operator   release       diagnostician
       └────────────┴────────┼────────┴─────────────┴──────────────┘
                         AWS ACCOUNT → NOVA DEPLOYMENT
```

Why the orchestrator is the main session and not a sub-agent: Claude Code sub-agents cannot spawn
other sub-agents, and the human approves actions in the main session. Putting orchestration there
keeps routing, memory and approval in one place.

## 2. Install in a repo (Claude Code)

```bash
# from the NOVA repo root
mkdir -p .claude/skills .claude/agents .claude/hooks .nova
cp -r nova-aws-deployment .claude/skills/
cp .claude/skills/nova-aws-deployment/agents/*.md .claude/agents/
cp .claude/skills/nova-aws-deployment/hooks/command_guard.py .claude/hooks/
# merge hooks/settings.example.json into .claude/settings.json
python3 .claude/skills/nova-aws-deployment/scripts/deploy_state.py init --env test
python3 -m unittest discover -s .claude/skills/nova-aws-deployment/tests   # guards self-test
```
Commit `.claude/` and `.nova/state.json` (it contains no secrets — keep it that way). Add
`tfplan`, `tfplan.json`, `*.tfstate*` to `.gitignore`.

Model choice: agents use `model: inherit`. Cheap read-only roles (aws-inspector) can be set to a
smaller model; keep the safety reviewer on the strongest model available.

On claude.ai (no sub-agents): the skill still works — play each role in turn, reading the agent
file as your instructions, and keep the report formats. The hook doesn't run there, so apply the
same deny/ask rules yourself.

## 3. Delegation protocol

Brief every specialist with this template; one hypothesis or task per brief:

```
ROLE:        nova-<name>
STATE:       <journal state + last_failure, from deploy_state.py show>
TASK:        <single task or hypothesis>
CONSTRAINTS: read-only | mutation authorized: <exact action>
CONTEXT:     <only the facts it needs: IDs, digests, plan sha, error text>
RETURN:      the report format in your agent file
```

Routing:
- Plan needed → terraform → safety-reviewer → (human if NEEDS_HUMAN/REVIEW/BLOCKED) → orchestrator runs `tf.sh apply`.
- Bedrock error → bedrock-diagnostician; it asks you for runtime-side probes → you route to ssm-operator.
- Worker crashed → ssm-operator (logs) → classify layer → the specialist for that layer.
- New release → docker-release → safety-reviewer (image) → ssm-operator (deploy, authorized) → verify.

Conflict rule: if two specialists disagree, don't average them. Get the evidence each relies on and
run the single command that distinguishes the hypotheses.

## 4. Memory model

| layer | where | lifetime | written by |
|---|---|---|---|
| working | conversation | session | everyone |
| deployment state | `.nova/state.json` via `deploy_state.py` | per environment | orchestrator |
| facts for rollback | `facts.*` in state (digests, bundle sha, plan sha, rollback) | per deployment | orchestrator |
| decisions | `decisions[]` in state (who approved what) | permanent | orchestrator, from human |
| lessons | `docs/lessons/YYYY-MM-DD-<slug>.md` | permanent | orchestrator |
| knowledge | skill `references/` | versioned with skill | skill maintainer |

Lesson file template:
```
# <title>
Symptom: <exact error>
Layer / category: <...>
Root cause: <confirmed | hypothesis pending AWS>
Fix: <what changed, PR/commit>
Prevention: <preflight check id | test name> — status: todo/done
```
When a lesson is general (applies to every customer), propose a change to the skill's references too.

## 5. Safety layers (defence in depth)
1. `command_guard.py` hook — denies never-allowed commands, asks for human-gated ones, catches edits
   that remove `prevent_destroy`/`ignore_changes` or add `Resource "*"` / `0.0.0.0/0`.
2. `tf.sh` — apply only a saved plan whose sha matches the guarded plan and `NOVA_APPROVED_PLAN_SHA`.
3. `plan_guard.py` — BLOCKED on unapproved destroy/replace of protected types, IAM wildcards, public ingress.
4. `deploy_state.py` — refuses skipping PLAN_REVIEW; refuses apply without a reviewed plan sha;
   refuses READY without rollback facts.
5. `nova-safety-reviewer` — intent vs diff.
6. Human — the only authority for destructive or widening changes.

IAM backstop (recommended): run the agent with an AWS profile whose policy denies
`ec2:TerminateInstances`, `ec2:DeleteVolume`, `kms:ScheduleKeyDeletion`, `s3:DeleteBucket` on NOVA
resources, and use a separate break-glass role for humans. Hooks are guardrails; IAM is a wall.

## 6. Typical flows

**Infrastructure change**
```
deploy_state advance PLAN
nova-terraform: tf.sh plan -> report (sha S)
nova-safety-reviewer: verdict on S
deploy_state advance PLAN_REVIEW ; record reviewed_plan_sha256 S ; decision "approved S" --by <human>
human: NOVA_APPROVED_PLAN_SHA=S scripts/tf.sh apply      (hook asks for confirmation)
deploy_state advance INFRASTRUCTURE_APPLY --evidence "apply ok, S"
nova-aws-inspector: verify levels 1-3
```

**Bedrock failure**
```
nova-bedrock-diagnostician: classify + local probe
nova-ssm-operator: bedrock_probe.sh on runtime (instance role)
human: console check
orchestrator: fill vantage table -> owner -> lesson -> preflight check
```

## 7. Extending the team
Add a specialist only when a recurring task needs a distinct tool set or knowledge file (e.g.
`nova-cloudwatch-analyst` once log volume grows, `nova-cli-developer` when building the engine).
Each new agent needs: one-line job, tool list (minimum needed), knowledge file, report format.
