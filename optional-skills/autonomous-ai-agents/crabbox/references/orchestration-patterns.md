# Crabbox Orchestration Patterns

Read only the pattern that matches the requested outcome. Patterns 1-3 require the agent-capable islo backend, patterns 4-7 stay local or use configured MCP servers, and pattern 8 uses the non-agent crabbox backend.

## Contents

1. Build a feature or fix
2. Review a PR
3. Refine a PR
4. Plan in parallel
5. Triage issues
6. Analyze logs
7. Research existing skills
8. Run or test remotely

## 1. Build a feature or fix

Preflight the issue and repository before starting a box. Check for existing linked PRs and active branches so the sandbox does not duplicate work.

```bash
ORG=example
REPO=service
ISSUE=42
BOX="build-${REPO}-${ISSUE}-claude-$(date +%s)"

./scripts/crabbox.sh new "$BOX" \
  --source "github://${ORG}/${REPO}" \
  --workdir "$REPO" \
  --agent claude \
  --task "Read issue #${ISSUE}. Implement the smallest complete fix, run the repository tests, commit on a focused branch, and open a PR that closes the issue."
```

For independent repositories, create one uniquely named box per repository. Use two agents only for contested designs or unusually large changes; treat the first sound PR as the candidate and use the other result as review input.

Poll the expected PR or branch with bounded backoff. Also inspect sandbox status and stop polling early on failed, stopped, or deleted states. Remove the box only after recording the PR or failure evidence.

## 2. Review a PR

Give the sandbox the PR URL and a bounded review rubric. Require it to inspect the diff, prior review context, repository instructions, and relevant tests.

Ask for:

- Inline findings first, capped and ordered by severity.
- A concise scorecard after the inline findings.
- `APPROVE`, `COMMENT`, or `REQUEST_CHANGES` only after evidence supports it.
- `COMMENT` for draft PRs.
- A partial-review note when the PR is too large to inspect completely.

Do not let an automated reviewer merge the PR or resolve human review threads.

## 3. Refine a PR

Start from the PR's existing head branch. Fetch unresolved review threads, group duplicate requests, and address each actionable cluster in a focused commit.

```bash
./scripts/crabbox.sh new "refine-${REPO}-${PR}-claude-$(date +%s)" \
  --source "github://${ORG}/${REPO}" \
  --workdir "$REPO" \
  --agent claude \
  --task "Address every actionable unresolved thread on PR #${PR}. Run focused tests, push to the same branch, and report each change. Do not merge or resolve reviewer threads."
```

If comments conflict or require product judgment, report the conflict instead of guessing.

## 4. Plan in parallel

Planning is read-only. Use Hermes `delegate_task` workers rather than provisioning boxes.

Useful independent roles:

- Requirements: extract acceptance criteria and out-of-scope work.
- Architecture: map affected files, dependencies, and established patterns.
- Risk: identify security, compatibility, migration, and CI risks.
- Prior art: inspect issues, PRs, and relevant history.
- Operations: cover rollout, monitoring, and rollback.

Merge results into one plan with requirements, architecture, phases, risk register, open decisions, and verification.

## 5. Triage issues

Use the configured Linear MCP server for issue data and the Slack MCP server only when issue context links to Slack. Keep this workflow read-only unless the user explicitly authorizes writes.

Cross-reference each issue against code, commits, and PRs. Classify it as `FIXED`, `STALE`, `ACTIVE`, `NEEDS_INFO`, `BLOCKED`, or `IN_PROGRESS`, attach a confidence level, and cite concrete evidence.

## 6. Analyze logs

Fetch logs locally through `terminal` or the configured observability MCP server. Summarize error signatures, time clusters, affected services or users, likely root cause, and the next diagnostic step.

For islo, use box logs for agent or exec output. For crabbox, use the captured run ID; `logs` does not accept a box ID. Hand a confirmed code defect to pattern 1 and issue-status questions to pattern 5.

## 7. Research existing skills

Search installed skills, the official optional-skills tree, and the public Skills Hub before proposing a new skill. Report the closest match, its path or identifier, the remaining gap, and whether to install, extend, or create.

A contributed Hermes skill must use a conventional `SKILL.md`, place helpers under `scripts/`, and include hermetic tests.

## 8. Run or test remotely

Choose crabbox when the diff already exists and only remote compute or a clean environment is needed.

```bash
export CRABBOX_BACKEND=crabbox

# Reuse a named lease.
./scripts/crabbox.sh new ci-box -- npm test

# Fresh ephemeral lease; it auto-releases when the command exits.
./scripts/crabbox.sh new -- npm test

# Inspect and remove one named lease.
./scripts/crabbox.sh status ci-box --json
./scripts/crabbox.sh list --json
./scripts/crabbox.sh rm ci-box -f
```

The wrapper maps the named commands to `crabbox run/status/stop --id NAME`, stripping the wrapper's compatibility `-f` before `stop`. It never maps `rm` to the fleet-wide `cleanup` command.

For a persistent box, create it directly with `crabbox warmup --slug NAME` or `crabbox run --keep --slug NAME`. Capture stdout, stderr, timing data, and the remote exit code as evidence.
