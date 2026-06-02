# Hermes-driven issue resolution prompt

Template for driving any single GitHub issue end-to-end (read → implement → self-QA → PR) through a `hermes -z --yolo` one-shot, with **dual-team clawteam coordination** for an audit trail.

## Architecture

Two clawteams per run:

| Team | Purpose | Lifetime |
|---|---|---|
| `issue-NN-fix` | Implementation audit trail — one milestone message per step | Per-issue (created at start) |
| `qa-validation` | Cross-issue QA verdicts — one verdict per issue | Long-lived (created once, reused) |

Hermes plays both implementer and reviewer:
1. **IMPLEMENT** — make the code change, log each step to `issue-NN-fix`
2. **QA** — re-read the diff, check acceptance criteria, post PASS/FAIL verdict to `qa-validation`
3. **SHIP** — only on PASS, push branch + open PR
4. **RETRY** — on FAIL, return to step 2 (max 2 retries) before giving up

The QA verdict gates the PR. No push without an explicit `PASS`.

## Template (fill in the `{{…}}` placeholders)

```text
You are completing {{REPO}} issue #{{ISSUE_NUMBER}} in {{REPO_DIR}}.

DUAL-TEAM DOGFOODING — process validation is mandatory:

Setup:
1. Call clawteam_team_discover. If "qa-validation" is NOT in the list,
   call clawteam_team_spawn name="qa-validation"
   description="Cross-issue QA verdicts".
2. Call clawteam_team_spawn name="issue-{{ISSUE_NUMBER}}-fix"
   description="Hermes-driven fix for #{{ISSUE_NUMBER}} {{ISSUE_TITLE_SHORT}}".

Issue #{{ISSUE_NUMBER}} ({{ISSUE_TITLE}}):
{{ISSUE_BODY_SUMMARY}}

Environment:
- Branch: {{BRANCH_NAME}}
- {{ENV_NOTES}}
- DO NOT push to main. DO NOT force-push. DO NOT --no-verify.

Process (call clawteam_inbox_send after each step, to the right team):

A. IMPLEMENT phase — to team="issue-{{ISSUE_NUMBER}}-fix" from="hermes":
   1. Read {{TARGET_FILES}}; identify the exact change locus.
   2. Apply the fix.
   3. Run {{TEST_COMMANDS}}.
   4. Commit on branch {{BRANCH_NAME}} with Closes #{{ISSUE_NUMBER}}.

B. QA phase — VERDICT to team="qa-validation" to="leader" from="hermes":
   5. Self-review the diff against the acceptance criteria.
   6. Post a verdict — exact format:
        "Issue #{{ISSUE_NUMBER}} QA verdict: PASS|FAIL.
         Checked: <bullets>. Findings: <list or none>.
         Diff size: +X/-Y lines."

C. SHIP phase — only if verdict=PASS:
   7. Push branch; open PR with a body that quotes the QA verdict.
   8. Final clawteam_inbox_send to team="issue-{{ISSUE_NUMBER}}-fix":
        "Shipped PR <url>".

If verdict=FAIL in step 6: return to step 2 (max 2 retries). After
2 retries still failing, post final FAIL to qa-validation and stop.

End with exactly one of:
  ISSUE_{{ISSUE_NUMBER}}_HERMES: PASS  url=<pr-url>
  ISSUE_{{ISSUE_NUMBER}}_HERMES: FAIL: <one-line reason>
```

## Why two teams

- **`issue-NN-fix` (per-issue)** lets you peek a single fix's audit trail in isolation. Each `clawteam_inbox_send` here is a build-log entry.
- **`qa-validation` (shared)** accumulates verdicts across many issues. After a batch run you can `clawteam inbox peek qa-validation --agent leader` and see every PASS/FAIL with findings — one screen.

## Verification

After a run, both should be true:

```bash
clawteam inbox peek issue-NN-fix --agent leader     # 7+ build-log messages, last one quotes PR URL
clawteam inbox peek qa-validation --agent leader    # one verdict line per issue this batch
gh pr view <NN-PR> --json state,mergeable           # OPEN + MERGEABLE if QA passed
```

## Cleaning up

The dev teams are throwaway:

```bash
clawteam team cleanup issue-NN-fix --force
```

Keep `qa-validation` — it's the long-running review history.

## See also

- `scripts/run-hermes-issue.sh` — driver that fills in this template per issue
- README "Hermes ClawTeam plugin" section for setup/install
