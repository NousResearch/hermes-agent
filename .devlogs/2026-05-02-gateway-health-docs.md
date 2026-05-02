# Gateway and Cron Health Skill Documentation

**Date:** 2026-05-02
**Author:** Unknown
**Branch:** docs/gateway-singleton-health-202605020310
**PR:** https://github.com/NousResearch/hermes-agent/pull/18640
**Issue:** https://github.com/NousResearch/hermes-agent/issues/18641

## Goal

Capture the operational lesson that Hermes gateway/scheduler health needs outside-in verification, especially in Docker or sandboxed environments where PID namespaces can make `hermes cron status` inconclusive.

## What Was Done

- Updated `skills/autonomous-ai-agents/hermes-agent/SKILL.md` gateway troubleshooting guidance.
- Regenerated `website/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent.md` from the bundled skill source.
- Wrote an implementation plan at `.hermes/plans/2026-05-02-gateway-health-docs.md`.
- Prepared and filed follow-up issue https://github.com/NousResearch/hermes-agent/issues/18641 for first-class gateway/scheduler health status.
- Opened PR https://github.com/NousResearch/hermes-agent/pull/18640 from fork branch `docs/gateway-singleton-health-202605020310`.

## Key Decisions

- Kept runtime watchdog scripts and local health artifacts out of the PR because they are environment-specific operations files, not upstream Hermes source files.
- Filed the remaining product work as an issue candidate rather than expanding this PR into runtime implementation.
- No ADR was added because the PR is documentation-only and changes no runtime architecture, APIs, or command behavior.
- Strict RED/GREEN TDD was not applicable because there is no production code change; validation used generator, diff review, static scan, and independent review.

## Validation

- `python website/scripts/generate-skill-docs.py` — passed; regenerated bundled skill docs.
- `python /work/.hermes-data/skills/software-development/requesting-code-review/scripts/static_scan_diff.py --rev origin/main...HEAD` — passed; no security/debug marker findings.
- `git diff --check origin/main...HEAD` — passed.
- `HERMES_TDD_EVIDENCE="N/A docs-only change; validation: python website/scripts/generate-skill-docs.py; static_scan_diff.py; independent review passed with wording cleanup applied" /work/.hermes-data/scripts/code_work_guard.py --mode final` — passed.
- Independent reviewer via `delegate_task` — passed; no security concerns or blocking logic/process issues. One wording precision suggestion was applied before pushing the PR.
- `gh issue create --repo NousResearch/hermes-agent --title "Add deployment-aware gateway/scheduler health status" --body-file /work/.hermes-data/tmp/hermes-gateway-health-issue.md` — passed; created https://github.com/NousResearch/hermes-agent/issues/18641.
- `gh pr create --repo NousResearch/hermes-agent --base main --head mikejflex:docs/gateway-singleton-health-202605020310 ...` — passed; created https://github.com/NousResearch/hermes-agent/pull/18640.

## What Skills and Tools Were Used

- `hermes-agent` for Hermes-specific configuration/troubleshooting context.
- `writing-plans` for the repo-local implementation plan.
- `test-driven-development` for the docs-only TDD applicability decision.
- `requesting-code-review` for static scan and independent review.
- `github-issues` and `github-pr-workflow` for issue/PR flow.
- `devlog` for this repo-local session log.

## Artifacts Updated

- `skills/autonomous-ai-agents/hermes-agent/SKILL.md`
- `website/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent.md`
- `.hermes/plans/2026-05-02-gateway-health-docs.md`
- `.devlogs/2026-05-02-gateway-health-docs.md`

## Related Repos

- `NousResearch/hermes-agent`

## Issues & Blockers

- No active blocker. GitHub API auth was recovered from the configured git remote token for PR/issue creation.
- An earlier broad issue search surfaced https://github.com/NousResearch/hermes-agent/issues/9069, but that FreeBSD-specific issue is not the same leftover product work, so a specific new follow-up issue was created.

## Key Learnings

- For Hermes gateway/cron troubleshooting, future agents should verify gateway liveness using the actual service/container supervisor and persisted run artifacts rather than relying on a single in-process status command.

## Next Steps

- Track PR review and CI for https://github.com/NousResearch/hermes-agent/pull/18640.
- Use https://github.com/NousResearch/hermes-agent/issues/18641 as the implementation boundary for first-class deployment-aware health/status work.

## Prompting Notes

- **Initial ask:** Review remaining health/watchdog work, file viable leftovers as a GitHub issue, and push PR-worthy changes.
- **Clarifications needed:** None; target repo was inferred from the Hermes Agent skill/runtime context and verified by remotes.
- **Corrections made:** Avoided committing local runtime scripts and local cron state to the upstream repo.
- **Scope drift:** Kept runtime health implementation out of this docs PR and converted it into an issue candidate.

## Session Quality

- **Faithfulness:** stayed on track — narrowed upstream PR to durable documentation and kept environment-specific work local.
- **Prompt patterns:** The ask was concise but depended on prior context; live repo/diff inspection was necessary to avoid pushing local-only artifacts.

---
*Generated by Hermes Agent — devlog skill*
