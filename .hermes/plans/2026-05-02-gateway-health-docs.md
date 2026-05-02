# Gateway Health Documentation Implementation Plan

> **For Hermes:** This is a small documentation/skill update. Do not use subagent-driven-development unless the scope expands beyond this plan.

**Goal:** Preserve the operational lesson that `hermes cron status` can be misleading across Docker/PID namespaces and gateway health should be verified outside-in.

**Architecture:** Update the bundled `hermes-agent` skill source so future Hermes sessions get the warning directly when troubleshooting gateway/cron behavior. Regenerate the derived website skill doc from the source skill.

**Tech Stack:** Markdown skill docs, generated Docusaurus docs.

**Spec Artifacts:**
- Canonical source doc: `skills/autonomous-ai-agents/hermes-agent/SKILL.md`
- Generated doc: `website/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent.md`
- ADR: not needed. This is documentation-only and does not change runtime architecture, user-facing command behavior, or APIs.

---

## Acceptance Criteria

- Gateway troubleshooting docs mention external supervision/health checks for long-running gateways.
- Docs warn that `hermes cron status` may be unreliable across Docker/container PID namespaces and should be corroborated with the service/container supervisor.
- The generated website skill doc reflects the source `SKILL.md` change.
- No local runtime artifacts under `/work/.hermes-data/scripts`, `/work/.hermes-data/ops`, or `/work/.hermes-data/cron/jobs.json` are committed.
- Remaining viable implementation work is filed as a GitHub issue instead of being included in this docs PR.

## Validation Plan

- Documentation check: inspect the diff and generated page to confirm source and generated docs match.
- Static/security scan: run the requesting-code-review diff scanner on the staged or unstaged diff.
- Generation check: run `python website/scripts/generate-skill-docs.py` and verify the generated doc is updated.
- Process guard: run the final code work guard with explicit note that RED/GREEN TDD is not applicable for docs-only changes.

## TDD Note

Strict RED/GREEN TDD is not applicable because this PR changes documentation only and no runtime behavior. The equivalent validation is source-doc update plus generated-doc regeneration and diff review.
