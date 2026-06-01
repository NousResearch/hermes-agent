# Codex for OSS and Hermes Evidence Pattern

These notes capture an evidence pattern from preparing a Codex for Open Source-style application around Hermes Agent and Hermes WebUI contribution work.

## Strategic Positioning

Position the user as an OSS maintainer-workflow builder, not only as a fork author. Strong themes include issue triage, PR review, release notes, security hardening, cross-platform reliability, and mobile approval flows.

Reusable narrative:

```text
I use Codex-style agents inside Hermes Agent and Hermes WebUI workflows to reduce invisible maintainer work: issue triage, PR review, release notes, security hardening, cross-platform reliability, and mobile approval flows.
```

## Evidence Observed In Source Session

The source session used GitHub handle `zapabob` and inspected public work in:

- `NousResearch/hermes-agent`
- `nesquena/hermes-webui`

Counts and states drift. Re-check them before quoting.

## Salvaged Or Cherry-Picked Credit

A closed PR can still be valid contributor evidence when maintainers salvage or cherry-pick the work.

Example observed:

- PR: `NousResearch/hermes-agent#29826`
- Upstream commit: `2c3ca475c055a493bc3c40c31c00e7ad2ce7f045`
- Commit title: `fix(cron): reject id mutation + validate output paths under OUTPUT_DIR`
- Commit message included: `Salvaged from PR #29826 by @zapabob`
- Commit author: `zapabob`

Safe wording:

```text
One closed Hermes Agent PR was salvaged/cherry-picked into upstream commit 2c3ca475c055 and credited to zapabob.
```

Only add release-line wording after checking tag containment again.

## Co-Author Credit

Co-author metadata is useful but narrower than direct merge or release inclusion. Treat it as contributor credit, not as proof that a feature shipped in a specific release.

Safe wording:

```text
Additional upstream commits include Co-authored-by credit for security and environment-hint work.
```

## Verification Commands

```bash
gh search prs --repo NousResearch/hermes-agent --author zapabob --limit 100 --json number,title,state,url,createdAt,updatedAt
gh search issues --repo NousResearch/hermes-agent --author zapabob --limit 100 --json number,title,state,url,createdAt,updatedAt
gh pr view 29826 --repo NousResearch/hermes-agent --json number,title,state,url,mergedAt,mergeCommit,reviewDecision
gh search commits zapabob --repo NousResearch/hermes-agent --limit 30 --json sha,commit,url
gh api repos/NousResearch/hermes-agent/commits/2c3ca475c055a493bc3c40c31c00e7ad2ce7f045
gh release list --repo NousResearch/hermes-agent --limit 100
```

## Draft Sentence Pattern

```text
My recent upstream submissions include security hardening, authentication boundaries, secret redaction, dependency and path safety, Windows behavior, media handling, test reliability, and mobile/WebUI-adjacent maintainer workflows. I describe open work as submitted contributions and reserve accepted or release-included wording for items verified from current GitHub metadata.
```
