---
name: tnt-devops-whitelist-ip
description: Use for TNT Jira IP whitelist tickets. Process and verify.
version: 0.1.0
author: Adnan Khafabi, Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [tnt, devops, jira, whitelist, security]
    related_skills: [tnt-devops, atlassian-cli]
---

# TNT DevOps IP Whitelist

Handle TNT Jira ISD/TIP/TDP IP whitelist tickets end-to-end using the approved TNT whitelist runbook. This skill is for whitelist scope only; do not create credentials, send secrets, or expand the request into other ticket work.

## When to Use

- User asks to process an IP whitelist request in a TNT Jira ISD/TIP/TDP ticket.
- Don't use for credential provisioning, non-whitelist requests, or unsupported ticket prefixes.

## Prerequisites

- Read the current canonical `tnt-devops` runbook `references/wiki/runbooks/whitelist-ip-jira-to-tnt-config.md` and applicable procedure before acting. Follow its current branch, exact two target files per environment, comparison policy, Jira sequence, approval gates, and verification requirements; it is authoritative if this skill's abbreviated notes differ.
- Load `atlassian-cli` instructions, including site/profile routing and internal-comment semantics.
- For Tilaka environment credentials, source `$HOME/tnt/env.sh`; never print its contents or values. Never use variables containing `ADNAN` without explicit permission.
- Use the workspace and repository location specified by the task/project context. For this Tilaka workspace, tnt-config is conventionally under `gitlab.tilaka.id/devops/infrastructure/tnt-config`.
- `uv` may be installed at `$HOME/.local/bin/uv`; use that path if it is not on `PATH`.

## Procedure

1. **Resolve exact issue/site.** Read ticket, status, description, and attachments using the profile/site appropriate to the Jira key. ISD routes through `cantikatnt`. Never rely on a URL's queue path alone to infer ticket details.
2. **Extract request evidence.** Download attached form into the task scratch workspace. Extract PDF text using Hermes runtime-compatible `uv run --with pypdf python ...`; keep the form/private data out of chat and logs. Identify exact client, public IP/CIDR, and environment. Private/local IPs are excluded from config and noted internally per runbook.
3. **Read-only inspect config.** Fetch current intended branch and inspect IP across all `whitelistip` files; compare precisely in the two target files only. Follow runbook policy for already-present IP, label variants, private IP, partial presence, mismatch, or ambiguous entity. Ask only for a material decision that is genuinely unresolved; do not ask for facts already explicitly in ticket/form or decisions recorded in the runbook. Do not proceed on IP/client/environment mismatch.
4. **Check prerequisites and current Jira state.** Follow ticket-prefix state rules exactly. Discover available Jira transitions and assignee by exact email/user search, not guessed IDs. Ensure current worktree is clean or preserve unrelated edits.
5. **Edit narrowly.** Add only the approved allow rule to the two environment target files using repository formatting. Do not edit unrelated OCR/Poc files. Run `git diff --check`; inspect full diff and confirm exactly two intended files and no lost existing rules.
6. **Verify deployment path before publishing.** Inspect the actual Job DSL/source for the correct environment job and expected effects. Source `env.sh` without exposing values. Query Jenkins read-only, traversing folders to find the job (for example `Tilaka/Infrastructure/Sandbox/WhitelistSB` or `Tilaka/Infrastructure/Production/WhitelistProd`). Do not treat an HTTP error at the Jenkins root or an incomplete root listing as proof the job is unavailable. If credentials fail or exact job cannot be found after traversing, stop before commit/push and report blocker—do not ask user to find an URL before doing this discovery.
7. **Commit and push only within approved runbook authorization.** For a qualified whitelist request, push the exact two-file change to the runbook's approved branch and use the runbook-prescribed commit format/attribution. A push triggers the environment Jenkins job; Production's routine backup/copy, `nginx -t`, and reload are authorized built-in effects when the current job/source confirms they are unchanged. Do not manually trigger a job or perform direct server changes outside the approved workflow.
8. **Verify exact build/revision.** Read Jenkins build state until terminal. Confirm correct environment job, SUCCESS, and built Git SHA equals the commit. Do not claim success if the build is running, failed, or on another revision. Do not confuse build SUCCESS with client connectivity testing.
9. **Update Jira only after verified success.** For ISD, add and read back an internal comment (`jsdPublic=false`) first, then transition to the runbook's L1 Work in Progress target, read back status, assign to the verified Cantika account, and read back assignee. For TIP/TDP follow their distinct ticket-comment/status rules in the runbook. Never mark ISD Done. If any Jira mutation fails, stop and report actual current state; no false success.
10. **Handoff clearly.** State ticket key, client/environment/IP, config commit, Jenkins job/build/revision/result, internal-comment verification, and Jira status/assignee. Mention untested client connectivity. Keep credentials and secret material out of chat.

## Pitfalls

- Jenkins foldered jobs may not appear in the root job listing; recursively inspect folder contents and compare the discovered full job path with Job DSL.
- User's phrase "whitelist this ticket" does not authorize unrelated credential creation or secret delivery requested by the same ticket.
- A matching IP in an unrelated file does not establish that the exact two environment targets already contain it.
- Never change production routing manually; rely only on the approved Git-triggered job and verify its exact build.
- Do not claim ticket completion before Jenkins and Jira readbacks both match expected state.

## Verification

- Form/ticket agree on client, IP/CIDR, and environment; ambiguity resolved.
- Only the two correct files changed, all original entries preserved, and `git diff --check` passes.
- Exact Jenkins environment job and its effects are verified from current source; build SUCCESS SHA equals pushed commit.
- Jira comment visibility, expected status, and ISD assignment are read back successfully.
- Report boundary between verified deployment and untested client connectivity; no credential values exposed.