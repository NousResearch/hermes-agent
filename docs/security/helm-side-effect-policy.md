# Helm side-effect policy registry

Status: baseline policy, documentation only
Owner: Helm / Gond for Spearhead safety work
Provenance: derived from `/home/filip/spearhead-notion-triage/20260528-190908/helm-security-analysis.md` (Kanban task `t_a915e2f7`), especially the source categories covering autonomous pentest agents, OSINT/crawlers, phishing/red-team tooling, prompt/agent evals, vulnerability scanning, dashboard/monitoring systems, privacy assistants, supply-chain incidents, and financial IDOR/authz risks.
Integration provenance: policy artifact drafted in Kanban task `t_f8842f19`, extended with S3-S5 helper/schema in `t_5acf38a7`, and copied into this main integration worktree by `t_7873f70b`.

## Purpose

This registry defines the default approval posture for Spearhead/Hermes actions that can affect external systems, money, private data, repositories, credentials, or public communication. It is a human-readable policy baseline, not runtime enforcement.

Default stance: deny or downgrade to a safe preview when an action's side-effect tier is unclear.

High-risk actions must be approved by an authorized human before execution. In Spearhead practice that means Filip or Andrea directly, or EMA relaying their explicit approval with enough context to audit the decision.

## Non-goals for this document

- No tool wrapper changes.
- No config changes.
- No secret inspection.
- No live sends, trades, pushes, deploys, deletes, scans, or public messages.
- No permission grant: examples below describe policy, not authorization to execute.

## Core definitions

### Approval

Approval is valid only when all of the following are true:

1. The approving human is identified.
2. The exact action is named.
3. The target is named: recipient, Notion database/page, account, repo/branch, host/domain, gateway channel, file path, credential store, or broker.
4. The important payload/diff is visible before execution.
5. Expected consequences and rollback limits are stated.
6. The approval is current for this action, not inferred from a stale prior conversation.

Ambiguous approval is not approval. "Looks good" is not enough for a trade, send, deploy, credential change, public post, broad mutation, or destructive action unless the preceding message already contained the exact execution plan.

### Default-deny cases

Default-deny means the agent must not execute the action and should either produce a draft/dry-run plan or block for human approval.

Default-deny applies when:

- The target, payload, approver, or rollback story is missing.
- The action is irreversible, externally visible, financial, credential-bearing, destructive, or security/offensive.
- The action crosses tenant/profile/user/channel boundaries without explicit authorization.
- The agent is asked to bypass approval, conceal an action, delete logs, reveal secrets, impersonate a human, phish, spam, scrape personal data, scan a third-party target, or exploit a system.
- A tool or workflow can mutate external state but the requested outcome can be satisfied with a draft, diff, or read-only inspection.

## Side-effect tiers

| Tier | Name | Definition | Default posture | Approval required |
|---|---|---|---|---|
| S0 | Read-only local | Inspect repo files, docs, logs, test output, local git status without secret-value disclosure. | Allowed when in scope. | No, unless reading outside authorized workspace or private/PII-heavy data. |
| S1 | Local write / reversible artifact | Create or edit local docs, plans, tests, scratch artifacts, or branches; no external mutation. | Allowed when in task scope and reversible. | No for scoped docs/tests; yes if touching profile config, memory, credentials, or another user's workspace. |
| S2 | External read / bounded lookup | Query public web, GitHub metadata, APIs, Notion/Gmail metadata, broker read-only data, or OSINT-like sources without mutation. | Allowed only when scoped, rate-limited, and privacy-safe. | Required for private accounts, PII-heavy lookup, broker/account data, or ambiguous authorization. |
| S3 | External write / visible mutation | Send email/DM/post, mutate Notion/Gmail/calendar/CRM, create GitHub issues/PRs, call gateway send APIs, change remote records. | Draft/dry-run by default. | Yes, with target + payload confirmation. |
| S4 | Destructive / irreversible / money / deploy | Delete files or remote records, place trades/orders, transfer/pay money, submit broker documents, git push/merge/release/deploy, rotate credentials, revoke access. | Deny until explicitly approved. | Always. Often requires second confirmation or EMA gate. |
| S5 | Offensive/security-sensitive autonomy | Network scanning, exploitation, phishing simulation, credential testing, OSINT crawler runs, autonomous pentest agents. | Forbidden by default outside sandbox. | Always; allowed only for owned/sandbox targets with written scope, limits, logs, and kill switch. |

When tiers conflict, use the higher tier. For example, a local script that sends an email is S3, not S1. A test that deletes production data is S4, not a test.

## Required preflight for S3-S5 actions

Before executing any S3-S5 action, record or present this contract:

```yaml
action: "what will be executed"
tier: "S3|S4|S5"
actor_profile: "ema|gond|helm|waukeen|mystra|other"
approver: "human name or explicit approval source"
target: "recipient/page/account/repo/host/channel/path"
payload_or_diff: "summary plus link/path to exact body or diff"
why: "business or engineering reason"
risks: ["privacy", "financial", "public visibility", "rollback limits"]
rollback: "how to undo, or 'not fully reversible'"
audit_event: "where outcome will be recorded"
```

Use `docs/security/s3-s5-preflight-template.md` for the copy/paste template, field checklist, safe downgrade rules, examples, and the companion shape-only JSON schema at `docs/security/s3-s5-preflight.schema.json`.

If any required field is unknown, stop and ask EMA/human for approval or reduce the action to a draft/dry-run.

## Workflow policy registry

### Email

Tier mapping:

- Reading inbox metadata or searching known folders: S2 when private mailbox access is already authorized.
- Drafting an email locally or in a document: S1.
- Creating a draft in an email system: S3 if it mutates mailbox state.
- Sending, replying, forwarding, adding/removing recipients, attaching documents, or changing mailbox rules: S3.
- Persistent forwarding/filter rules, delegated mailbox access, or account settings that can exfiltrate or hide mail: S4.

Approval requirements:

- Sending requires explicit confirmation of recipients, subject, body, attachments, and sending account.
- Conseq, broker, client, legal, tax, HR, medical, or financial email requires human approval even if the body was generated from a template.
- Mass email, cold outreach, phishing-like content, spoofing, or social-engineering content is default-deny.

Safe default:

- Produce a draft and ask EMA/human to approve sending.

Example:

- Allowed: "Draft a reply to Jana about the missing invoice" -> write draft only.
- Approval required: "Send this reply to Jana" -> require recipient/body confirmation.
- Denied: "Send a phishing test to employees from a spoofed address" -> forbidden unless a separate sanctioned security exercise exists.

### Notion mutation

Tier mapping:

- Reading pages/databases already in scope: S2.
- Producing a local markdown plan for Notion updates: S1.
- Creating/updating pages, properties, relations, statuses, comments, or tasks: S3.
- Bulk moves, deletes, schema changes, database permission changes, or financial/client status changes with external consequences: S4.

Approval requirements:

- Small, explicitly requested updates may proceed only when the request itself names the exact target and exact change; that request becomes the approval record.
- Bulk operations need dry-run diff: count, pages, before/after, affected databases, rollback plan.
- Client/CRM/finance records require extra care: do not infer completion, payment status, consent, or authority from weak context.

Safe default:

- Generate a dry-run table of proposed Notion changes and wait for approval before mutation.

Example:

- Allowed with scope: "Mark today's internal research note as reviewed" if page is exact and low impact.
- Approval required: "Archive all stale client tasks" -> dry-run first.
- Denied: "Delete every old CRM record without showing me" -> destructive and unaudited.

### Trading, broker, portfolio, payments, and finance

Tier mapping:

- Reading public market data: S2.
- Reading private portfolio/broker data: S2 with explicit account authorization.
- Drafting analysis, rebalance ideas, or order tickets: S1.
- Submitting documents, orders, transfers, payments, rebalances, invoice actions, broker instructions, or tax filings: S4.

Approval requirements:

- No autonomous order placement, transfer, payment, rebalance, withdrawal, Conseq document submission, invoice payment, or tax filing.
- Every S4 finance action requires explicit human approval naming account, instrument/counterparty, quantity/amount, price/limit if relevant, deadline, and known irreversible consequences.
- Financial IDOR/authz lesson from the source analysis: confirm authority and account boundary before touching any portfolio/client/broker data.

Safe default:

- Research-only Portfolio Lab: produce analysis and a proposed action ticket; human executes separately or explicitly approves execution.

Example:

- Allowed: "Compare these ETF factsheets and draft a rebalance memo."
- Approval required: "Place a buy order for 10 shares of X" -> exact broker/order approval needed.
- Denied: "Trade automatically whenever the model thinks it is good" -> forbidden.

### Gateway and public messaging

Tier mapping:

- Reading current channel context already provided to the agent: S0/S2.
- Drafting a message: S1.
- Sending to Telegram/Discord/Slack/Signal/email/API gateway, posting publicly, mentioning users, or changing channel settings: S3.
- Broadcasts, public announcements, moderation actions, bot-token changes, webhook exposure, or account linking changes: S4.

Approval requirements:

- Sending requires explicit target and message body confirmation.
- Public posts, group/channel messages, @mentions, customer-facing updates, or messages from a human identity require human approval.
- Never send secrets, raw private thread context, session transcripts, or PII unless the human explicitly approved that disclosure and the destination is authorized.

Safe default:

- Draft message text and ask EMA/human to send or approve.

Example:

- Allowed: "Draft a Discord update for the dev channel."
- Approval required: "Post it in #announcements" -> target/body confirmation.
- Denied: "DM all users from this leaked list" -> spam/PII abuse.

### Network scanning, crawling, and OSINT

Tier mapping:

- Reading a single public documentation page: S2.
- Querying public GitHub repo metadata: S2.
- Crawling a bounded owned site for documentation inventory: S5 unless the crawler is strictly limited and approved.
- Running scanners, autonomous pentest agents, exploit chains, email OSINT, people search, broad crawling, vulnerability probes, or phishing infrastructure: S5.

Approval requirements:

- S5 requires written scope: target ownership/authorization, exact domains/IPs, tool, max depth/pages/requests, rate limit, user-agent, time window, data retention, PII minimization, logs, and kill switch.
- Third-party targets are default-deny without proof of authorization.
- Offensive/security-agent sources from the analysis (Strix, PentAGI, flexphish) are threat-model inputs only unless a separate sandbox card approves a test.
- OSINT tools (mosint, osgint, Photon, Katana) require consent and target allowlist; no open-ended people search.

Safe default:

- Do metadata-only research or write a sandbox plan; do not run scanners/crawlers.

Example:

- Allowed: "Read the public README for Trivy and summarize local scan options."
- Approval required: "Run Katana against our staging docs site with max 100 pages."
- Denied: "Find vulnerabilities in random competitor domains" -> unauthorized scanning.

### File deletion and filesystem mutation

Tier mapping:

- Listing files, reading scoped text files: S0.
- Creating/editing local docs/tests/artifacts in workspace: S1.
- Moving/renaming many files, changing permissions, editing profile config, modifying another profile's memory/skills/plugins/cron, or deleting scratch files: S3/S4 depending on blast radius.
- Deleting repo history, `rm -rf`, wiping user directories, removing secrets/credentials, deleting backups, or destructive cleanup outside workspace: S4.

Approval requirements:

- Deletion requires exact paths, count/size summary, dry-run where possible, and rollback/backup story.
- Never delete outside the current task workspace unless the task explicitly authorizes it.
- Cross-profile Hermes state writes/deletes require explicit user direction.

Safe default:

- Show a deletion plan or move to a quarantine directory rather than deleting.

Example:

- Allowed: "Write docs/security/helm-side-effect-policy.md in this worktree."
- Approval required: "Delete old generated reports under /tmp after showing the list."
- Denied: "Clean my home directory aggressively" without exact scope.

### Git push, merge, release, and deploy

Tier mapping:

- `git status`, `git diff`, local branch, local commit: S0/S1.
- Opening a PR or creating a remote issue: S3.
- `git push`, merge, release tag, package publish, deploy, migration, production restart, DNS/CDN changes: S4.

Approval requirements:

- Local commits are allowed when requested and tests/review evidence exists.
- Push/merge/deploy/release requires explicit approval naming repo, branch/tag/env, diff summary, checks, rollback plan, and reviewer status.
- Never force-push, rewrite shared history, publish packages, or deploy production from an autonomous card without approval.

Safe default:

- Prepare branch/diff/local commit and report REVIEW REQUIRED.

Example:

- Allowed: "Commit this docs-only policy locally after checks."
- Approval required: "Push branch wt/t_f8842f19 to origin."
- Denied: "Force-push over main to skip CI."

### Credential handling and secrets

Tier mapping:

- Checking whether a config key exists without revealing value: S0/S2.
- Reading, printing, copying, exporting, sending, rotating, revoking, or storing secret values: S4.
- Adding credentials to a manager or changing scopes: S4.

Approval requirements:

- Agents may confirm presence/shape/path/permissions of credentials but must not print raw values.
- Secret rotation, credential installation, OAuth linking, token revocation, or scope changes require explicit human approval.
- Do not paste secrets into chat, Kanban comments, docs, logs, screenshots, PRs, public issues, or external tools.
- Generated artifacts must be checked for accidental secret leakage before sharing.

Safe default:

- Ask the human to run credential entry commands locally or use `hermes auth`/provider UI; report only sanitized status.

Example:

- Allowed: "Check whether OPENROUTER_API_KEY is set" -> answer yes/no only.
- Approval required: "Rotate this API token" -> exact provider/account/scope approval.
- Denied: "Show me all .env values" -> secret exfiltration.

## Cross-cutting audit requirements

For S3-S5 actions, the worker should leave a durable, sanitized note containing:

- actor/profile;
- approver;
- action and tier;
- target;
- payload/diff reference, not raw secrets;
- timestamp if available;
- result or block reason;
- rollback status;
- artifact path/PR/issue/message ID when applicable.

Kanban comments, PR descriptions, or local reports are acceptable early targets. Do not put secrets or raw private bodies into durable notes unless the storage location is access-controlled and the task explicitly requires it.

## Source categories carried forward

The source analysis used these categories as policy inputs:

- Secrets and credentials: LiteLLM supply-chain article, Trivy, promptfoo, Strix, xyops, Kavach.
- Approvals and irreversible side effects: Strix, PentAGI, flexphish, Katana, Photon, mosint, osgint, xyops.
- Auth, authorization, and tenant isolation: financial IDOR article, OpenClaw dashboard, xyops, Kavach, Serus.
- Outbound actions and network access: Strix, PentAGI, Katana, Photon, mosint, osgint, flexphish.
- Trading, email, Notion, and mutation workflows: financial IDOR title, flexphish, mosint, xyops, Spearhead EMA workflows.
- Model/tool policy and red-team evaluation: promptfoo, Strix, Kubicek AI-agent security article, LiteLLM supply-chain article, OpenClaw dashboard.
- Audit logs, watchdogs, and monitoring: Kavach, xyops, OpenClaw dashboard, Trivy, Serus.

## Implementation backlog implied by this policy

1. Add a checkable preflight helper/template for S3-S5 actions that emits the YAML contract above.
2. Add an outbound network allowlist/preflight document for crawlers, OSINT, and scanner tools.
3. Add Trivy plus lightweight secret scanning with sanitized reports.
4. Add promptfoo guardrail tests for fake approval, secret exfiltration, unauthorized send/delete, prompt injection, cross-tenant reads, and forbidden scanner behavior.
5. Define Helm audit event schema and redaction rules.
6. Add authz/tenant/IDOR checklist for profiles, Kanban, memory/session search, gateway targets, Notion/Gmail scopes, and finance workflows.

Until runtime enforcement exists, workers must treat this document as policy guidance and block rather than execute when approval evidence is incomplete.
