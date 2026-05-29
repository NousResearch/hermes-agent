# Closure summary: <source name>

Task: `<kanban-task-id>`
Artifact directory: `<absolute artifact directory>`
Primary artifact: `source-spike.md`

## Status

CLOSE_READY: no
Verdict: <ADOPT | ADOPT SELECTIVELY | SPIKE | MONITOR | NO_ADOPT | REJECT>

## Provenance

- Source inspected: <URL/source identifier>
- Retrieval method: <git ls-remote + shallow clone | GitHub API/raw | browser | local file | other>
- Revision inspected: <commit/tag/release/date, or `not available: <reason>`>
- Files/areas inspected: <README, LICENSE, src paths, docs, etc.>

## License/access

- License: <license conclusion, including unknown/mixed if applicable>
- Access: <public | auth required | paywalled | private/local | other>
- Third-party license/access risks: <none known | list risks | not reviewed>

## Import audit summary

Use when the source recommends importing/adapting third-party skill, MCP, workflow, agent rule, hook, or executable prompt content. If not applicable, state `Not applicable: no importable automation content`.

- Scanner command: <exact `third-party-import-redflag-scan` command, preferably with `--summary-json`, or `not run: <reason>`>
- Scanner exit code: <0 | 1 | 2 | 3 | not run>
- Finding counts: <total/info/warn/error and notable by_code counts>
- Audit status / risk level: <pass|review|warn|malicious|error> / <low|medium|high>
- Credentials/env: <declared env/bins/config/install fields; referenced env; undeclared env mismatches; credentials needed>
- External services: <domains, paid/billable service notes, data-flow boundaries, or none>
- Network-write/API mutations: <findings and approval requirement, or none>
- Accepted findings: <accepted with rationale, or none>
- Rejected/remediated findings: <rejected/remediated with rationale, or none>

## Verdict and extraction

<One paragraph or bullets explaining what to adopt/reject/monitor and why. Make clear whether this is pattern extraction, implementation candidate, or no-op closure.>

## Specialist routing

- Handoff: <NO_HANDOFF | Gond implementation | research | security | UX | data | Notion/admin | other>
- Reason: <why this lane is or is not needed>

## Downstream decision

- Next action: <close | create follow-up card | needs EMA/Filip approval | monitor | backlog>
- Follow-up scope if any: <bounded scope, non-goals, evidence required>

## Notion closure

- Notion status: <no Notion writes performed | Notion update required | Notion already updated | Notion deliberately out of scope>
- Bulk-edit approval: <not requested | approved by ... | not applicable>

## Non-goals confirmed

- No external installs performed unless explicitly stated.
- No Notion bulk edits performed unless explicitly approved.
- No code imported/copied unless explicitly stated.
- No credentials or private tokens logged.

## Risks/gaps

- <remaining uncertainty or `none known`>
