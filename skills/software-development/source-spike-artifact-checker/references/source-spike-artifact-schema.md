# Source-Spike Closure Artifact Schema

This is the minimum closure schema enforced by `scripts/source_spike_checker.py`.

## Files

- `source-spike.md`: detailed extraction artifact.
- `closure-summary.md`: standalone closure gate and downstream handoff.

## Required gates

| Gate | Required evidence | Checker signal |
|---|---|---|
| Artifact reference | `source-spike.md` exists or is explicitly referenced | source-spike artifact reference |
| Provenance source | Source URL or source identifier | provenance: source URL/identifier |
| Provenance retrieval | Retrieval/read method | provenance: retrieval method |
| Provenance revision | Commit/tag/release/date or explicit unavailable rationale | provenance: revision/commit/rationale |
| License | License conclusion, including unknown/mixed if applicable | license/access: license |
| Access | Public/auth/paywall/private/local access conclusion | license/access: access |
| Verdict | `Verdict:` line or `## Verdict` section | verdict |
| Specialist routing | `NO_HANDOFF` or named specialist/lane/profile/domain | specialist routing |
| Downstream decision | close/follow-up/approval/monitor/backlog/next action | downstream decision |
| CLOSE_READY | `CLOSE_READY: yes` or `CLOSE_READY: no` | CLOSE_READY explicit |
| Notion closure | no writes / update required / already updated / out of scope | Notion closure hygiene |
| Import audit summary | If third-party automation is importable/adopted: scanner command, exit code, finding counts, `audit_status`, `risk_level`, credentials/env declarations and mismatches, external service domains, network-write/API-mutation findings, and accepted/rejected findings | import audit sidecar/handoff evidence |

## Import audit summary fields

When the source-spike recommends importing or adapting third-party skill/MCP/workflow/agent-rule/hook/executable prompt content, run or cite `third-party-import-redflag-scan` with `--summary-json` and record these fields in `closure-summary.md`:

- Scanner command and exit code.
- Finding counts: total, info, warn, error, and notable `by_code` counts.
- Derived `audit_status`: `pass`, `review`, `warn`, `malicious`, or `error`.
- Derived `risk_level`: `low`, `medium`, or `high`.
- Credentials/env: declared env/bins/config/install fields, referenced env, and undeclared env mismatches.
- External services: service domains, known paid/billable service documentation, credentials needed, and data-flow notes.
- Network-write/API mutation findings: destination, method/payload boundary if known, and approval requirement.
- Accepted/rejected/remediated findings with reviewer rationale.

`malicious` is a static local-audit status for high-risk patterns, not an external malware verdict. Do not add VirusTotal/external telemetry unless separately approved.

## Closure rules

- `CLOSE_READY: yes` is invalid if any gate is missing.
- A failed checker result should become either a summary patch or a Kanban block with the missing gate named.
- Checker pass does not certify the technical quality of the source extraction; it only certifies handoff completeness.
- Notion bulk edits remain out of scope unless separately approved.
