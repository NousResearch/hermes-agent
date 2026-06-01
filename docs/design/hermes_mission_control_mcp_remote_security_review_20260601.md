# Hermes Mission Control MCP Remote Security Review

Date: 2026-06-01
Status: Phase 5 planning only

This document is the pre-implementation security review for a future remote
OAuth-protected Mission Control MCP bridge. It does not implement OAuth, remote
transport, public routes, or a ChatGPT connection. The current safe default
remains the Phase 4 local/stdout bridge in `hermes_cli/mission_control_mcp.py`.

## Purpose

A future remote OAuth bridge would let an authorized external MCP client read
narrow Mission Control state without using Codex as a relay for status,
worker-result, audit, or next-prompt context. The intended path is:

```text
ChatGPT -> local/stdio or future OAuth MCP bridge -> narrow Mission Control tools -> Hermes dashboard/API state
```

ChatGPT must connect only to a dedicated Mission Control MCP bridge. It must
not connect to the broad Hermes tool registry, dashboard internals, terminal
surfaces, browser-control tools, credential stores, payment/customer systems,
publishing systems, or worker/Codex/Hermes-run dispatch paths.

Remote OAuth is deferred because the current local/stdout bridge has no
network boundary, token lifecycle, issuer/audience validation, remote audit
schema, remote rate limits, reverse-proxy contract, or recovery procedure.
Those controls must exist before any remote endpoint or ChatGPT connector is
implemented.

## Existing Safe Default

- Transport: local stdio only.
- OAuth: disabled.
- Remote/public endpoint: absent.
- Tool surface: narrow Mission Control-only allowlist.
- Packet writes: local packet/audit artifacts only.
- Packet safety posture: `dry_run=true`, `review_required=true`,
  `trusted_for_execution=false`.
- Worker result text: untrusted display data, not executable.
- Broad Hermes registry: not inherited.
- Execution paths: absent for shell, browser/mouse/keyboard/computer-use,
  Codex runs, Hermes runs, worker starts, email, publishing, payments, delete,
  and production mutation.

## Deployment And Network Assumptions

- The dashboard remains private and should continue to bind to loopback by
  default.
- The future remote MCP bridge is a separate service surface, not a dashboard
  route and not a reuse of the dashboard session token.
- The bridge runs under the operator's Hermes profile and reads only the
  Mission Control facade/packet helpers needed by its allowlisted tools.
- Public internet exposure is forbidden until a later explicit security review
  approves it.
- A private network such as Tailscale/Tailnet is the recommended first remote
  boundary if remote transport is ever enabled.
- If a reverse proxy is introduced later, it must terminate TLS, preserve
  request identity metadata in controlled headers, strip untrusted forwarded
  headers from clients, and enforce path-level routing to the Mission Control
  MCP bridge only.

## Binding And Proxy Rules

- Default transport remains local/stdout.
- Remote HTTP/SSE is disabled unless an explicit reviewed config flag exists.
- Loopback bind is the only default bind for any future HTTP/SSE listener.
- Binding to `0.0.0.0` or `::` is forbidden unless a reviewed config requires
  an explicit operator acknowledgement and the deployment is behind an approved
  private network or TLS reverse proxy.
- Browser origins are irrelevant for stdio. If browser transport is ever added,
  allowed origins must be explicit and must not wildcard.
- Health endpoints must not reveal tool lists, token state, file paths, user
  names, provider names, credential paths, dashboard tokens, OAuth tokens, API
  keys, or recent packet/audit contents.
- The broad dashboard API must not be exposed as the remote MCP bridge. The
  bridge must have its own authorization, audit, rate limits, scopes, and tool
  allowlist.

## Forbidden Surfaces

The remote bridge must not expose:

- Broad Hermes tool registry.
- Terminal, arbitrary shell, command execution, file mutation, or patch tools.
- Browser, mouse, keyboard, computer-use, or autonomous-control tools.
- Codex-run, Hermes-run, worker-start, queue dispatch, or packet execution.
- Email send, bulk outreach, publishing, payment, customer, delete, or
  production mutation paths.
- Dashboard session token, API server key, OAuth tokens, cookies, SMTP/Gmail
  credentials, model provider keys, payment/customer credentials, credential
  paths with sensitive values, or raw authorization headers.

Blocked tool names/classes include `send_email`, `publish_video`,
`activate_payment`, `delete_files`, `run_unbounded_codex`, `run_codex`,
`start_codex`, `start_worker`, `start_hermes_run`,
`autonomous_computer_use`, `browser_control`, `mouse_control`,
`keyboard_control`, `start_bulk_outreach`, `arbitrary_shell`,
`reveal_secret`, and `update_credentials`.

## Threat Model

### Assets

- Hermes state.
- Mission Control packets.
- Approval audit logs.
- Worker results.
- Repo/status paths.
- Dashboard session token.
- `API_SERVER_KEY`.
- OAuth tokens.
- Gmail, OpenAI, Anthropic, payment, and customer credentials.
- Production data.
- Publishing and outreach paths.

### Threat Actors

- Unauthorized internet client.
- Compromised browser or chat session.
- Malicious prompt or worker result text.
- Malicious MCP client.
- Stolen OAuth token.
- Confused deputy or tool misuse.
- Local malware/process.
- Accidental operator error.

### Threats And Mitigations

| Threat | Mitigations |
| --- | --- |
| Token leakage | No token logging, redaction tests, profile-local token storage, restrictive file permissions, short-lived access tokens where feasible, rotation and revocation. |
| Prompt injection through worker result text | Worker output remains labeled untrusted display data; no execution tools; schema validation; output redaction; no packet-to-execution path. |
| Tool name spoofing | Static allowlist, unknown-tool rejection, blocked-name tests, no dynamic import from broad registry. |
| Accidental exposure of broad Hermes tools | Dedicated module/service; tests assert broad tools absent; no inheritance from `tools/registry.py` or Codex MCP server. |
| Remote execution escalation | No shell, terminal, browser, Codex-run, Hermes-run, worker-start, send, publish, payment, delete, or production mutation tools. |
| Public dashboard exposure | Keep dashboard private; do not reuse dashboard auth; bridge is separate; loopback/private-network defaults. |
| Write-action abuse | Read-only default; packet-write tools deferred for remote; if later enabled, require write scopes, confirmation, audits, and local packet-only semantics. |
| Packet-to-execution confusion | Preserve `dry_run=true`, `review_required=true`, `trusted_for_execution=false`; UI/API labels must state packets are review artifacts. |
| Audit log poisoning | JSON schema validation, bounded fields, redacted previews, actor/source/session fields, append-only records, controlled error handling. |
| Rate-limit abuse | Per-tool limits, actor/source counters, burst caps, lockout and suspicious activity flags. |
| Credential path disclosure | Redact credential-like fields and sensitive paths; never include token store paths in outputs. |
| Stale/replayed approvals | Timestamped audit events, nonce/request ids for future write confirmations, short-lived sessions if feasible. |

## Future Remote Scope Matrix

Initial remote posture should be read-only and conservative. Packet-write tools
are deferred for remote use until a second review validates confirmation gates,
operator UX, and abuse handling.

| Tool | Remote | Scope | Class | Risk | Rate limit | Audit | Redaction | Confirm | Sensitivity | Failure behavior |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `get_project_status` | Defer, eligible first | `mission_control.read.project_status` | Read | Medium | 30/min actor, 120/min service | Required | Required | No | Medium | Return controlled warning, no partial secrets |
| `get_open_tasks` | Defer, eligible first | `mission_control.read.tasks` | Read | Medium | 30/min actor, 120/min service | Required | Required | No | Medium | Return controlled warning |
| `get_latest_worker_results` | Defer, eligible first | `mission_control.read.worker_results` | Read | High | 15/min actor, 60/min service | Required | Required | No | High | Return untrusted-data warning |
| `get_repo_status` | Defer, eligible first | `mission_control.read.repo_status` | Read | Medium | 30/min actor, 120/min service | Required | Required | No | Medium | Return `not_probed` or controlled warning |
| `get_approval_gates` | Defer, eligible first | `mission_control.read.approvals` | Read | Medium | 30/min actor, 120/min service | Required | Required | No | Medium | Return controlled warning |
| `get_recent_audit_log` | Defer, second read batch | `mission_control.read.audit` | Read | High | 10/min actor, 30/min service | Required | Required | No | High | Return bounded redacted page or warning |
| `list_mission_packets` | Defer, second read batch | `mission_control.read.packets` | Read | Medium-high | 20/min actor, 60/min service | Required | Required | No | Medium-high | Return bounded redacted page |
| `get_mission_packet` | Defer, second read batch | `mission_control.read.packets` | Read | High | 20/min actor, 60/min service | Required | Required | No | High | Return redacted not-found or warning |
| `save_next_codex_prompt` | No, local-only until review | `mission_control.write.prompt_packet` | Write local packet | High | 5/min actor if later enabled | Required | Required | Required | Medium-high | Reject and audit rejected request |
| `import_worker_result` | No, local-only until review | `mission_control.write.worker_result_packet` | Write local packet | High | 5/min actor if later enabled | Required | Required | Required | High | Reject and audit rejected request |
| `save_block_flag_packet` | No, local-only until review | `mission_control.write.block_flag_packet` | Write local packet | Medium-high | 5/min actor if later enabled | Required | Required | Required | Medium | Reject and audit rejected request |

Read-only eligibility does not mean enabled today. It means the tool may be
considered first when remote transport, OAuth, audit, rate limits, and binding
controls exist. All packet-write tools stay local-only for the initial remote
scope.

## OAuth And Token Plan

- Provider/client model: use an explicit OAuth provider/client configuration
  for the Mission Control MCP bridge. Do not reuse dashboard session tokens or
  outbound MCP client tokens without review.
- PKCE: require PKCE for public clients and any authorization-code flow where a
  client secret cannot be protected.
- Token storage: store bridge tokens under the operator's Hermes profile in a
  dedicated auth directory, separate from packet/audit data and separate from
  dashboard session state.
- File permissions: token files must be owner-readable/writable only, and
  parent directories must reject group/world-writable modes where possible.
- Rotation: support access-token expiry, refresh-token rotation, and operator
  initiated rotation.
- Revocation: provide an operator command or documented manual step to revoke
  all bridge tokens and force reauthorization.
- Refresh tokens: never log refresh tokens; encrypt at rest if a platform
  keychain or equivalent profile-local secure store is available.
- Scope naming: use narrow scopes such as
  `mission_control.read.project_status`, `mission_control.read.tasks`,
  `mission_control.read.worker_results`, `mission_control.read.repo_status`,
  `mission_control.read.approvals`, `mission_control.read.audit`,
  `mission_control.read.packets`, and future write scopes only after review.
- Audience/issuer validation: verify issuer, audience, expiry, not-before when
  present, subject, client id, and scopes on every remote invocation.
- Per-invocation verification: no cached authorization decision may bypass
  per-call scope and tool checks.
- Logging: logs must include token ids or hashes only if needed for incident
  response; never log raw tokens, authorization headers, cookies, or secrets.
- Redaction tests: add tests for token-shaped strings, provider keys, dashboard
  session tokens, authorization headers, and credential-like path/value fields.
- Corrupt store recovery: fail closed, ignore unreadable/corrupt token entries,
  write a redacted audit event, and require reauthorization.
- Emergency disable: add a future kill switch such as
  `mission_control.mcp_remote.enabled=false` or an environment override that
  disables remote transport before token parsing.

## Audit And Rate-Limit Plan

Every future remote MCP call must write a structured audit event, including
failed authorization and rejected tool attempts.

Required audit fields:

- `event`: `remote_mcp_call`, `remote_mcp_rejected`, `remote_mcp_rate_limited`,
  `remote_mcp_token_revoked`, or `remote_mcp_config_changed`.
- `timestamp`: UTC ISO-8601.
- `request_id`: generated per invocation.
- `actor_id`: OAuth subject or stable pseudonymous id.
- `client_id`: OAuth client id.
- `source`: remote address or trusted proxy source after proxy normalization.
- `session_id`: bridge session id, never dashboard session token.
- `tool`: requested tool name.
- `scope_required` and `scopes_present`.
- `risk_class`: low, medium, medium-high, or high.
- `decision`: allowed, denied, rate_limited, malformed, or error.
- `request_preview`: bounded, redacted, schema-shaped preview.
- `response_preview`: bounded, redacted result summary.
- `error_code`: controlled symbolic error when relevant.
- `safety`: booleans for `dry_run`, `review_required`,
  `trusted_for_execution`, and `executes_or_dispatches`.

Rate-limit policy:

- Apply per-actor and service-wide limits.
- Use lower limits for audit reads, packet reads, and all future writes.
- Treat unknown tools, blocked tools, malformed inputs, and repeated auth
  failures as suspicious activity.
- Lock out or cool down actors after repeated invalid-token, wrong-scope, or
  blocked-tool attempts.
- Keep alerts local by default: write audit flags and surface dashboard status.
  External notifications require a later review because send paths are
  forbidden in this phase.
- Rotate audit logs by size or age and preserve enough history for incident
  review. Never rotate by deleting the only recent copy without operator
  visibility.

## Failure Modes And Rollback

- Invalid, expired, wrong-audience, or wrong-scope tokens fail closed.
- Unknown tools and blocked tool names fail closed and are audited.
- Missing packet/status sources return controlled warnings, not stack traces.
- Redaction failures fail closed for remote responses.
- Token store corruption disables affected sessions and requires
  reauthorization.
- Rate-limit state corruption fails toward deny for remote calls.
- Reverse-proxy misconfiguration should be treated as deployment-blocking.
- Emergency rollback is to disable remote transport, revoke bridge tokens,
  stop the remote bridge process, keep local/stdout available, and inspect
  audit logs.

## Phase 6 Test Plan

Future implementation tests should cover:

- OAuth is required for every remote call.
- Invalid token rejected.
- Wrong scope rejected.
- Expired token rejected.
- Unknown tool rejected.
- Broad Hermes tools absent.
- Blocked dangerous tool names absent.
- Secret redaction for request previews, responses, errors, and audits.
- Prompt injection in worker result text cannot trigger execution.
- Packet writes, if ever remote-enabled, remain `dry_run=true`,
  `review_required=true`, and `trusted_for_execution=false`.
- Rate limits enforced per actor and service.
- Audit written for every allowed, denied, malformed, and rate-limited remote
  call.
- Dashboard token never appears in remote bridge outputs or audits.
- No arbitrary shell, file mutation, browser control, send, publish, payment,
  delete, Codex-run, Hermes-run, worker-start, or production mutation tool is
  reachable.
- Network binding defaults to local/stdout or loopback-only for future remote
  transport.

## Recommended Next Phase

Phase 6 should add an inert policy/schema and tests before any server code.
The policy should encode the scope matrix, remote-disabled defaults, blocked
tool names, rate-limit defaults, and audit-required flags. Tests should assert
that all write tools are remote-disabled by default and that no forbidden tool
can appear in the policy.

## Phase 6 Implementation Note

Phase 6 adds `hermes_cli/mission_control_mcp_policy.py` as a static inert
remote policy/schema for the Phase 5 matrix. The policy does not start a
server, define a remote transport, implement OAuth, expose a public route, or
connect ChatGPT.

Remote remains disabled for every tool. Read-only Mission Control tools are
marked only as `eligible_first` or `deferred` until future OAuth, audit,
rate-limit, and binding controls exist. Packet-write tools remain
`local_only`, remote-disabled, confirmation-required, audited, redacted, and
non-dispatching.

The Phase 6 tests enforce that forbidden tool names and action classes are
absent, every entry has audit/redaction/rate-limit metadata, no tool executes
or dispatches, no tool exposes secret material, and the policy tool set stays
aligned with the Phase 4 local MCP allowlist.

The next safest step is local MCP client E2E validation against the existing
stdio bridge. Remote OAuth implementation planning should remain separate from
remote exposure and should not add a public endpoint.
