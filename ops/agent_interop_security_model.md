# Hermes Agent Interop Security Model

Status: draft / low-risk implementation artifact  
Source reference: `/Users/travis/Documents/obsidian/References/2026-04-25 Agent Interop Background Computer Use and Permission-Hardened Daemons.md`

## Purpose

This document turns the three referenced mechanisms into a safe Hermes/OpenClaw adoption boundary:

1. Agent-to-Agent Protocol: live-session peer communication, not disposable subagent spawning.
2. Background computer-use: non-foreground-hijacking app operation, only after observability and approvals exist.
3. Permission-hardened daemon: execution policy, budget telemetry, and durable identity/state reporting.

The immediate goal is not to enable new remote control. The immediate goal is to define the gates that must exist before enabling it.

## Non-goals

- Do not install or enable `hermes-a2a` blindly.
- Do not grant A2A peers access to private memory, diary-like context, credentials, or full session transcript by default.
- Do not enable background control of Messages, mail, browser profiles, or personal apps without explicit approval gates.
- Do not store secrets in Markdown, memory, logs, reports, or A2A transcripts.

## Threat model

### Assets

- Live Hermes session identity and persona.
- User private messages and platform-specific context.
- Persistent memory, skills, Obsidian notes, and credential file paths.
- Tool access: terminal, files, browser, Telegram, cron, MCP, and future A2A peers.
- Cost and token budget for long-running agent work.

### Main risks

- Prompt injection through inbound A2A messages.
- Impersonation of trusted peers.
- Replay of previously valid webhook payloads.
- Cross-session privacy bleed: A2A context entering normal chat or memory without labels.
- Outbound leakage of credentials, private messages, or sensitive file contents.
- Tool escalation: remote peer convinces live session to call high-impact tools.
- Cost exhaustion through repeated inter-agent messages.
- Hidden background GUI action that surprises or disrupts the user.

## Required controls before A2A enablement

### Identity and authentication

- Maintain an explicit `allowed_peers` list. Each peer needs:
  - stable peer id
  - display name
  - allowed ingress routes
  - allowed outbound routes
  - policy profile
  - human owner/contact
- Require Bearer token authentication for API ingress.
- Require HMAC signature for webhooks.
- Include timestamp and nonce in signed payloads.
- Reject payloads outside a short clock-skew window.
- Keep replay nonce cache at least as long as the skew window.

### Privacy boundary

- Store A2A transcripts separately from normal chat history.
- Label every A2A message with source peer, target agent, route, thread id, and policy profile.
- Do not add A2A content to persistent memory automatically.
- Do not include private user-session tail in A2A replies unless the local user explicitly requested that handoff.
- For Hermes/OpenClaw, treat peer messages as external input even if both agents are local.

### Tool and permission policy

Every inbound A2A message must resolve to an execution policy before tools are available.

Policy fields:

- `read_roots`: allowed filesystem read prefixes.
- `write_roots`: allowed filesystem write prefixes.
- `blocked_roots`: sensitive paths that remain blocked even under broad roots.
- `allowed_toolsets`: toolsets the peer may request indirectly.
- `external_send_allowed`: whether the session may send to Telegram/email/social surfaces because of this peer.
- `dangerous_command_mode`: deny, ask, or scoped allow.
- `credential_policy`: never plaintext; file path only if already authorized.
- `max_tokens_per_day`: budget ceiling for peer-originated work.
- `max_tool_calls_per_message`: bounded tool loop limit.

Default policy should be read-only, no external sends, no credentials, no shell writes.

### Prompt-injection and output controls

Inbound filter:

- Strip or quarantine instructions that claim to override Hermes system/developer/user instructions.
- Preserve the raw message in transcript for audit, but pass a sanitized body to the live session.
- Mark all peer-provided content as untrusted.

Outbound filter:

- Redact credential-like strings, bearer tokens, API keys, cookies, and private key blocks.
- Redact paths under known credential directories unless sharing a credential file path is explicitly allowed.
- Block raw private message dumps unless approved by the local user.

### Observability

Every A2A-triggered task should emit a reportable event with:

- event id and timestamp
- source peer and target agent
- route and conversation id
- policy profile used
- toolsets enabled
- read/write scope used
- external systems contacted
- approval decisions
- token/cost estimate when available
- redaction count
- result status

## Background computer-use gates

Background computer-use is useful for QA/demo/app inspection, but high-risk for personal apps.

Minimum gates:

- separate `background_computer_use` toolset, disabled by default
- visible session indicator / report entry
- app allowlist
- foreground-hijack prevention check
- screenshot/video artifact retention policy
- no Messages/mail/social actions without explicit approval
- stop/interrupt path tested before use

Recommended first use cases:

- visual QA on a local dev app
- demo capture
- hidden browser read-only inspection

Avoid first:

- personal messaging
- file deletion or account settings
- payment, purchasing, admin panels

## Token budget and Auto-Concise policy

Implement gradually. Avoid fake precision.

- Track approximate input/output tokens by session and by peer-originated task.
- Define daily budget by route: normal chat, cron, A2A, background computer-use.
- At 70% of budget, switch long-running autonomous work to concise progress notes and lower-context prompts.
- At 90%, stop nonessential autonomous expansion and ask/notify.
- Reports should say whether token counts are measured or estimated.

## Adoption sequence

1. Keep this document and schema as review artifacts.
2. Add offline validators for policy/report artifacts.
3. Add task observability fields before enabling new ingress.
4. Prototype A2A transcript store and peer policy loading with no live ingress.
5. Add HMAC/Bearer verification tests.
6. Add inbound/outbound redaction tests.
7. Only then wire a disabled-by-default A2A platform adapter.
8. Background computer-use remains a later, separately gated experiment.

## Definition of ready for live A2A trial

- Security model reviewed.
- Peer policy schema validated offline.
- Transcript storage separated from normal memory.
- Auth/signature/replay tests pass.
- Redaction tests pass.
- Task observability includes peer, policy, tools, scopes, approvals, and budget.
- Manual kill switch exists.
- First peer is local/test-only, not public internet exposed.
