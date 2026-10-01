# Runtime defaults

Status: display, approval and configuration-access decisions agreed. Other
values below are inherited baseline defaults under the preservation policy;
backend-specific compatibility is checked before implementation.

Implemented: native Telegram streaming default off (other agreed display defaults
already native), medium reasoning, compression at 0.85 with three tail user
messages, approvals off, quiet restart/transcript notices and local STT selection.
Explicit operator values remain supported. No custom Telegram renderer or config
loader was added. Shared conversation defaults and product rules are implemented; deployment
files are prepared. Live server validation remains pending.

Use the employee runtime configuration as the comparison baseline, adapting it
to native Hermes settings and the separately agreed providers. Do not copy the
hosted runtime or custom display implementation. Fixed product behavior remains
code-owned; display and operational preferences use native configuration.

## Telegram display — agreed

```yaml
display:
  platforms:
    telegram:
      tool_progress: "off"
      interim_assistant_messages: true
      show_reasoning: false
      streaming: false
```

Show the employee's ordinary spoken progress updates and completed replies,
without native per-tool breadcrumbs, arguments or command previews. Retain native
rendering. This supersedes the unaccepted compact `new`/accumulate/cleanup
proposal; those settings were not approved.

## Employee baseline defaults

- Main-agent reasoning effort: medium.
- Compression threshold: 0.85; retain at least three tail user messages.
- Execution approvals: agreed off (`approvals.mode: off`), matching the employee.
  Session-command and MCP reload confirmations also default off
  (`approvals.destructive_slash_confirm: false`, `approvals.mcp_reload_confirm: false`).
  Previously these commands asked even with execution approvals off; they now run
  immediately unless the operator explicitly enables their confirmation setting.
  Prompt-level action authorization and unconditional native command restrictions
  remain in effect.
- Shared group/thread sessions. Observe unmentioned group messages as context
  where native access and Telegram visibility permit. Mentions/replies are the
  baseline trigger, with native chat/topic overrides below; do not hardcode this
  mode for every group.
- Quiet operational notices: reference disables gateway restart notifications
  and transcript echo. These are separate from assistant progress updates.

Do not copy the reference's fixed context length or provider-specific caching
parameters without checking the selected Codex model and backend support.

## Channel response policy

Response mode must remain configurable per conversation location, not hardcoded
to mentions-only for every group. Native Telegram supports a mention requirement
with free-response exceptions for chats/topics. Use the [hosted settings UI](hosted-settings.md); native configuration remains authoritative.
The settings are `telegram.require_mention`, `telegram.free_response_chats` and
`telegram.free_response_topics`. Preserve native accepted formats, parsing and
reload/restart behavior. Allowlisting a group and choosing when to respond are
separate settings. Avoid competing environment overrides that hide saved edits.

## Administration

Use native dashboard, configuration, CLI commands, authentication and secret
management. The hosted settings UI adds focused routes over these owners and
shared-admin password rotation. There are no fork config/auth/guide write
prohibitions. Selected native UI controls are hidden.

The model tool filter and disabled skill loading remain product differences.
Memory/provider settings are native; Railway config selects Hindsight and
Browser Use. Use native SOUL/personality for identity. Final prompt exceptions
remain open. Prompt-affecting changes follow native cache-safe boundaries.
