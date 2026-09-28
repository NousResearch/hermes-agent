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
with free-response exceptions for chats/topics. Use the native dashboard Config editor; no custom toggle or adapter is needed.
The settings are `telegram.require_mention`, `telegram.free_response_chats` and
`telegram.free_response_topics`. Preserve native accepted formats, parsing and
reload/restart behavior. Allowlisting a group and choosing when to respond are
separate settings. Avoid competing environment overrides that hide saved edits.

## Administration and fixed rules

Reuse the native dashboard, including API-key management, Telegram setup and
configuration editing. Do not port the hosted dashboard or build member accounts.
Use native authentication; Railway access/auth topology is a deployment check.

The effective runtime enforces the fixed tool, knowledge and memory contracts
even if configuration asks for incompatible values. Keep one native config
resolution path with explicit product-policy enforcement, not per-consumer
fallbacks or an unrelated second configuration stack. Unsupported overrides
must be visible as fixed/ignored or rejected, not silently presented as applied.

Model choice, employee name and custom instructions remain configurable. Prompt-
affecting edits follow native cache-safe session boundaries; do not rewrite a
warm conversation's prompt merely because a setting was saved. Credentials,
channel access and service endpoints remain deployment/operational settings.
The chosen provider defaults and required authentication routes are in
[deployment](deployment.md). Never silently select Nous-managed services when
one of those direct providers is unavailable.
