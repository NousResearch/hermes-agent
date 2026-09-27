# Person identity

Status: behavior agreed; runtime implementation pending.

## Decision

Use stable person records scoped to the employee's Hermes profile. Platform
identities map to those records; display names are labels, never identity keys.

- A Telegram sender keeps the same personal memory across DMs and groups for
  the same employee. Resolve the sender through trusted native channel metadata,
  not names supplied in message text or model-generated identifiers.
- Link identities across Telegram, CLI, desktop and other channels explicitly.
  Never merge people because their display names match.
- Local CLI/desktop conversations bind a configured owner to a person record.
  That record can be explicitly linked to the owner's Telegram identity.
- Keep identity mappings and personal memory isolated between Hermes profiles.

## Model-facing behavior

Preserve employee-style sender labels and the `memory` tool's person selector.
A selector resolves against recorded participants; unknown or ambiguous labels
produce corrective errors rather than writing to another person's profile.
Load only the current speaker's personal profile as current-user context, in
[the agreed position](tool-surface.md) after their original message content.

Unattended work must not invent a current person. Preserve its shared-memory
default when no authenticated person is bound.

## Native integration

Use existing Telegram sender/session metadata and the shared gateway/memory
paths. Do not replace the Telegram adapter to introduce person records. Exact
local persistence and identity-linking configuration remain implementation
work. Person identity alone does not grant action permissions or authorize
cross-person disclosure.

## Verification when implemented

Verify one sender has the same profile in DM and group contexts, two people
with the same display name remain distinct, explicit links preserve identity
across local and messaging surfaces, and switching speakers or Hermes profiles
does not load or mutate the wrong personal memory.
