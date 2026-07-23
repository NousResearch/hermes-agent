# Adapter boundaries

Every adapter call receives one exact connected-account record and declares
capabilities such as `contacts.read`, `messages.read`, or `messages.send`.
Unsupported operations fail explicitly. There is no default-account fallback.

- Facebook wraps the existing verified local CRM/browser domain; migration
  opens the legacy database read-only and never enables write actions.
- Telegram Communication covers private contacts/dialogs and is separate from
  Telegram News sources, stories, claims, and editorial state.
- VK follows the same account-scoped read contract.
- Dating adapters are read-only and require a named, user-confirmed pilot plus
  a separate browser profile and test account.
- Only `FakeCommunicationAdapter` can execute the test outbox. Production
  adapters cannot cross that boundary during this workflow.

Inspect adapter health and capabilities through the CLI. Do not call adapter
Python APIs directly from the skill.
