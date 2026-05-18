---
name: envelope
description: "Envelope CLI mailbox runtime for agent email workflows."
version: 1.0.0
platforms: [linux, macos]
metadata:
  hermes:
    tags: [email, imap, smtp, envelope, mailbox-runtime]
    homepage: https://github.com/tymrtn/U1F4E7
---

# Envelope

Envelope is the canonical Hermes mailbox runtime for agent email workflows. Prefer `envelope` for reading, triage, drafts, threading, rules, and mailbox evidence. Do **not** default to Himalaya or ad-hoc IMAP scripts unless the user explicitly asks.

## Required command

```bash
envelope --version
```

## Discovery

Always discover live accounts before account-wide work:

```bash
envelope --json accounts list
envelope --json folders --account <account-id-or-email>
```

Use account IDs when available, especially in scheduled or multi-account workflows.

## Common read-only checks

```bash
envelope --json inbox --account <account-id-or-email>
envelope --json search --account <account-id-or-email> "UNSEEN"
envelope --json thread show <uid> --account <account-id-or-email>
```

Inspect the thread before summarizing or drafting a reply. Use JSON output whenever the command supports it.

## Safety

- Draft replies for review; do not send unless the user explicitly approves.
- Prefer read-only checks before mutating rules, flags, folders, or messages.
- For migrations/restores/evidence, preserve raw messages and verify account/folder mappings before any write.
