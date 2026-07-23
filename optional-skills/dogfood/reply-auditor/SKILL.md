---
name: reply-auditor
description: Pre-send audit for assistant replies. Detects unsourced "done/fixed/listo/working" claims before the agent commits them to the user. Companion to iron-laws (Law 1: Show, don't claim). Load when a response is approaching completion-claim shape.
version: 1.0.0
author: shootzjmr
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [dogfood, self-correction, audit, iron-laws, bash]
    related_skills: [iron-laws, self-audit]
---

# Reply Auditor

> Born from the same failure that gave us Iron Law 1.
> This is the mechanical enforcement of that law.

## When to load

Load this skill when **any** of these are true:

- The draft response contains a word like `done`, `fixed`, `listo`,
  `working`, `ready`, `configured`, `complete`, `set up`.
- The response is longer than ~300 characters.
- The agent is about to declare a service healthy / a task complete.
- The agent is about to push, rotate credentials, or run a migration.

If none apply, do not load — the cost of a false-positive audit is
real conversational friction.

## What "auditing" means

Pass the draft response (file or stdin) through:

```bash
bash scripts/verify_done.sh <draft.md>
# or
echo "$DRAFT" | bash scripts/verify_done.sh -
```

The script returns:

- **Exit 0** + `✓ No unsourced completion claims` — reply is clean.
- **Exit 1** + line-by-line claims — agent must either add evidence
  or rephrase.

## What counts as "proof"

The auditor accepts these as evidence that a claim is grounded:

| Evidence | Example |
|----------|---------|
| Inline code | `` `exit code 0` `` |
| Fenced code block | ```` ``` ... ``` ```` |
| Dollar command | `$ docker ps` |
| HTTP status | `HTTP/1.1 200`, `HTTP %{http_code}` |
| Exit code | `exit code 0` |
| System marker | `ENOENT`, `Permission denied`, `Connection refused` |
| Prompt context | `root@hostname:`, `$` followed by command name |
| Visual marker | leading `✓`, `✗`, `❌` |
| API response | `Status: 200`, `Status: 404` |

A claim is allowed if **any** of these appears within `VERIFY_DONE_WINDOW`
(default 5) lines before or after the claim.

## Anti-patterns the auditor catches

| Phrase | Why it fails |
|--------|--------------|
| "Listo, ya quedo todo configurado." | No command output shown |
| "Done. Working." | Decorative, no proof |
| "It's set up." | No diff, no file, no output |
| "Fixed." | No patch, no test, no run |
| "I configured Vaultwarden." | No config file diff, no log line |

## Anti-patterns the auditor does NOT catch (be aware)

These pass the regex by design — they're false-positive-prone:

- Third-party quoted output (e.g., a vendor's "Setup complete" message)
- Citations from external docs that say "done" in flowing prose
- Lines where the proof is *visually* before but lexically far

See `references/false-positives.md` for the full list and how to suppress.

## The workflow

1. Compose the reply.
2. Save to a temp file (or pipe via stdin).
3. Run `bash scripts/verify_done.sh <file>`.
4. If exit 1 → fix the flagged lines before sending.
5. If exit 0 → send.

In Hermes runtime this maps to:

- Pre-send hook: agent runs audit automatically before
  `send_message()` for any reply > 300 chars.
- Manual: agent calls the script from a tool execution step.

## Why not just check logs later?

Because the user reads the message first. By the time a log audit
finds a contradiction, the user has already been misled. The damage
is done: trust ticks down, even if the lie is later corrected.

Pre-send audit is a tradeoff: the auditor occasionally false-positives
on innocent language ("I haven't done X yet"), and that costs us
conversational friction. We accept that cost because the cost of
sending a confident-sounding claim without proof is much higher.

See `references/why-this-design.md` for the full rationale.

## Acceptance test

```bash
bash tests/test_verify_done.sh
```

Runs eight scenarios:

1. Bare claim → flag (exit 1)
2. Claim with `$ docker ps` → pass (exit 0)
3. No claims → pass
4. Mixed (only one bare claim) → flag the bare one
5. Exit code as proof → pass
6. HTTP status as proof → pass
7. Empty message → pass
8. `--help` flag works

## Related

- `iron-laws` (Law 1 lives there; this skill is its operational companion)
- `references/false-positives.md` — when the regex is wrong
- `references/why-this-design.md` — design rationale

## Provenance

Originally shipped in `hermes-iron-laws v0.1.0` as a log-scraper.
That design produced false positives on every INFO line that
contained the word "done" or "ready". Rewritten in v0.2.0 to audit
only agent-voice messages. Promoted to standalone skill v1.0.0 here.

Companion projects:
- https://github.com/shootzjmr/hermes-iron-laws
- https://github.com/shootzjmr/reply-auditor

## License

MIT — same as the parent project.
