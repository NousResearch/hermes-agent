---
name: meeting-prep-card
description: Use when preparing privacy-safe meeting context cards.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [meetings, productivity, privacy, briefings]
    related_skills: []
    requires_toolsets: [terminal]
    category: productivity
---

# Meeting Prep Card

## Overview

Meeting Prep Card creates a concise, source-backed context card before a meeting using a synthetic or pre-normalized JSON fixture. It is designed for privacy-sensitive workflows: it renders public-safe Markdown or JSON, hides meeting links, hashes source IDs, and redacts risky identifiers before output.

This skill does **not** read calendars, mailboxes, chat archives, CRMs, or note systems directly. Use separate, explicitly approved read-only collection steps to prepare a normalized fixture, then run this renderer locally.

## When to Use

Use this skill when the user wants to:

- Prepare a one-screen meeting context card from already-normalized source data.
- Test a privacy-safe card shape before wiring live source adapters.
- Convert notes, email metadata, chat excerpts, CRM summaries, or manual context into a read-only meeting brief.
- Produce a local Markdown or JSON artifact that can be reviewed before sending or posting anywhere.

Do **not** use this skill for:

- Live data ingestion without separate approval.
- Sending messages, drafting outbound communication, or changing calendar/CRM records.
- Persisting raw private message bodies, raw source IDs, or meeting URLs.
- Large relationship dossiers; keep the output to one actionable screen.

## Privacy Contract

Treat every output as user-visible by default.

The helper script enforces these rules:

1. Fixtures must be synthetic or pre-sanitized before commit.
2. Markdown and JSON pass through the same sanitizer.
3. Raw URLs are rendered as labels such as `[meeting link available — hidden]`.
4. Phone numbers, chat IDs, email addresses, token-like strings, bearer tokens, and Slack broadcast mentions are redacted.
5. Source IDs are never shown directly; public JSON uses short `safe_ref` hashes instead.
6. The renderer is read-only. It never sends messages, posts to Slack, edits calendar events, creates tasks, or writes CRM records.

If strict mode fails, stop and inspect the fixture. Do not bypass strict mode for user-visible output.

## Fixture Schema

The fixture shape is:

```json
{
  "now": "2026-05-28T12:00:00+00:00",
  "events": [
    {
      "id": "evt_acme",
      "title": "Acme partner intro",
      "start": "2026-05-28T15:00:00+00:00",
      "end": "2026-05-28T15:30:00+00:00",
      "status": "confirmed",
      "attendees": [
        {"name": "Host", "email": "host@example.internal"},
        {"name": "Sam Contact", "email": "sam@acme.example"}
      ],
      "location": "Video call",
      "url": "https://meet.example/redacted"
    }
  ],
  "evidence": [
    {
      "source": "notes",
      "source_id": "raw_SYNTHETIC1234",
      "label": "prior meeting note",
      "timestamp": "2026-05-21T11:00:00+00:00",
      "confidence": "clear",
      "kind": "open_loop",
      "text": "Host promised to send an intro and request the latest deck.",
      "people": ["sam@acme.example"],
      "company": "Acme"
    }
  ]
}
```

Recommended source names: `calendar`, `notes`, `email`, `chat`, `crm`, `manual`.

Recommended evidence kinds: `last_touch`, `open_loop`, `waiting_on`, `decision`, `risk`, `doc`, `context`, `note`.

## How to Run

From a repo checkout or installed skill directory:

```bash
python3 optional-skills/productivity/meeting-prep-card/scripts/meeting_prep_card.py \
  --fixture optional-skills/productivity/meeting-prep-card/templates/sample_fixture.json \
  --event-id evt_acme \
  --format markdown \
  --strict
```

For JSON:

```bash
python3 optional-skills/productivity/meeting-prep-card/scripts/meeting_prep_card.py \
  --fixture optional-skills/productivity/meeting-prep-card/templates/sample_fixture.json \
  --event-id evt_acme \
  --format json \
  --strict
```

When installed into a user skill directory, adapt the path:

```bash
python3 ~/.hermes/skills/productivity/meeting-prep-card/scripts/meeting_prep_card.py \
  --fixture /path/to/synthetic_fixture.json \
  --format markdown \
  --strict
```

## CLI Options

| Option | Purpose |
|---|---|
| `--fixture PATH` | Required normalized JSON fixture. |
| `--event-id ID` | Render one event from the fixture. |
| `--format markdown\|json` | Output format; Markdown is default. |
| `--include-internal` | Include internal-only meetings that normally skip. |
| `--max-chars N` | Markdown character budget; default 1500. |
| `--now ISO` | Override fixture time for deterministic tests. |
| `--output PATH` | Write local output instead of stdout. |
| `--strict` | Fail if rendered output contains risky patterns. |

## Procedure

1. Gather source context only inside the approval scope already granted.
2. Normalize into the fixture schema without committing raw private data.
3. Run the Markdown command with `--strict`.
4. If the card is too long, lower `--max-chars` or reduce evidence items in the fixture.
5. Run the JSON command with `--strict` if another system needs structured output.
6. Inspect both outputs before sending or posting anywhere.
7. Stop before live data reads, cron scheduling, public posting, or external mutations unless the user explicitly approves that exact boundary.

## Output Shape

Markdown includes:

- meeting title and time;
- external attendees by safe display name;
- hidden meeting-link label, never a raw URL;
- top warning when context is missing or stale;
- last touch;
- up to two open loops or risks;
- up to two context bullets;
- suggested move;
- source counts and read-only mutation status.

Public JSON includes the same sanitized card plus safe source references. It intentionally excludes raw `source_id` fields and raw meeting URLs.

## Common Pitfalls

1. **Committing real fixtures.** Keep real source snapshots out of git. Commit only synthetic examples.
2. **Treating JSON as private.** JSON often gets logged, copied, or attached. It must be sanitized like Markdown.
3. **Using raw source IDs as evidence pointers.** They may be retrievable handles. Use `safe_ref` hashes in public artifacts.
4. **Letting a card imply approval.** Suggested moves are not actions. External sends require exact target, channel, and draft approval.
5. **Expanding into live adapters too early.** Prove fixture output first, then design read-only adapters with separate tests and approval gates.

## Verification Checklist

- [ ] Fixture is synthetic or pre-sanitized.
- [ ] `--strict` passes for Markdown.
- [ ] `--strict` passes for JSON.
- [ ] Output contains no raw URLs, phone numbers, email addresses, chat IDs, tokens, or source IDs.
- [ ] Output states no messages were sent and no external records changed.
- [ ] The user has separately approved any live read or external posting step.
