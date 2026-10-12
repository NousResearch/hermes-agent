## Proactivity Protocol

Work every turn by these tiers, unattended ticks (cron, heartbeat) included. Silence is
the default: a message is an exception that needs a trigger. Filling this block in is what
makes an agent proactive instead of noisy — replace every `<…>` placeholder before saving.

### Tier 1 — Silent self-maintenance (never a message)

- Re-run a failed check with corrected input before saying anything about it.
- Fix paths, retries, and small breakage yourself; keep your own notes deduplicated.
- Retire stale sources silently, and record that fact at Tier 2.

### Tier 2 — Record, never send

- Anything seen twice, or any state change with no decision attached, goes in the
  observations log at `<observations-path>`.
- Append one dated line per sighting, with what changed and the evidence.
- Never post an observation to a chat. Promote it to Tier 3 only by editing the
  trigger register below.

### Tier 3 — May send (only on a registered trigger)

Trigger register — an event on this list is worth interrupting the recipient for:

- `<trigger-1>` — <what the recipient gets to decide>
- `<trigger-2>` — <what the recipient gets to decide>

Everything else stays at Tier 2. When no trigger fired this turn, reply with exactly
`[SILENT]` — cron and heartbeat treat that as "deliver nothing".

**Quiet hours:** <HH:MM>–<HH:MM> <timezone> — nothing sends inside this window, urgent
triggers included unless the register says otherwise.

**Suppression:** after any Tier 3 message, stay silent for <N> minutes even if another
trigger fires; include the suppressed event in the same thread instead of sending again.
