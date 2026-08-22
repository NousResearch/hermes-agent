# Continuity JSON — Schema & Parsing Pitfalls

This reference documents edge cases discovered when building a continuity-file reader for the LAH Stack mcporter system. If you maintain or extend a continuity reader (`readContinuityFiles`, `getContinuitySummary`, or equivalent), study these carefully.

## Pitfall 1: File naming filter

The continuity reader filters files by name — usually checking for `CONTINUITY` in the filename. But smoke-lock files may use `SMOKE_LOCK` instead:

```
✅ docs/mcporter/CLOE_CLI_OPENCLAW_BRAIN_WIRING_ENABLEMENT_CONTINUITY.json     (caught)
❌ docs/mcporter/CLOE_OPENCLAW_BRAIN_RUNTIME_ALIGNMENT_SMOKE_LOCK.json          (missed)
```

**Fix:** Accept both patterns in the filter:
```javascript
if (!entry.name.includes('CONTINUITY') && !entry.name.includes('SMOKE_LOCK')) continue;
```

## Pitfall 2: `new_verdict` vs `final_verdict`

Some continuity files use `new_verdict` instead of `final_verdict` for the current state. The reader must parse both:

```json
// Standard continuity:
{ "final_verdict": "CLOE_CLI_OPENCLAW_BRAIN_WIRING_READY_OPERATOR_ENV_REQUIRED" }

// Smoke lock (no final_verdict — uses new_verdict):
{ "new_verdict": "CLOE_CLI_OPENCLAW_BRAIN_WIRING_LIVE_VALIDATED" }
```

**Fix:** Accept `new_verdict` as fallback:
```javascript
const finalVerdict = data.final_verdict || data.new_verdict || null;
```

## Pitfall 3: Sort by timestamp, not (just) alphabetically

Alphabetical sort by `mission` name is unreliable — a smoke lock may not follow the `CLOE_CLI_*` naming convention of earlier continuity files:

```
Alphabetical (descending)            Correct (by timestamp desc)
─────────────────────────            ─────────────────────────
CLOE_OPENCLAW_...SMOKE_LOCK  ← 1st   CLOE_OPENCLAW_...SMOKE_LOCK     (t=03:28)
CLOE_CLI_OPENCLAW_...ENABLEMENT  ← 2nd CLOE_CLI_OPENCLAW_...ENABLEMENT (t=03:05)
CLOE_CLI_LLM_BRAIN_...V1              CLOE_CLI_LLM_BRAIN_...V1
```

But alphabetical can also be _wrong in the other direction_ if the smoke lock's name starts with a letter that sorts after the CLI entries. **Always use timestamp-based sorting**:

```javascript
results.sort((a, b) => {
  if (a.timestamp && b.timestamp) {
    return b.timestamp.localeCompare(a.timestamp);
  }
  if (a.timestamp && !b.timestamp) return -1;
  if (!a.timestamp && b.timestamp) return 1;
  return b.mission.localeCompare(a.mission);
});
```

**Key rule:** every continuity JSON should have a `timestamp` field. Without it, the file sorts below any timestamped file, which is almost certainly wrong.

```json
"timestamp": "2026-07-05T03:28:00.000Z"
```

## Pitfall 4: Adding new metadata fields

A smoke lock may carry fields that older continuity files don't have (provider, model, patch_commit, cloe_live_marker, runtime_container). The reader's parser must be extended to extract these:

```javascript
results.push({
  // ... existing fields ...
  provider: data.provider || null,
  model: data.model || null,
  patch_commit: data.patch_commit || null,
  cloe_live_marker: data.cloe_live_marker || null,
  runtime_container: data.runtime_container || null,
});
```

If the summary display function (`getContinuitySummary`) also needs updating, add display logic for the new fields:

```javascript
if (f.patch_commit) {
  lines.push(`  Patch: \`${f.patch_commit}\``);
}
if (f.provider) {
  lines.push(`  Provider: ${f.provider}`);
}
```

## Pitfall 5: Summary block based on latest verdict

After listing continuity files, the summary may include a block that reacts to the latest record's verdict. These verdict-inspection checks must be kept in sync with the actual verdict strings used:

```javascript
const latest = files[0];
if (latest?.final_verdict) {
  if (latest.final_verdict.includes('LIVE_VALIDATED') || latest.final_verdict.includes('BRAIN_WIRING_LIVE_VALIDATED')) {
    lines.push('✅ **Brain**: Connexion OpenClaw brain validée en live.');
  }
}
```

Add new detection patterns as new verdict strings are introduced. Don't assume all legacy patterns will match.

## Verification checklist

When you change a continuity reader, run these checks in order:

- [ ] `node --check <reader-file>` — syntax
- [ ] `node --test <test-file>` — all existing continuity tests pass
- [ ] Smoke: run `node cloe.mjs "où on en est sur ClawX ?"` (or equivalent) and verify:
  - The most recent continuity file is listed first
  - Its verdict is correct (parsed from `final_verdict` or `new_verdict`)
  - New fields (provider, model, etc.) appear in the output
  - The summary block at the bottom reflects the latest state accurately