# Provider/auth/fallback change reporting pattern

Use this reference when a Strategic-mode task touches live Hermes provider routing, auth credentials, fallback chains, or gateway reloads.

## Trigger

Apply after an approved provider/auth/fallback change, especially when the work also affects tests, gateway command routing, or fleet-wide runtime config.

## Required sequence

1. Confirm the change was explicitly approved or falls under a direct capital-letter execution instruction.
2. Back up the affected config/auth files before editing.
3. Make the smallest routing/auth change that fixes the root issue.
4. Remove stale failed credentials only when they are clearly redundant and the active credential path is preserved.
5. Verify with live commands, not assumptions:
   - provider/status/auth listing
   - config assertions for default provider/model and fallback order
   - relevant unit tests
   - command alias/registry checks if command routing changed
   - gateway service health if a reload/restart was approved
6. Write a concise governance report under `~/.hermes/governance/reports/` with:
   - changes made
   - backup location
   - verification outputs summarised
   - remaining non-blockers
   - explicit note for any provider left unavailable by design

## Output style

Final user report should be concise and execution-backed. Do not over-explain. Include exact file paths and real verification results. Separate:

- fixed
- verified
- remaining non-blockers

## Pitfalls

- Do not describe provider/fallback changes as routine. They are must-ask changes unless the user has explicitly authorised execution.
- Do not record transient provider failures as durable negative rules. Capture the fix path and verification pattern instead.
- Do not claim the whole fleet is healthy from config assertions alone. Distinguish config hygiene, service status, test status, and unresolved external-provider limits.
