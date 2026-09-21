# Reference Comparison Validation Pattern

Use this when Sahil asks KENSEI to validate an analysis against architecture/reference documents or a fleet/system blueprint.

## Pattern

1. Load the governing strategic skill/context first.
2. Read the named reference documents directly from disk; do not validate from memory.
3. Verify live state only where the analysis makes live-state claims, e.g. running services, profile counts, SOUL/config presence, cron/webhook evidence.
4. Separate three judgement layers:
   - Reference fit: does the current system match what the source docs expect?
   - Operational reality: does live state support the claim?
   - Execution readiness: are recommendations safe, scoped, and complete enough to act on?
5. For every recommendation, check whether it maps to a stated gap. If the bottom line names a gap but no recommendation fixes it, call that out as incomplete.
6. Treat excluded or deferred broken items as still-open risks unless independently verified fixed. Never phrase them as "resolved" just because they were excluded from scoring.
7. Tighten unsafe recommendation wording:
   - Prefer audit/classify/port/approve over "scrub/delete" for profiles, configs, skills, or state.
   - Prefer templates/body conventions before schema/tooling changes unless schema work is explicitly requested.
   - Webhook/event recommendations should default to task/alert creation, not destructive or publishing actions.
8. Preserve approval gates: provider/fallback/auth, profile deletion, service deactivation, broad config sweeps, service restarts, and destructive actions require explicit approval.

## Common gaps to look for in multi-agent fleet comparisons

- Profile count/profile sprawl mismatch vs live `~/.hermes/profiles`.
- "All profiles have SOUL.md" overclaim; distinguish active leads from variants/aliases.
- Specialist gateway dispatcher safety: only Kensei should own cron/kanban dispatch; specialist gateways should not dispatch.
- Skill curation leakage: specialists should have narrow curated skill sets; Kensei can stay broad by design.
- Knowledge accumulation gap: memory is not the same as a structured wiki/runbook/provenance layer.
- Webhook security: HMAC, event allowlist, replay/timestamp checks, payload limits, rate limits, and approval gates.
- Cross-pollination governance: do not blindly copy skills; review, adapt, record provenance, test load, then curate.
- Fleet cleanup completeness: reconcile profile dirs, systemd services, Discord bot apps/tokens, platform configs, skill dirs, memory dirs, cron references, kanban assignees, and channel permissions.

## Output shape

Keep the final judgement direct:
- Valid / partially valid / invalid.
- Corrections to factual claims.
- Which recommendations stand.
- Missing recommendations or unaddressed risks.
- A corrected bottom line Sahil can use as the decision record.
