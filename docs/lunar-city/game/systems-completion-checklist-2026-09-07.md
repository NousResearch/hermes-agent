# Lunar City systems completion checklist

Reconciled against the preserved historical design handoff, September 1 handoff, video-reference
roadmap and strategy-game diagnostic at historical commit
`0d9a879ce487cd6886c742ab06828547882c708c`, plus current runtime source. Historical checklist
claims are not current acceptance evidence. This work follows the user's expanded instruction
to flesh out alerts, personalities and worker interactions.

| System | Recovered implementation | Current execution slice |
| --- | --- | --- |
| Identity and observed population | Exact connection/profile/session/subagent/Kanban identities, bounded readers, freshness expiry | Preserve; new systems consume snapshots only |
| Movement and activity | State-to-room mapping, arrival clips, Recast paths, LOD and animation budget | Preserve; local gestures must yield immediately to movement/work/stale state |
| Alerts | Source health, inspector evidence, capped floating status badges | Fix default-working badge defect; add bounded open/acknowledged/resolved/reopened incident ledger and exact-entity focus |
| Personality | Exact group/title metadata and stable visual signatures | Add explicitly authored local traits/cadence; optional read-only exact-owner SOUL inspection, never inferred SOUL claims |
| Worker relationships | Exact parent ownership preserved by subagent identity; no relationship UI | Expose actual parent/delegation relations; distinguish group membership from actual collaboration |
| Ambient social behavior | Garden/rest destination only | Add bounded paired gestures for nearby resting workers, mutual facing, cooldown/cancellation, explicit local-social label; no generated backend speech |
| Leader dialogue and commands | Exact-owner text/voice, persistent leader sessions, broker confirmation/revalidation/readback | Reuse; no independent command or messaging path |
| Building activity | Authored legacy prop clips; no current observed-state orchestration | Inspect usable existing clips/sockets; hook only actual authored clips to observed activity; replacement buildings without clips remain static |
| Camera/accessibility/performance | Focus/follow, source-health fallback, efficient tier, reduced motion, budgets/context recovery | New behavior follows these controls and remains keyboard accessible |
| Art acceptance | Review exports and runtime contracts exist | Root owns topology/material/rig QA; incomplete models are not promoted |
| Product acceptance | Renderer/WebGL checks only | Packaged Electron, authenticated gateway end-to-end, clean package SHA and 30-minute hardware session remain separate acceptance work |

The original handoff's SOUL and garden social notes are aspirations, not a detailed behavior
specification. Authored traits and ambient pair rules are new local presentation choices under
the current user instruction. They never assign work, change authority, clear real blockers,
create achievements, send messages or call external actions. Alert acknowledgement is local UI
state; only a fresh healthy observation can establish recovery. Source disappearance or stale
data cannot prove success.

Execution ownership: alert ledger/status truth in an isolated delegated slice; personality
module/read-only panel in another; parent game agent owns social controller, scene wiring,
relationship UI, integrated viewer scenarios and end-to-end checks. Changes stay in this
workspace; no publication or live backend mutation is included.

## Validation checklist

- [x] Truthful badges for stale/unknown/heartbeat and new states.
- [x] Alert deduplication, acknowledgement, fresh resolution and reopening with exact identities.
- [x] Stable authored personality and exact-owner SOUL reads; late response isolation.
- [x] Paired visible gestures/facing; interruption by work/disconnect/reduced motion/quality.
- [x] No snapshot mutation or backend calls from ambient behavior.
- [x] Real viewer scenarios for alerts and pair transitions, plus runtime tests.
- [x] Existing feature tests/typecheck/build and focused handoff receipts updated.

Integrated implementation and actual browser receipts: `systems-implementation-handoff-2026-09-07.md`. The comprehensive recovered draft matrix supersedes the preliminary inventory above. Packaged/live and final model acceptance remain open.
