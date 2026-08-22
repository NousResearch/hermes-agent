# CLOE Live-Gate Bounded Proof Pattern (in-process, read-only)

Established during Lot 3 resume → Lots 4-8 + Lot 1 re-verify (2026-08-05,
CLOE_HIGH_ROI_BUSINESS_CAPABILITY_GRAPH_V1). Each business capability's
`live_verified` dimension at a new deployed SHA is proven by a BOUNDED
IN-PROCESS proof: pure modules imported from the deployed worktree, fixed
deterministic inputs, zero network/LLM/mutation. Artifact persisted → real
digest computed → verdict imported via `importLiveGateVerdict` + append +
rebuildIndexes.

## When to use
- SHA-rebind demoted a `live_verified` dim (see
  `cloe-evidence-recollection-and-live-import-pattern.md`) and the operator
  authorizes re-import at the new SHA.
- A new lot needs its live gate without touching providers or the container.

## Proof construction rules
1. Import the capability's canonical module from the DEPLOYED worktree
   (HEAD == deployed SHA). Module registry:
   `src/self-audit/evidence/high-roi-capabilities.mjs`.
2. Exercise FAIL-CLOSED branches first — they never touch disk/network:
   invalid input, missing config, placeholders, insufficient data.
3. Fixed inputs only — no Date.now(), no random, no env dependence.
   VERIFY determinism: run the proof TWICE and `cmp` the artifacts; both runs
   must be byte-identical before the digest is usable.
4. Zero network, zero LLM, zero writes, zero container env toggles. Record a
   `safety` block in the artifact (network_calls=0, writes_performed=false,
   provider_mutations=0, paid_traffic_activation=false).
5. Persist the artifact to `/tmp/<lot>-proof-raw.json`.
6. digest = sha256(JSON.stringify(raw_result, compact, insertion order)) —
   the exact serialization the proof printed (excluding any self-referential
   digest field the script adds AFTER hashing).

## Capability module map (8 business caps, 131f3506)
| capability | module | safe proof surface |
|---|---|---|
| campaign-memory | `src/services/campaign-memory-writer.js` | writeCampaignMemory(mode='read_only', memoryEventsDir=<tmp>) → governance_mode `read_only_dryrun`, files_after=0; invalid metadata → blocked_validation; secret content → blocked; isCampaignMemoryWriteAvailable |
| provider-statistics-read | `src/services/exoclick-stats.js` | operator Option C real read-only GET replay (below) |
| tracking-attribution | `src/services/canonical-tracking-context.js` + `crakrevenue-tracking-url-builder.js` | context build → URL-safe serialize → click/conversion identity → UNATTRIBUTED fail-closed → link composition with aff_sub1 |
| approval-budget-safeguards | `src/services/lahb-approval.js` | `delete process.env.LAHB_URL` → submitApprovalRaw throws LAHB_URL_REQUIRED (fail-closed, throws BEFORE getSecret — zero network) |
| governed-campaign-creation | `src/services/campaign-creation-orchestrator.js` | isCreativeAssetImportReady fail-closed branches: null cell → MISSING_ASSET_ID; no generation / QUEUED → NOT_YET_GENERATED; UNKNOWN_OUTCOME; placeholder asset_id → PLACEHOLDER_ASSET_ID (all return BEFORE readAssetMetadata) |
| zone-site-selection | `src/decision/zone-decision-engine.js` | evaluateZone returns `{ok, decision:{action_type, reason_codes, requires_approval}}` — READ `.decision`! invalid → INVALID_INPUT; missing ids → WATCH_ZONE MISSING_TARGET_IDENTIFIER; low sample → WATCH_ZONE INSUFFICIENT_SAMPLE; loss cap → REVIEW_ZONE PER_ZONE_LOSS_CAP; assert requires_approval=true on EVERY decision |
| creative-inventory-lineage | `src/services/creative-factory-orchestrator.js` | runCreativeFactoryOrchestrator: `{}` → CREATIVE_FACTORY_INVALID_INPUT (9 missing fields); valid payload → OPERATOR_REVIEW_READY, cells + test_matrix, deterministic=true read_only=true |
| business-health | `src/services/business-health-report-core.js` | validateBusinessHealthInput({}) → field errors; valid input → buildBusinessHealthReport → deriveRiskLevel → renderBusinessHealthSummary (all pure) |

## Pitfalls
- **evaluateZone output is nested**: `{ok, decision}` — extract
  `decision.action_type` / `decision.requires_approval`, not top-level fields
  (a top-level extraction silently yields `{}` in the artifact).
- **creative-factory validatePayload**: requires 9 string fields
  (campaign_id, campaign_name, objective, vertical, geo, device,
  audience_hint, base_lah_url, offer_slug) + angle_inputs entries with
  `title` and `rationale` (+ optional angle_id); objective must be a valid
  objective (e.g. 'diversify'). Its agents are rule-based (0 network/LLM) so
  the FULL orchestrator path is deterministic and safe.
- **campaign-memory original live gate mutates the container** (env toggle
  CLOE_CAMPAIGN_MEMORY_EVIDENCE_WRITE_ENABLED + real writes) — do NOT re-run
  it for a re-verification; write a new in-process read_only proof instead.
- **campaign-memory secret check**: SECRET_PATTERNS are sk-…, *_API_KEY,
  -----BEGIN, Bearer — a 'password=…' string does NOT match; use an
  '*_API_KEY='-style content. The block lands as governance
  blocked_validation (not blocked_secret) when the content validator catches it.
- **lahB-approval**: never call submitApproval/submitApprovalRaw with a real
  LAHB_URL (network POST + cooldown file writes). The fail-closed proof REMOVES
  LAHB_URL from the env first.

## Import + operator protocol
1. Write the import script: `importLiveGateVerdict` (pure builder!) + loop
   `appendReceipt` + `rebuildIndexes` once; pass `now: new Date(certified_at)`;
   full 64-hex graph_hash; real digest.
2. Operator review: script sha256 pinned. Run 1 → appended=1/duplicate=0/
   rejected=0; run 2 → appended=0/duplicate=1 (idempotency proof). Verify the
   single receipt's bindings (source_sha == deployed_sha, graph_hash full,
   digest, runtime_identity, deterministic observed_at).
3. Publisher dry-run → REQUIRE demotions=0, rejected=0, conflicts=0 →
   promote → hot reload (wait > 15 s) → container ID/StartedAt/RestartCount
   unchanged → ALREADY_CURRENT → chain integrity ok.

## Option C — operator-authorized real read-only provider replay
When a demoted live gate needs fresh evidence from the real provider (e.g.
provider-statistics-read after the mapping fix), the operator may authorize ONE
bounded read-only replay: real ExoClick GET /statistics/a/zone, canonical clean
campaign 8308460, single-day window 2026-05-12, rows=0 acceptable honest empty,
identity POST /v2/login only, EXOCLICK_LIVE_ENABLED=false, token read from .env
via split key name (never printed, never on argv). Persist the sanitized raw
artifact → digest = sha256 of the canonical raw result → import. STOP if the
call would mutate state or exceed the exact read-only scope.

## Batching multiple gates
Multiple live gates (e.g. 6 caps in one continuation) can be imported in ONE
script (array of verdicts, loop importLiveGateVerdict+append, rebuildIndexes
once), then ONE publisher promotion (+N promotions / 0 demotions). Per-lot
certification artifacts may reference the same final graph; each gate still
gets its own receipt and gate_id.
