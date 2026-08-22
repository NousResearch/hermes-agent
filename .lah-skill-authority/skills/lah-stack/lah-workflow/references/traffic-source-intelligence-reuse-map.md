# Traffic Source Intelligence — Reuse Map (S0–S4 mission class)

Verified 2026-08-14 on deployed SHA f7502ae (origin/main), mission
LAH_TRAFFIC_SOURCE_INTELLIGENCE_S0_S4_ARCHITECTURE_AND_REUSE_V1.
Applies to any mission in the S0 (source evidence projection) / S1 (source entity
resolution) / S2 (classification) / S3 (context×proposition learning) / S4
(contextual experiment design) class. REUSE > EXTEND > BUILD — these facts prevent
rebuilding primitives that already exist.

## Data-model facts (do NOT rebuild)

- **canonical-traffic-context.js** — PRIMARY source-context primitive. Already has:
  `TRAFFIC_GRANULARITIES ['CAMPAIGN','ZONE','SITE','VISIT']`, `field_state`
  OBSERVED/TARGETED/MISSING (MISSING ≠ zero via realValue()), `field_provenance`
  per dim, `identity_quality`, `deriveContextId`, `compareTrafficContexts`.
  Dims include traffic_source, campaign_id, zone_id, site_id, domain, geo,
  device, format, placement, channel. `buildCanonicalTrafficContext({targeting,
  observed, provenance})` is the honest MISSING-preserving projection.
- **experiment-record-builder.js** — `buildContextualExperimentRecord` schema
  ALREADY carries traffic_source, zone_id, site_id, domain, landing_family,
  landing_variant, angle_id, creative_id, allocation_arm, proposition_id. These
  dims are just unpopulated in practice. S3 = populate, not new schema.
- **coverage-matrix.js** — `BENCH_CELL_DIMENSIONS` ALREADY includes
  `SOURCE_VERTICAL_BENCH: ['vertical']` and
  `SOURCE_VERTICAL_EXPERIENCE_BENCH: ['vertical','experience']`.
  **PITFALL: bench_type strings are UPPERCASE** ('VERTICAL_BENCH'). Passing
  lowercase ('vertical_bench') silently yields 0 cells because
  `benchTypes.includes(p.bench_type)` filters everything — no error is raised.
- **coverage projection requirements** — `projectRecordToCells` needs
  `recordMatchesContext` (record dims must equal non-null context dims) AND
  `resolveRecordOffer` (map.offers must contain provider_offer_id === record.offer_id).
  Records with unresolvable offers or zone-dim mismatch produce no cells.
- **Provider surface (ExoClick)** — `/statistics/a/zone` = authoritative totals
  (Σ cost reconciles to campaign within ~0.2%); `/statistics/a/site` exposes
  `site_hostname` (1,350 rows for T03; UNDER-reports network-targeted campaigns:
  Σcost $15.03 vs $20.07 zone truth). `zone-statistics-contract.js` already parses
  site_id/domain from raw rows (`readFirst(rawRow, ['site_id','siteid','site'])`
  and `['domain','site_domain','site_url']`) but the values are dropped at store.
- **Provider surface (CrakRevenue stats)** — `affiliate_info2` = zone_id,
  `affiliate_info3` = campaign_id (ad_campaign). Zone→arrivals join proven:
  T03 506 zones / 31,314 hits joined = 100% of arrivals.
- **T03 known truths (acceptance fixture)** — campaign 8539232, direct provider,
  3 arms (A 10138 OurDream PPS $44, B 8780 Jerkmate, C 8517 WannaHookup),
  conversion zone **5688510** (1 SOI, ARM A, unpaid), dominant spend zone
  **5554566 = ad-maven.com** (34.6% spend). `spend_per_arm = UNKNOWN` is
  AUTHORITATIVE (ONE_CAMPAIGN_LAH_ROUTER, no per-variation spend surface) — never
  reconstruct from EVEN rotation or arrivals. These truths live in EVIDENCE
  RECEIPTS (`/home/deploy/evidence/lah-t03-postmortem-business-safety-learning-v1/`,
  `.../lah-t03-direct-provider-exploration-preparation-v1/`), NOT in
  experiment-memory.json (T03 memory records are campaign-level, proposition_id
  null on ARM A).
- **scope-identity-gate.js (PR #803)** — statuses SCOPE_PROPOSITION_VALID /
  STALE / UNKNOWN / AMBIGUOUS. Offer 10138 currently resolves to TWO propositions
  (prop_ourdream_84f0eb12 main + prop_ourdream_bounded_discovery_bf7e9ca0) →
  any scope declaring the old bounded hash gets AMBIGUOUS fail-closed. Reconciled
  target = prop_ourdream_84f0eb12.

## Techniques

- **Pre-merge gate replay (proven #803):** verify a PR's NEW pure-function gate
  BEFORE merge by running it from a temp worktree (`git worktree add /tmp/pr803-wt
  <head-sha>` + node_modules symlink) against CURRENT live data (canonical
  inventory → buildBusinessOfferMap → validateScopePropositionIdentity). This
  proves VALID vs AMBIGUOUS and surfaces production drift (stale candidate_scope)
  before the gate ever lands.
- **Acceptance replay on verified data (proven T03):** prove an S-layer design
  read-only by building records with `buildContextualExperimentRecord` (verified
  postmortem numbers only) and projecting with `buildCoverageMatrix` using
  SOURCE_VERTICAL_BENCH. Discipline: spend null (arm spend UNKNOWN), downstream
  events ≠ paid conversions (SOI lead = downstream_events 1, paid_conversions 0),
  nothing promoted (all cells PROBING).
- **Read-only probe pattern:** source canonical checkout `.env` (`set -a; .
  .../.env; set +a`), then `node --input-type=module -e` with ABSOLUTE imports
  from the deployed worktree (`openclaw-runtime-wt-deploy/lah-openclaw-mvp/`).
  Grep the container env for the live flag proof
  (`docker inspect ... | grep -c 'EXOCLICK_LIVE_ENABLED=true'` → 0).
- **Coverage verification gotcha:** `git status` on the canonical checkout may be
  on a stale work branch; inspect `openclaw-runtime-wt-deploy/` (deployed SHA) for
  production truth, and remember canonical `data/` is bind-mounted INTO the
  container (untracked data files are live but not in git) — see
  lah-stack-local-operator quirks.

## Invariants (mission doctrine)

1. zone_id != source_entity (multiple zones can share an entity; NEVER merge on
   similarity alone). 2. missing != zero; UNKNOWN is a valid answer. 3. No spend
   per arm reconstructed without provider authority. 4. Unpaid downstream signal
   != paid conversion != economic proof. 5. No global GOOD/BAD site/context —
   conclusions are contextual. 6. No classification without evidence provenance
   (no intuition/domain-name guessing). 7. No mass crawl before observed
   inventory proves value. 8. Fail-closed on ambiguous identity/provenance.
