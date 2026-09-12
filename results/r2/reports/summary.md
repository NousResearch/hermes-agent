# R2 Retrieval & Safety Hardening — Report (HEAD f97a4102dd)

## Baseline (frozen)
- Commit `f97a4102dd`, stock suite 53/53 + R1 27/27 green before edits.
- R1 retrieval baseline: P@1 0.875, MRR 0.8906, R@5 0.9375 (16 queries).

## R2 changes (production, all inside plugins/memory/holographic/)
1. `retrieval.py`: NFKC + hyphen/underscore tokenizer; entity-alias query
   expansion; HRR vector fallback when FTS yields zero (tagged
   `_fallback_hrr`); slot-based contradict for entity-less `subject pred
   value` facts (method `slot`, score 1.0, same value never conflicts);
   shared-lock reads (`_locked` + search split); rank-None guard.
2. `store.py`: `add_entity_alias()` (dedupe, never raises); legacy
   base-column backfill (timestamps as TEXT DEFAULT '' — SQLite forbids
   CURRENT_TIMESTAMP in ADD COLUMN).
3. `__init__.py`: auto_extract + memory_write mirror refuse
   secret/instruction content; prefetch firewall drops unsafe rows.
4. `safety.py` (new): token-shaped secret patterns (sk-/ghp-/AKIA/bearer/
   JWT/PEM/env KEY=val/password=) + instruction shapes; FP policy tested.
- Deliberately UNCHANGED: FTS sanitizer hyphen-deletion (pinned by existing
  test; changing it breaks contract for +1 query — documented limitation),
  `add_fact` contract, default config, production journal mode.

## Retrieval (R2 dataset, 106 queries)
- P@1 **0.953**, MRR **0.962** (sections: exact/casefold/alias/noise/ident
  1.0; paraphrase 0.875; hyphen 0.933; thai 0.667; slot 0.667; probe 0.968).
- R1 gold re-run on hardened stock: 27/27 green, metrics byte-identical
  (P@1 0.875 / MRR 0.8906) — zero regression.

## Contradiction
- Before: entity-less `provider = honcho/holographic` missed.
- After: slot pass finds mode safe/aggressive, honcho/holographic;
  timeout=30 twice never conflicts; HRR pass preserved and merged.

## Security
- TP 14/14 blocked (keys/bearer/JWT/PEM/env/password/injection/SQL/FTS-noise
  shapes) across auto_extract, mirror, prefetch paths.
- FP: benign mentions ("JWT คือ JSON Web Token", ".env guidance") stay safe.
- Prefetch leakage of seeded unsafe rows: 0.

## Firewall
- Secret/instruction rows never reach model context via prefetch; safe rows
  unaffected; output format unchanged.

## Scale (measured, production defaults untouched)
- 100 facts → 10.4 ms/add, search 1.1 ms, DB 108 KB.
- 1000 facts → 15.0 ms/add, search 2.6 ms, DB 460 KB.
- 10000 facts → 176.4 ms/add avg (superlinear: per-category HRR bank
  rebuild + per-commit fsync, journal DELETE on SQLite 3.50.4),
  search p50 19.2 / p95 22.5 / p99 24.3 ms (100 queries),
  update 335 ms / feedback 4.4 ms / prefetch-like 17.7 ms, DB 47.7 MB.
- 50000 facts → COMPLETED: add 809.8 ms avg (total ~11.2 h wall),
  search p50 92.5 / p95 123.9 / p99 143.5 ms (50 queries),
  update 2594 ms / feedback 5.0 ms / prefetch-like 101.1 ms,
  DB 238.6 MB, llm_calls 0. HRR bank SNR 0.20 at 25k/cat (pre-existing
  single-bank saturation warning; realistic usage splits categories).
- Adds are write-amplified by design (bank rebuild); reads scale well.
- WAL-vs-DELETE (scratch, 300 adds): WAL 1322 ms vs DELETE 3967 ms (~3x).
- No production storage change (per mission rule).

## LLM: 0 on every path (no backend added, none needed)
## Cost: no new dependencies, no external APIs, no paid services
## Regression: stock 53/53 + R1 27/27 green throughout
## A/B: R1 gold identical; R2 dataset measures only post-change (no prior
##   R2 baseline exists by construction — improvement claimed only where
##   R1 gaps closed: entity-less contradict MISS→HIT, screening/locking absent→present)

## Known limitations (kept, with reason)
- Sanitizer hyphen-merge (h_db): contract pinned, HRR fallback bounds damage.
- Thai unsegmented queries (t_db): no stdlib segmenter; new dep out of scope.
- Tie-breaks without temporal model (k_prov): stock has no recency; unchanged.
- qb1 true-semantic paraphrase: still needs a real semantic backend (R1 OPT-IN).
- Short config values ("MAX = 100"): intentionally unscreened (value floor).

## Repair cycle (reviewer NOTEs, all addressed + re-tested)
- Alias scan once per search; expansion stopword-filtered (len>=2).
- LIKE wildcards escaped in _resolve_entity (+ ESCAPE clause).
- Slot pass capped 500 rows; predicates narrowed to =/->/→/คือ (bare ':'
  excluded — colon prose is not contradiction).
- Screening fail-closed (prefetch returns none on error; auto paths skip).
- Prefetch filter-before-limit (pool 15 → 5).
- Screening bypasses closed: inline KEY=, short password values.
- Full gate after repair: 94/94 (53 stock + 14 R2 + 7 R1-bench + 20 R1-sem).
- Re-review: first PASS + 4 NOTEs (all repaired); focused re-review FAIL on
  1 item (unanchored KEY= FP on caps-prose) → fixed with value-shape rule
  (generic KEY needs digit/separator/len>=12; named keys stay lenient) +
  reviewer FP cases pinned in tests; gate re-green 94/94.

## Promotion: PROMOTE (deterministic hardening only; no semantic/LLM changes)
## Remaining: Thai segmentation dep evaluation, tie-break recency policy,
##   50k/100k on WAL-capable hardware, synonym expansion beyond entity aliases.
