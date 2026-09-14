# Feature Parity campaign ledger contract

The ledger is the source of truth for capability identity, product disposition,
publication ownership, consumer wiring, and release evidence. Narrative EPIC
tables and packet implementation maps are projections of this file; they do not
override it.

## Root fields

```json
{
  "schema_version": 1,
  "campaign": {
    "id": "discord-feature-parity",
    "tracker": 79564,
    "expected_capability_ids": ["M1", "M2"],
    "forbidden_growth_paths": ["plugins/platforms/discord/adapter.py"],
    "contract_sha256": "<sha256 of canonical rows>"
  },
  "snapshot": {
    "upstream_sha": "<40-hex commit>",
    "captured_at": "<UTC timestamp ending in Z>"
  },
  "capabilities": []
}
```

The digest is calculated over the ordered list of each row's `id`, `name`, and
`product_state`. A renamed or repurposed row therefore fails validation rather
than silently changing the campaign's meaning.

## Capability row

Required fields:

- `id`, `name`, and `source_anchor`;
- `product_state`: `accepted`, `existing`, `pair_gap`, `conditional`,
  `deferred`, or `rejected`;
- `delivery_state`: `gap`, `candidate_blocked`, `candidate_unwired`,
  `candidate_open`, `on_main_unverified`, `released`, or `superseded`;
- `implementation_paths`, `test_paths`, `consumers`, `publications`, and
  `artifact_evidence`.

Conditional/deferred/rejected/pair-gap rows also require `decision`.
Candidate delivery states require one authoritative publication. A candidate is
`unwired` until the real caller/effect path is named and tested.

## State semantics

`candidate_blocked` means code exists behind a still-unresolved product,
dependency, collision, or authority gate.

`candidate_unwired` means a module or request builder exists, but no accepted
runtime consumer proves the user-visible capability.

`candidate_open` means one current PR contains implementation, tests, and
consumer wiring.

`on_main_unverified` means the exact merged commit exists but terminal release
evidence is incomplete.

`released` requires:

- exact merged commit SHA;
- head-bound CI receipt;
- live-system receipt when the campaign requires one;
- two independent exact-head approvals.

The following are deliberately invalid delivery states:
`implemented_in_packet`, `implemented_locally`, `package_green`, `patch_ready`,
and `branch_exists`.

## Publication authority

Each active row has exactly one publication with `role: authoritative`.
Related work is `dependency`; obsolete work is `superseded`. One authoritative
PR may not own two rows. This forces a visible decision when high-velocity work
collides instead of allowing two branches to claim the same class.

## CI invocation

```bash
python scripts/ci/validate_feature_parity_ledger.py \
  docs/architecture/feature-parity/<platform>.json
```

Campaign-specific conformance tests should import the same validator and assert
semantic invariants that are unique to the approved specification.
