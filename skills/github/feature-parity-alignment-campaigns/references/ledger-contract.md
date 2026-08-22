# Feature Parity campaign ledger contract

A campaign has two different authorities:

1. `contracts.json` is the append-only semantic registry. It authorizes ordered
   capability identity and product disposition.
2. `<platform>.json` is the mutable delivery ledger. It records current code,
   publication, wiring, and release evidence for one registered revision.

A ledger cannot authorize a semantic change by recomputing its own digest.

## Contract registry

The repository registry lives at
`docs/architecture/feature-parity/contracts.json`.

```json
{
  "schema_version": 1,
  "contracts": {
    "example-feature-parity": [
      {
        "revision": 1,
        "repository": "example/project",
        "tracker": 123,
        "contract_sha256": "<sha256>",
        "previous_contract_sha256": null,
        "authority": {
          "kind": "issue",
          "number": 123,
          "url": "https://github.com/example/project/issues/123"
        }
      }
    ]
  }
}
```

Revisions are contiguous and append-only. Revision 1 has a null predecessor;
each later revision points to the prior digest and includes a non-empty `reason`.
Repository and tracker identity do not change across revisions.

The digest covers the ordered list of each row's `id`, `name`, `product_state`,
and `source_anchor`.

## Ledger root

```json
{
  "schema_version": 1,
  "campaign": {
    "id": "example-feature-parity",
    "repository": "example/project",
    "tracker": 123,
    "contract_revision": 1,
    "expected_capability_ids": ["M1", "M2"],
    "forbidden_growth_paths": ["plugins/platforms/example/adapter.py"],
    "contract_sha256": "<registered sha256>"
  },
  "snapshot": {
    "upstream_sha": "<lowercase 40-hex commit>",
    "captured_at": "<RFC 3339 UTC timestamp ending in Z>"
  },
  "capabilities": []
}
```

`expected_capability_ids` is required, non-empty, unique, and ordered. The
capability list must match it exactly.

## Capability row

Every row requires:

- `id`, `name`, `source_anchor`, `product_state`, and `delivery_state`;
- `implementation_paths`, `test_paths`, `consumers`, `publications`, and
  `artifact_evidence`, even when empty;
- `decision` for `pair_gap`, `conditional`, `deferred`, or `rejected` product
  states.

Paths are canonical repository-relative POSIX paths. Consumer identifiers use
`<path>:<symbol>`.

## Product states

- `accepted` — approved campaign scope.
- `existing` — behavior already exists but still requires evidence and
  publication truth.
- `pair_gap` — paired product or authority decision remains unresolved.
- `conditional` — implementation is gated by an explicit condition.
- `deferred` — no production, test, or consumer paths may accumulate.
- `rejected` — no production, test, or consumer paths may accumulate.

Decision-gated states cannot advance to `candidate_open`,
`on_main_unverified`, or `released`.

## Delivery states

- `gap` — no active authority; requires `gap_reason` or a product `decision`.
- `candidate_blocked` — one open authoritative PR plus a non-empty `blocker`.
- `candidate_unwired` — implementation and tests exist, no consumer exists, and
  `wiring_gap` explains the missing runtime path.
- `candidate_open` — one open authoritative PR with exact `head_sha`,
  implementation paths, tests, and runtime consumers.
- `on_main_unverified` — authoritative PR is merged and exact merged SHA is
  recorded, but terminal evidence remains incomplete.
- `released` — exact merged SHA and all terminal evidence are complete.
- `superseded` — no active authority; requires `superseded_by`.

Artifact-only pseudo-states such as `patch_ready`, `branch_exists`, and
`implemented_in_packet` are invalid.

## Publication authority

An active row has exactly one authoritative publication, and it must be a pull
request in `campaign.repository`.

```json
{
  "kind": "pull_request",
  "number": 456,
  "role": "authoritative",
  "state": "open",
  "author": "contributor",
  "url": "https://github.com/example/project/pull/456",
  "head_sha": "<required for candidate_open>"
}
```

For `on_main_unverified` and `released`, state is `merged` and
`merge_commit_sha` equals `merged.commit_sha`.

## Terminal release evidence

```json
{
  "merged": {
    "repository": "example/project",
    "commit_sha": "<lowercase 40-hex commit>"
  },
  "release_evidence": {
    "ci": {
      "url": "https://github.com/example/project/actions/runs/123",
      "commit_sha": "<same merged commit>"
    },
    "live_receipt": {
      "path": "receipts/example.json",
      "sha256": "<lowercase 64-hex file digest>",
      "commit_sha": "<same merged commit>"
    },
    "reviews": [
      {
        "reviewer": "reviewer-one",
        "url": "https://github.com/example/project/pull/456#pullrequestreview-1",
        "commit_sha": "<same merged commit>"
      },
      {
        "reviewer": "reviewer-two",
        "url": "https://github.com/example/project/pull/456#pullrequestreview-2",
        "commit_sha": "<same merged commit>"
      }
    ]
  }
}
```

Reviewers are distinct and neither may be the authoritative PR author.
Repository validation confirms that the live receipt exists within the
repository and matches its declared SHA-256.

## Repository validation

Run the validator without positional ledgers to discover every JSON ledger in
`docs/architecture/feature-parity/` except `contracts.json`:

```text
python scripts/ci/validate_feature_parity_ledger.py --repository-root .
```

Repository validation additionally rejects duplicate campaign IDs, tracker
issues, and authoritative pull-request ownership across ledgers.
