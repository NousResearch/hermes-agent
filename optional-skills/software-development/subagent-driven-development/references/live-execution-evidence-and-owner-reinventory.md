# Live Execution Evidence and Canonical-Owner Reinventory

Use this reference when integrating a receipt-grounded Rust plan whose effectful path crosses sandbox, verification, memory, replay, or lifecycle crates.

## 1. Reinventory owners against the integration HEAD

An inventory is a statement about one source generation, not a durable fact. Parallel branches may add owner APIs after an earlier reviewer reported a gap.

Before implementing a new lifecycle, replay, receipt, verifier, or store abstraction:

1. Record the current integration `HEAD`.
2. Search the current source for the exact desired owner symbols and facade reexports.
3. Inspect the implementation, not only public docs or an earlier summary.
4. Compare the inventory's recorded HEAD with the current HEAD.
5. Classify each need as `reuse directly`, `thin adapter`, `owner-native extension`, or `genuine gap`.

If a prior report says “no canonical API exists” but current source exports the API, treat the report as stale and reuse the owner. Never preserve a duplicate abstraction merely because it was reasonable on an earlier branch.

## 2. Receipt shape is not execution proof

The following are configuration or shape signals only:

- backend kind says `Container`;
- execution mode string says `real_sandbox`;
- a caller supplies a structurally valid capability receipt;
- expected containment mechanisms are enumerated;
- check outputs look successful;
- an adjudication object can be constructed from those outputs.

None proves that the current commands ran in the claimed runtime.

A verified effectful result must require owner-backed evidence tied to the current execution, such as a post-execution runtime observation that binds:

- runtime and container identity;
- requested and resolved digest-pinned image;
- observed containment mechanisms;
- current command/run lineage;
- cleanup/finalization outcome;
- a content digest verified by the canonical backend owner.

## 3. Safe backend contract pattern

Give the execution-backend trait a fail-closed evidence method whose default is false. Only the real backend may return true, and only after it has captured and verified current-execution evidence.

Synthetic/fake backends used in unit tests should remain able to test orchestration, but must not satisfy publication-complete or `SucceededVerified` predicates. Assert this explicitly.

Do not let an arbitrary caller closure or report parameter upgrade a fake backend into live evidence. Caller-supplied receipts may be compared to owner evidence, but must not replace it.

## 4. Three evidence levels

Keep these claims separate:

1. **Deterministic unit evidence** — argv restrictions, receipt validation, fake backend orchestration.
2. **Host preflight evidence** — runtime exists and rootless/cgroup/seccomp capabilities are observable.
3. **Live effectful evidence** — the digest-pinned image ran the actual fixture and denial matrix under the required controls.

An ignored live test is not a pass. Report its exact ignored status and classify live proof as unavailable until it executes. Never upgrade level 1 or 2 into level 3.

## 5. Effectful closure checklist

A passing sandbox command is not the full learning loop. Before calling the lane complete, verify that the same lineage reaches:

- durable preflight and one-shot permit-use receipts before the effect;
- typed patch validation/application and before/after tree digests;
- actual required checks and bounded outputs;
- independently persisted verification artifacts;
- CEA attribution plus canonical CEA-store persistence, or explicit causal unavailability;
- canonical export envelope;
- child-first run-bundle publication and index-last recovery;
- procedure lifecycle owner receipt;
- retained-input replay or explicit replay unavailability;
- terminal projection that cannot succeed from mock, fixture, synthetic, stale, degraded, or indeterminate evidence.

A thin composition report may reference these owner artifacts, but must not recreate their truth.
