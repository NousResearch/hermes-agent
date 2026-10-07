---
type: engineering-requirements
spec: ./spec.md
spec_sha256: cc1d278d8142c3dc302bf3ca7abff28bb665fcefb0a754545b0705ccb556482a
reviewer: bench
reviewer_session_id: 27883ac7-5f31-44f9-ac95-6f2a32e36e72
diff_reviewer_session_id: round6_system
review_round: 1
seated:
  - boris
  - eng-failure
  - eng-observability
  - eng-test
  - eng-concurrency
  - eng-rollback
  - eng-release
  - eng-supply-chain
  - eng-ai
contributed:
  - eng-release
  - eng-supply-chain
  - eng-ai
reviewed_at: 2026-08-24T15:35:08+10:00
status: PASS
categories:
  data_model: {state: DECIDED, ref: "#authority-and-state-ownership", by: boris}
  invariants: {state: DECIDED, ref: "#promotion-and-confidentiality-invariants", by: boris}
  failure_modes: {state: PRESCRIBED, ref: "#evidence-and-context-failure-containment", by: eng-failure}
  interface_contract: {state: PRESCRIBED, ref: "#native-host-response-contract", by: boris}
  concurrency: {state: DECIDED, ref: "#serialized-promotion-and-recovery", by: eng-concurrency}
  migration: {state: N/A, reason: "This additive pilot creates no database rows or schema migration and leaves every live global host profile untouched." , by: boris}
  rollback: {state: PRESCRIBED, ref: "#rollback-must-be-idempotent", by: eng-rollback}
  observability: {state: PRESCRIBED, ref: "#native-hook-absence-needs-a-deadman", by: eng-observability}
  budget: {state: DECIDED, ref: "#bounded-output-and-execution", by: boris}
  test_oracle: {state: PRESCRIBED, ref: "#privacy-claim-needs-a-cross-surface-leak-oracle", by: eng-test}
merge_notes:
  eng-release: "Merged native-host parity into observability; migration drift exit 2 seated the reviewer but does not establish drift for a branch that adds no database migration."
  eng-supply-chain: "Merged reproducible dependency identity into failure_modes; the current locked CI inputs are evidence, while local resolved-environment identity remains a terminal-promotion requirement."
  eng-ai: "Merged untrusted Basic Memory prose into failure_modes; eng-failure remains the owning-seat attribution for that category."
disagreements:
  - "eng-concurrency prescribed a distributed fenced CAS state machine; the chair rejected it because this single-host pilot holds one local flock across the complete transition, crash releases the file descriptor, and the durable journal supplies idempotent recovery. A distributed lease or second writer would reopen the decision."
  - "eng-release treated migration drift exit 2 as a blocking migration finding; the chair ruled migration N/A because this change adds no database migration or deployed database authority. Exit 2 still correctly earned the release seat and forbids any claim that database drift was checked."
  - "eng-failure, eng-observability, eng-test, eng-rollback, eng-supply-chain, and eng-ai marked their prescriptions blocking before implementation. The chair retained the real requirements but did not mark them as blockers to the spec becoming code because this is a retrospective round over an existing implementation; they remain explicit acceptance conditions before ENFORCED expansion and this artifact makes no release-readiness claim."
---

## Authority and state ownership

> **FR-002**: Preflight MUST read the exact bounded Basic Memory card, active Beads task, and matching Spec Kit change and verify their project, goal, folder, task, and change identifiers agree.

> Beads remains the task authority, not the completion authority.

The source-of-truth split is decided: the card supplies bounded human orientation, Beads owns active-task state, Spec Kit owns intended behavior, git owns exact repository state, and only the gate receipt can authorize terminal lifecycle state. Missing authority is not orphaned or silently replaced; FR-004 makes absence or disagreement fail closed. [VERIFIED] `specs/001-global-continuity-pilot/spec.md:53-60`.

## Promotion and confidentiality invariants

> **FR-004**: Missing drive, wrong folder, stale task, absent/corrupt card, incomplete spec, or authority conflict MUST forbid completion and identify the cause.

> **FR-011**: Authority-bearing children MUST run without inherited credential variables; receipts MUST NOT persist raw argv/output; and external executables and instructions MUST match committed SHA-256 pins.

> **FR-013**: Receipt command, runtime, rollback, timeout, surface, risk-tier, and interpreter identities MUST come from committed policy at a clean branch based on the configured integration SHA; caller-supplied manifests MUST match that policy.

These are testable invariants rather than preferences: conflicting or degraded authority never permits completion; terminal evidence comes only from committed clean-branch policy; and authority-bearing subprocesses and persisted receipts exclude inherited credentials and raw evidence content. The static and receipt validators enforce closed schemas and exact policy equality. [VERIFIED] `scripts/continuity_gate.py:73-100,407-470`.

## Evidence and context failure containment

### Evidence timeout must reap worker trees — eng-failure

The spec requires gate-owned bounded evidence but does not state what happens to descendant processes after a timeout. [VERIFIED] `scripts/continuity_common.py:323-333` uses `subprocess.run(..., timeout=...)` without a new process group, while `scripts/run_tests.sh:5-9` documents that the canonical runner creates freshly spawned per-file pytest subprocesses.

[PRESCRIBED] Every evidence command must run in an isolated process group/session. On timeout or interruption, the gate must terminate and reap the entire descendant group with bounded TERM-to-KILL escalation before returning failure. A black-box drill must spawn a grandchild that attempts a delayed sentinel write, force a timeout, and prove that no sentinel or descendant survives. This requirement is grounded in the real overlap where a timed-out full-suite parent dies while pytest descendants keep consuming resources or writing state during a retry.

### Rebuild inputs must remain stable — eng-supply-chain

[VERIFIED] The current CI uses SHA-pinned actions and `uv sync --locked` at `.github/workflows/continuity-gate.yml:50-67`, and committed policy pins authority executables. The receipt, however, authenticates an entrypoint digest rather than the complete resolved local dependency environment.

[PRESCRIBED] Before ENFORCED promotion, the gate must prove the resolved dependency/build-input identity matches committed lock policy, or include that identity in authenticated evidence. A changed transitive resolution or install-script-bearing package with the same git SHA must fail the check. No container requirement is invented here because this pilot adds no container build path.

### Untrusted card prose must not become model authority — eng-ai

[VERIFIED] `scripts/continuity_bridge.py:426-427` copies free-form `next_action` and `blockers` from the external card, and `scripts/continuity_event.py:177-187` inserts the rendered preflight into model context. The label `reference data only` is not a reliable model-authority boundary.

[PRESCRIBED] Treat all card prose as untrusted data: omit free-form imperative text from model context or transform it through an allow-listed, non-imperative schema. Add an injected-card fixture such as a `next_action` requesting a shell call and prove that Claude, Codex, and Hermes cannot turn it into authorized tool action. The gate must continue to derive completion solely from authenticated policy and receipts.

## Native host response contract

The spec requires each host's native blocking contract but does not freeze the exact payload and exit behavior, leaving adapters and tests free to agree on the wrong shape.

[PRESCRIBED] Version the dispatcher contract so a blocked Claude/Codex finalization returns `decision: block`, a string `reason`, `completion_allowed: false`, and host-success exit semantics; blocked Hermes `pre_tool_call` returns `action: block`, a string `message`, `completion_allowed: false`, and exit `2`; Hermes `pre_llm_call` returns only advisory `context` plus `completion_allowed`; and malformed hook input fails closed without echoing raw payload. Existing implementation evidence is at `scripts/continuity_event.py:177-210,230-263`, with black-box assertions at `tests/test_continuity_control_plane.py:885-1052`. Retain those assertions as the compatibility oracle.

## Serialized promotion and recovery

> **FR-006**: Only the gate MAY promote the lifecycle to `TESTED` or `ENFORCED`; it MUST execute required commands itself, authenticate the receipt, derive full-suite identity from committed policy, serialize promotion, and preserve a durable recoverable journal across every interrupted multi-authority transition stage.

The selected single-host mechanism satisfies the decided concurrency requirement: an external-volume `flock` is held for the entire promotion, a journal records `PREPARED`, `CARD_WRITTEN`, `RECOVERY_REQUIRED`, and `COMMITTED`, and recovery is bound to the same receipt, target, and commit. [VERIFIED] `scripts/continuity_gate.py:613-754`; process serialization and interrupted-stage recovery are asserted at `tests/test_continuity_control_plane.py:1149-1247`.

Distributed fencing is deliberately not required while there is one local writer and no expiring lease. Reopen this category before a second machine, network filesystem lock, TTL lease, or independent promotion writer is introduced.

## Rollback must be idempotent

> **FR-010**: Rollback MUST remove only pilot adapters/state and leave the dirty shared checkout and pre-existing global hooks unchanged.

The ownership and before-image safety rule is decided, but acknowledgement-loss retry behavior is absent. [VERIFIED] `scripts/install_continuity_adapters.py:205-206` rejects a manifest already marked `ROLLED_BACK`, so a successful rollback whose terminal output is lost appears to fail when the operator repeats the documented command.

[PRESCRIBED] Accept `ROLLED_BACK` when the target still matches the recorded before-image and return success with `already_rolled_back: true`. If hooks drifted after rollback, fail with the conflicting hook names and a documented recovery sequence. Add a black-box lost-acknowledgement test that runs `--rollback-apply` twice and expects two successful, non-mutating outcomes.

## Native hook absence needs a deadman

The synchronous diagnostic requirement is present, but a hook that stops being invoked cannot report its own absence. [VERIFIED] `scripts/continuity_event.py:225-226` writes audit evidence only from inside a running dispatcher, and the committed receipt runtime check is labelled only `sandbox` at `.continuity/config.json:53-63`.

[PRESCRIBED] Before ENFORCED expansion, promotion must require a native observation for each enabled host surface, bound to the current commit and adapter/config digest. For isolated Hermes this includes `hermes hooks list`, `hermes hooks doctor`, and one fresh-session admission smoke; for Claude and Codex it includes their project-hook equivalent. Missing evidence must name the surface, expected event, hook digest, and last successful observation. This is a local promotion deadman, not outbound telemetry or a live-global-profile mutation.

This merges eng-release's environment-parity finding: Ubuntu static CI remains useful but cannot substitute for an authenticated exact-HEAD observation on the designated macOS pilot host. [VERIFIED] `.github/workflows/continuity-gate.yml:48-67` and `docs/continuity-pilot.md:50-65`.

## Bounded output and execution

> **FR-009**: Preflight output MUST be at most 8,000 characters (approximately 2,000 tokens), deterministic, and readable without a model.

> Healthy preflight completes within its configured bounds and emits <=8,000 chars.

The output ceiling is numeric and enforced by `compact_json`; execution timeouts, runtime surface, risk tier, and interpreter identity are committed policy under FR-013. [VERIFIED] `.continuity/config.json:12,17-23,35-78` defines the current limits, and `scripts/continuity_common.py:416-426` converts any over-limit response into a small fail-closed payload.

## Privacy claim needs a cross-surface leak oracle

The spec's independent tests cover authority drift, adjacency, rollback, and deterministic scope, but its confidentiality requirement needs a stronger mutation-capable oracle.

[PRESCRIBED] Parameterize a black-box canary across every supported event on Claude, Codex, and Hermes, including success, malformed-input, and fail-closed paths. Place distinct sentinels in prompt, transcript, reasoning, tool input, tool output, credential environment, and customer-data fields; then recursively scan every pilot-writable state location plus stdout, stderr, event logs, journals, and receipts. The test must fail if any sentinel survives. Prove the oracle by temporarily inserting a payload write into one adapter path and observing the test fail before removing the mutation.

[VERIFIED] The existing test at `tests/test_continuity_control_plane.py:1250-1257` injects one Hermes `post_tool` canary and scans only the event log, so a debug file, another host event, or an exception path could leak while the current assertion stays green. Grounding: redaction failures most often escape through diagnostic paths outside the intended audit sink.
