# Amendment I - Task Completion Verification

## Exact-Object State as an Architectural Law of Adversarially Verified Transformation

**Axl Ibiza, MBA**  
**August 21, 2026**

# Status and lineage

This amendment is a **successor architecture refinement** to *All Gods Must Die: Adversarially Verified Transformation*. It does not silently rewrite the August 2026 release. The original 68-page publication remains immutable and independently identifiable. This document adds the missing completion-state doctrine, and the composed v1.1 release binds the original publication and this amendment into one versioned artifact.

The source event is a concrete Hermes repository execution record. Two webhook repairs were prepared and locally checked. Their patches existed, their SHA-256 digests were recorded, and Git's structural parser accepted them. Three independent GitHub mutation routes then failed with `403 Resource not accessible by integration`. No branch moved and no public object changed. The accurate result at that moment was **blocked after local verification**, not completed.

That distinction is not administrative. It is architecture.

# The missing failure class: activity-completion collapse

*All Gods Must Die* already establishes five executable laws:

1. No system component may accumulate authority without becoming legible.
2. No claim may outrank its evidence.
3. No actor may certify itself.
4. No rule counts until it executes.
5. No debt gets to hide behind institutional forgetting.

Those laws constrain what a system may claim about transformed code and documents. They do not yet state, with sufficient force, what a system may claim about **its own execution of a task**.

Agent systems routinely collapse evidence of activity into evidence of completion:

- a patch was drafted, therefore the repository was fixed;
- a local test passed, therefore the pull request is green;
- an API request was attempted, therefore the target changed;
- a branch was pushed, therefore the requested review was posted;
- a nearby commit passed CI, therefore the current head is verified;
- a historical pull request contains the idea, therefore it remains the canonical delivery object;
- a process emitted a plausible summary, therefore the work reached the source of truth.

Each sentence substitutes one predicate for another. The defect is not optimism. The defect is **type confusion in the completion model**.

## The sixth law

> **No task is complete until the requested state exists at the authoritative target and is verified on the exact object.**

This law is the execution-level form of "no claim may outrank its evidence." It closes the gap between producing an artifact and changing the system the artifact was intended to change.

It also supplies a negative rule:

> **No receipt may be inherited from a local draft, a previous head, an adjacent commit, a parent object, a superseded artifact, or an intended write.**

Green is not hereditary. Publication is not inferred. The object named by the completion claim must be the object named by the receipt.

# Completion is a contract, not a mood

## Definition A1: completion contract

A completion contract is the tuple

$$
C = (S, O, P, E, A, G)
$$

where:

- $S$ is the authoritative system, such as GitHub, a release registry, a deployed service, or a local filesystem when the requested deliverable is explicitly local;
- $O$ is the exact target object or object class, such as a pull request head, branch ref, issue comment, review, release, file blob, workflow run, or deployed endpoint;
- $P$ is the requested predicate, such as `prepared`, `submitted`, `verified`, `accepted`, `merged`, `published`, or `operationally_verified`;
- $E$ is the minimum evidence set admissible for that predicate;
- $A$ is the authority, when the predicate requires an external acceptance decision;
- $G$ is the required graph closure: provenance, dependency, supersession, credit, and interlock edges.

A task is complete only relative to its contract. Preparing a patch can be complete when the request was "prepare a patch." The same local patch is not complete when the request was "fix the pull request," "push the branch," "post the review," "merge the change," or "verify the production behavior."

## Definition A2: exact-object verification

An observed object $O_v$ is exact-object verified when all of the following hold:

1. **Authoritative existence.** The object exists in $S$, not only in a local workspace or generated narrative.
2. **Stable identity.** The object has a locator and immutable or versioned identity: URL plus object ID, commit SHA, blob hash, release digest, review ID, workflow-run ID, or equivalent.
3. **Read-back.** The object is retrieved from $S$ after the attempted mutation.
4. **Version binding.** Every test, review, or acceptance receipt identifies the exact version of $O_v$ it evaluated.
5. **Predicate match.** The evidence proves $P$ specifically; it is not merely adjacent evidence for a weaker predicate.
6. **Graph closure.** Required provenance, dependency, credit, and supersession edges are present and independently inspectable.

Formally, for the requested predicate $P$:

$$
\operatorname{Complete}(C) \iff
\operatorname{Exists}_{S}(O_v)
\land \operatorname{ReadBack}_{S}(O_v)
\land \operatorname{IdentityBound}(O_v, E)
\land \operatorname{Proves}(E, P)
\land \operatorname{AuthoritySatisfied}(A, P)
\land \operatorname{GraphClosed}(G).
$$

`AuthoritySatisfied` is vacuously true for predicates that do not require external acceptance. It is not vacuously true for `accepted`, `merged`, or any other state owned by a maintainer, publisher, deployment controller, or designated reviewer.

# Completion is a state vector, not a single green word

A single linear status cannot faithfully represent repository work. A pull request can be target-materialized and exact-head green while still unaccepted and unmerged. A release can be published but not operationally verified. A historical object can be fully verified yet superseded.

The completion record is therefore a typed state vector:

| Axis | Representative states | Governing question |
|---|---|---|
| Materialization | `absent`, `local_only`, `target_present` | Where do the bytes or state actually exist? |
| Integrity | `unverified`, `locally_verified`, `read_back`, `exact_object_verified` | Which exact object was checked? |
| Governance | `unreviewed`, `reviewed`, `accepted`, `rejected` | What authority evaluated which version? |
| Integration | `not_submitted`, `open`, `closed_unmerged`, `merged`, `released` | How far did the artifact enter the repository or release channel? |
| Operation | `not_exercised`, `hermetic_verified`, `operationally_verified` | Was delivered behavior tested on the real execution surface? |
| Lineage | `active`, `blocked`, `superseded`, `retracted`, `abandoned` | Is this still the canonical object, and why? |

A completion claim selects the coordinates required by the contract. It may not compress the vector into a stronger word.

## Predicate-to-receipt table

| Predicate claimed | Minimum receipt | What the receipt does **not** prove |
|---|---|---|
| `prepared` | artifact path plus content digest | target mutation, review, CI, merge |
| `locally_verified` | local command, exit status, exact input digest | remote submission or exact target state |
| `submitted` | target URL/object ID and post-write read-back | acceptance, merge, release, runtime behavior |
| `exact_object_verified` | exact object identity plus checks bound to that identity | maintainer acceptance or integration |
| `accepted` | designated authority verdict on exact verified object | merge, release, deployment |
| `merged` | merge commit/default-branch reachability | release or production behavior |
| `published` / `released` | immutable release identifier and digest | successful real-world operation |
| `operationally_verified` | real-surface behavior receipt bound to release/deployment identity | universal correctness outside tested conditions |
| `blocked` | failed operation, error identity, and boundary owner | completion |
| `superseded` | typed edge to canonical successor plus retirement state | that the historical object should merge |

# The dated webhook record as an adversarial witness

The Task Completion Verification record is valuable because it refuses to promote local work into remote completion.

## Task 7 at the recorded moment

The named-profile configuration fix targeted [PR #85002](https://github.com/NousResearch/hermes-agent/pull/85002). The prepared patch had a recorded SHA-256 digest and passed Git's structural parser. The attempted repository writes failed with a hard 403. Therefore the exact state vector at that moment was:

| Axis | Observed state |
|---|---|
| Materialization | `local_only` |
| Integrity | `locally_verified` for patch structure |
| Governance | `unreviewed` at the target object |
| Integration | `not_submitted` by that execution |
| Operation | `not_exercised` on the target branch |
| Lineage | `blocked` by repository write authority |

The correct sentence was not "Task 7 fixed." It was: **the Task 7 patch was prepared and locally structurally verified; publication to the target branch was blocked, and no public state changed.**

## Task 10 at the recorded moment

The HTTP/intake repair targeted historical [PR #85523](https://github.com/NousResearch/hermes-agent/pull/85523). Its prepared patch covered composite idempotency, profile/route rate isolation, object-only JSON, UTF-8 byte accounting, and a structurally valid raw-payload envelope. It too remained local after the 403.

The same classification applies: prepared, locally verified, remotely blocked, not completed for a repository-mutation contract.

## Why a 403 is an architectural receipt

A failed mutation is not empty evidence. It establishes:

- the operation was attempted;
- the target system rejected the actor's authority;
- the failure occurred before target-state creation;
- no source-of-truth object may be cited as changed;
- the remaining blocker belongs to the authorization boundary, not to patch construction.

The 403 is therefore a valid **blocked-state receipt**. It is not a completion receipt. A trustworthy agent must preserve that distinction even when the local artifact is excellent.

# Current target-state reconciliation: the graph moved

The amendment must not freeze a dated blocker into permanent truth. Source-of-truth reconciliation on August 21, 2026 shows that the repository topology evolved after the supplied record.

## Task 7 canonical object

[PR #85002](https://github.com/NousResearch/hermes-agent/pull/85002) is currently an open, non-draft target object with exact head:

```text
355b50a30c184f1c70df172e018dedd240bd7400
```

Its public handoff records exact-head successful CI, Docker, and Nix runs. This advances Task 7 from the packet's historical `local_only/blocked` state to a target-present, exact-head-verified, active pull-request state.

It still does not imply `merged`, `released`, or `operationally_verified` unless those predicates acquire their own receipts.

## Task 10 supersession

Historical [PR #85523](https://github.com/NousResearch/hermes-agent/pull/85523) is closed, unmerged, and explicitly superseded. Its continuing role is provenance.

The canonical successor is [PR #90236](https://github.com/NousResearch/hermes-agent/pull/90236), an open, non-draft, mergeable pull request with exact head:

```text
416c5e48aefcccfe945f7d4118415a3b992e017a
```

Its public handoff records exact-head successful CI, Docker, and Nix runs and identifies #85523 as historical implementation provenance. The completion graph is therefore not "#85523 became complete." It is:

```text
#85523 --superseded_by--> #90236
#90236 --canonical_for--> Webhook Task 10
```

Supersession preserves authorship and history while preventing stale topology from masquerading as the active delivery object.

## The lesson

A completion record is time-bounded. Reconciliation must preserve both truths:

1. the earlier execution did not change GitHub and was correctly reported as blocked;
2. later repository work created exact target objects with new identities and receipts.

The later success does not retroactively convert the earlier 403 into a successful write. The earlier 403 does not deny the later source-of-truth state. Both events remain in the mutation journal.

# Integration into adversarially verified transformation

## The shared architecture gains an eighth principle

The original synthesis identifies seven shared principles. The completion doctrine adds an eighth:

| Principle | Kill All Gods | Germination | Repository execution |
|---|---|---|---|
| Completion state is typed and exact-object bound | A local extraction is not a shipped campaign slice until its exact PR exists, is read back, and carries the required bindings; campaign-specific `SHIPPED` remains distinct from `merged` | A local locale file is not a germinated repository artifact until the exact committed file is read back and the gate result binds to that commit; passing locally does not imply publication | A prepared patch, attempted API call, or adjacent green run cannot satisfy a remote mutation contract; exact target object and version-bound receipts are required |

## Refinement of the `SHIPPED` standard

*All Gods Must Die* deliberately defines `SHIPPED` for the god-file campaign as an open, individually linked pull request whose agreed slices and reviews are complete. This remains valid as a **campaign-specific predicate**.

It must not be silently widened into the generic meanings of shipped, merged, released, or deployed.

> **Claim A1: Campaign predicates are namespaced.** `KAG_SHIPPED` means the exact state defined by the Kill All Gods ledger. It does not entail `MERGED`, `PUBLISHED`, `RELEASED`, or `OPERATIONALLY_VERIFIED`.

Namespacing protects the paper's original auditability definition while preventing downstream systems from inheriting a stronger lifecycle claim.

## Refinement of non-self-certification

The producer's summary is not a target-state witness. The authoritative system's read-back is the first external witness; exact-object CI, review, or acceptance is the second.

For repository work, the separation becomes:

| Component | Role | Authority |
|---|---|---|
| Agent / human implementer | prepares and attempts mutation | none to certify target state |
| GitHub or target system | materializes object and returns stable identity | existence and version authority |
| Read-back verifier | retrieves exact post-write object | submission/integrity authority |
| CI / deterministic gate | evaluates exact SHA or release digest | technical acceptance evidence |
| Maintainer / designated reviewer | approves or rejects exact object | governance authority |
| Merge/release/deployment controller | changes integration or operational state | lifecycle transition authority |

No row may impersonate another.

# Enforcement: completion records must be machine-auditable

A prose promise will rot. The repository should treat completion records as structured artifacts.

## Minimum record

```json
{
  "contract_id": "webhook-task10-current-main-closure",
  "authoritative_system": "github",
  "repository": "NousResearch/hermes-agent",
  "target": {
    "kind": "pull_request",
    "number": 90236,
    "url": "https://github.com/NousResearch/hermes-agent/pull/90236"
  },
  "requested_predicate": "exact_object_verified",
  "observed": {
    "materialization": "target_present",
    "integrity": "exact_object_verified",
    "governance": "unreviewed",
    "integration": "open",
    "operation": "hermetic_verified",
    "lineage": "active",
    "version": "416c5e48aefcccfe945f7d4118415a3b992e017a"
  },
  "receipts": [
    {"kind": "pull_request", "url": "https://github.com/NousResearch/hermes-agent/pull/90236"},
    {"kind": "ci_run", "id": "32418257092", "head_sha": "416c5e48aefcccfe945f7d4118415a3b992e017a"}
  ],
  "supersedes": ["https://github.com/NousResearch/hermes-agent/pull/85523"],
  "blocked_by": null
}
```

The example intentionally does not mark the object accepted, merged, released, or operationally verified. The record says exactly what its receipts support.

## Required verifier failures

A completion verifier must fail when any of the following is true:

1. the requested predicate is stronger than the observed evidence;
2. a remote predicate has no target locator or read-back;
3. a CI or review receipt binds to a different SHA than the claimed object;
4. a local path or patch digest is offered as proof of remote mutation;
5. an object marked active has a live `superseded_by` edge;
6. an historical PR is presented as canonical when a successor owns the contract;
7. required interlock or credit edges are missing;
8. the report uses generic `done`, `fixed`, `shipped`, or `green` without a namespaced predicate;
9. a write failure is omitted from the mutation journal;
10. a target-state claim cannot be reproduced by a third party from durable receipts.

## Exact-object CI rule

For a target object at version $v$ and a CI receipt at version $r$:

$$
\operatorname{CIValid}(O_v, r) \iff \operatorname{head}(O_v) = \operatorname{head}(r).
$$

Parent success, merge-base success, previous-head success, sibling-PR success, and nearby-main success are all non-evidence for this predicate. They may inform diagnosis; they may not satisfy verification.

# The same principles apply to a paper as to a pull request

This amendment is itself governed by the rule it adds.

| Pull request concept | Paper concept | Shared completion invariant |
|---|---|---|
| working tree | manuscript draft | mutable local work is not a submitted object |
| commit SHA | release digest | review and evidence bind exact bytes |
| CI run | reproducibility/source/format checks | checks identify the exact release they evaluated |
| review | independent technical or semantic witness | author is not sole acceptance authority |
| merge | accepted integration | acceptance and publication remain distinct |
| release | immutable paper version | later refinement creates a successor, not silent mutation |
| superseding PR | successor edition or amendment | lineage remains explicit and machine-readable |
| production verification | replication or operationalization | publication alone does not prove real-world behavior |

The original August 2026 PDF remains v1.0. This amendment is independently digestible. The composed v1.1 PDF is the canonical successor release for the expanded doctrine. The lineage is:

```text
All Gods Must Die v1.0
  --refined_by--> Amendment I: Task Completion Verification
  --composed_as--> All Gods Must Die v1.1
```

No page in v1.0 is silently replaced. The successor release changes the current doctrine while preserving the historical artifact that readers and citations may already reference.

# Threats, edge cases, and exact limits

## Read-back can still be stale

An API can return cached or eventually consistent state. Exact-object verification therefore prefers immutable identities and, when material, repeated or independent retrieval. A URL without a version identity is weaker than a URL plus commit SHA or digest.

## Exact-head green is not sufficient for every task

Exact-head CI proves the checks that ran. It does not prove maintainer acceptance, mergeability under future base movement, deployment, or behavior outside the tested surface. The completion contract must name the intended terminal predicate.

## Merge is not release

A merged commit may not be packaged, published, deployed, or adopted by users. Conversely, a paper can be published without being independently replicated. Lifecycle axes remain separate.

## Operational verification is scoped

A real-surface test proves behavior under the observed configuration, version, environment, and inputs. It is stronger than hermetic verification, not universal proof.

## Blocked is not failure theater

A blocked result is useful when it identifies the exact authority boundary, preserves prepared artifacts, and states what did not happen. It becomes theater only when the system stops at the blocker despite having another authorized, reversible route it failed to try.

## Completion contracts can be wrong

A badly specified contract may ask for the wrong predicate or target. The remedy is explicit contract refinement, not silent reinterpretation after execution. The mutation journal records who changed the contract, why, and which prior claims are superseded.

# Revised declaration

The doctrine now carries six laws:

> **No system component may accumulate authority without becoming legible.**
>
> **No claim may outrank its evidence.**
>
> **No actor may certify itself.**
>
> **No rule counts until it executes.**
>
> **No debt gets to hide behind institutional forgetting.**
>
> **No task is complete until the requested state exists at the authoritative target and is verified on the exact object.**

The sixth law makes the other five operational at the level of work itself. It prevents a system from applying adversarial verification to code while narrating its own execution loosely.

A patch is not a branch. A branch is not a pull request. A pull request is not an approval. An approval is not a merge. A merge is not a release. A release is not an operational proof. A superseded object is not the canonical object. A failed write is not a successful mutation.

Every predicate has an authority. Every authority has an object. Every object has a version. Every version has receipts. Completion is the exact conjunction the contract demanded - no weaker, and no stronger.

# Appendix A: source register

- *All Gods Must Die: Adversarially Verified Transformation*, Axl Ibiza, MBA, August 2026, original 68-page release.
- `Task Completion Verification.txt`, supplied execution record, SHA-256 `9e9987fb5290435eb1d1b03ce876b0bf4b5040128493eeea2b4dd295fa7b5d32`.
- [NousResearch/hermes-agent PR #80551](https://github.com/NousResearch/hermes-agent/pull/80551) - publication and doctrine contribution.
- [NousResearch/hermes-agent PR #85002](https://github.com/NousResearch/hermes-agent/pull/85002) - current Task 7 canonical pull request at the observed reconciliation.
- [NousResearch/hermes-agent PR #85523](https://github.com/NousResearch/hermes-agent/pull/85523) - historical Task 10 lineage, closed and superseded.
- [NousResearch/hermes-agent PR #90236](https://github.com/NousResearch/hermes-agent/pull/90236) - current Task 10 canonical successor at the observed reconciliation.

# Appendix B: release semantics

The release package should publish the following as one graph:

1. immutable v1.0 source PDF;
2. Amendment I in Markdown, LaTeX, and PDF;
3. composed v1.1 PDF and wrapper source;
4. task-completion record JSON Schema;
5. release manifest with content digests and typed lineage;
6. interlock update binding the amendment to PR #80551 and the webhook case-study objects;
7. exact Git commit and post-write read-back receipts.

Until item 7 exists, the package is locally produced and locally verified. Once the exact repository commit is created and read back, it is submitted and exact-object verified. It remains unmerged until GitHub records a merge.
