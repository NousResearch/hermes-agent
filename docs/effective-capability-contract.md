# Effective Capability at Consuming Boundaries

Status: Proposed architecture contract

> **Configuration requests. The consuming boundary proves. The effect settles.**

Hermes has many ways to express intended capability: configuration flags,
provider and model catalog entries, installed plugin metadata, deployment
plans, remembered routes, authenticated accounts, process identifiers, and
resolved filesystem paths. Those values are useful inputs to selection and
planning. None of them, by itself, proves that the exact operation is
currently available and authorized at the boundary that will perform it.

This contract defines the distinction between configured intent and effective
runtime capability. It composes the repository's existing identity,
generation, proof-scope, realization, and settlement architecture rather than
creating a parallel authority model.

## State model

Capability-sensitive paths should preserve these states as separate facts:

```text
configured
  -> resolvable
  -> authenticated
  -> policy-admitted
  -> currently effective
  -> exercised
  -> settled
```

- **Configured**: a user, administrator, manifest, or plan requested the
  capability.
- **Resolvable**: the exact actor and target can be located uniquely in the
  current namespace.
- **Authenticated**: the current principal or credential claimant is accepted
  where authentication applies.
- **Policy-admitted**: current policy authorizes this operation and scope.
- **Currently effective**: the exact live substrate can perform the operation
  now, under the current identity and generation.
- **Exercised**: the consuming boundary attempted the operation using that
  proof.
- **Settled**: terminal postconditions prove what actually committed or became
  observable.

An earlier state never implies a later one. A transition may also regress at
runtime because a capability was revoked, a credential rotated, a plugin or
peer disappeared, a process restarted, a route changed, or a policy revision
superseded the one previously observed.

## Normative rules

### 1. The side-effect owner is the authority boundary

The component that will perform the operation must obtain or revalidate the
current effective-capability proof at the last responsible boundary. Upstream
selection and preflight may narrow candidates and improve diagnostics, but
cannot authorize a later side effect on their own.

A proof produced by one component is transferable only when the consumer can
verify its full scope against current authoritative state.

### 2. The proof must cover the exact effect

The semantic proof must bind, as applicable:

- actor or owner identity;
- operation or capability;
- exact target or resource identity;
- route, profile, tenant, or other namespace dimensions;
- current process, adapter, activation, deployment, or credential generation;
- policy or configuration revision used for admission;
- realized substrate identity, such as the loaded plugin, live peer, bound
  target, installed artifact, or repository installation;
- observation time and any expiry or invalidation condition; and
- a typed verdict.

The proof must not contain secrets. A non-secret claim identifier or digest is
preferred where credential identity is required.

This is a semantic contract, not a requirement that every subsystem import one
universal `EffectiveCapabilityProof` class. Subsystems may use native types so
long as they preserve the same authority dimensions and verdicts.

### 3. Proof consumption and effect must be atomic with respect to invalidation

A check performed before an asynchronous gap, redirect, retry, lock release,
process handoff, registry replacement, or target substitution may already be
stale. The consumer must either:

1. validate and exercise the capability inside one boundary that excludes
   relevant invalidation; or
2. revalidate the exact proof after reacquiring authority and immediately
   before the effect.

If the operation resolves a different actor, target, route, or generation than
the proof names, it must fail closed or obtain a new proof. It must not widen
scope or fall back to ambient/default state.

### 4. Failure states remain distinct

At minimum, capability-sensitive consumers must be able to distinguish:

- **effective** — the exact operation is currently available and authorized;
- **absent** — an optional capability is not installed, configured, or
  present;
- **denied** — policy or authorization explicitly refuses the operation;
- **ambiguous** — more than one current claimant or target matches;
- **unavailable** — the intended capability exists conceptually but its live
  substrate cannot currently be reached or realized; and
- **stale** — the supplied proof names a superseded identity, generation,
  policy revision, or substrate.

Subsystems may add richer reasons such as unsupported, malformed, quarantined,
or externally managed. They must not collapse these states into one boolean or
misreport ordinary optional absence as an operational failure.

### 5. Coordinates and projections cannot become authority by reuse

The following can select or describe candidates but cannot independently
authorize mutation:

- configured booleans or mode names;
- display names and unqualified profile names;
- catalog membership or a declared feature flag;
- path, file, PID, port, or directory existence;
- an installed package or plugin manifest;
- successful authentication without operation-specific authorization;
- success on an adjacent or metadata-only API;
- a deployment plan that a physical consumer silently reconstructs or widens;
  or
- a prior capability verdict from an older generation.

Legacy or inferred values may remain usable for display, diagnostics, and
explicitly read-only degradation. They do not acquire mutation authority merely
because no exact proof is available.

### 6. Exercise is not settlement

A capability may be effective when exercised and still fail to settle. Exit
zero, request acceptance, process existence, file presence, or a successful
transport write are events, not terminal proof. The operation's completion
contract must separately establish the required postconditions and preserve a
receipt for the exact identity and generation that performed the effect.

## Acceptance matrix

Every concrete adoption should include the applicable adversarial cases below.

| Scenario | Required result |
|---|---|
| Capability is configured but an optional plugin is absent | Return ordinary `absent`; do not expose or invoke the capability and do not flood error logs. |
| A capability is revoked after initial selection | The consuming boundary observes the revocation and refuses use. |
| Two current claimants match the same durable selector | Return `ambiguous`; never choose first/current/default. |
| Credentials rotate or an adapter/process restarts | Reject the old generation as `stale`; require a current claimant. |
| The intended target is not yet realized or reachable | Return `unavailable` and fail closed; never widen to a fallback bind, route, repository, or deployment. |
| Policy changes between preflight and effect | Revalidate at the consumer and return `denied` or `stale`. |
| A redirect, retry, or handoff changes the actor or target | Invalidate the prior proof and obtain a new exact proof. |
| The operation reports success but required postconditions are absent | Record `exercised` but not `settled`; do not report completion. |
| A read-only legacy route lacks exact authority | Preserve degraded read/display behavior; reject mutation. |

Tests should exercise the real consuming boundary, not only a helper predicate
or configuration parser. At least one test should invalidate or replace the
capability after initial selection to prove that the implementation does not
cache intent as authority.

## Review checklist

For each capability-sensitive change, reviewers should be able to answer:

1. What source expresses configured intent?
2. What source of truth establishes current effective capability?
3. Which exact actor, operation, target, namespace, policy, and generation does
   the proof bind?
4. Which component owns the side effect and consumes the proof?
5. Can any asynchronous or process boundary invalidate the proof before use?
6. Are absence, denial, ambiguity, unavailability, and staleness represented
   distinctly?
7. Do revocation, replacement, rotation, and duplicate-claim tests reach the
   real consumer?
8. What postcondition turns exercise into settlement?

A design that cannot answer these questions is still describing configured
intent, not effective capability.

## Existing interlocks

This contract composes existing architecture owners:

- #90866 — proof-carrying state from source to side effect;
- #90142 — monotonic qualified identity;
- #90144 — proof scope equals mutation scope;
- #90145 — durable generation fencing;
- #90049 — typed completion and settlement proofs;
- #90150 — built artifacts and real peers belong to the system under test;
- #90200 and #90230 — operation-specific GitHub repository-object authority;
- #91230 — exact-object task-completion verification;
- #89252 — current credential-claimant and adapter-generation proof;
- #91316 — immutable deployment admission consumed by physical executors.

Concrete evidence already demonstrates the rule:

- #91695 rechecks live Developer Mode at every browser-control selection and
  revokes already-attached controllers;
- #91720 waits for and binds the actual configured Nix target rather than
  treating configuration as realization; and
- #91828 treats a missing optional Desktop plugin half as inventory absence,
  while removing stale registrations when a previously realized entry
  disappears.

Those fixes are examples, not substitutes for class-wide adoption. Each
remaining subsystem keeps its existing implementation owner and must apply this
contract at its own consuming boundary.

## Non-goals

- This contract does not require speculative or expensive probing when the
  first real operation is the only authoritative capability signal. In that
  case, the operation result must be classified truthfully and must not be
  presented as success or as evidence about a different target.
- It does not require one process-global registry or one cross-language proof
  type.
- It does not remove configured-state reporting. Status surfaces should expose
  configured and effective state separately when both are useful.
- It does not make every temporary outage a permanent denial. Recovery may
  obtain a fresh proof; it may not reuse a stale one.
