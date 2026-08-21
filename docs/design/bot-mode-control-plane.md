# Bot Mode control plane

## Status

Foundation contract. The first implementation is shadow-only: it does not
change runtime authorization, delivery, cancellation, provider selection, or
Desktop behavior.

Tracking issue: #91911.

## Problem

Bot Mode currently derives identity and authority independently from profile
names, canonical `Bot Chat` titles, gateway profile routes, Desktop connection
state, session-store placement, runtime provider/model/tool configuration,
background completion ownership, and local versus remote teammate delivery.

Those coordinates locate a candidate actor. They are not a current proof that
the actor owns the profile, runtime generation, credential, or side effect.
That produces one recurring class of defects: stale identity, configured intent
that differs from effective capability, lost completion ownership, incomplete
cancellation, and remote trust-boundary drift.

> Coordinates select. Current proof objects authorize.

## Foundation contracts

### `BotAddress`

```text
(install_id, gateway_instance_id, connection_id, profile_id)
```

Profile names and handles remain editable aliases. Durable room membership,
sessions, notifications, routines, mutations, cancellation targets, delivery
events, and audit records should converge on this exact address.

### `BotExecutionContext`

One authenticated turn binds:

```text
BotAddress
profile_config_revision
session_id / session_key / turn_id / task_id
authenticated_principal
source platform / user / chat / thread
runtime_snapshot_id
capability_grant_id / revocation_epoch
cancellation_scope_id
budget_id
trace_id / inbound_event_id / parent_event_id / hop_count
contract_version
```

The context is constructed only after authentication, route resolution, and
current owner resolution. Consumers must not recreate it from prompt text,
environment variables, cached roster rows, display names, or session titles.

### `RuntimeCapabilitySnapshot`

One immutable snapshot records the exact profile/runtime generation:

```text
grant_id
profile_id / profile_config_revision
runtime_snapshot_id / revocation_epoch
configured / requested / effective provider
effective model / API mode / reasoning / service tier
credential source identity (never secret material)
fallback reason
capabilities
```

The initial capability vocabulary is intentionally smaller and more stable than
the tool registry:

```text
local.read
local.write
network.read
network.write
external.message
peer.message
process.spawn
credential.use
profile.configure
destructive
```

Tools, MCP servers, and provider actions project onto these classes. Prompt
content may request an operation; it cannot manufacture a grant.

### `BotPolicyDecision`

A content-free result records the decision ID, operation, verdict, reason,
required capability, and bounded constraints. The evaluator fails closed on
profile, profile-revision, grant, revocation-epoch, runtime-snapshot, or
capability mismatch.

### `ShadowPolicyComparison`

During migration:

```text
legacy_allowed
policy_allowed
matches
effective_allowed = legacy_allowed
```

A mismatch is evidence. It cannot grant or deny work until the consuming
boundary explicitly migrates and its parity matrix is green.

## Current `message_agent` mapping

Phase 1 records the current #91802 gate order exactly, including two
asymmetries. It does not endorse or silently repair them.

Injection (`ensure_message_agent_tool`):

1. disabled Bot protocol denies;
2. an already-present schema allows before title/install revalidation;
3. otherwise the exact `Bot Chat` title is required;
4. then a Bot-Mode-managed install is required.

Dispatch (`message_agent_tool`):

1. the exact `Bot Chat` title is required;
2. a Bot-Mode-managed install is required;
3. the protocol toggle and schema presence are not consulted.

Parity tests import the current implementation and prove the typed mapping
tracks those semantics. A later boundary-migration PR must select one canonical
contract and delete the superseded gate in the same change.

## Rollout

1. **Foundation:** immutable proof objects, pure fail-closed evaluation, exact
   legacy mapping, and parity tests. No production call sites.
2. **Authenticated shadow construction:** build context at gateway ingress,
   derive one effective snapshot, and emit bounded content-free mismatch
   evidence while legacy remains authoritative.
3. **Boundary migration:** migrate `message_agent`, tool/MCP injection, profile
   mutation, group interruption, peer delivery, completion ownership, and
   destructive/file/network operations one at a time.
4. **Durable settlement:** extend the shared delivery ledger into an
   at-least-once, idempotent Bot mailbox and add a cancellation tree spanning
   room turns, member turns, delegated children, background jobs, waits, and
   queued events.
5. **Canonical read model:** Desktop, mobile, API, logs, and `/status` consume
   one revisioned projection of exact owner, lifecycle, effective runtime,
   capability, budget, pending work, and terminal failure reason.

## Interlocks

Preserve existing contributor ownership and compose it onto this spine:

- #91802 — structured `message_agent`;
- #90329 — source-qualified cross-connection room identity;
- #89455 — effective toolset/MCP runtime truth;
- #90954 — completion-loss evidence and immediate stopgap;
- #91889 — visible group interruption and late-reply fencing;
- #91862 — authenticated Bot roster/read model;
- #88819 — peer credential redirect boundary;
- #91832 — permanent configuration refusal; and
- #73923 — shared delivery-ledger substrate.

## Acceptance invariants

The completed class must prove:

1. no profile crosses another profile without an explicit current grant;
2. no side effect commits without a decision tied to the exact context;
3. every accepted event reaches a terminal state;
4. cancelled or superseded work cannot append a late visible reply;
5. displayed owner/runtime/capability equals what actually ran;
6. message content cannot increase authority;
7. restart between delivery states does not lose the event;
8. duplicate or reordered delivery does not duplicate a side effect;
9. remote redirects cannot move credentials to another origin; and
10. deleting a profile prevents late work from resurrecting it.

## Non-goals of this PR

No production wiring, transport rewrite, durable inbox schema, Desktop change,
approval UI, provider/catalog behavior change, profile-ID migration, or switch
away from legacy authority.
