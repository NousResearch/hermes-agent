# Ownership Policy: Authenticated Dashboard Session REST APIs

**Status:** Proposed policy (open for maintainer review)
**Source issue:** NousResearch/hermes-agent#79498 (design; driven by #62549, transport fix PR #79488)
**Applies to:** The two HTTP surfaces that expose authenticated session data:

1. **Web dashboard** — `hermes_cli/web_server.py` + `hermes_cli/web_routers/sessions.py` (15 endpoints; shared session-token / OAuth / basic auth gate)
2. **OpenAI-compatible API server** — `gateway/platforms/api_server.py` (11 `/api/sessions*` endpoints plus the model/run surface; per-profile `API_SERVER_KEY` Bearer)

Both bind to loopback by default and **fail closed** on non-loopback binding unless an auth provider is configured (`website/docs/user-guide/features/web-dashboard.md`, §auth-gated mode).

This policy defines who owns these APIs, who may change them, how access to sessions is authorized, how sessions must be managed over their lifecycle, how changes are reviewed and approved, and how security incidents are escalated. It is written to close the gaps identified in the research findings (see §7). Changes to session data through any other surface (CLI, TUI, gateway adapters) are out of scope except where they share the accessors discussed in §2.6 and §4.3.

---

## 1. Ownership & Responsibility

### 1.1 Primary owner

The **Dashboard & Platform APIs team** (the maintainers responsible for the web dashboard and the gateway API server) owns both surfaces end to end. This is the group with merge rights over `hermes_cli/web_server.py`, `hermes_cli/web_routers/sessions.py`, `hermes_cli/dashboard_auth/`, `gateway/platforms/api_server.py`, `hermes_state.py` (session accessor layer), and `hermes_state_common.py` (schema). If the repo later gains a `CODEOWNERS` file, this team's GitHub handle(s) must be listed against those paths.

| Responsibility | Owner |
|---|---|
| Functional changes (routes, handlers, behavior) | Dashboard & Platform APIs team |
| Security controls (auth gate, token handling, authorization, key rotation) | Dashboard & Platform APIs team, reviewed by Security |
| Session data model & schema (incl. `user_id`) | Dashboard & Platform APIs team, co-signed by State/Storage owner |
| Incident response (SEV) | On-call of the Dashboard & Platform APIs team; Security for `severity-critical` (see §5) |
| Documentation (`web-dashboard.md`, this policy) | Dashboard & Platform APIs team |

### 1.2 Escalation owners

- **Security review owner:** the project's Security point of contact (the role named in `SECURITY.md` reporting process).
- **Overall repo maintainers:** retain final decision authority on any policy conflict.

### 1.3 Single accountable owner

Every change touching a session endpoint must have **exactly one accountable engineer** (the author or the merge-merging reviewer). Ambiguity about who is accountable is itself a defect; a PR with no named owner is not mergeable (§4).

---

## 2. Access Control Rules

These rules govern Create, Read, Update, Delete (CRUD) on session rows through both surfaces. Two authorization models currently exist — **shared token/key** (no per-user principal) and **per-user principal** (dashboard OAuth/basic login). Until per-user authorization ships, the shared-key model is treated as *administrator-equivalent* and governed by §2.4.

### 2.1 Authenticated principal — the model this policy commits to

The project must converge on a single, consistent boundary for session ownership. This policy adopts:

> **An authenticated dashboard user is a first-class principal. Each session is owned by the principal who created it. Anonymous/shared-token access to session data is restricted to an explicit admin capability, never the default.**

Consequence (consistent across *every* read and mutation, including the message-write and model/run surfaces of §2.2):
- A principal may list, read, update, archive, pin, delete, export, import, fork, chat into, and prune **only sessions they own** (rows where `sessions.user_id = <principal>`).
- Anonymous or shared-token requests may access sessions **only through the admin capability** in §2.4.
- Any endpoint or shared accessor that does not enforce this is a security defect and must be fixed before it is considered production-safe.

### 2.2 CRUD matrix

| Operation | Allowed for | Notes / required control |
|---|---|---|
| **Create** (POST `/api/sessions`) | Any authenticated principal; anonymous only via admin capability | Handler **must write `user_id = <principal>`** on every created row. Today it does not (see §2.5, §7 gap #2) — this is an acceptance blocker for the migration. |
| **Import** (POST `/api/sessions/import`, dashboard) | Any authenticated principal for rows it owns; anonymous only via admin capability | Import is a bulk external insert with its own trust domain. Every imported row **must** receive `user_id = <importing principal>` (or an explicit admin-owner backfill, audited per §3.4). Import must never bypass ownership scoping — imported rows must be attributable, and cross-profile data must not be pullable via import. |
| **Read** (GET list / detail / messages / export / search / stats / latest-descendant) | Owner of the row; admin capability via shared token/key | `list_sessions_rich`, `search_sessions`, `count_sessions`, and `resolve_session_id`/`get_session` must be restricted by `user_id` at the point of use (route layer) — see §2.6 for where scoping lives. |
| **Update** (PATCH rename/archive/pin/end_reason; `?profile=`) | Owner of the row; admin capability | PATCH must scope to the row's `user_id`. `?profile=` selection (see §2.4) may open non-owned profiles **only** under the admin capability. |
| **Fork** (POST `/api/sessions/{id}/fork`, API server) | Owner of the parent row; anonymous only via admin capability | Fork is a lineage-branching write that creates a new child row. The child's owner is the **forking principal** (who may differ from the parent's owner); `user_id` is written on fork-created rows too, consistent with the Create acceptance blocker. Only the parent row's owner may fork it. |
| **Chat / model-lock** (POST `/chat`, `/chat/stream`, `/model`, API server) | Owner of the session; admin capability | These write *into* a session. Message writes and model-lock are scoped to the session's owner, matching the row-level matrix. An anonymous/shared-key caller may not write into a session it does not own except under the admin capability. |
| **Model / run surface** (POST `/v1/chat/completions`, `/v1/responses`, `/v1/runs`, API server) | Admin capability (per-profile `API_SERVER_KEY`) | This surface creates/runs models under a profile's key. Treat it as admin-capability-governed today (§2.4); a per-user principal may reach it only where their account maps to the profile. |
| **Delete** (DELETE one, bulk-delete, delete empty) | Owner of the row; admin capability | Delete is per-row authorization like any other. On delete, disk transcript cleanup must be scheduled/executed; it must not silently leave orphaned transcript files (see §2.7). |
| **Prune** (POST prune, DELETE empty, cleanup) | Admin capability (shared-token/key mode); owner-scoped otherwise | Pruning is a bulk, cross-session operation — do not scope it to a single owner by default; require the admin capability or an explicit owner filter. |

### 2.3 Routes are not exempt

The admin capability is **not** achieved by merely holding the gate token or API key at the middleware layer; it must be re-checked in each handler that mutates or reads session data. "It passes the middleware" is never sufficient authorization for a session row.

### 2.4 Admin capability & profile selection

- The shared session token (dashboard) and the per-profile `API_SERVER_KEY` (API server) are defined as the **admin capability** for their surface: a request authenticated with either may access any session in the profile and may select any profile. This is today's effective behavior (§2 shared-key = superuser) and is acceptable **only** because these credentials are per-deployment secrets, not per-user. It is a documented risk, not a feature for normal users.
- `?profile=` remains caller-controlled, but its use must be gated: an anonymous/shared-token request may select any profile; a per-user principal may select only the profile to which their account maps (or must be denied unless an explicit admin grant exists).
- **Credentials are the capability.** Anyone who holds the shared token or an `API_SERVER_KEY` is granted admin-equivalent access to that profile. Therefore these secrets are governed by the same rules as the master deployment credentials (§3.2): least-spread, rotated, logged, and never committed.

### 2.5 Legacy `user_id IS NULL` rows

Rows created before this policy (notably `api_server` POST-created rows with `user_id = NULL`) have no proven owner. The policy is the "no safe compatibility rule" resolution:

1. **Migration must assign an owner**, not leave rows ambiguous. During rollout, run a backfill that attributes each `NULL`-row to the profile's configured admin/owner account (the deployment operator) and records the backfill in the audit log.
2. **Do NOT show `NULL`-owned rows to all users by default** (history leak), **and do NOT strand them forever hidden** (orphans a user's history). The chosen rule is: backfill to the profile owner, then apply normal ownership rules thereafter.
3. Until backfill runs, any code path that would display or mutate `NULL`-owned rows under a per-user principal must be disabled (fail closed).

### 2.6 Where authorization lives

Adding `user_id` scoping to the **shared** `SessionDB` accessors (`list_sessions_rich`, `search_sessions`, `count_sessions`) changes semantics for every consumer (CLI, TUI, gateway, dashboard). **Decision:** authorization scoping lives at the **route layer** (dashboard router and API-server handlers), which passes the authenticated principal into the accessor as an explicit ownership filter. The shared accessors gain an *optional* `user_id` filter parameter (default `None` = unrestricted, preserving existing local CLI/TUI/gateway behavior), **not** a hard-coded default that would break local single-user tooling. The route layer always supplies the filter for HTTP requests.

### 2.7 Delete must also purge transcripts

Hard DELETE of a session row must also schedule on-disk transcript cleanup. Today cleanup is deferred to a later prune pass with an inconsistency risk (row gone, transcript file remains). Policy: a delete either (a) synchronously removes the associated on-disk transcripts, or (b) enqueues a cleanup job and logs the file paths for the run; it must never leave transcript files with no owning session and no follow-up. The same applies to fork/import cleanup on failure — a partially imported or orphaned fork must not strand transcript files.

---

## 3. Session Lifecycle Requirements

### 3.1 Timeouts

- **Dashboard session token:** enforce an idle timeout. A dashboard session token that has been idle for **X minutes (default 30)** or has exceeded its absolute TTL (**default 8 hours**) must be rejected and re-issued on next authenticated request. Configurable via deployment config; defaults in place out of the box.
- **API-server bearer key (`API_SERVER_KEY`):** keys do not expire by idle, but must be **rotated** — at minimum every 90 days and immediately on any suspected exposure (§3.3).
- Non-loopback binding **fails closed** unless an auth provider is configured (already enforced; keep it). This default is non-negotiable.

### 3.2 Credential handling

- Never store secrets in plaintext in the repo, logs, or `state.db`.
- Never commit `API_SERVER_KEY` or session tokens.
- Rotate the shared dashboard token and per-profile keys on a defined cadence (90 days) and on personnel change or exposure.
- Limit token spread; a per-user principal is strongly preferred over sharing one deployment-wide token.

### 3.3 Revocation

- **Logout revokes the principal's dashboard session token** immediately (token store invalidation).
- Add a "revoke all sessions for principal X" capability (admin), which invalidates the principal's tokens and marks their active rows `expiry_finalized` per the existing messaging lifecycle pattern (`gateway/session.py`, `gateway/run.py`, `docs/session-lifecycle.md`). This is today's gap — logout currently does not cascade to dashboard session rows.
- On suspected `API_SERVER_KEY` exposure: rotate immediately, invalidate the old key, and revoke any sessions created under the affected profile that cannot be attributed, per §2.5.

### 3.4 Audit logging

Every session-affecting operation must produce an audit record with at least: timestamp, principal (or `admin:<token-scope>` for shared-token/key access), operation (CRUD / import / fork / prune / export / chat-write / model-lock / profile-switch), target session id(s), and outcome (success/denied). Audit records must be:
- Append-only and non-repudiable in practice (write-only from the API layer).
- Retained for a defined period (**minimum 180 days**, or as law/enterprise policy requires — longer is safer given these are conversation transcripts).
- Searchable so `who created/mutated/deleted which session, when` is answerable (this closes the audit gap — no such trail exists today).

No destructive or bulk operation (bulk-delete, prune, profile switch, admin read of another profile) may run without an audit record.

---

## 4. Review & Approval Process for API Changes

### 4.1 Mandatory review

Any change to a session endpoint, its auth gate, its authorization logic, the shared session accessors (when touched), or the session schema requires:

1. **Code review by at least one other Dashboard & Platform APIs team member.**
2. **A named accountable owner** on the PR (the author or the merge-merging reviewer) — a PR without a named owner is not mergeable (§1.3).
3. **Security review** if the change touches auth, authorization, token handling, key rotation, or any cross-profile data access. Security sign-off is recorded on the PR.

### 4.2 Security-impacting changes require escalation

Classify each change as **normal**, **security-relevant**, or **security-critical** (§5). Security-relevant and security-critical changes require the Security owner's review **before merge**, not after.

### 4.3 Backwards-compatibility gate

Because the shared accessors are used by CLI/TUI/gateway/dashboard, any change that alters their default behavior must:
- Be additive (new optional parameter) rather than changing existing defaults, where feasible;
- Declare the compatibility impact in the PR description;
- Not change `NULL`/`user_id` semantics without the §2.5 migration plan attached.

### 4.4 Tests required

- Per-endpoint tests for the **owner-can / non-owner-denied / admin-capability** matrix, covering create, import, fork, chat/model-lock, delete, prune, and `?profile=` selection.
- Tests for **cookie/token mode and loopback mode** (the issue's explicit requirement).
- Tests asserting **create writes `user_id`**, **fork writes `user_id` on the child**, **import assigns `user_id`**, and **delete triggers transcript cleanup**.
- A regression test proving `?profile=` selection is denied without the admin capability.

### 4.5 Documentation

Merging a user-visible change to these APIs requires updating `website/docs/user-guide/features/web-dashboard.md` and this policy in the same PR (or an explicitly tracked follow-up issue). A change that is shipped but undocumented is not complete.

---

## 5. Escalation Path for Security Issues

### 5.1 Severity classification

| Severity | Definition | Examples |
|---|---|---|
| **Severity 1 (critical)** | Active or imminent cross-user/cross-profile data exposure or takeover on a public deployment | Unauthenticated enumeration of another profile's sessions; per-user auth bypass; key exposure in logs/repo |
| **Severity 2 (high)** | Significant unauthorized read/write/delete within a profile without per-user separation | Shared-token/key access to data outside its documented scope; `?profile=` cross-profile access without the admin capability; import/fork writing rows with a misattributed owner |
| **Severity 3 (medium)** | Localized inconsistency or hardening gap, no cross-user exposure today | Deferred transcript cleanup leaving orphans; missing audit record on a delete; undocumented endpoint |
| **Severity 4 (low)** | Cosmetic / doc / hygiene | Stale docs, missing comment, non-blocking lint |

### 5.2 Escalation steps

1. **Identify & contain immediately.** Reporter or on-call disables/restricts the affected surface (e.g. revert the change, rotate keys, bind loopback, revoke tokens). Containment precedes diagnosis.
2. **File/raise the issue** referencing the affected endpoint(s) and severity. Critical (S1) and high (S2) issues bypass the normal review queue (§4) and go straight to the Dashboard & Platform APIs team on-call **and** the Security owner.
3. **Notify Security and repo maintainers** for S1/S2. For S1, involve maintainers within **1 hour**; for S2 within **1 business day**.
4. **Root-cause & fix** with the accountable owner named; apply the §4 review gate for the fix.
5. **Post-incident:** write a short write-up (cause, blast radius, affected sessions/users, remediation, prevention), attach it to the issue, and log the audit trail. Document any new control that should be added back into §2–§3.

### 5.3 Never-tolerate list (report at S1/S2 regardless of deployment)

- Session data readable/writable by a principal other than its owner without the admin capability.
- `user_id` scoping absent on any HTTP session route (list/read/delete/update/prune/import/fork/chat/model-lock).
- `?profile=` selecting a profile not authorized for the caller.
- Import or fork creating rows with a misattributed or absent `user_id`.
- Session tokens or `API_SERVER_KEY` committed, logged, or otherwise exposed.

---

## 6. Enforcement & Ownership of This Policy

- This policy file is owned by the Dashboard & Platform APIs team and must be **reviewed at least annually** and on any security-impacting change.
- The Security owner reviews the policy's access-control and escalation sections.
- A breach of the §2 explicit rules (e.g. shipping a route with no ownership scoping) must be treated as a Severity 2 incident.

---

## 7. Mapping to research findings (traceability)

| Research gap (§5 of findings) | Where addressed here |
|---|---|
| 1. No authenticated-principal model | §2.1, §2.2, §2.3 |
| 2. `user_id IS NULL` legacy rows | §2.5 |
| 3. Shared-key = implicit superuser | §2.4, §3.2 |
| 4. `?profile=` caller-controlled | §2.4, §5.3 |
| 5. Shared `SessionDB` blast radius | §2.6, §4.3 |
| 6. CRUD asymmetry (create no user_id, deferred delete cleanup) | §2.2, §2.7 |
| 7. Profile-trust boundary ambiguity | §2.1 (explicit choice: per-user principal + admin capability) |
| 8. No expiry/revocation/audit tied to principals | §3.1–§3.4 |
| 9. No incident/escalation owner | §1.1, §1.2, §5 |

---

*End of policy. Proposed for adoption by the Dashboard & Platform APIs team and Security.*
