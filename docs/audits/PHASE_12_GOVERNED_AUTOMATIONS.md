# Phase 12 — Governed Automations

Phase 11 surfaced the runtime's scheduler and refused to offer *create*. This closes that
gap without opening the hole the refusal was protecting.

---

## 1. The problem being solved

`cron.jobs.create_job` takes a free-text prompt and a schedule. A control plane calling it
directly would hand an agent a **recurring instruction that NOVA never compiled and no
policy reviewed**, running on a timer, indefinitely. Every other agent behaviour on this
platform goes through spec → policy → `pre_tool_call`; an automation created that way
would go around all three.

So an automation became a spec.

```
Tenant
  ↓  automations/*.yaml  or  POST /platform/v1/automations
AutomationSpec              nova/spec/automation.py
  ↓  shape validation
NOVA compiler               nova/automations/compile.py
  ↓  agent exists · permissions ⊆ agent · knowledge ⊆ agent · channels grant agent · schedule parses
CompiledAutomation          carries job_kwargs + tenant-salted digest
  ↓  adapter, the only package that may name the runtime
Hermes cron job             cron.jobs.create_job under use_cron_store(profile)
```

There is no second path. `create_automation` on the runtime contract takes a
**`CompiledAutomation`**, never a prompt — a method that accepted free text would reopen
the hole by API design.

## 2. What an automation declares

| Field | Purpose |
|---|---|
| `id`, `title` | Identity, reviewable in a diff |
| `agent` | Who runs it. Its profile is where the work executes, so its policy is the ceiling |
| `schedule` | Parsed by the **runtime's** parser, never a second grammar |
| `objective` | The instruction. Bounded at 4000 chars — longer is a procedure, and belongs in the knowledge base |
| `enabled` | Created paused when false, so a job is never briefly live |
| `permissions` | Business actions the work needs. **Subset of the agent's grant** |
| `knowledge` | Corpora it may read. Subset of the agent's |
| `channels` | Connections it may reach. Must already grant the agent |
| `reason`, `owner` | Governance metadata, carried into the audit record |

Tenant is not a field: it comes from the bundle, so an automation cannot name a tenant
other than the one loading it.

## 3. What is enforced where

This is the part worth being precise about rather than implying.

**At compile time, by NOVA:** the agent exists and is enabled; declared permissions,
knowledge and channels are a subset of what that agent already holds; the schedule parses;
the objective is present and bounded.

**At runtime, by Hermes:** what the agent may actually do. The work executes inside the
owning agent's profile, so the agent's compiled `nova-policy.json` and the fail-closed
`pre_tool_call` hook are the real ceiling.

**The consequence, stated plainly:** declaring *fewer* permissions on an automation than
its agent holds is a **declaration, not a narrowing**. The runtime still allows the agent
everything the agent may do. The subset check stops an automation being a
privilege-escalation path — it can never reach past its agent — but it does not sandbox an
automation *below* its agent. Doing that needs a derived profile per automation, the way
Phase 10 derives one per channel grant. That is not built here and is not claimed.

## 4. Security decisions

| Decision | Why |
|---|---|
| Compiler is the only path to the scheduler | A raw-prompt API would be the Phase 11 hole with extra steps |
| `create_automation` takes a compiled object | Makes the safe path the only representable one |
| Permissions must be a subset | An automation asking for what the tenant never granted is an escalation on a timer |
| Schedule parsed by the runtime's parser | A second grammar would drift from the thing that decides when jobs run |
| `deliver="local"`, never an origin | The runtime defaults `deliver` to `origin` whenever an origin is passed, which would route output through a channel this automation was never reviewed for |
| Created paused when disabled | Not created-then-paused; never briefly live |
| Tenant-salted digest | Identical declarations in two tenants are not the same reviewed artifact |
| Delete drops the registry entry | Provenance does not accumulate for automations the runtime no longer has |
| Registry failures are swallowed | The runtime owns existence; a provenance write that failed must not turn a successful act into an error to reconcile |
| Every intent reaches a terminal phase | An exception used to leave `intent` with no `committed`/`failed`, so the log read as though the act *might* have happened |
| 404, not 403, for another tenant's automation | Distinguishing them confirms ids |
| `create` is not in `AUTOMATION_ACTIONS` | Creating is a collection-level act through the compiler, not a verb on something that exists |

### Isolation

Structural, as in Phase 11: a cron store is a profile directory, a NOVA agent *is* a
profile, an agent belongs to one tenant. Cross-agent reads and writes return
`None`/`False` from the runtime's own API because it is a different file. The registry is
additionally tenant-checked — one comparison, defence in depth.

### RBAC

| Route | Role |
|---|---|
| `GET /automations` | viewer |
| `POST /automations` (create) | **admin** |
| `POST /automations/{id}/decide` (pause/resume/delete) | **admin** |

`WRITE_ROUTES` refuses undeclared routes outright, so a new write is unreachable until
somebody declares its role. Tested: a viewer is refused at `ControlAPI.write` for create,
pause, resume and delete, and the runtime is never reached.

### A note on testing RBAC over loopback

`nova serve` treats **any loopback caller as local admin**, by design
(`nova/control/server.py:113`) — anyone on the host can read the principals file off disk
anyway. So a `curl` from `127.0.0.1` proves nothing about roles. The role gate is asserted
where it is enforced, in `ControlAPI.write` with a viewer principal.

## 5. Control Center

The Automations screen gained: **Declare** (a form that posts an AutomationSpec),
**delete behind a confirmation** that names the automation, **governance on each card**
(declared by, reason, permissions, digest), and an explicit line separating the two facts:

> *Declared. The runtime has recorded no execution yet.*
> *Declared, and the runtime has recorded an execution.*

An automation created outside NOVA shows "Created outside NOVA — no declaration on
record" rather than being given invented provenance.

A compiler refusal surfaces verbatim in the form — *"declares permission(s) crm_refund
that agent 'operations' does not hold"* — which is more useful than anything the form
could guess in advance.

## 6. Validation

| Check | Result |
|---|---|
| `tests/platform/test_governed_automations.py` | **35 passed** |
| `tests/platform/test_automations.py` (Phase 11) | **25 passed** |
| Full platform suite | **752 passed** |
| kanban + platform, per file | see final report |

### Live chain

Control Center → NOVA API → compiler → policy check → Hermes cron API → runtime, all
against a real `nova serve`:

```
1. create (201)                     5. resume (200)
2. ungranted permission (400)       6. delete (200)
3. invalid schedule (400)           7. cross-site Origin (403)
4. pause (200)                      8. non-JSON body (415)

automation.declared | intent    → committed
automation.decision | intent    → committed   (pause, resume, delete)
```

Browser-driven: declare → card with governance → compiler refusal surfaced → delete
confirmation → removed. No console errors.

### Execution remains unproven

**No scheduled execution was observed, and nothing claims one was.** Firing requires a
running Hermes gateway — the ticker has no standalone daemon — and this environment's
proxy refuses provider egress, so an agent-backed job could not complete even if the
ticker ran. Accordingly the screen reports *"Nothing is running these schedules"*, every
card says *"The runtime has recorded no execution yet"*, and no test asserts that a
schedule fires.

## 7. Remaining limitations

1. **Execution is unproven** (§6). The distinction between declared and verified is
   surfaced rather than papered over, but it remains a distinction this deployment sits
   on the wrong side of.
2. **No per-automation sandbox.** Declared permissions are validated, not narrowed (§3).
3. **No edit.** Changing an automation means delete and re-declare, which produces a new
   digest and a clean audit pair. In-place update would need a diff story and a digest
   transition rule.
4. **Bundle-declared automations are validated but not materialised by `nova apply`.**
   `automations/*.yaml` compiles at load time — a bad one fails `nova validate` — but
   applying a bundle does not yet create the cron jobs. Control-API creation is the live
   path. Wiring it into `apply` needs a reconcile story (what happens to an automation
   deleted from the bundle) that is deliberately not guessed at here.
5. **Schedule phrases are not checked at bundle load**, because `load_bundle` has no
   runtime. They are checked wherever a runtime exists — `nova validate`/`apply` and the
   control API.
6. **`trigger_job` (run now) is still withheld.** It executes the objective immediately;
   it belongs with the execution story, not the declaration one.
