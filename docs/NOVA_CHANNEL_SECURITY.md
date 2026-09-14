# NOVA Channel Security Model

A channel is the first thing in this product that a stranger can send bytes to. Everything
else — the control plane, the work board, the knowledge index — is reached by someone who
already holds a credential. A connected WhatsApp number is reachable by anybody who knows the
number, and the message they send becomes the text a worker reads.

So this document is written as a threat model rather than a feature list, and it is explicit
about which defences are the runtime's, which are NOVA's, and which do not exist yet.

---

## 1. Trust boundaries

```
  anyone on the internet
          │
          │  (1) provider authentication + webhook signature   ← RUNTIME
          ▼
   provider adapter
          │
          │  (2) sender allowlist, default-deny                ← RUNTIME
          ▼
   inbound routing
          │
          │  (3) route → agent, refused if agent not served    ← RUNTIME, fed by NOVA
          ▼
   the agent's profile
          │
          │  (4) tool policy, knowledge scope, delegation      ← NOVA (Phases 2–5)
          ▼
   the model
```

**NOVA owns (3)'s input and nothing else on this path.** That is deliberate: (1) and (2) are
implemented, tested and battle-worn in the runtime, and a second implementation would be a
second thing to get wrong.

---

## 2. What the runtime already defends against

Verified by reading the implementation during the Phase 9 audit; file references are in
`NOVA_CHANNEL_AUDIT.md`.

| Threat | Defence | Where |
|---|---|---|
| Forged webhook | HMAC-SHA256, Svix-compatible signatures, constant-time compare | `gateway/platforms/webhook.py` |
| Replay of a captured webhook | Signed timestamp checked against a replay window; unparseable → reject | `_timestamp_fresh` |
| Duplicate delivery | Idempotency cache, 1-hour TTL, bounded | `webhook.py:178` |
| Oversized body | Cap checked **before** the body is read | `webhook.py` module docstring |
| Unknown sender | **Default-deny.** No allowlist → denied; an adapter's own policy is trusted only when it is a real allowlist, never when it is `open` | `base.py:1889` |
| Cross-profile credential theft | `get_secret` returns the scoped value or the default, never another profile's; **raises** when unscoped under multiplexing | `agent/secret_scope.py:110` |
| Lost response after a crash | Durable obligation ledger; an ambiguous send is redelivered with a visible recovered marker rather than silently | `gateway/delivery_ledger.py` |

**NOVA adds nothing to this list and must not claim to.**

---

## 3. What NOVA adds

### 3.1 The grant — a connection may reach only declared agents

The runtime routes a conversation to a profile. Nothing in it says *which* profiles a
connection is allowed to reach, so a route edited by hand could point a customer WhatsApp
line at an internal finance agent and the runtime would deliver it correctly.

NOVA's `allowed_agents` is that missing statement, and it is enforced twice:

1. **At parse time.** A route naming an agent outside the grant is refused, with the
   connection id and the agent named.
2. **In the runtime's own fail-closed check.** The grant compiles into
   `gateway.multiplex_profile_allowlist`, and `gateway/run.py` refuses a route whose target
   profile is not served: *"Rejecting profile route %r: target profile %r is not served"*.

The second one is the one that matters. A validation somebody bypasses by editing YAML is
advice; a runtime that will not deliver is a control.

### 3.2 Credentials NOVA cannot hold

A channel declaration **cannot express a secret**. `token`, `api_key`, `secret`, `password`,
`credential` and `app_secret` are refused at parse time, at any nesting depth, and the
compiler refuses them a second time before handing settings to the runtime. Any existing
`token:` in the runtime's own platform config is **stripped** from the file NOVA writes.

Credential *variable names* come from the provider catalogue and an author cannot override
them — an author who could choose the name could choose one the adapter never reads, and the
failure would present as a broken channel rather than a misconfiguration.

Values live in `<profile>/.env`, which is on `materialize.NEVER_WRITE`. Verified on a live
deployment: the value appears **zero times** in the control API response and **zero times** in
NOVA's audit log, which does not even record the variable's name.

### 3.3 Connecting a channel is model-visible, and logged write-ahead

An inbound message becomes the text a worker reads, exactly as a work item's body does. So
`channel.connected` is a `MODEL_VISIBLE_KIND` and goes through `audit.model_visible_change` —
intent before, committed after — recorded against the **named human** who connected it, not
the process. That is what answers *"when did this channel start reaching that agent"* after
an incident, which is the question nobody can reconstruct later without a record.

### 3.4 Authorization on the control plane

| Action | Role | Why |
|---|---|---|
| Read `/channels` | `viewer` | Operational state, and it contains no credential — only variable names and which are absent |
| `POST /channels/apply` | `admin` | Makes the runtime start delivering to agents |

Separate tables, as Phase 8 established: reading what is connected and changing who it
reaches are different permissions. The write inherits the CSRF defences, the body cap, the
JSON content-type requirement and the audit-or-refuse rule already built for work decisions.

---

## 4. Residual risk — stated, not solved

These are real and they are **not** fixed by this phase. Listing them is the point.

| # | Risk | Status |
|---|---|---|
| 1 | **The default profile shares the process environment.** Under multiplexing, secondary profiles are credential-scoped and fail closed; the default profile falls back to `os.environ`. Two agents both holding channel credentials are isolated only if neither is the default profile. | **Live.** Same boundary the Phase 6 readiness audit found for model credentials. Mitigate by giving every channel-bearing agent a named profile. |
| 2 | **Inbound is not durable.** Outbound has a ledger; inbound has a 1-hour idempotency cache. A message arriving while the gateway is down is lost unless the provider retries. | **Live.** NOVA must never claim "durable messaging". |
| 3 | **Sender identity is not authorization.** The runtime's allowlist is per-platform env (`TELEGRAM_ALLOWED_USERS`), not per-NOVA-connection. NOVA does not yet model which *humans* may reach a channel. | **Live.** Today: set the platform allowlist. |
| 4 | **A webhook provider needs a public HTTPS endpoint.** WhatsApp Cloud requires one; exposing it is a deployment decision with its own attack surface, and the AWS module does not open it. | **Live.** Surfaced as `needs_public_endpoint` on the provider and in the dashboard. |
| 5 | **No per-channel rate limit in NOVA.** The runtime rate-limits some adapters; NOVA adds no cost ceiling, and cannot — a token ceiling is structurally impossible on this runtime (`BUDGET_ENFORCEMENT_AUDIT.md`). | **Live.** Cost control is the provider's console. |
| 6 | **No human-approval gating per channel yet.** The phase brief sketches "human approval required for discounts, refunds, contract changes". The policy layer can express tool approvals today; binding them *per channel* is not built. | **Not built.** Recommended next phase. |
| 7 | **Prompt injection from an inbound message.** A stranger's text reaches a model. The existing tool policy and knowledge scoping bound what that model may then *do*, which is the real mitigation; nothing sanitises the text itself. | **Live and inherent.** Any channel product has this. The defence is that the agent's permissions are declared and fail closed. |

---

## 5. What an attacker gets from each thing they might steal

- **The channel declaration file.** Agent names, conversation ids, provider names. No
  credential — by construction, tested.
- **The generated gateway config.** The same, plus which agents the gateway serves. No
  credential — the merge strips `token` and `api_key`.
- **NOVA's audit log.** Who connected what and when. No credential, and not even the
  variable names.
- **A control-plane viewer token.** The channel list, including which variables are missing.
  Useful reconnaissance; no secret, and no ability to connect or reroute.
- **`<profile>/.env`.** Everything for that one agent's channels. This is the file the whole
  design pushes secrets into precisely so that it is the *only* one worth stealing, and it is
  the one NOVA never writes, never reads the values of, and never archives by default.

---

## Correction: one tier of this enforcement is gone on current Hermes

Found while merging the NOVA work onto a newer Hermes snapshot, and recorded here rather
than quietly fixed, because this document made a claim that is no longer true everywhere.

**What changed upstream.** `multiplex_profile_allowlist` has been removed from
`GatewayConfig`. The served-profile set is now computed in a single chokepoint,
`hermes_cli.profiles.profiles_to_serve`, as *default plus every live named profile
directory*. Configuration no longer narrows it.

**What that costs NOVA.** A NOVA agent id *is* a Hermes profile name, and NOVA materialises
one profile per agent — so on those versions every agent is served and the allowlist would
narrow nothing even if it were still read. The sentence above describing the grant as
"enforced by the runtime's own fail-closed check" was true when written and is now only half
true.

**What still holds, unchanged.** `gateway.profile_routes` is **replaced** by NOVA on every
apply, not merged. A revoked connection loses its route, and an inbound conversation with no
route does not reach the agent. The runtime's rejection of a route to an unserved profile is
also still present (`gateway/run.py`) — what changed is only how the served set is computed.
So a revoked grant still stops delivery; it stops it with one mechanism rather than two.

**What NOVA does about it.** `nova/runtime/hermes/channels.py` now probes
(`supports_serve_allowlist()`) instead of assuming. Where the field exists the key is written
and the backstop is real. Where it does not, the key is **omitted** — an ignored key in a
config file reads as a live restriction to whoever opens it next — and the apply returns a
warning naming what is and is not enforced. `tests/platform/test_channels_integration.py`
asserts the key is present *if and only if* the runtime has the field.

Restoring the second tier would mean changing Hermes core, which this project deliberately
does not do for a control NOVA can express another way. The honest position is one tier,
stated plainly.
