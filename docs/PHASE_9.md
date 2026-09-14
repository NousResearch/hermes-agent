# Phase 9 — The Channel Layer

> **Correction (later than this document).** `multiplex_profile_allowlist` was removed
> from Hermes' `GatewayConfig`; the served-profile set is now computed from the profile
> directory and configuration cannot narrow it. The grant is still enforced — by NOVA
> replacing `gateway.profile_routes` — but by one mechanism rather than two. See
> `docs/NOVA_CHANNEL_SECURITY.md`, *Correction: one tier of this enforcement is gone*.

Connecting the customer's existing communication channels to the workforce they already have.
**Core patches: still 1.**

The brief's most important instruction was *do not start by building WhatsApp — investigate
first*. That instruction was right, and following it changed the phase completely.

---

## What the audit found

[`NOVA_CHANNEL_AUDIT.md`](NOVA_CHANNEL_AUDIT.md), written before a line of channel code:

> Hermes already has a complete, production-grade messaging platform system: **22 bundled
> platform plugins**, a registry with a documented plugin seam, per-profile credential
> scoping that fails closed, channel→profile routing, default-deny access control, webhook
> signature verification with a replay window, and a durable outbound delivery ledger.

The runtime's own guide opens with *"This requires **zero changes to core Hermes code**."*
That sentence answered the seam question before I could get it wrong.

So the phase stopped being "build channels" and became "expose channels a customer can buy."

---

## 1. What Hermes already had

- 22 platform plugins (`plugins/platforms/`) plus 12 built-in adapters, behind
  `ctx.register_platform()` — a public, documented plugin API.
- **Channel→agent routing.** `gateway.profile_routes` matches platform + conversation +
  workspace + thread, most-specific-first, and **rejects a route to a profile it does not
  serve**.
- **Per-profile credential scoping that fails closed.** `get_secret` raises rather than read
  `os.environ` when unscoped under multiplexing.
- Webhook HMAC-SHA256 + Svix, a signed-timestamp replay window, an idempotency cache, a body
  cap checked before reading, and per-adapter rate limiting.
- **Default-deny inbound.** No allowlist means denied; an adapter's own policy is trusted
  only when it is a real allowlist, never when it is `open`.
- A **durable outbound delivery ledger** that distinguishes "never started" from "crashed
  mid-send" and refuses to silently resend an ambiguous one.

## 2. What NOVA reused

All of it. NOVA wrote **no adapter, no transport, no protocol code, no webhook receiver, no
message queue and no conversation store.**

The load-bearing discovery, verified rather than assumed: NOVA writes
`<home>/profiles/<agent_id>/` and the runtime resolves `get_profile_dir(name)` to exactly
that path. **A NOVA agent id *is* a runtime profile name**, so a declared channel routes to a
NOVA agent with no mapping table and no id translation.

## 3. What NOVA had to build

The four things the runtime deliberately does not have:

| Built | Why the runtime does not have it |
|---|---|
| **The grant** — `allowed_agents` per connection | A route says where a message goes; nothing said where it was *allowed* to go |
| **A customer vocabulary** | Configuration was YAML + env + a setup wizard, shaped for an operator |
| **Credential indirection that cannot express a secret** | `PlatformConfig.token` can carry a raw token from YAML |
| **A write-ahead audit of who connected what** | The runtime logs traffic; it does not write NOVA's tamper-evident record |

Plus the surfaces: a control-plane read route, an admin-only write, a CLI, and a Channel
Center in the existing dashboard.

## 4. The channel contract

Provider-neutral, in `nova/channels/`, with **no runtime word anywhere in it**:

```
Provider      id · label · transport · required_env · capabilities · VERIFICATION
ChannelSpec   id · provider · display_name · enabled · allowed_agents · routes · settings
ChannelRoute  agent · conversation · workspace · thread          (most-specific-first)
```

Deliberately minimal. There is no `connect()`, no `disconnect()`, no `send_message()` and no
`receive_message()` — the brief offered those as a sketch and the audit showed they would be
a second, thinner copy of `BasePlatformAdapter`. NOVA declares; the runtime connects.

`Capability.supported` is **tri-state**. `None` means unknown, which is a different and more
useful statement than `False`, and the audit's rule was that unknown is never rendered as yes.

### Verification is part of the contract

Every provider carries how its support was established: `source_read`, `local_validated`, or
`field_validated`. A channel list is the most quotable page in a product — a customer reads
"WhatsApp ✓" and signs a contract on it. If the tick means "a plugin directory exists", the
first real deployment discovers what the tick actually meant. A test asserts that no provider
claims field validation, so promoting one requires evidence, not an edit.

## 5. Security model

Full threat model in [`NOVA_CHANNEL_SECURITY.md`](NOVA_CHANNEL_SECURITY.md). The two that
matter most:

**The grant is enforced twice.** NOVA's parser refuses a route outside `allowed_agents`, and
the grant compiles into `gateway.multiplex_profile_allowlist` so the **runtime's own
fail-closed check** refuses it too. A validation somebody bypasses by editing YAML is advice;
a runtime that will not deliver is a control.

**A channel declaration cannot express a secret.** `token`, `api_key`, `secret`, `password`,
`credential` and `app_secret` are refused at parse time at any nesting depth; the compiler
refuses them again; and any existing `token:` in the runtime's platform config is stripped
from the file NOVA writes.

## 6. Tenant and routing model

```
channel connection ──grant──> allowed_agents ──compiles──> multiplex_profile_allowlist
        │                                                            │
        └── routes ──compiles──> profile_routes ──runtime checks──> served?  else REJECT
```

One tenant per deployment, unchanged from every earlier phase. A route that escaped NOVA's
validation still cannot deliver, because the runtime will not serve a profile outside the
compiled allowlist.

## 7. API surface

| Route | Method | Role | Notes |
|---|---|---|---|
| `/platform/v1/channels` | GET | `viewer` | No credential — only variable names and which are absent |
| `/platform/v1/channels/apply` | POST | `admin` | Inherits Phase 8's CSRF defences, body cap and audit-or-refuse |

CLI: `nova channels providers | list | plan | apply`.

## 8. Dashboard experience

A **Channels** section in the existing Control Center, in the house style — no separate visual
system. Columns: the channel, its status *with its evidence level beside it*, **the workers it
may reach** with the routes spelled out, and **exactly which credential variable each agent
still needs**. Live screenshot in the commit.

The column that matters is who an inbound message reaches, not which protocol carried it,
because the product idea is that a customer is not buying integrations — they are connecting
the places they already talk to the workforce they already have.

## 9. Tests and counts

| Suite | Count | What it covers |
|---|---|---|
| `test_channels.py` | 37 | grant, credential refusals, routing, merge, failure modes |
| `test_channels_integration.py` | 9 | **the runtime's own** parser, matcher and config loader |
| `test_control_write.py` (added) | 3 | viewer/admin split, no credential in the read route, 501 |
| **Platform total** | **661** | passing with **and without** the runtime installed |

The integration tests use `gateway.profile_routing` and `gateway.config.load_gateway_config`
directly. No stub stands in for the thing that will consume NOVA's output.

## 10. Real-provider validation status

> ### LOCAL VALIDATION
>
> **No channel has been connected to a real provider.** No Telegram bot token, no Slack app,
> no Meta Business account exists in this environment.

What *was* proven end to end, on a live runtime home:

- NOVA compiled a two-agent Telegram connection and **the real `load_gateway_config()`
  accepted it** — multiplexing on, allowlist exactly the granted agents, routes parsed with
  correct specificity.
- **The real `match_profile_route()` resolved every case correctly**: the escalations group →
  `operations`, a thread inside it → `customer-support`, everything else → `customer-support`,
  and an unconnected platform → nothing.
- An operator's `max_concurrent_sessions: 7` and `quick_commands` **survived the apply**.
- The written config contained **no credential field**, and `PlatformConfig.token` was `None`.
- Supplying `TELEGRAM_BOT_TOKEN` in each agent's `.env` flipped the status to *connected*; the
  value appeared **0 times** in the API response and **0 times** in the audit log, which does
  not record even the variable name.
- `channel.connected` was written **write-ahead** (intent → committed), model-visible, against
  the named human `priya-ops`.

## 11. Remaining limitations

Stated so nothing downstream over-claims:

1. **No provider has been field-tested.** Everything reads `source_read`.
2. **Inbound is not durable.** Outbound has a ledger; inbound has a 1-hour idempotency cache.
   **NOVA must never claim "durable messaging".**
3. **The default profile shares the process environment** under multiplexing. Give every
   channel-bearing agent a *named* profile.
4. **No per-message audit in NOVA.** Deliberate: it would copy customer conversation content
   into a second store.
5. **No per-channel human-approval gating.** The brief sketches "approval required for
   discounts, refunds, contract changes". Tool approvals exist; binding them *per channel*
   does not.
6. **No sender-identity model in NOVA.** Which *humans* may reach a channel is still the
   runtime's per-platform env allowlist.
7. **Connect is declarative, not interactive.** There is no OAuth dance in the dashboard; a
   customer declares a channel and an operator places the credential.
8. **The gateway must be restarted** to pick up a channel change.
9. **MS Teams and Discord are catalogued but unopened beyond their manifests.**

## 12. Recommended next phase

**Per-channel approval policy** — binding the existing tool-approval machinery to a channel,
so "refunds require a human on WhatsApp but not on the internal Slack" is declarable. It is
the highest-value remaining item, it is the one the brief's own UX sketch asks for, and every
part it needs (policy compilation, the escalation gate, the Phase 8 write path, the audit)
already exists. The work is the binding, not the mechanism.

Second: **field-validate one provider**, almost certainly Telegram — a bot token is free and
the transport is long-polling, so it needs no public endpoint. One real connection converts
the whole catalogue's evidence level from an argument into a fact.
