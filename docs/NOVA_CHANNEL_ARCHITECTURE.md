# NOVA Channel Architecture

> **Correction (later than this document).** `multiplex_profile_allowlist` was removed
> from Hermes' `GatewayConfig`; the served-profile set is now computed from the profile
> directory and configuration cannot narrow it. The grant is still enforced — by NOVA
> replacing `gateway.profile_routes` — but by one mechanism rather than two. See
> `docs/NOVA_CHANNEL_SECURITY.md`, *Correction: one tier of this enforcement is gone*.

**One sentence:** the runtime already does messaging; NOVA owns the tenant boundary, the
per-connection grant, the customer's vocabulary and the audit record, and compiles all four
into configuration for a system that already works.

---

## The layering

```
         customer                    "connect Telegram, let Sales and Support use it"
             │
    ┌────────▼─────────┐
    │  channels.yaml   │            provider · display name · allowed_agents · routes
    │  (declaration)   │            NEVER a credential — refused at parse time
    └────────┬─────────┘
             │            nova/channels/       provider-neutral, no runtime vocabulary
    ┌────────▼─────────┐
    │  ChannelSpec     │            the grant is enforced here, first gate
    └────────┬─────────┘
             │            nova/runtime/hermes/channels.py   ← the ONLY runtime-aware module
    ┌────────▼─────────┐
    │  gateway config  │            platforms · profile_routes · multiplex allowlist
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │  Hermes gateway  │            adapters · transports · webhook auth · delivery ledger
    └──────────────────┘            22 platform plugins, none written by NOVA
```

`nova/channels/` contains no runtime word. `nova/runtime/hermes/channels.py` is the whole of
NOVA's runtime-specific channel knowledge — 200 lines of translation — which is what makes a
second runtime adapter additive rather than a rewrite.

## The translation, in full

| NOVA concept | Compiles to | Why that key |
|---|---|---|
| connection (enabled) | `gateway.platforms.<provider>.enabled` | the runtime's own platform switch |
| route → conversation | `profile_routes[].chat_id` | a chat, a channel, an address |
| route → workspace | `profile_routes[].guild_id` | a Slack workspace, a Discord guild |
| route → thread | `profile_routes[].thread_id` | matched inside its parent conversation |
| route → agent | `profile_routes[].profile` | **a NOVA agent id *is* a runtime profile name** |
| `allowed_agents` (union) | `gateway.multiplex_profile_allowlist` | the runtime refuses a route to an unserved profile |
| any credential | **nothing** | values live in `<profile>/.env`, which NOVA never writes |

That fifth row is the load-bearing one. NOVA writes `<home>/profiles/<agent_id>/` and the
runtime resolves `get_profile_dir(name)` to exactly that path — verified, not assumed — so a
declared channel routes to a NOVA agent with no glue, no mapping table and no id translation.

## Message flow (Phase I)

```
external user
     │
     ▼  provider delivers: long-poll │ socket │ webhook │ bridge
adapter                                                      ── RUNTIME
     │
     ▼  webhook signature + timestamp + idempotency          ── RUNTIME
inbound intake
     │
     ▼  sender allowlist, DEFAULT-DENY                       ── RUNTIME
authorization
     │
     ▼  match_profile_route(platform, chat, thread)          ── RUNTIME, config by NOVA
routing                                                          most-specific-first
     │
     ▼  target profile ∈ served set?  else REJECT            ── RUNTIME, grant by NOVA
the grant
     │
     ▼  profile secret scope installed                       ── RUNTIME
the agent's profile
     │
     ▼  tool policy · knowledge scope · delegation limits    ── NOVA (Phases 2–5)
the model
     │
     ▼  obligation recorded BEFORE the send                  ── RUNTIME (delivery ledger)
response
     │
     ▼
external user
```

Auditable transitions: NOVA records `channel.connected` (write-ahead, model-visible, against
the named human) when a channel begins reaching an agent. **Per-message auditing in NOVA does
not exist** — the runtime logs the traffic, and duplicating a message log inside NOVA would
mean copying customer conversation content into a second store. That is a deliberate omission,
not an oversight, and `PHASE_9.md` lists it as a limitation.

## What NOVA deliberately did not build

| Not built | Because |
|---|---|
| A NOVA adapter per provider | 22 working adapters exist behind a documented plugin seam |
| A NOVA webhook receiver | the runtime verifies HMAC, binds timestamps and de-duplicates |
| A NOVA message queue | a second dispatcher, in front of a runtime with durable outbound delivery |
| A NOVA conversation store | it would copy customer conversation content into a second place |
| Provider entries nobody opened | a catalogue entry is a commitment; a list is not a product |

## Extending it

**A new provider** is an entry in `nova/channels/providers.py` — id, label, transport, the
variable names its adapter reads, capabilities with their evidence level — plus one line in
`PLATFORM_NAMES`. No adapter, no transport code. The bar for adding one is that somebody
opened the runtime plugin and read what it needs, which is why the catalogue is shorter than
the runtime's list.

**A second runtime** implements `apply_channels` and `channel_readiness` on the contract.
Everything in `nova/channels/` is reused unchanged.
