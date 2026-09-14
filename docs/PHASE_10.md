# Phase 10 — Per-channel approval

> **Correction (later than this document).** `multiplex_profile_allowlist` was removed
> from Hermes' `GatewayConfig`; the served-profile set is now computed from the profile
> directory and configuration cannot narrow it. The grant is still enforced — by NOVA
> replacing `gateway.profile_routes` — but by one mechanism rather than two. See
> `docs/NOVA_CHANNEL_SECURITY.md`, *Correction: one tier of this enforcement is gone*.

*"A refund needs a human when it comes in on WhatsApp, but not on the internal Slack."*

**Core patches: still 1.**

The requirement is easy to say and the obvious implementation does not work here. Finding
that out first is most of what this phase was.

---

## The audit finding that shaped everything

The runtime's policy hook — the fail-closed `pre_tool_call` that every NOVA governance claim
rests on — receives exactly this:

```
tool_name, args, task_id, session_id, tool_call_id, turn_id, api_request_id, middleware_trace
```

**It is never told which channel the turn arrived on.** The session *key* encodes the
platform (`gateway/session.py`: `parts = [namespace, source.platform.value, …]`), but the
session *id* the hook receives does not.

So a per-channel rule evaluated inside the hook would be a rule that never fires: enforcement
in name only, which this project has a standing rule against shipping. The alternatives were
to patch the runtime's hot path to carry a source (a core patch, on a path that also serves
board tasks which have no channel at all), or to find a different shape.

## The shape that works

The runtime *does* enforce one compiled policy per profile. And a profile is exactly what the
channel layer already routes to. So:

> A channel that tightens approval derives an agent.

`operations` reached over `acme-support-telegram` becomes
`operations__acme-support-telegram` — the same agent, same instructions, same tools, same
knowledge, built with `dataclasses.replace` so it cannot drift — with the channel's extra
approvals compiled into its own policy, and the channel's routes pointed at it.

**No new enforcement mechanism.** The hook that refuses is the hook that already refused.

```
channels.yaml                          compiled
─────────────                          ────────
approval:                              profiles/operations__acme-support-telegram/
  required_for: [send_external_email]    └─ nova-policy.json
                                            approval_actions: { send_external_email: [email_send] }

gateway.profile_routes[].profile  ->  operations__acme-support-telegram
gateway.multiplex_profile_allowlist ->  [..., operations__acme-support-telegram]
```

Proven through the real decision function, not by reading the compiled file:

```
operations                        allow: ['email_send', 'erp_stock_query']
operations__acme-support-telegram allow: ['email_send', 'erp_stock_query']   ← identical

email_send   internally -> allow             on Telegram -> require_approval
execute_code internally -> deny              on Telegram -> deny
```

## The defect this phase found in itself

The first working version **widened an agent's reach**, which is the exact opposite of what a
channel declaration may do.

`nova/policy/decide.py` tests approval actions **before** the allow-list. So a tool that is
*not granted* but *is* named by an approval action resolves to `require_approval` rather than
`deny`. Declaring `approval.required_for: [send_external_email]` on a channel therefore moved
`email_send` from **deny** to **require_approval** for an agent that had never been granted
it — granting the tool, behind a human gate, by connecting a messaging app.

Caught by running it, not by reading it:

```
Same tool (email_send), two postures:
   internal (operations)      -> deny             rule=not-in-allowlist
   on Telegram (derived)      -> require_approval rule=approval-required   ← widened
```

*Fixed:* `reachable_approvals()` drops any approval naming a tool the agent cannot already
reach, and records it as dropped rather than silently ignoring it. Dropped rather than
refused because one channel usually grants several agents and only some hold the tools;
refusing would make the common case unwritable. Two tests pin it.

### A related finding, deliberately **not** changed

The same ordering means a **tenant-wide** `requires_approval: true` action is reachable by any
agent with a human's sign-off, including agents whose allow-list was written to exclude it.
I changed this, and 14 tests failed — among them one whose comment reads
`("crm_refund", "approve"),  # tenant-wide approval` on an agent without that permission.

That is a deliberate, tested design: **permissions are what an agent may do alone; approval
actions are what it may do with a human.** Coherent, and not mine to redefine as a side effect
of a channel feature. So the change was reverted and the guard lives only in the channel
layer, where the promise "a channel may tighten, never loosen" is mine to keep.

It is recorded here because a reviewer should decide it on purpose rather than inherit it by
accident.

## The cost, stated rather than hidden

A derived agent is a separate profile, so it has **its own session and memory namespace**. The
same person talking to "the support agent" on Telegram and through a ticket is talking to two
profiles, and they do not share conversation history.

For a channel strict enough to need extra approvals that separation is usually right — an
external conversation and an internal one are different trust contexts — but it is a
consequence a customer should meet in a document, not discover in a transcript. `nova apply`
prints it as a warning every time.

## Also fixed here

Readiness reported the **base** agent as missing a channel credential, while the profile the
adapter would actually read was the variant. An operator would have been told the credential
was in place while every message failed. Found on a live deployment; regression-tested.

## Tests

| Suite | Count |
|---|---|
| `test_channel_approval.py` | 17 |
| **Platform total** | **680**, passing with and without the runtime |

Five existing tests changed because the example bundle now materialises a third agent. Each
new expectation is correct behaviour — a variant is skipped when its base is disabled, and a
bundle with no policy cannot declare a channel approval — and the reason a third agent exists
is written down once, as `EXAMPLE_AGENTS` in `conftest.py`, rather than as a literal in five
places.

## Limitations

1. **Approval is per channel, not per conversation.** A VIP chat and a general one on the same
   channel share a posture. Per-route approval would multiply profiles further.
2. **One profile per (agent, channel) that tightens.** Ten channels tightening the same three
   agents is thirty profiles. Fine at the scale this product targets; not free.
3. **No history sharing across variants**, as above.
4. **A channel can only tighten.** Deliberate, and unlikely to change: a permission that could
   be widened by connecting a messaging app would make the channel list part of the security
   review.
5. **Still no real-provider validation.** Nothing in Phase 9 or 10 has met a live provider.

## Recommended next phase

**Field-validate Telegram.** A bot token is free, the transport is long-polling so it needs no
public endpoint, and one real connection would convert the whole channel catalogue's evidence
level from `source_read` to `field_validated` — including this phase's approval gate, which
has never been exercised by a real inbound message.

It is now the single highest-value thing available, and it is small.
