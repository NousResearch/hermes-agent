# Audit and Receipt-Chain Plugins

This page documents the integration seam for plugins that build **tamper-evident audit
trails** of agent activity: signed, hash-chained receipts of what a tool did, intended to
survive scrutiny by someone who was not there when it ran.

This is a different goal from tracing. Tracing answers *what did we observe and log?*.
A receipt chain answers *what can we still prove happened, to a party who does not trust
the operator?*. The two use the same hooks; only the guarantees differ.

Signing and anchoring live outside core. The observer hook contract already carries
everything a receipt chain needs, so these plugins integrate with **zero core changes**.
See [#487](https://github.com/NousResearch/hermes-agent/issues/487) for the discussion that
settled this, and [#49371](https://github.com/NousResearch/hermes-agent/pull/49371) for the
proposed opt-in, local-only execution-receipt substrate, which is explicitly not a signing
or proving feature and composes with the plugins below rather than replacing them.

## The three-hook pattern

Register from your plugin's `register(ctx)`:

```python
def register(ctx):
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("transform_tool_result", on_transform_tool_result)  # optional
```

| Hook | Role in a receipt chain |
| --- | --- |
| `pre_tool_call` | Sign the *authorization*: what the agent intended to do, and under which policy, before it ran. |
| `post_tool_call` | Sign the *outcome*: arguments, result, status, duration. This is the minimum viable receipt. |
| `transform_tool_result` | Optional. Attach a receipt id or proof reference to the result the model sees. |

The `post_tool_call` payload carries the fields a receipt needs:

```python
def on_post_tool_call(**kwargs):
    tool_name = kwargs.get("tool_name")        # what ran
    args = kwargs.get("args")                  # with which inputs
    result = kwargs.get("result")              # producing what
    status = kwargs.get("status")              # success or failure
    duration_ms = kwargs.get("duration_ms")    # for how long
    session_id = kwargs.get("session_id")      # correlation
    task_id = kwargs.get("task_id")
    tool_call_id = kwargs.get("tool_call_id")
```

These hooks fire on every tool dispatch across the CLI, gateway, cron, and subagents, so a
plugin registered once covers every surface.

Note that two payload surfaces are documented. The observer contract above
(`hermes.observer.v1`) includes `status`, `error_type`, `error_message`, `session_id`, and
`tool_call_id`; the older callback signature in
[Event hooks](../../website/docs/user-guide/features/hooks.md) lists only `tool_name`,
`args`, `result`, `task_id`, and `duration_ms`. Read every field with `kwargs.get(...)` and
degrade gracefully when one is absent, rather than depending on either list being complete.

## Rules that matter more for receipts than for tracing

**Accept `**kwargs`.** Payload fields are additive. A receipt plugin that unpacks named
positional parameters will break on the next field Hermes adds.

**Never return a value from `pre_tool_call` unless you mean to block.** Returning
`{"action": "block", "message": "..."}` blocks the tool. A recorder must return `None`.
If your plugin is both a policy engine and a recorder, keep the two decisions explicit.

**Stay fail-open.** Hermes catches callback exceptions, logs a warning, and keeps the agent
loop running. Rely on that as a backstop, not as your error handling: swallow inside the
callback so a signing or network failure cannot stall an agent or flood the log.

**Do not silently paper over gaps.** If a receipt could not be written, that absence should
be detectable later rather than hidden. An audit trail that quietly drops entries under load
is worse than no audit trail, because it looks complete.

## The property a hash chain does not give you

Worth stating plainly, because it is the reason receipt chains are documented here as an
integration pattern rather than shipped in core.

A hash chain, verified only against itself, proves internal consistency: no entry was
deleted or reordered *within the copy you are holding*. It does not prove that copy is the
original. An operator with write access to the log can fork the chain at any point, rewrite
every entry after the fork, and internal verification still passes, because the rewritten
chain is self-consistent by construction.

Closing that gap requires something outside the log:

- **Signatures** prove *who* wrote each entry, so an attacker without the key cannot
  fabricate entries even with full write access. Keeping the signing key out of the agent's
  address space raises the bar further.
- **An external anchor** binds the log to something the operator does not control: a trusted
  timestamp, a transparency log with inclusion and consistency proofs, independent
  cosigners, or a public chain.
- **Deterministic canonicalization** (RFC 8785 JCS is the common choice) so that two
  independent implementations hash identical bytes. Without it, a record is only verifiable
  by the implementation that wrote it, which defeats the purpose.

Evaluate any receipt plugin against those three questions.

One limit applies to every implementation on this page, and is worth stating so nobody
buys more than is on offer. A hook proves that what *was* recorded has not been altered. It
cannot prove completeness, because an agent that never dispatches through the hook is never
seen by it. Completeness needs a control outside the agent process, such as forced egress.

## Implementations

External projects implementing this pattern, alphabetically. Hermes does not endorse any of
them; each has its own threat model, and the list is a starting point for evaluation rather
than a recommendation.

- [AgentLedger](https://github.com/dembovvski/agentledger): Ed25519-signed, hash-chained
  receipts with cryptographic identity for multi-agent systems, plus an offline
  verification CLI. MIT.
- [asqav](https://www.asqav.com/): agent governance with server-side signing, so the
  signing key never enters the agent process.
- [nobulex](https://github.com/arian-gogani/nobulex): bilateral receipts, signed before and
  after execution, implementing the OWASP Agentic Skills Top 10 AST09 pattern. MIT.
- [Provenrail](https://provenrail.com): off-box append-only sink with an independent
  server-side receipt chain, RFC 3161 trusted timestamps, and an RFC 6962 transparency log
  with independent witness cosignatures, so a record can be checked against the operator
  rather than only against itself. MIT client and verifier, AGPL server.
- [Signet](https://github.com/Prismer-AI/signet): a proof layer for cryptographically
  verifying agent actions. Apache-2.0.

Cross-implementation spec work, including `policy_attestation` and shared canonicalization
test vectors, was discussed in
[LangChain RFC #35691](https://github.com/langchain-ai/langchain/issues/35691).

## Related

- [Observer hooks](README.md) for the full `hermes.observer.v1` contract.
- [Monitoring](monitoring.md)
