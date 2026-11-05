# ACP session controls: remaining scope after upstream integration

## Upstream ownership

PR #139701 retains its three original published commits as provenance. This
refresh integrates upstream rather than reinstating the competing gateway-control
mixin. #143600 supplies gateway model/reasoning pickers and live synchronization;
#141338 supplies edit policy/modes; #148949 supplies the service-tier picker,
owner eligibility, reload/reconnect/peer refresh, old-owner fallback and
attach/race protections. Those are upstream capabilities, not new work here.
Gateway and embedded option IDs are `reasoning_effort` and `service_tier`.

## Retained scope

- Durable `session.mutate service_tier` uses the existing policy/receipt authority,
  validates owner-supplied choices and extends `session_pins`, not dataclass pin
  fields. Explicit picks survive followed-profile refresh, cold restore and
  independent forks. Normal disables a tier even on an unsupported route.
- ACP native/text speed controls and peer `config.set fast` writes to an ACP
  policy share that durable authority. Other surfaces retain their runtime
  behavior. Existing reset semantics are preserved.
- Embedded typed controls retain idle locking, persisted selections across
  compression tips, model-first labels and endpoint-aware catalog lookup.
  Model reconstruction must not implicitly enable a paid tier.
- Gateway goals use existing generic slash forwarding, with only recognized
  trailing harness metadata removed. Multiline user goal text and arbitrary XML
  survive. Cold-local goal commands target the retained conversation.
- Embedded goals use existing GoalManager scheduling, budget and judge, human
  input priority, pause/clear revocation and cancellation scopes. No new engine.

## Verification and integration

Use a PM-owned isolated Python 3.14 dev/test/acp environment and disposable homes.
Run `scripts/run_tests.sh` with retries disabled, including upstream picker,
reload, peer-sync, edit-mode and attach-order witnesses. Exercise real ACP stdio
and the authenticated owning gateway with deterministic loopback inference.
Regenerate gateway contracts and command parity through canonical generators;
run `python scripts/check` and shared TypeScript checks. Linux-native transport
proof does not establish GUI/Windows acceptance. Independent review and
publication remain separate from implementation.

## Contributor provenance

Reasoning behavior adapts JackySnake's #105652 commit `4ad4de849e6d17cde92162871f5b37c82a0333cf` (`JackySnake <yanguir@126.com>`); its config inheritance is already upstream. Fast session scope, override preservation, persistence/fork/rebuild and reset behavior adapt our #88303 commits `aa30f2c209ddcee29fc64b0fc9b7edcf647afd93`, `80899001265be0b268eedddf5b8af54d254be970`, and `4d8c0ce88f55f77ac8d47267235d84d19cd50c33`. The old boolean implementation is not transplanted because current code supports several canonical tiers and a new gateway ownership architecture. No skills code is included. Preserve JackySnake and ensu/Ran coauthor credit on the integrating commit; source patch inventory is in the Task B receipt.
