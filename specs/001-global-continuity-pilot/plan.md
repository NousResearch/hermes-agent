# Implementation plan

## Architecture

One standard-library-oriented Python control plane reads four human-inspectable
authorities: repository state, the Basic Memory card, the Beads task JSONL/CLI, and
the Spec Kit change. `continuity_bridge.py` produces bounded preflight JSON;
`continuity_gate.py` verifies exact-state receipts and controls terminal lifecycle
promotion; `continuity_event.py` is the sole host dispatcher.

Project-level Claude and Codex adapters and an owned, reversible sandbox Hermes adapter
call the dispatcher. No global host file is edited. External state uses atomic writes
and a serialized promotion journal under the configured Storage Unit root. Gate-owned
evidence runs from closed command manifests in a credential-minimized environment.
Tests use temporary repositories and black-box dispatcher fixtures.

## Risk and rollback

Risk tier is T3 because hook lifecycle and external storage are involved. Failures are
fail-readable: a session may continue in degraded mode, but completion is forbidden.
Rollback restores only the managed Hermes hook before-image after checking ownership,
then archives the external pilot directory; it does not touch the shared checkout.
