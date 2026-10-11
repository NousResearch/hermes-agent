# Default-off coordinator integration (bounded candidate)

`kanban.provider_lanes.enabled` defaults to `false`. Neither this code nor its
unit tests change live configuration, policy, credentials, or concurrency.
Standalone ticks and CLI daemon dispatch resolve the profile's read-only config;
the gateway watcher passes its explicit served-profile settings. The existing
ready/review/released-triage dispatch path shares one reserve/spawn/bind adapter.

Enabled admission preserves explicit provider/model/effort pins. This candidate
does not guess a stage from an assignee, substitute an API-billed provider, or
rewrite an alias. Unknown or unsupported pins defer with no failure-budget
charge. Native Anthropic and API OpenAI transports are rejected. The accepted
subscription provider identifiers are `openai-codex` and
`claude-subscription-directsdk-experimental`; the exact lane/model must also be
present in the operator's read-only router. Admission is not an entitlement or
runtime attestation. Missing transport/plugin/auth still needs separate work.

The host-shared ledger is `<kanban_home>/kanban/provider-lanes.db`. Reservations
count physical workers, with lane caps 2 Claude + 2 Codex and total cap 4. Memory
is sampled inside the ledger transaction; unknown memory or untracked existing
workers in any discovered board prevents admission. Pending/unknown spawn
outcomes retain capacity. A terminal card does not free a slot while the recorded
PID/fingerprint remains live. Observed provider/model fields remain NULL: requested
values are never presented as runtime provenance.

Activation is NOT authorized by merging this candidate. Every dispatcher using
the shared host must be upgraded/configured together and legacy dispatchers
must be drained first. Mixed enabled/disabled legacy processes cannot be made
safe by a lock used only by the enabled path. Do not delete the ledger during
rollback while workers may exist. Disable new admissions and drain/reconcile
physical workers before restoring the prior worker cap.

Deferred scope (not claimed complete): profile pin resolution; credential and
subscription transport compatibility/entitlement; canonical model aliases;
ordinary-stage cross-provider routing; independent build/review provenance;
provider-response runtime observation hook; four-real-model-worker/memory and
allowance proof. Activation still requires the recorded Keion diff gate.

Verification commands (isolated workspace environment):

    env -u INVOCATION_ID /absolute/workspace/test-env/bin/python -m pytest -q \
      tests/hermes_cli/test_kanban_lane_coordinator.py \
      tests/hermes_cli/test_kanban_provider_lanes.py \
      tests/hermes_cli/test_kanban_lane_liveness.py
    python scripts/check

Unsetting INVOCATION_ID only for the test subprocess avoids inheriting the agent
service's managed-systemd topology in legacy mocked-Popen tests. No production
scope checks are removed or weakened.
