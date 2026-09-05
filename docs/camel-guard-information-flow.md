# CaMeL Guard Information-Flow Model

This document states the security properties enforced and tested by the
current Hermes CaMeL Guard plugin. It is an executable information-flow model,
not a claim of machine-checked formal verification.

## Labels and boundary

- `T` — trusted control: the current user message's text fields.
- `U` — untrusted data: tool-result payloads and retrieved content.
- `S` — declassified provenance: untrusted source tool names only.
- `C` — a sensitive capability requested by a tool call.
- `A(T)` — the capability plan produced from trusted control only.
- `E` — a sensitive external or persistent effect.
- `P` — persisted policy trace metadata.

The guard is an integrity boundary. It does not hide `U` from the main model;
Hermes still needs retrieved data to answer the user. It prevents instructions
inside `U` from becoming authority for `E`.

## Invariants

### I1. Payload noninterference

For a fixed trusted message and fixed presence of taint:

```text
classify(T, U1, S1) = classify(T, U2, S2) = classify(T)
```

Neither untrusted payload bytes nor source names enter the classifier request.
`S` is intentionally declassified only into block messages and metadata traces
so an operator can identify the data source. Changing `U` or `S` cannot change
the authorization boolean.

### I2. Capability separation

```text
allow(C) iff C is in A(T) and C is not explicitly denied
```

Authorization for one capability never authorizes another. The test suite
checks the complete cross-product of the current capability vocabulary.

### I3. Complete mediation

Every mapped sensitive native or connector tool reaches `pre_tool_call` before
dispatch. Under taint in `enforce` mode:

```text
E occurs iff allow(C)
```

Direct dispatch and both native sequential and concurrent executor paths are
covered. The mapping suite separately checks current process, computer-use,
project, kanban, messaging, desktop, generation, and MCP mutation families.

### I4. Fail-closed policy

A timeout, provider exception, malformed JSON, missing/extra field, invalid
capability, duplicate capability, or wrong field type produces a
`fallback_read_only` plan. `enforce` blocks the effect; `monitor` reports what
would have been blocked.

The plugin validates the classifier object itself. Safety does not depend on
Hermes having the optional `jsonschema` package installed.

### I5. Confinement and deterministic concurrency

Turn state is keyed by session/task scope plus turn id. Taint and decisions do
not cross sessions, and session reset/end removes only the matching scope.
Concurrent sensitive calls share one immutable capability plan; classifier
resolution is single-flight and all workers observe the same decision.

### I6. Persistence non-disclosure

Tracing is separately opt-in, bounded, and impossible in `off` mode:

```text
P excludes T, U, tool arguments, and tool results
```

Only time, scope ids, mode, tool/capability, outcome/reason, classifier status,
and declassified source tool names may persist.

## Executable evidence

`tests/plugins/test_camel_guard_information_flow.py` implements the invariants
above through the real plugin loader under an isolated `HERMES_HOME`. It uses
adversarial payload variations, the full capability-separation matrix, the
current sensitive-tool family table, malformed classifier responses, separate
session scopes, trace canaries, and 64 concurrent decisions.

`tests/plugins/test_camel_guard_plugin.py` supplies integration evidence for
actual direct tool dispatch plus Hermes' sequential and concurrent executors,
including the native `make_tool_result_message()` contract.

## Explicit non-claims

- This is not a proof of the main model's natural-language noninterference.
- It does not prevent the main model from repeating attacker text in an answer.
- It does not automatically classify future tool names whose effects are not
  represented in the policy map; mapping coverage must evolve with Hermes.
- `monitor` is deliberately non-enforcing.
- Operator-authorized dangerous effects remain possible by design.
