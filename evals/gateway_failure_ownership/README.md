# Gateway failure-writer ownership

Offline, full gateway-handler integration: `_handle_message` → session preparation
and lease → production `_run_agent` / TurnRunner → real AIAgent → loopback HTTP/SSE
→ on-disk SQLite → gateway persistence/delivery → exception fallback.

Run from any checkout using its Python environment:

```sh
.venv/bin/python evals/gateway_failure_ownership/probe.py "$PWD" /tmp/ownership.json
scripts/run_tests.sh -j 1 tests/gateway/test_failure_writer_ownership.py
```

The first argument selects the production checkout, allowing the same fixture to
A/B another worktree. The second is the JSON receipt. The temporary HERMES_HOME is
recorded in that receipt. Inherited credentials are removed; socket connections
are restricted to loopback, tools are disabled, and all state is temporary.
No live account, messaging service, or paid provider is contacted.

## Controlled fault boundaries

- The local peer returns a real HTTP 400 for the provider-failure turn.
- A subclass raises during AIAgent construction for the pre-agent failure cases.
  All successful construction and conversation behavior is production code.
- The voice-policy callback raises after the agent and gateway persistence steps.
  Neither the handler nor its exception writer is mocked or invoked directly.
- A raw baseline read raises `TranscriptReadError` to verify fail-closed admission.

Authorization is allowed for the fixture source, topic recovery is disabled,
provider routing selects the local peer, and the agent cache is evicted per turn.
This proves handler reachability with controlled exception sources, not a claim
that any particular real Telegram or provider failure caused the reporter's rows.
The initial warmup is essential: a first-turn `session_meta` row can separate the
two user rows and mask replay coalescing.

## Evidence

Base `869228cab4a8276d3b4c78da9d9939670c47bd0f`: **6/16** checkpoints.
Fixed implementation: **16/16**, with 14 accepted user rows, same-ID failure retry
not reinserted, and unreadable-baseline input refused. Both runs made only real
loopback requests, with no blocked external connection attempts.

The original 15-checkpoint invariant was RED on base (6/15), with every controlled
exception boundary reached. The added sixteenth checkpoint tests baseline-read
refusal; base does not perform that raw read and incorrectly proceeds.
The matrix includes two separately accepted identical platform inputs, identical
keyless inputs with identical timestamps, HTTP failure and recovery, keyed/keyless
post-persistence exceptions, pre-agent failures, same-delivery failure retry,
separate identical pre-agent inputs, and a healthy follow-up.

Two invariant tests are retained. The second uses real SQLite compaction archives
and durable compression lineage with keyed/keyless input. It proves both parent
and successor-only ownership; disabling successor lookup makes it RED (3 rows
instead of 2). A new ambient observed row must not masquerade as the accepted
input; that negative control was also RED/GREEN verified. It does not run
provider-driven compaction. Independent review
identified baseline-read, content-projection, archive, and successor-only edge
cases; ownership now avoids content comparison and includes durable lineage.

Targeted canonical runner: 25 passing tests in four files (ownership,
`test_42039_duplicate_user_message`, silence tokens, retry replacement). Broad
gateway tests are left to CI, not a local publication gate. No historical rows are
rewritten and no storage-wide deduplication is introduced.
