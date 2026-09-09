# DirectSDK ownership recovery: focused contribution

This contribution builds on teknium1's [DirectSDK PR #105863](https://github.com/NousResearch/hermes-agent/pull/105863)
at `7408a29e6de23addada9c25c19c809d6e202f9db`. It does not replace that provider,
introduce another relay, or qualify the old standalone plugin release.

## Reproduced defects and corrections

| Boundary | Counterexample | Focused correction |
| --- | --- | --- |
| Truncated tool response | A `max_tokens` response ends inside `{"path":"`; parsing aborts before terminal usage/reason | Retain incomplete arguments as a non-executable length-retry signal, read the terminal metadata, omit invalid signed replay blocks |
| Partial-step observability | Hermes handles `length` before normal intake, skipping `post_api_request` and usage recording | Record that response before existing truncation recovery; no duplicate hook on the final response |
| Denied continuation | The loop rejects admission but the finalizer sends a separate summary request | Preserve the already-received partial answer when the tail is a length-continuation scaffold; leave unrelated summary policy unchanged |
| Canonical whitespace | A whitespace-only edit still restores old signed text because both projections are stripped | Compare exact visible text, including old carriers' actual native text; preserve unchanged signed prefixes and rebuild edited visible history |
| Cancellation teardown | Darwin returned `EPERM` during repeated cancellation of an exited subprocess | Recheck the owned child's exit state; a permission error for a live child still propagates |
| Context-window exhaustion | Native context stop was translated to output `length`, so ordinary Hermes never invoked compression | Preserve the stop, save partial output/usage, route through existing Hermes overflow recovery, and require a new admission without refunding the completed generation |

All native/subscription authentication remains CLI-owned. The existing request-scoped
relay and Hermes tool executor remain in place. There are no Claude-specific core types,
new retry limits, new budgets, new dependencies, or changes to published plugin assets.

## Offline evidence and limits

The added ordinary-loop test uses the real `AIAgent.run_conversation`, provider-client
factory, streaming path and relay. Only the native executable and remote HTTP service
are synthetic fixtures. It observes budget admission, step callbacks, pre/post request
hooks, request counts and the assembled continuation.

- Allowed continuation: two host provider invocations, one forwarded request each;
  both native recovery attempts blocked; assembled answer retained.
- Hard admission denial: one forwarded request; partial answer returned without the
  finalizer's extra model call and with an incomplete outcome. This is distinct from
  normal Hermes wrap-up policy.
- Cancellation after the partial step: one forwarded request, interrupted result.
- Incomplete tool arguments: previous text and usage retained with a length finish;
  no signed replay carrier for the invalid call. The separate non-streaming tool-retry
  path now returns to outer iteration admission instead of reusing the first permit.
  Both text and cut-off-tool fixtures pass allow/deny/cancel (six cases); no tool executes.
- A usage fold confirming compression recovery clears the existing preflight latch
  before truncation restarts, as it does for a complete response.
- Context overflow: the actual host-loop fixture invokes existing overflow routing,
  saves partial output before compression, and sends only compressed canonical history
  on the next admitted generation. Allow/deny/cancel pass (three cases). The compressor
  algorithm itself is stubbed to verify routing, not requalified by this test; cancellation
  before recovery prevents both compression and the next generation.
- A queued redirect during context recovery must also retain the completed
  generation's budget charge. The focused delta covers that branch. The synthetic
  native fixture now ignores historical assistant seed frames, matching the real
  native replay protocol; final remote validation of that correction is required.
- Whitespace edit and legacy stripped carrier: canonical visible text replaces stale
  native blocks; unchanged prefixes retain their original blocks.

The existing real-native admission eval was also run once in a sterile environment:

| Measurement | Result |
| --- | --- |
| Native executable | Claude Code `2.1.263` |
| SHA-256 before/after | `ef5d2909c8af49f31ab6d5487e90316777bc2fac170adfe8160716caa8aaf4f9` |
| Fault cases | 10/10: final, tools, output limit, complete-tool output limit, context limit, thinking, refusal, HTTP error, disconnect, cancellation |
| Forwarded requests | Exactly one per case |
| Native output-limit recovery | Blocked before forwarding |
| Cancellation disconnect | Approximately 0.051 seconds |
| Network | macOS sandbox denied non-localhost outbound traffic |
| Authentication | Synthetic fixture credential, temporary homes; real credential directories denied |
| Subscription generations | Zero |

The 10-case native eval's `tool_max` uses complete JSON; it is **not** evidence for
incomplete arguments. The added regression supplies the missing cut-off JSON case.
The native eval and ordinary-loop fixture are complementary offline evidence, not an
installed subscription canary or proof of service terms, billing entitlement or release readiness.

## Reproduction

Use the repository's existing test runner with a development environment and isolated
`HERMES_HOME` (the runner supplies isolation). Keep local workers bounded:

```sh
scripts/run_tests.sh -j 1 --file-retries 0 tests/providers/test_claude_oauth_directsdk_admission.py
scripts/run_tests.sh -j 1 --file-retries 0 tests/providers/test_claude_oauth_directsdk_replay.py
scripts/run_tests.sh -j 1 --file-retries 0 tests/providers/test_claude_oauth_directsdk.py
scripts/run_tests.sh -j 1 --file-retries 0 tests/agent/test_turn_finalizer_iteration_limit_exit.py
```

The existing `evals/directsdk_admission.py /path/to/claude` uses synthetic peers.
Run it under an OS localhost-only outbound policy with a temporary home and no real
account access. No vendor request is needed for the offline reproduction.

## Still required; do not infer qualification

- Exact contribution-head remote CI and changed-surface independent semantic review.
- Independent review and real-native affected-case readback of context-overflow routing;
  existing compressor algorithm evidence remains separate from the new routing fixture.
- Complete imported/edit/resume/persistence and request-accounting readback, then the
  separately authorized visible installed subscription canary.
- Upstream PR #105863's OSV job currently fails during SARIF upload on a GitHub
  installation rate limit ([run](https://github.com/NousResearch/hermes-agent/actions/runs/34351282262/job/102469694612)).
  This is not attributed to the contribution and is not called green.

The [recovery tracker](https://github.com/100yenadmin/hermes-claude-agent-sdk/issues/1)
and its milestone #3 remain open. The [vendor-native admission request](https://github.com/anthropics/claude-code/issues/92936)
remains open as a preferred long-term control. No merge, publication, migration or
full-ownership claim is authorized by this packet.
