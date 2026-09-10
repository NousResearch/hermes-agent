# Hermes LLM Egress Firewall and Performance Router Design

**Date:** 2026-08-25

**Status:** Approved for implementation planning

**Scope:** Hermes agent, auxiliary LLM calls, Relay, profile routing, and the benchmark harness

## Objective

Allow Hermes to use remote models for task-scoped private source snippets and diffs without
silently exporting unrelated workspace data, while selecting the fastest reliable model for each
profile from all installed local Ollama models, all currently callable free Nous models, and the
approved GPT/Codex models.

The design optimizes successful wall-clock completion, not raw token speed. Privacy policy,
identity, exact-head constraints, and task safety remain hard gates and cannot be traded for lower
latency.

## Non-goals

- Encoding source as Base64, binary, or another reversible representation is not a privacy control.
- The router does not weaken approval, merge, CI, worktree, or repository-identity gates.
- Benchmark scores do not authorize a model to perform tasks outside its privacy class.
- Phase one does not claim process-wide containment for standalone utilities that bypass the agent
  and auxiliary clients.

## Design decisions

### 1. Source-bound egress firewall

Add `agent/llm_egress_firewall.py` as the canonical policy and receipt owner. It classifies each
destination as `local_process`, `loopback`, `remote`, or `unknown`. Unknown destinations use remote
policy and fail closed.

The security classifier is intentionally stricter than the existing timeout-oriented local endpoint
helper. LAN addresses, container DNS, Tailscale addresses, ACP subprocesses, missing URLs, and
unqualified hostnames are not automatically trusted as local.

Remote source content is admissible only when Hermes created its provenance while resolving an
exact `read_file` or `@file:path:start-end` slice. A grant binds:

- canonical real path and workspace-relative display path;
- inclusive line interval;
- SHA-256 of the approved bytes;
- byte count;
- session, turn, and request identities;
- policy digest.

Terminal output, user-pasted path labels, plugin-generated references, compressed summaries, and
arbitrary tool output cannot create file provenance. The firewall rechecks the canonical sensitive
path policy and content hash immediately before transmission.

Every remote-bound string is scanned with forced credential redaction semantics. If scanning would
change the payload, the request is rejected instead of silently changing source code. The firewall
also enforces serialized-byte and conservative token ceilings. Rejection selects an eligible local
route or returns a typed blocked result; it never sends a broader substitute prompt.

### 2. Enforcement points

Phase one enforces the same policy at every centralized agent path:

1. The final logical request passed to `_perform_api_call` in `agent/conversation_loop.py`, after
   middleware and transport shaping.
2. The final Relay-rewritten body in `agent/relay_llm.py`, immediately before each sync, async, and
   streaming provider callback.
3. Direct auxiliary call branches in `agent/auxiliary_client.py`, including compression, title,
   vision, search, review, curator, reference aggregation, and mixture-of-agents calls.
4. Trusted provenance production in `agent/context_references.py`, `tools/file_tools.py`, and the
   tool-result attachment boundary in `agent/tool_executor.py`.

Provider SDK wrappers are reserved for phase-two defense in depth. Standalone utilities must adopt
the centralized wrapper before Hermes claims process-wide LLM egress closure.

### 3. Content-free receipts

Each decision appends a receipt under the profile-local Hermes state directory. Receipts contain
only request identity, destination class, provider/model, byte and token counts, approved slice
hashes and ranges, payload hash, policy digest, decision, and reason codes. They never contain raw
prompts, source, tool results, credentials, or absolute home-directory paths.

The ledger uses `O_APPEND | O_CREAT | O_NOFOLLOW`, mode `0600`, and a process lock. A remote send
fails closed if its allow receipt cannot be recorded. The hot path avoids per-request `fsync` unless
the configured threat model requires crash-durable proof.

### 4. Exhaustive candidate inventory

Every benchmark run takes a timestamped inventory rather than relying on a static model list.

Local candidates are every model returned by `ollama list`. The initial inventory contains:

- `hermes-compression-fast:latest`
- `hermes-cron-fast:latest`
- `hermes-qwen3-fast:latest`
- `glm-4.7-flash:latest`
- `qwen3-coder-next:q4_K_M`
- `qwen3.6:27b`
- `qwen3.6:35b-a3b`
- `qwen3-coder:30b`
- `qwen3.5:4b`
- `qwen3:4b-instruct`

Free Nous candidates are every model returned by the live Portal recommendations endpoint plus any
additional free model reported by the authenticated inference catalog. The initial Portal roster is:

- `upstage/solar-pro4:free`
- `meituan/longcat-2.0:free`
- `tencent/hy3:free`
- `poolside/laguna-s-2.1:free`
- `stepfun/step-3.7-flash:free`
- `poolside/laguna-xs-2.1:free`

Remote paid candidates are GPT/Codex 5.4 mini, 5.5, 5.6 Luna, 5.6 Terra, and 5.6 Sol. Codex Spark
is included only when Hermes can invoke it through a supported provider. Catalog-only, unauthorized,
rate-limited, or unavailable models remain visible in the report with a typed reason rather than
being silently omitted.

### 5. Controlled benchmark corpus

The harness runs identical bounded tasks for each eligible model and records prompt, tool schema,
reasoning control, timeouts, and expected receipt. The corpus covers:

- title generation and short classification;
- fast compression and long-context compression;
- review and review-comment triage;
- curator and reference aggregation;
- skill and profile selection;
- GitHub comment audit, CI diagnosis, and repair planning;
- narrow coding with deterministic tests;
- merge-conflict analysis and exact-head maintenance planning;
- orchestration and multi-step tool use;
- vision only for models that advertise and successfully demonstrate vision support.

Deterministic parsing, deduplication, identity checks, receipt validation, and merge eligibility stay
in code and are not benchmarked as LLM responsibilities.

Each task has a machine-checkable success receipt. Free-form judging is supplemental and cannot
turn a failed deterministic outcome into a pass. Source-bearing cases use synthetic secret canaries
and the firewall so the benchmark also proves that forbidden data produces zero remote requests.

### 6. Metrics and selection objective

For each model/task pair, the harness records:

- deterministic task success and tool-call accuracy;
- time to first output, p50/p95 completion time, and total wall time;
- no-output stalls, timeouts, retry rate, malformed output, and reasoning loops;
- input, cached-input, and output tokens where available;
- estimated API cost;
- local peak resident memory, model-load time, and approximate energy pressure;
- context-limit and capability failures;
- privacy eligibility and firewall decision.

The primary score is expected successful completion time:

`expected_wall_time = observed_wall_time / bounded_success_probability`

Candidates that miss the task's minimum correctness, tool accuracy, context, or privacy threshold
are ineligible regardless of speed. Among eligible models, the router selects the lowest expected
wall time subject to the lane's cost ceiling. A materially faster paid model may therefore be the
primary rather than only a fallback.

Cost is reported separately and as a cost-per-success measure. GPT-5.6 Sol requires a measured
quality or reliability advantage that offsets its incremental cost; it is not selected merely
because it is the largest model.

### 7. Profile-specific routing

Each Hermes profile receives an independent ordered route, reasoning level, context budget, output
budget, no-output deadline, total deadline, and concurrency weight. Auxiliary roles such as
compression, title generation, review, curator, reference aggregation, skill selection, and
approvals are tuned separately instead of inheriting a convenient global model.

The required routing matrix covers every configured profile and every invocation surface, including:

- curator and reference aggregation;
- primary review, independent review, and review-comment triage;
- primary, retry, fallback, circuit-breaker probe, and local privacy fallback routes;
- context-window admission, context packing, compaction, and compression;
- title and conversation-summary generation;
- MCP discovery, tool selection, and tool-result interpretation;
- approval classification and approval-response handling;
- Skills Hub discovery, skill selection, and skill execution support;
- vision and image-bearing tool results;
- coding, CI audit, typed CI repair, PR feedback repair, merge maintenance, and auto-merge;
- cron, Kanban orchestration, task assignment, and new-conversation model selection.

The configuration compiler fails if a declared profile or invocation surface lacks an explicit
benchmark-backed primary route, privacy-compatible fallback, context window, and timeout policy.
An intentional shared model is permitted only when each affected row independently selects it from
its own results; shared configuration for convenience is not evidence.

The route table records the benchmark artifact and policy digest that justified each choice. New
conversations use the winning route for their selected profile. Existing conversations keep their
model unless the current call trips a circuit breaker or the operator explicitly migrates them.

### 8. Stalls, retries, and circuit breaking

Each candidate has a learned startup envelope and a hard no-output deadline. A stale call is aborted
before the current multi-minute reconnect interval. Hermes permits at most one retry when the error
is demonstrably transient and the remaining deadline can accommodate it. The next attempt uses a
different eligible route; it cannot bounce repeatedly between two failing models.

Circuit state is keyed by provider, model, task class, and endpoint. Repeated stale failures open
the circuit with bounded exponential cooldown. A half-open probe is small and side-effect free.
Local model load contention is admission-controlled so oversized models cannot thrash unified
memory or starve the strict runtime process.

### 9. Concurrency and runtime protection

Concurrency is computed from measured resident memory, model load state, CPU/GPU pressure, and
provider quotas. Local inference uses weighted admission rather than one uniform worker count.
When a protected strict runtime process is present, the local budget drops to its conservative
profile and remote eligible lanes can absorb work. The router never launches the protected runtime.

Remote task lanes may run concurrently up to their provider quota and local orchestration capacity.
Benchmarks begin at low concurrency, then increase until p95 latency or failure rate crosses a
configured threshold. The highest stable point becomes the profile cap.

### 10. Deployment and rollback

Rollout is staged:

1. Firewall shadow mode on synthetic and local-only traffic, recording deny reasons without sending
   newly eligible private content.
2. Enforced firewall for one non-mutating profile with fake-provider and canary verification.
3. Controlled remote source-slice enablement for approved profiles.
4. Benchmark all candidates and publish a signed routing artifact.
5. Canary the new routes on title/compression/read-only review lanes.
6. Expand to repair and orchestration profiles after latency, correctness, and egress receipts pass.

The previous route table remains an atomic rollback target. A policy or benchmark artifact mismatch
fails closed to the conservative local route. Routing rollback does not disable the firewall.

## Test strategy

### Firewall unit and integration tests

- Classify loopback/in-process separately from LAN, Tailscale, container DNS, public, malformed, and
  missing endpoints.
- Allow only the exact canonical path/range/hash grant; deny adjacent lines, symlink escapes,
  modified files, sensitive paths, and forged provenance.
- Deny private keys, bearer credentials, API-key patterns, JWTs, URL credentials, split secrets,
  and synthetic canaries with zero provider calls.
- Test byte/token boundaries, fail-closed scanner errors, receipt contents, file permissions, and
  concurrent append safety.
- Prove request middleware, execution middleware, Relay rewriting, auxiliary calls, streaming, and
  all supported provider modes cannot bypass the gate.
- Prove firewall metadata never enters provider messages or invalidates prompt-cache prefixes.

### Benchmark and router tests

- Freeze a small deterministic corpus in CI; keep full live provider benchmarking operator-run.
- Validate model discovery, unavailable-model reporting, schema normalization, scoring, tie-breaking,
  cost ceilings, and privacy eligibility.
- Use fake clocks and providers to prove no-output abort, one-retry maximum, circuit open/half-open,
  and absence of retry loops.
- Replay historical stale traces for `qwen3:4b-instruct`, `qwen3.5:4b`, and
  `qwen3-coder:30b` to verify bounded recovery.
- Load-test weighted local admission and assert the protected runtime profile reduces concurrency.
- Require the route artifact to bind the corpus version, inventory, environment, policy digest,
  measurements, and generated profile configuration.

## Acceptance criteria

- A forbidden canary or unapproved source byte results in zero remote HTTP requests.
- Every allowed remote source slice has a content-free durable receipt bound to its exact bytes.
- Every installed local model and currently callable free Nous model appears in the benchmark report.
- Every specialized profile has an independently justified route and fallback.
- Stale providers terminate within their task deadline without reasoning or reconnect loops.
- The selected routing table improves measured expected completion time without reducing required
  correctness, privacy, CI, approval, or merge safety gates.
