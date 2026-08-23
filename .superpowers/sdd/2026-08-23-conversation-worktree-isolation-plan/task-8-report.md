# Task 8 — Guarded fast compression lane

## Scope

- Existing profile role: `performance-sentinel`.
- No profiles, local configuration, services, TradingBot sources, or subagents changed.
- Main chat route, natural-language routing, and default compression inheritance remain unchanged.

## TDD evidence

- RED: the initial focused Python run produced three expected failures: the fast-lane resolver was absent and phase timing telemetry had no contract.
- RED: the request-level wire test then failed with `KeyError: 'max_tokens'`, proving the generic auxiliary builder omitted the configured local fast-lane cap.
- Review RED: `7 failed, 1 passed` proved the first implementation certified configuration before the effective route, suppressed inherited reasoning, treated pre-dispatch liveness as first progress, and did not independently certify fallback caps.
- Review GREEN: `495 passed` across fast-lane, compression telemetry, context-compressor, continuity, fallback-budget, progress-timeout, worker-isolation, auxiliary-client/concurrency, TUI compaction status, and worktree-lineage tests.
- Desktop GREEN: focused compaction lifecycle test passed (`6 passed`); ESLint, Prettier, and TypeScript checks passed.

## Latency and safety contract

- A bounded output request is sent only after the concrete primary provider/model has been resolved once and matches the explicit compression route (including a `summary_model` override), with `reasoning_effort: none` and a positive `max_output_tokens`.
- Fallbacks drop primary-only cap/non-reasoning fast controls. A configured fallback gets a cap only when its own provider, model, `reasoning_effort: none`, and positive output cap match the actual fallback destination; auth-refresh retries are re-certified against their actual destination too.
- Inherited, `auto`, unknown, drifted, reasoning, malformed, and boolean-cap routes stay uncapped. Their pre-existing `reasoning_effort` behavior is unchanged.
- Compression telemetry is content-free and carries queue wait, prompt build, provider dispatch, first provider response/chunk, summary generation, and commit timings. Prompt and summary bodies are not logged. A delayed-first-chunk regression proves pre-dispatch liveness cannot set TTFP.
- The structured checkpoint, redaction, tail, archive/commit, fallback, cooldown, and compression-lineage worktree behavior remain on their existing paths.
- Desktop clears a stale compacting flag only from trusted server terminal or resumed-turn evidence, never from a timer.

## Static checks

- Ruff (`--no-cache`), `git diff --check`, desktop ESLint (0 errors), Task 8 scoped Prettier, and TypeScript passed.
- The repository-wide desktop Prettier baseline remains non-green in 16 unrelated pre-existing files; none are Task 8 files and none were modified.

## Commit

- `d7428c757a perf(compression): add guarded fast summary lane`
- Follow-up review correction: pending commit at report-write time.
