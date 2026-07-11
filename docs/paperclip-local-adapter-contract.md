# Paperclip Local Adapter Contract

Hermes can run as Paperclip's local execution backend without giving either
system the other's authority. Paperclip owns workflow state, attempts, policy,
and approvals; Hermes owns execution and technical evidence; the adapter only
translates between them.

## Required launch controls

Paperclip launches classic/headless chat and may pass:

```sh
hermes chat --cli -Q -q "$PROMPT" \
  --source paperclip \
  --session-id "$RUN_SESSION_ID" \
  --disable-fallback-model
```

`--session-id` creates a fresh session with the exact supplied identifier. It
accepts 1–128 ASCII letters, digits, dots, underscores, and hyphens, beginning
with a letter or digit. It fails if the identifier is invalid, already exists,
is combined with `--resume` or `--continue`, is sent to the TUI, or the session
store is unavailable. The identifier is claimed with an atomic insert before
provider execution, so two concurrent workers cannot merge their transcripts
under one ID. Transcript cleanup independently rejects path- or drive-shaped
legacy IDs and uses the same safe filename component as request-dump creation.
A caller that wants existing history must use `--resume` instead.

`--disable-fallback-model` is a hard stop, not a routing preference. It disables
both credential-time fallback and the runtime fallback chain. The equivalent
process-level control is `HERMES_DISABLE_FALLBACK_MODEL=1`; it is read before
agent/provider construction so direct `AIAgent` callers receive the same
behavior. Hermes also rechecks the hard-stop latch at the activation boundary,
so a cached-agent or model-switch refresh cannot reintroduce fallback traffic.

Paperclip can bound tool output for one child process without modifying user
configuration:

- `HERMES_TOOL_OUTPUT_MAX_BYTES`
- `HERMES_TOOL_OUTPUT_MAX_LINES`
- `HERMES_TOOL_OUTPUT_MAX_LINE_LENGTH`

Valid positive integers override `tool_output` configuration. Invalid values
fail closed to Hermes' built-in limits rather than silently accepting a larger
configured budget.

## Completion invariant

A process exit code, non-empty text, or syntactically present tool call is not a
completion receipt. When a provider returns a complete, malformed, or truncated
XML/DSML or JSON tool-call envelope as assistant text with no executable tool
call, Hermes retries once. A second occurrence returns an explicit blocked
result and marks the turn failed. The same classifier is used for max-iteration
summaries. JSON coverage includes canonical OpenAI function objects, direct
Hermes/Gemma function objects, lists, and truncation at every structural
boundary. Prose and complete JSON that merely document a tool schema are not
rejected.

Paperclip and its adapter must independently enforce the same invariant; this
Hermes check is defense in depth, not a reason to trust stdout blindly.

## Session database compatibility

Legacy session databases can predate `sessions.parent_session_id`. Hermes adds
missing columns before creating `idx_sessions_parent`, preserving existing rows
and allowing the same database to open after an upgrade. Unknown future schema
versions still follow the normal session-store compatibility policy.

## Cutover and rollback

Use a stable executable indirection such as `~/.local/bin/hermes` in the
adapter. Before moving that symlink:

1. run the focused flag, fallback, final-response, tool-limit, and legacy-state
   tests;
2. run `scripts/run_tests.sh` in full;
3. record the old symlink target and candidate commit;
4. perform a no-provider help/capability probe;
5. move the symlink atomically and verify `hermes --version` plus
   `hermes chat --help`;
6. roll back immediately on malformed-final acceptance, session collision,
   fallback drift, schema-open failure, or regression-suite failure.

Do not embed Portfolio OS business logic or target-repository mutation in
Hermes core. That authority remains outside this integration contract.
