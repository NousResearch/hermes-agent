# Zero-Content Truncation Recovery Design

## Problem

The long-lived WhatsApp session can outgrow the effective Ollama context
window. Ollama then returns `finish_reason="length"` with no assistant text.
Hermes currently treats that response like a genuine partial answer, appends a
synthetic continuation prompt, and repeats the same request four times. The
user receives an empty or fragmentary response while the session becomes less
well formed on every retry.

The observed incident had 163 active messages, repeated role-alternation
repairs, four zero-content length completions per turn, and no automatic
session expiry. Hermes was configured for a 131,072-token context and an
8,192-token output allowance while the shared Ollama runtime was observed at a
32,768-token context. A direct local probe established 65,536 tokens as a
working runtime size.

## Scope

This change covers only:

- zero-content `length` completion recovery;
- reliable local context and output limits;
- automatic session rotation; and
- regression coverage for those behaviors.

Delegation, specialist routing, emergency escalation, normal tool use, and
provider fallback remain unchanged.

## Runtime Configuration

The live Hermes configuration will use:

- `model.context_length: 65536`;
- `model.ollama_num_ctx: 65536` so the requested Ollama window is explicit;
- `model.max_tokens: 2048` to reserve predictable response headroom;
- daily session rotation at 04:00 Europe/Amsterdam; and
- idle session expiry after 1,440 minutes.

The matching checked-in operational example will be updated if it projects
these settings. Secrets and household identifiers must not enter the branch.

## Recovery Behavior

Hermes will distinguish two `finish_reason="length"` cases:

1. A non-empty assistant response is a genuine partial response. Existing
   continuation behavior remains intact.
2. An empty assistant response is context/output starvation. Hermes must not
   append a continuation prompt because doing so makes the request larger.

For the second case, Hermes will request conversation compression once and
retry from the compressed, role-valid history. The recovery is bounded to one
compression retry per user turn. If the retry is also an empty length
completion, Hermes returns a short actionable response telling the user to
send `/new`. It does not run the generic four-continuation loop.

The implementation will reuse the existing compression restart mechanism and
turn retry state. It will not delete sessions automatically or silently drop
the current user request.

## Data Flow

1. The provider response is normalized.
2. The conversation loop sees `finish_reason="length"`.
3. If assistant content is non-empty, existing continuation handling runs.
4. If assistant content is empty and recovery has not run, Hermes compresses
   the current conversation and retries once.
5. If the bounded retry is also empty, Hermes persists coherent state and
   returns the `/new` recovery instruction.

Synthetic continuation messages are never persisted for zero-content length
responses.

## Error Handling

- Compression failure uses the existing compression failure reporting and
  ends with the same `/new` instruction.
- A zero-content response with no finish reason remains a provider/stream
  error and follows existing stream recovery; it is not reclassified.
- A partial tool call retains the existing truncated-tool-call safeguards.
- No recovery path activates cloud escalation or changes provider routing.

## Tests

Regression tests will prove:

- an empty `length` completion requests compression instead of continuation;
- only one compression retry occurs per user turn;
- a second empty `length` completion returns the `/new` instruction;
- no synthetic continuation prompt is persisted in that path;
- a non-empty `length` completion retains existing continuation behavior;
- truncated tool-call handling is unchanged; and
- supported session-reset configuration accepts the selected daily/idle
  settings.

Focused conversation-loop and configuration tests will run first, followed by
the relevant gateway/session suite and the repository's standard test command
in proportion to runtime.

## Deployment and Verification

Implementation occurs on branch `hermes/truncation-recovery` in an isolated
worktree. After tests pass:

1. update the approved live configuration;
2. validate the YAML through Hermes configuration loading;
3. restart `hermes-gateway.service` once;
4. verify the unit is active, WhatsApp reconnects, and the effective model
   configuration is 65,536/2,048; and
5. run a clean-session WhatsApp smoke check without sending unsolicited
   outbound content.

Rollback restores the prior configuration, reverts the focused code change,
and restarts the gateway once. Existing session data is retained.
