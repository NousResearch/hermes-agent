# Hermes ↔ Plannotator Integration Specification

> For Hermes: treat this as the source-of-truth product and integration spec for rebuilding the Plannotator integration cleanly.

## 1. Goal

Integrate Plannotator into Hermes so a user can open a live browser review/annotation session from chat, submit feedback in the browser, and have Hermes continue working from that feedback in the same conversation flow.

This must work reliably in Telegram topic/thread contexts and must not leave stray local web servers running.

## 2. Product requirements

### 2.1 Primary user experience

Hermes must support these user-visible workflows:

1. Review the last assistant message
   - User invokes `/plannotator_last` or `/plannotator-last`
   - Hermes sends a browser URL immediately
   - User reviews in Plannotator
   - User presses `Send Annotations`
   - Hermes treats the returned annotations as the user's latest input and continues working

2. Review a code diff / review target
   - User invokes `/plannotator_review` or `/plannotator-review`
   - Optional review target is accepted as command arguments
   - Hermes sends URL immediately
   - Hermes continues from returned feedback

3. Annotate a specific artifact
   - User invokes `/plannotator_annotate` or `/plannotator-annotate`
   - Artifact path or target is provided
   - Hermes sends URL immediately
   - Hermes continues from returned feedback

### 2.2 Feedback semantics

Plannotator feedback is not a terminal report.
It is user input.

After feedback is received, Hermes must:
- treat it as the user's latest instruction on the active task
- continue the task by incorporating the feedback
- not stop at a summary unless the feedback explicitly asks only for a summary

### 2.3 Server lifecycle

Plannotator sessions must not leave open web servers behind.

After feedback submission or approval:
- the local Plannotator server must stop promptly
- the supervising process must exit
- the local fixed port must be free again

If the session hangs or shutdown fails:
- Hermes or the bridge supervisor must terminate lingering listeners on the dedicated port

### 2.4 Telegram ergonomics

Telegram only auto-highlights slash commands up to the dash in many contexts.
Therefore Hermes must support both:
- dashed commands:
  - `/plannotator-last`
  - `/plannotator-review`
  - `/plannotator-annotate`
- underscore aliases:
  - `/plannotator_last`
  - `/plannotator_review`
  - `/plannotator_annotate`

Telegram-facing docs/help should prefer underscore forms.

## 3. Non-goals

This integration does not require:
- a remote Plannotator SaaS API
- a mandatory external Plannotator server
- a cloud77-specific router dependency inside Hermes core
- skill-prompt-only orchestration for primary slash-command behavior

Hermes may support skills for documentation and fallback guidance, but the primary slash-command flow must be deterministic and native.

## 4. Hard requirements extracted from prior attempts

These are learned requirements from previous failed or partial implementations.

### 4.1 What worked

1. Stable host per session/chat
   - Reusing a fixed host per Telegram chat/thread improves browser continuity
   - It avoids needless cookie/settings resets

2. Supervisor PID model
   - The launcher must expose a PID whose lifetime matches the browser review session
   - Hermes can reliably wait on that

3. Transcript fallback for `last`
   - Native `plannotator last` is not sufficient in Hermes Telegram sessions
   - Hermes must be able to derive the last assistant artifact itself

4. Local bridge cleanup
   - Pre-launch stale-port cleanup helps recovery from past crashes
   - Post-feedback cleanup is still required even if pre-launch cleanup exists

5. Native gateway command path is better than skill-only prompting
   - Skill prompting is too indirect for core slash commands
   - Direct native gateway handling is more reliable

### 4.2 What did not work

1. Manual bridge launch as proof of inline integration
   - Launching `start_session.py` manually only validates artifact generation and shutdown
   - It does not prove Hermes is waiting and continuing from returned feedback

2. Relying on skill text to make the model call the native tool
   - This is non-deterministic
   - The assistant can echo instructions instead of executing the tool

3. Using `send_message(target='origin')` inside a waiting inline flow in gateway mode
   - In live gateway use, this can interact badly with interrupt/pending-message logic
   - Result observed: the waiting run was interrupted and the Plannotator child got SIGTERM before user input

4. Assuming transcript shape is always `.jsonl`
   - Hermes Telegram sessions may store usable transcripts in:
     - `~/.hermes/sessions/session_<session_id>.json`
   - Fallback resolution must support both transcript shapes

5. Assuming templates are present in the gateway shell env
   - Native command handlers must seed sane defaults when local bridge integration is present

## 5. Design principles

1. Deterministic command routing
   - Slash commands must map directly to native handlers
   - Do not require the LLM to decide whether to call Plannotator for these commands

2. Hermes owns conversation semantics
   - Plannotator returns feedback
   - Hermes decides how feedback becomes user input and how work continues

3. Bridge stays operator-owned
   - Hermes core must stay generic
   - The bridge/launcher contract can be local and operator-specific

4. Gateway sends the immediate URL itself
   - The slash command handler should send the URL through the platform adapter directly
   - Avoid a tool-driven synthetic send that can interfere with active-session interrupt tracking

5. Native `last` must degrade gracefully
   - Try launcher-native `last` only if explicitly desired
   - Hermes fallback from session transcript must be first-class and expected

## 6. Architecture

## 6.1 Layers

1. Gateway slash-command layer
   - Parses `/plannotator_last`, `/plannotator_review`, `/plannotator_annotate`
   - Routes directly to native gateway handlers

2. Hermes Plannotator orchestration layer
   - Resolves action and fallback strategy
   - Resolves launcher templates
   - Creates or identifies the artifact to review
   - Performs prepare → send URL → launch → wait → continue

3. Launcher / bridge layer
   - Starts Plannotator locally
   - Publishes a stable URL
   - Supervises the session process
   - Stops Plannotator after feedback
   - Emits structured `KEY=value` lines

4. Plannotator runtime
   - Hosts the browser UI
   - Captures annotations and feedback
   - Exits after the decision is complete

## 6.2 Command ownership

The following gateway commands must be first-class native commands:
- `plannotator-last`
- `plannotator-review`
- `plannotator-annotate`

The following aliases must resolve to those canonical commands:
- `plannotator_last` → `plannotator-last`
- `plannotator_review` → `plannotator-review`
- `plannotator_annotate` → `plannotator-annotate`

Skills may exist for:
- operator guidance
- manual fallback procedures
- CLI usage notes

But not for primary Telegram slash-command control flow.

## 7. User-facing flows

## 7.1 `/plannotator_last`

### Inputs
- current chat/thread context
- optional freeform user instruction (e.g. what to look at)

### Flow
1. Gateway resolves command to native handler
2. Hermes determines the active session transcript
3. Hermes resolves the last assistant message
4. Hermes writes a temporary markdown artifact
5. Hermes prepares a fixed host/URL
6. Hermes sends the URL to the current chat/thread directly through the platform adapter
7. Hermes launches the actual annotate session pinned to the same host
8. Hermes waits for completion
9. Hermes extracts returned feedback
10. Hermes reinjects that feedback as the next effective user input
11. Hermes continues the task

### Output
- immediate URL message
- later continuation response based on the feedback

## 7.2 `/plannotator_review`

### Inputs
- optional review target argument
- current workspace/repo context

### Flow
Same as above except the artifact comes from review target / diff instead of transcript fallback.

## 7.3 `/plannotator_annotate`

### Inputs
- absolute artifact path or explicitly supported target syntax

### Flow
Same as above except the artifact path is supplied directly.

## 8. Interface contracts

## 8.1 Gateway command interface

### Required behavior
The gateway handler must:
- set session env context before orchestration
- seed default local bridge templates if absent
- send the URL directly via adapter, not via `send_message_tool`
- wait synchronously for the session result
- continue work from feedback

### Pseudocode contract

```python
result = native_plannotator_flow(command, args, source, event)

if result.kind == "prepared":
    adapter.send(chat, url_message)
    wait_result = launch_and_wait(...)
    if wait_result.feedback:
        event.text = feedback_as_user_input(wait_result.feedback)
        return _handle_message_with_agent(event, source, session_key)
    return wait_result.user_message
```

## 8.2 Native orchestration interface

Hermes should expose a clean internal orchestration API independent of the user-facing slash command.

Suggested internal interface:

```python
run_plannotator_flow(
    *,
    kind: Literal["last", "review", "annotate"],
    source: SessionSource,
    event: MessageEvent,
    review_target: str | None = None,
    artifact_path: str | None = None,
    continue_after_feedback: bool = True,
) -> PlannotatorFlowResult
```

Suggested result structure:

```python
@dataclass
class PlannotatorFlowResult:
    success: bool
    url: str | None
    host: str | None
    log_path: str | None
    prepared_message: str | None
    completed: bool
    timed_out: bool
    feedback_detected: bool
    feedback_markdown: str | None
    error: str | None
```

## 8.3 Launcher contract

The launcher/bridge must support these actions:
- `prepare`
- `review`
- `annotate`
- optional `last`

### Required stdout contract

The launcher should emit parseable lines:
- `HOST=...`
- `URL=...`
- `PID=...`
- `LOG=...`

Minimum required:
- `URL=...`

Recommended required for Hermes inline flows:
- `HOST=...`
- `URL=...`
- `PID=...`
- `LOG=...`

### PID semantics
`PID` must identify a supervisor process whose lifetime spans:
- actual Plannotator session runtime
- browser feedback submission
- final log flush
- local server shutdown

It must not be a short-lived bootstrap shell.

### Environment contract
Hermes may pass:
- `PLANNOTATOR_HOST`
- `PLANNOTATOR_REMOTE=1`
- `PLANNOTATOR_PORT=<fixed-port>`
- session identity env vars

## 8.4 Transcript-resolution contract for `last`

Hermes must support these transcript shapes:
- `~/.hermes/sessions/session_<session_id>.json`
- `~/.hermes/sessions/<session_id>.jsonl`

Suggested resolution order:
1. session metadata lookup from active session key
2. inspect `session_<session_id>.json`
3. inspect `<session_id>.jsonl`
4. choose the latest non-empty assistant message content
5. render to temp markdown artifact

## 8.5 Feedback parsing contract

Hermes must parse the returned log and detect a feedback section when present.

Expected marker:
- `# File Feedback`

Bridge trailer lines such as:
- `[bridge] child exited with return code ...`

must be stripped from the extracted feedback block.

Suggested normalized structure:

```python
{
  "feedback_detected": True,
  "feedback_markdown": "# File Feedback\n...",
  "next_step_instruction": "Treat feedback as the user's latest input and continue the work."
}
```

## 9. State machine

### States
1. Idle
2. Preparing
3. URL sent
4. Waiting for feedback
5. Feedback received
6. Continuing work
7. Completed
8. Failed
9. Timed out

### Valid transitions
- Idle → Preparing
- Preparing → URL sent
- URL sent → Waiting for feedback
- Waiting for feedback → Feedback received
- Waiting for feedback → Timed out
- Feedback received → Continuing work
- Continuing work → Completed
- any → Failed

### Invariants
- only one active Plannotator session per session/thread by default
- URL send happens before wait
- fixed host remains stable between prepare and launch
- server is down after Completed, Failed, or Timed out cleanup finishes

## 10. Failure handling

## 10.1 Configuration missing
If templates are missing:
- first try gateway-side default seeding to the local bridge path
- if no bridge exists, return a clear operator-facing error

## 10.2 Port already in use
If the fixed port is busy:
- kill stale Plannotator listeners before launching
- if the port is owned by an unrelated process, fail loudly

## 10.3 Premature interruption
The URL-send step must not register as a new inbound message for the current session.

Implementation rule:
- do not use `send_message_tool(target='origin')` from inside the waiting gateway-native Plannotator flow
- send through the platform adapter directly

## 10.4 No feedback submitted
If the user closes the browser or never submits:
- time out after configured duration
- shut down the session
- return a clear timeout message

## 10.5 Feedback present but continuation fails
If Hermes receives feedback but continuation fails:
- persist the feedback in the transcript
- return a message explaining that feedback was received but work continuation failed
- do not lose the feedback content

## 11. Operational defaults

Recommended defaults for this setup:
- fixed port: `19432`
- remote mode: enabled
- stable host: derived from platform/chat/thread identity
- timeout: 3600s
- shutdown grace: ~1.5s after decision before force cleanup

## 12. Testing requirements

## 12.1 Unit tests
Must cover:
- transcript resolution from both `.json` and `.jsonl`
- feedback block extraction
- bridge trailer stripping
- default template seeding
- direct gateway command routing for slash commands
- underscore and dash alias resolution

## 12.2 Integration tests
Must cover:
- gateway command sends URL immediately
- gateway command then waits for completion
- returned feedback is converted into follow-up user input
- `_handle_message_with_agent` is called again with feedback text

## 12.3 Process/bridge tests
Must cover:
- supervisor PID stays alive during session
- feedback submission triggers clean exit
- lingering listener cleanup frees port `19432`

## 12.4 Live acceptance criteria
A real Telegram acceptance test passes only if:
1. `/plannotator_last` sends a URL immediately
2. the browser remains open until user submits
3. no premature termination occurs before user input
4. after submission, Hermes posts a continuation response based on the feedback
5. port `19432` is no longer listening afterward

## 13. Recommended implementation shape

## 13.1 Keep the generic tool
Keep `plannotator_session` as a reusable native tool for LLM-driven usage and non-slash-command flows.

## 13.2 Add a dedicated gateway-native orchestration path
For slash commands, the gateway should bypass LLM tool selection and call the orchestration path directly.

This is the key architectural lesson from previous attempts.

## 13.3 Keep skills as documentation/fallbacks
Keep Plannotator skills, but reduce their responsibility to:
- documentation
- alternative manual usage
- operator onboarding

Do not make them the primary runtime path for Telegram slash commands.

## 14. Explicit “what worked / what didn’t” summary

### Worked
- stable per-chat host
- transcript fallback for `last`
- supervisor PID waiting model
- post-feedback cleanup of port `19432`
- direct native gateway command routing
- direct adapter send for immediate URL message

### Did not work
- manual bridge launch as evidence of inline integration
- skill-prompt-only slash command routing
- model-only compliance with “use native tool first” instructions
- tool-internal `send_message(target='origin')` while the gateway is waiting
- assuming transcript storage shape is only `.jsonl`
- assuming launcher templates will already exist in the runtime environment

## 15. Open questions

1. Should `/plannotator_annotate` accept only absolute paths, or also named artifacts from a workspace abstraction?
2. Should review flows support both current local diff and explicit PR URL with different command names or flags?
3. Should Hermes persist the prepared URL message ID and later edit it on completion?
4. Should Hermes support multiple concurrent Plannotator sessions per thread, or force one-at-a-time semantics?

## 16. Immediate recommendation

If implementing from scratch, do this order:
1. Build direct gateway-native command handlers
2. Implement transcript fallback for `last`
3. Implement prepare → direct adapter send → waited launch flow
4. Parse feedback into normalized user-input semantics
5. Implement server cleanup guarantees
6. Add underscore aliases for Telegram
7. Add live acceptance test checklist

This order avoids repeating the main historical mistake: trying to solve a deterministic chat-control problem with skill prompting and model compliance instead of a native command path.
