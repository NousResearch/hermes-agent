# Gateway-owned fresh local sessions

The composed gateway listener can create fresh local **CLI, TUI, or GUI-policy** sessions without
starting `tui_gateway`'s separate agent runtime. The existing `GatewayRunner` / `TurnRunner`
own execution, approval waiters, tool calls, session storage, and canonical event delivery.

A server-authenticated identity with `session:create` can call:

```json
{"jsonrpc":"2.0","id":1,"method":"session.create","params":{"request_id":"stable-client-create-id","source":"cli"}}
```

The result preserves `session_id`, `stored_session_id`, `messages`, `message_count`, and
`info`, and includes the authority's normal resume snapshot fields. Both session IDs are
the same stable persisted **logical** ID. Compression and reset may rotate the private
physical transcript target without changing either client ID. Creation is lazy with respect
to the agent, but reserves the real
SessionDB/SessionStore identity before returning. One canonical SQLite transaction commits
the session row, route, and private `gateway.local_policy.v1:<stored-id>` creation receipt.
Repeating the request ID for the same principal and profile reuses that identity and frozen
policy, including after daemon death. Omit the ID for a new session; clients that need
lost-ACK retry protection must retain it. A failed transaction publishes no live session.

Attach another authenticated viewer with `session.resume({session_id})`, and submit
through `prompt.submit({session_id, submission_id, text})`. The accepted receipt is durable
admission, not inference completion. Closing every viewer leaves the execution and its
pending approval alive. A new viewer can resume the same identity and answer through
`approval.respond({session_id, execution_generation, prompt_id, choice})`.

## Deliberate behavior change: shared operator access

Verified operators of the same gateway profile may attach to canonical local sessions
created through another operator authentication path. This replaces the creator-subject-only
access fence for those operators: the existing dashboard gate already grants operator access,
so changing from native CLI/TUI to an authenticated dashboard must not create a second runtime
or prevent attachment solely because the raw identities differ. This is shared access, not
shared identity: creation receipts keep the original owner, and admissions and retries keep
the submitting principal.

The server issues `session:operator` only at these existing trusted boundaries:

- Native interactive bootstrap ticket redemption (not worker-adoption tickets).
- The dashboard WebSocket upgrade gate, including authenticated dashboard login tickets,
  the process-lifetime internal credential for server-spawned PTY clients, and the verified
  legacy session token when that authentication path is enabled.
- The HTTP mutation gate, after a verified dashboard Session, native HTTP owner grant,
  or permitted legacy session token has been accepted.

A client cannot request this scope in RPC parameters or an identity capability list.
Operator scope does not replace profile, instance, per-operation capability, subscription,
revision, or generation checks. It does not upgrade messaging, room, worker, or service
credentials, or permit remote first-claim adoption of unowned historical transcripts. If
restricted dashboard identities are introduced, these issuance gates must reflect that
restriction rather than treating every accepted dashboard credential as an operator.

Cross-subject local submissions carry a private, server-written profile/session/principal
binding for queued execution and recovery. It is not exposed in model requests or public
session snapshots. This change does not expose the gateway listener remotely or implement
OpenAI API-to-local session affinity; those remain separate integration paths.

## Deliberately limited compatibility

- `source` accepts `cli` (default), `tui`, or `gui`. These select the existing agent
  platforms `cli`, `tui`, and `desktop`, respectively. The native local routing identity
  remains server-owned `Platform.LOCAL`; source never grants messaging/native trust.
  Default TUI selection folds in `project`; GUI folds in `project` and `desktop_ui`.
  GUI policy is independent of `HERMES_DESKTOP` and of the attaching viewer's identity.
- Optional flat creation fields: `cwd` (existing absolute gateway-local directory),
  `model` (nonempty model identifier on the daemon's configured provider), and `toolsets`
  (explicit array of established toolset names, including an empty array). Unknown or
  policy-filtered toolsets reject instead of silently disappearing. Explicit toolsets
  override the default surface additions; CLI/TUI cannot explicitly request `desktop_ui`.
  `cwd`, model selection, source and effective toolsets are captured at creation; attach
  cannot change them. A conflicting repeat `request_id` returns `invalid_params`.
- Provider/base URL overrides, reasoning/service-tier, skills, cwd worktree creation,
  seeded history, profile switching, YOLO, and other launch options still return
  `invalid_params`. Provider credentials/routing and reasoning/service-tier defaults still
  use the existing gateway resolution lifecycle; this is not full launch-option parity.
  No supported launch field mutates daemon-wide configuration or process environment.
- Fresh creation requires a server-authenticated identity with `session:create`, supplied
  by the gated dashboard or native bootstrap path. The verified legacy `?token=` route
  stamps a session-token identity and also supports creation. Callers cannot provide an
  identity in RPC parameters, and operator scope does not promote resume-only capabilities
  to `session:create`.
- `session.list({limit})` returns authorized **live** authority sessions (`scope: "live"`),
  not the full historical session picker. `session.info({session_id})` is a lightweight,
  authorized view of source, current model, lazy-agent status, and profile ID.
- `ping({})` returns `{pong: true}`. `runtime.describe({})` exposes only authority identity,
  epoch, and the narrow implemented creation contract. It does not claim full runtime
  protocol readiness, reveal credentials/paths, start an agent, or renew a turn lease.
- Cold restart restores valid private local policies and server-owned routes before execution.
  Creation receipts remain bound to their authenticated creating principal and profile;
  verified operators in that profile may access the canonical session without replacing
  that binding. Caller source/native context cannot reconstruct or authorize a local route.
  Missing, malformed, foreign-profile or mismatched policy/identity fails closed, including preclaim checks;
  historical pre-extension sessions without a receipt are not silently treated as CLI.
- Never-started queued local inputs resume through the same authority FIFO after bootstrap
  readiness. Interrupted started inputs become `unknown`, are never replayed, and pause
  their followers. Reattachment can inspect the unknown snapshot without running it.
- Compression publication atomically advances the receipt's private `entry.session_id` and
  route alongside the child transcript. The receipt retains its creation `session_id`, policy,
  principal/profile binding, and explicit approved `lineage`. Recovery validates compression
  steps with the existing canonical child selector and reset steps with the existing
  `_reset_from` boundary; it never adopts arbitrary forks or copied foreign receipts.
- The canonical SessionStore local reset commits the new row, old-row closure, receipt and
  route before changing the in-memory entry. Reset deliberately starts empty history while
  preserving the frozen launch policy and queued admission owner. It does not replay an
  interrupted execution or cancel later accepted inputs. Creating an independent conversation
  still uses `session.create` with a new request ID; this is not a new revision-fenced reset RPC
  or a change to the multi-view `/new` contract.
- Admission targets, digests, claim generations, subscriber identity and uncertain outcomes
  remain on the stable logical row. Thus lost-ACK retries keep their existing receipts after
  either transition, and shared controls/stream callbacks keep the same execution owner.
  Clients continue resuming the creation ID, not an internal physical continuation ID.

## Verification

`tests/gateway/test_operator_cross_surface.py` exercises native-created/dashboard-attached
and dashboard-created/native-attached sessions through real bootstrap and dashboard
login/callback/cookie/ticket paths in a disposable gateway process. The identity provider and
model endpoint are test fixtures. It verifies client disconnect/reconnect, FIFO admission and
retry, retained attribution and model context, unchanged system messages, and private
operator provenance absent from public responses and model requests. It does **not** kill
and restart the gateway process, directly compare simultaneous live fanout, or exercise
shared approval/clarify across those two authentication methods.

`tests/gateway/test_session_operator_scope.py` covers queued operator input across a fresh
authority epoch using a stub executor and a paused scheduler. This is unit-level recovery
evidence, not real cross-surface cold-process restart certification. The existing recovery
fixtures described below provide separate coverage; their results must not be presented as
a new end-to-end cross-authentication gateway-crash test.

`tests/gateway/test_local_session.py` launches a disposable process with temporary HOME
and HERMES_HOME, the production composed HTTP/WS listener, real single-use authenticated
tickets, a loopback OpenAI-compatible model, the real TurnRunner/AIAgent, and the real
terminal tool. It proves persisted identity and same-agent reattachment; closes both
viewers while approval is pending; reconnects and consents before deleting only an owned
temporary directory; checks canonical history; and rejects forged authentication/source/
profile and unsupported launch fields. A second case runs the real clarify tool,
reattaches after all viewers close, and verifies the answer reaches the next loopback
model request. No native messaging allow-all credential authorizes the execution; a
separate negative control enables messaging allow-all only while proving reconstructed
local source/profile objects still fail closed.

`tests/gateway/test_session_policy.py` adds three simultaneous CLI/TUI/GUI turns against
an owned loopback model. A barrier holds their first requests concurrently; real terminal
calls write separate files in three owned working directories. The next requests expose
the correct cwd results, distinct requested models and surface-specific tool schemas.
Reconnect preserves each agent; launcher environment carriers and config bytes do not
change. This is loopback integration evidence, not native launcher or vendor evidence.
`tests/gateway/test_local_session_recovery.py` proves rollback-before-publication and cold
receipt reuse, then starts separate ordinary daemons on the same temporary database using
real control-socket tickets and authenticated WebSockets. A fixture-only scheduler barrier
kills the first daemon after an input commits but before claim; another model request is
held after its real claim. Fresh daemons execute only the authorized queued input, preserve
history/source/model/toolsets despite changed defaults, and refuse unknown, foreign,
missing and corrupt policy cases. A further restart executes nothing twice. All inference
is loopback-only; this is not native launcher or compression-transfer evidence.

`tests/gateway/test_local_session_lineage.py` adds rollback and isolation invariants, plus
three separate ordinary daemon processes. Its first-process fixture invokes the production
agent rotation publication on the real cached AIAgent and the production SessionStore reset,
then kills the daemon after a durable queued admission. Cold unmodified daemons preserve
compressed history, reset's empty-history boundary, policy/cwd, admission identities and
unknown-state blocking. Unrelated fork work and foreign receipts stay unclaimed. The model
peer verifies recovered cwd in the actual prompt and frozen model/toolsets; this fixture does
not execute a post-restart cwd tool effect, generate a compression summary, or exercise a
native launcher. It proves the production publication/recovery boundary, not those separate
surfaces.
