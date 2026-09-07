# gateway/ — messaging gateway, adapters, delivery

Applies on top of the root `AGENTS.md`. Long-form: `website/docs/developer-guide/gateway-internals.md`.
New platform adapter: follow `gateway/platforms/ADDING_A_PLATFORM.md` step by step.

## Shape

`gateway/run.py` is the facade; phases live in `run_*.py` (startup, adapters, inbound, turn, busy,
goals, notifications, shutdown, ...), sessions in `session*.py`, slash handlers in
`slash_commands_*.py` mixins, authorization in `authz_mixin.py`, adapters in `platforms/<name>.py`
over `platforms/base.py`. `builtin_hooks/` is the extension point for always-registered gateway
hooks (none shipped). The gateway reads user YAML **raw** (`run.py` + `config.py`), not through
`DEFAULT_CONFIG` — a key the CLI sees but the gateway doesn't means you're on the wrong loader
(`hermes_cli/AGENTS.md`). Each adapter picks a base toolset (Telegram → `"messaging"`).

Slash commands: handlers are looked up by name through `_command_handler_table`; a command is
listed in `_IDLE_COMMANDS` or `_PLAIN_COMMANDS` (works mid-run) in `run_busy.py`. No
`if canonical == ...` chains. Registry + adding a command: `hermes_cli/AGENTS.md`.

## The gateway has TWO message guards — both must bypass approval/control commands

While an agent is running, an inbound message passes two sequential guards: (1) the **base
adapter** (`platforms/base.py`) queues it in `_pending_messages` when `session_key in
self._active_sessions`; (2) the **runner** (`run.py`/`run_busy.py`) intercepts `/stop`, `/new`,
`/queue`, `/status`, `/approve`, `/deny` before they reach `running_agent.interrupt()`. Any new
command that must reach the runner while the agent is blocked (approval prompts) MUST bypass BOTH
guards and be dispatched inline — never via `_process_message_background()`, which races session
lifecycle.

## Streaming delivery contract (stream-is-the-message adapters)

Adapters with `draft_stream_is_message = True` (relay Slack native streaming) keep ONE cumulative
native stream per turn; the stream IS the final message. Four invariants, each from a live
duplicate-final incident (NS-658 canary ledger, hermes#85796 / gateway-gateway#210); violating any
re-creates a duplicate or frozen stream:

1. **Draft frames are prefix-stable.** Frame N must be a string prefix of frame N+1. Never mutate
   drafts per tick — no fence-closing (`ensure_closed_code_fences`), cursor suffix, segment-state
   resets at tool boundaries, or mrkdwn conversion. A non-prefix frame forces a whole-snapshot
   re-append ("stacked copies"). The finalize path may still transform the real final.
2. **The consumer declares the final; the adapter never guesses.** `finish(final_text)` carries the
   completed `final_response` (verifier footer, completion explainer included). New post-stream
   augmentation MUST ride this payload — mutating `final_response` after the seal re-opens the
   `delivered_final_matches` mismatch → corrective duplicate send.
3. **Interim sends carry `metadata["_interim_send"] = True`.** Any consumer-side `adapter.send()`
   that is not the turn-final (commentary, segment-tail flushes) must set it or seal-interception
   seals the live stream with interim text. Seal-interception exists at BOTH egress doors (`send()`
   and `send_for_platform()`); a new egress door needs the same two checks.
4. **Reconcile by edit, never by plain send.** A lane delivering a final beside a sealed stream
   (queued follow-ups, media-accompanied finals) first tries `edit_message` on the consumer's
   `message_id`; plain `send()` only when no editable message exists. A sealed native stream is a
   regular message — `chat.update` works (live-verified).

Contract tests: `tests/gateway/test_stream_final_contract.py` (mutation-checked). Slack ground
truth: `chat.*Stream` speaks STANDARD markdown, not mrkdwn; `stopStream.markdown_text` APPENDS;
`startStream`/`stopStream` are Tier 2 (~20/min). Check `draft_stream_is_message is True` —
MagicMock adapters in older tests auto-create truthy attributes.

## Background process notifications

`terminal(background=true, notify_on_complete=true)` starts a gateway watcher that detects
completion and triggers a new agent turn. Verbosity: `display.background_process_notifications`
(or `HERMES_BACKGROUND_NOTIFICATIONS`): `concise` (default; one line, failures append an output
tail), `all` (running updates + final raw output), `result` (final raw output only), `error`
(final raw output only on non-zero exit), `off`.

Cron deliveries are NOT mirrored into the target gateway session — they land in their own cron
session with a header/footer frame so the main conversation's role alternation stays intact
(`cron/AGENTS.md`).

## Gateway lifecycle vs. the Desktop app

`hermes serve` (control plane, desktop-spawned child) dies with the app — by design. The messaging
gateway (`gateway run`) SURVIVES the app: the serve backend's `/api/gateway/*` endpoints spawn it
detached (`_spawn_hermes_action` — `start_new_session` / `DETACHED_PROCESS`), so `before-quit`'s
SIGTERM never reaches it and bots keep running. The known breach is the Windows shim-unlock
teardown (`taskkill /T /F` on venv-shim holders, #85265), which exists to let updates proceed and is
replaced by #92091's `pause-for-update`. Do NOT "fix" gateway-dies-with-app by re-parenting the
gateway under the backend, and do NOT "fix" update locks by widening the tree-kill. Gateways stamp
`code_sha`/`code_version` into `gateway_state.json` (`status.py`) so the updater can verify a fleet.

## Profiles and secrets in adapters

- **Token locks.** An adapter that connects with a unique credential (bot token, API key) calls
  `acquire_scoped_lock()` from `gateway.status` in `connect()`/`start()` and `release_scoped_lock()`
  in `disconnect()`/`stop()`, so two profiles cannot share one credential. Canonical:
  `plugins/platforms/irc/adapter.py`.
- **Multiplex profile-scoped env reads MUST fail closed — never borrow from `os.environ`**
  (`agent/secret_scope.py`; #72348, #86905). Under `gateway.multiplex_profiles`, `os.environ` holds
  the DEFAULT profile's values; a secondary profile's `.env` exists only in its secret scope,
  installed per turn by `_profile_runtime_scope`. All profile-level env config — credentials
  (`app_secret`, tokens) AND authorization (`FEISHU_ALLOWED_USERS`, `{PLATFORM}_ALLOW_ALL_USERS`,
  `GATEWAY_ALLOW_ALL_USERS`, `group_policy`, `allow_bots`) — is read scope-aware: adapters via
  `_get_scoped_secret()` (canonical fail-closed copy: `plugins/platforms/feishu/adapter.py`),
  gateway authz via `_auth_env()` / `_platform_gate_env()` (`authz_mixin.py`). Scope installed +
  multiplex active → a scoped miss returns the **default**, NEVER `os.environ` (a leaked allowlist
  skips the allow-all check and silently rejects every secondary-profile sender, #86905). The
  unscoped default-profile path (`UnscopedSecretError`) and single-profile deployments keep the
  `os.environ` read — there it IS the profile's own value. `_get_scoped_secret` is copy-pasted
  across ~15 adapters: when touching one, verify fail-closed semantics and never reintroduce the
  `except _UnscopedSecretError: val = os.getenv(...)` fallback-after-miss shape.

## Tests

`tests/gateway/`. Adapter tests exercise the real delivery path with a fake transport; never
assert on hardcoded platform lists or command counts (root: no change-detectors). Session-key and
guard behaviour are invariants worth a test; platform API quirks belong in connector comments +
tests, not in prose.
