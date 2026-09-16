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

The idle completion watcher also drains `watch_match` / `watch_disabled`; no user follow-up is
required. Notify-off drains these without waking. Transport failures are retried; unavailable
durable completion owners/transports do not spend delivery attempts. Profile-namespaced process
events keep their owning adapter, including raw progress/final notices. Adapter acceptance is
at-least-once admission, not proof that a model turn or final outbound reply completed. API-server
async completions remain durable delivery rows, never autonomous new model turns. Require the
`admit_internal_event` receipt for completion/watch injection: a handler returning None is not
acceptance. Refused admission refunds every claimed batch sibling without spending an attempt;
actual delivery errors keep their bounded retry policy. Recognized raw API routes resolve after
persisted messaging origins and defer quietly when unavailable; malformed routes still warn.

Cron execution has its own session. Eligible continuable deliveries may mirror or seed the
reply-facing conversation: origin, origin-less home fallback, user-written bare-platform home,
or opted-in explicit targets. `all` expansions do not gain home mirror eligibility. Mirrored
briefs are labelled user turns appended at a turn boundary, preserving role alternation
(`cron/AGENTS.md`).

## `/login` (off-turn, paired DM only)

`/login` is registered in `hermes_cli/commands.py` with `busy_policy="dispatch"` and
`desktop="settings"`, listed in `run_busy.py::_PLAIN_COMMANDS`, and handled by
`GatewayLoginCommandsMixin` (`gateway/slash_commands_login.py`). It refuses outside a paired DM:
`chat_type in {"dm","private"}`, a truthy `chat_id`, and a platform whose `"dm"` really is a paired
conversation — ntfy, raft and a2a all report `chat_type="dm"` for a broadcast topic, a channel and
an agent peer, so posting a consent link there would publish it.

**It binds the whole install.** The sign-in writes the singleton `providers.nous`, so whoever
approves the code owns this gateway's inference and connectors for every chat it serves. Slash
gating is opt-in (`gateway/slash_access.py`): with no `allow_admin_from` set, every user allowed to
DM the bot can run it. Operators of shared gateways must set it.

The handler returns its ack at once and drains `anon_auth.run_sign_in` on a **private single-worker
executor** (`_login_executor`), never the shared 10-thread gateway pool — a promotion wait can last
the code's full expiry, and a live worker in the shared pool makes shutdown skip the SessionDB
close/checkpoint. One attempt per process, stamped with the identity that started it: the same
identity's second `/login` supersedes (the loser ends with the superseded copy pushed into its own
chat); a different identity is refused. `task.cancel()` cannot interrupt a blocking poll inside a
worker thread, so shutdown and supersede both work through `attempt.cancelled`, which
`wait_for_promotion` now polls on a ≤1 s tick. A shutdown-cancelled attempt pushes nothing — the
task is dying and its adapters may already be gone — but if the server had already completed the
transfer it still persists, because that transfer is irreversible.

On completion the handler **evicts** every cached agent still on `nous/welcome` and clears any
session model override pinned to it. It does not switch agents in place and does not write a model
override: `settle_after_upgrade` already moved `model.default`/`model.base_url` in the config, every
turn re-resolves the config and the credentials, and `_agent_config_signature` already forces a
rebuild when the route changes — so an in-place swap buys no cache warmth and an override would pin
an expiring access token that nothing refreshes.

## Gateway lifecycle vs. the Desktop app

`hermes serve` (control plane, desktop-spawned child) dies with the app — by design. The messaging
gateway (`gateway run`) SURVIVES the app: the serve backend's `/api/gateway/*` endpoints spawn it
detached (`_spawn_hermes_action` — `start_new_session` / `DETACHED_PROCESS`), so `before-quit`'s
SIGTERM never reaches it and bots keep running. The known breach is the Windows shim-unlock
teardown (`taskkill /T /F` on venv-shim holders, #85265), which exists to let updates proceed and is
replaced by #92091's `pause-for-update`. Do NOT "fix" gateway-dies-with-app by re-parenting the
gateway under the backend, and do NOT "fix" update locks by widening the tree-kill. Gateways stamp
`code_sha`/`code_version` into `gateway_state.json` (`status.py`) so the updater can verify a fleet.

## Profile scope (adapters, turns, and everything between turns)

- **Token locks.** An adapter that connects with a unique credential (bot token, API key) calls
  `acquire_scoped_lock()` from `gateway.status` in `connect()`/`start()` and `release_scoped_lock()`
  in `disconnect()`/`stop()`, so two profiles cannot share one credential. Canonical:
  `plugins/platforms/irc/adapter.py`.
- **Multiplex profile-scoped env reads MUST fail closed — never borrow from `os.environ`**
  (`agent/secret_scope.py`; #72348, #86905). Under `gateway.multiplex_profiles`, `os.environ` holds
  the DEFAULT profile's values; a secondary profile's `.env` exists only in its secret scope,
  bound per profile *activity* by `_profile_runtime_scope` (turn, callback, eviction, tick — never
  only "per turn"). All profile-level env config — credentials
  (`app_secret`, tokens) AND authorization (`FEISHU_ALLOWED_USERS`, `{PLATFORM}_ALLOW_ALL_USERS`,
  `GATEWAY_ALLOW_ALL_USERS`, `group_policy`, `allow_bots`) — is read scope-aware: adapters via
  `gateway.platforms._shared.get_scoped_secret` (the ONE implementation; adapters import it as
  `_get_scoped_secret`; `extra_or_secret` / `seed_extra_from_env` / `env_is_connected` build on it),
  gateway authz via `_shared.platform_gate_env` (imported by `authz_mixin.py` as `_auth_env`). Scope installed +
  multiplex active → a scoped miss returns the **default**, NEVER `os.environ` (a leaked allowlist
  skips the allow-all check and silently rejects every secondary-profile sender, #86905). The
  unscoped default-profile path (`UnscopedSecretError`) and single-profile deployments keep the
  `os.environ` read — there it IS the profile's own value. Never re-implement the reader in an
  adapter (the `try get_secret / except UnscopedSecretError: os.getenv` shape drifts into a
  fallback-after-miss leak); import the shared one. `tests/gateway/test_shared_platform_boilerplate.py`
  asserts every plugin's `_env_enablement` reads only through it.
- **Scope is bound per profile activity, not per turn.** `run.py::_profile_runtime_scope(home)`
  wraps a routed turn (`run_turn.py::_profile_scope_for_source`); the same binding wraps every path
  that touches a session's home, secrets or terminal scope with no turn on the stack: release and
  eviction (`run_agent_cache.py::_run_release_in_profile_scope` — TTL, LRU and memory-pressure
  eviction all route through it so `on_session_end`/memory flush hit the OWNING profile's provider),
  shutdown (`run_shutdown.py::_finalize_session`), post-turn media delivery
  (`platforms/base.py::_media_delivery_scope`), deferred callbacks (pickers, reactions — capture the
  routed home at command time, re-enter the scope in the callback), notifiers and outbound webhooks.
  Resolve the owning home from the session record (`profile_home`, `agent:<profile>:` key), never
  from `os.environ`, which holds the launch profile. Why: eviction that flushed under the launch scope
  wrote a secondary profile's memories into the default profile's store, silently.
- **Hooks and observers register per served profile.** `builtin_hooks/`, `agent/shell_hooks.py::
  register_from_config` and lifecycle observers are prepared under each profile's scope at startup
  and on profile add/remove; idempotence keys include the profile, and `hooks/` paths resolve at call
  time (a module constant freezes to the first importer).
- **Adapter YAML never reaches `os.environ` under multiplex.** `platforms/_shared.py::
  apply_yaml_bridge` seeds `PlatformConfig.extra` and skips the environ write under a secondary
  profile's scope; gate/allowlist reads go through `platform_gate_env`. A `if not os.getenv(X):
  os.environ[X] = …` bridge is first-profile-wins across the process — test two profiles with
  conflicting flags before touching precedence.
- **Unserved is reported, never silent.** Shared-ingress platforms (WhatsApp bridge, Relay) run on
  the default profile only; a secondary enabling one is logged once with the remedy and stamped
  into runtime status (`run_adapters.py::_note_unserved_secondary_platform`). `needs_attention` is
  set and cleared at the single writer (`_update_platform_runtime_status`) on the connect path.

## Tests

`tests/gateway/`. Adapter tests exercise the real delivery path with a fake transport; never
assert on hardcoded platform lists or command counts (root: no change-detectors). Session-key and
guard behaviour are invariants worth a test; platform API quirks belong in connector comments +
tests, not in prose.
