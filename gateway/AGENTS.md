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

The agent-thread approval notification implementation is in
`run_turn_runner_approval_notify.py`. `TurnRunner._approval_notify_sync` forwards to it.

The shared `send_exec_approval` implementation is in
`platforms/base_approval_prompt.py`. Relay approval rendering and resolution
are in `relay/approval.py`. Discord, Slack, Telegram and Feishu approval
rendering and callbacks are in their platform plugin `approval.py` siblings.
Discord's approval view is created there by `create_exec_approval_view`;
its adapter keeps the existing public view registration and SDK patch seams.

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

`terminal(background=true, notify_on_complete=true)` arms a watcher on the gateway loop
(`run_notifications.py::arm_process_watcher`) that starts a new turn on completion; verbosity is
`display.background_process_notifications`. Injection requires the `admit_internal_event` receipt
(a handler returning None is not acceptance); refused admission refunds every claimed batch sibling
without spending an attempt. Long form: `website/docs/developer-guide/gateway-internals.md`
§ Background process notifications.

## `/login` (off-turn, paired DM only)

`GatewayLoginCommandsMixin` refuses outside a real paired DM (ntfy, raft and a2a report `dm` for
broadcast surfaces) and binds the WHOLE install's `providers.nous`, so shared gateways must set
`allow_admin_from`. It drains `anon_auth.run_sign_in` on its private single-worker `_login_executor`
(never the shared pool), allows one attempt per process, and on completion evicts agents still on
`nous/welcome` instead of writing a model override. Long form: `gateway-internals.md` § `/login`.

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

- **One identity per inbound event, canonicalized FIRST.** `gateway/session_identity.py::resolve_identity`
  answers "which bot received it / who may admit it / where does it run" ONCE per event and pins a
  frozen `RoutingIdentity` on the source (wire-invisible, like `_transport_adapter_ref`). Every
  ingress path calls the canonicalize seam before it derives a key: the adapter side
  (`platforms/base.py::_canonicalize`, called by `platforms/base_processing.py::handle_message`,
  `platforms/base_text_batching.py::_enqueue_text_event`, Telegram photo /
  album routing, every `_source_session_key`, and
  `platforms/base_busy.py::_handle_message_while_active`) and the runner side
  (`run_adapters.py::_canonicalize` — the per-profile / default message, busy and platform-event
  handlers, the adapter auth-check callback, `run_inbound_admission.py::_hm_admit_event`). No key derivation
  before it; an unresolved identity under multiplexing (route to an unserved profile) is dropped
  with one WARNING at the first seam it reaches, never keyed into `agent:main`. `_transport_owner`,
  `_authorization_home_for_source`, `_resolve_profile_home_for_source`, `_session_key_profile` and
  `_resolve_profile_for_key` read the identity when present and fall back to their old chain only
  for sources nothing resolved (restored rows, hand-built sources). Never derive a second answer
  next to the identity; extend the object. `transport_profile` ≠ `runtime_profile` is normal
  (shared bot → routed satellite). A source copy goes through `session_identity.replace_source`
  so the identity travels with it (`_apply_topic_recovery` does).
- **Intake vs delivery adapter.** `authz_mixin.py::_intake_adapter_for(source)` is the bot that
  RECEIVED the event (live transport ref → relay socket → identity's `transport_profile`); it gates
  intake policy (ignored channels, relay fronting, re-dispatch of a still-live event) and fails
  closed to `None` for a source with no live provenance. `_delivery_adapter_for(source)` is the bot
  that ANSWERS — the receiving bot whenever known, else the unique owner of `(platform,
  runtime_profile)` via `_adapters_for_profile`. Never read `self.adapters[platform]` for a source;
  never add a third resolver. The matrix (`tests/gateway/test_multiplex_transport_matrix.py`):

  | Topology | runtime | intake | delivery |
  |---|---|---|---|
  | per-credential bot, no route | owner | owner adapter | owner adapter |
  | shared credential → satellite (`profile_routes`) | routed | receiving adapter | receiving adapter (`_is_shared_bot_satellite` after restart) |
  | shared bot → profile that owns its own bot | routed | receiving adapter | receiving adapter (conversation continuity; #70625's "routed bot" reading was not adopted) |
  | secondary-owned bot → `default` (`bot_profile`) | default (`agent:main`) | receiving adapter | receiving adapter |
  | restored / hand-built, no live provenance | stored `source.profile` | **`None`** | unique owner of `(platform, runtime)`; a disconnected secondary → `None`, never the default bot |
- **Identity survives the process.** `SessionEntry.transport_profile` (routing index +
  `sessions.transport_profile`, nullable, reconciled by `SCHEMA_SQL`) persists the receiving bot
  next to the key namespace; the namespace says where a lane RUNS, the column says which bot may
  DELIVER to it. Anything reviving a session from durable state (auto-resume, heartbeat restore,
  plugin injection, background-process events) reads `entry.origin` through
  `authz_mixin.py::_restored_source`, which re-pins a `RoutingIdentity(transport=None)` via
  `session_identity.restore_identity`; `_delivery_adapter_for` then delivers through that bot's
  adapter or nothing (never the default bot by heuristic). Rows without the column (pre-PR-5)
  keep the `_is_shared_bot_satellite` fallback. Deferred callbacks capture the identity/home at
  command time (`/model` picker); `_run_in_executor_with_context` carries the scope over thread
  hops. Relay: `_with_scope` echoes the routed `profile` on every outbound frame and `follow_up`
  derives it from the key namespace so the connector stamps it on the next `passthrough_forward`.
  Ambient `get_active_profile_name()` reads in `gateway/` are boot-only and marked
  `# launch profile, pre-identity`; a path with an identity reads `identity.runtime_profile`.
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
  shutdown (`run_shutdown.py::_finalize_session`, with notices in
  `run_shutdown_notices.py` and process teardown in `run_shutdown_processes.py`), post-turn media delivery
  (`platforms/base.py::_media_delivery_scope`), deferred callbacks (pickers, reactions — capture the
  routed home at command time, re-enter the scope in the callback), notifiers and outbound webhooks.
  Resolve the owning home from the session record (`profile_home`, `agent:<profile>:` key), never
  from `os.environ`, which holds the launch profile. Why: eviction that flushed under the launch scope
  wrote a secondary profile's memories into the default profile's store, silently.
- **`api_server` rebuilds the agent per request but not the memory provider.** The adapter bypasses
  `TurnRunner` and the agent cache (per-request callbacks, model route, ephemeral prompt), so
  `platforms/api_server_memory_sessions.py` parks each session's initialised `MemoryManager` between
  requests (exclusive check-out in `_create_agent`, check-in in the turn's `finally`, keyed by profile
  home + `agent.session_id`; idle/LRU eviction shuts down under the owning home) and
  `AIAgent(memory_manager=...)` adopts it without a second provider init. Without it the previous
  turn's queued recall never reaches the next request (#120116).
- **`multiplex_profiles: false` is not "no scope ever".** A native hosted room serving a second
  profile flips the process-wide guard (`tui_gateway/launch_profile_policy.py::
  activate_multi_profile_hosting`) inside the gateway process, after the adapters were wired; every
  standalone entry point (`run_turn.py::_profile_scope_for_source`, the primary adapter's message /
  busy / platform-event handlers via `run_adapters.py::_standalone_scoped`) then binds the launch
  profile's OWN scope through `run_turn.py::_standalone_launch_scope` — `.env` over the env frozen at
  activation, never a `.env`-only rebuild (systemd / `op run` keys have no file) and never live
  `os.environ`. Gate a new standalone path on that helper, not on the config flag (#112878).
- **Hooks and observers register per served profile.** `builtin_hooks/`, `agent/shell_hooks.py::
  register_from_config` and lifecycle observers are prepared under each profile's scope at startup
  and on profile add/remove; idempotence keys include the profile, and `hooks/` paths resolve at call
  time (a module constant freezes to the first importer).
- **Adapter YAML never reaches `os.environ` under multiplex.** `platforms/_shared.py::
  apply_yaml_bridge` seeds `PlatformConfig.extra` and skips the environ write under a secondary
  profile's scope; gate/allowlist reads go through `platform_gate_env`. A `if not os.getenv(X):
  os.environ[X] = …` bridge is first-profile-wins across the process — test two profiles with
  conflicting flags before touching precedence.
- **Unserved is reported, never silent.** Shared-ingress platforms (Relay) run on the default
  profile only; a secondary enabling one is logged once with the remedy and stamped into runtime
  status (`run_adapters.py::_note_unserved_secondary_platform`). The WhatsApp bridge is per profile:
  a paired secondary gets its own adapter and port (`plugins/platforms/whatsapp/bridge_ownership.py`),
  an unpaired one is reported as `whatsapp_unpaired`, and a secondary never adopts or signals a
  bridge its own pidfile (pid + start time + port) does not identify. `needs_attention` is
  set and cleared at the single writer (`_update_platform_runtime_status`) on the connect path.
- **One launch-home identity.** "Does this task serve a routed profile?" compares the override
  with `hermes_constants.get_routing_process_hermes_home()` (`agent/secret_scope.py::
  serves_routed_profile` and `_is_process_home`, `tools/environments/local.py::_is_routed_home`,
  `hermes_cli/env_loader.py::_process_hermes_home`), never with `os.environ["HERMES_HOME"]` read
  live: an embedding host that mirrors the served profile into the env var per turn (Hermes
  WebUI) pins its own home with `pin_process_hermes_home()`, and without a pin the resolver is
  `get_process_hermes_home()` unchanged. Do not add another routing decision that compares
  against `get_process_hermes_home()` directly; that resolver is for process-level assets.

## Tests

`tests/gateway/`. Adapter tests exercise the real delivery path with a fake transport; never
assert on hardcoded platform lists or command counts (root: no change-detectors). Session-key and
guard behaviour are invariants worth a test; platform API quirks belong in connector comments +
tests, not in prose.
