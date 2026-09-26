# Messaging gateway

The root guidance applies. Read `website/docs/developer-guide/gateway-internals.md` and `gateway/platforms/ADDING_A_PLATFORM.md`. `gateway/run.py` is the facade. Behavior belongs in topical `run_*.py`, `session*.py`, `slash_commands_*.py`, and `platforms/` modules.

## Inbound control and streaming

An inbound message crosses two busy guards: the base adapter's active-session queue and the runner's busy-command dispatch. A control or approval command that must work while the agent is blocked bypasses both and dispatches inline. Sending it through `_process_message_background()` races session teardown.

Adapters with `draft_stream_is_message = True` have one cumulative native stream per turn:

1. Draft frame N is a prefix of frame N+1. Apply markdown transforms, fence closing, and final decoration only at finalization.
2. The consumer declares completion with `finish(final_text)`. The adapter does not infer a final frame, and post-stream additions enter `final_text` before the seal.
3. Every non-final `adapter.send()` carries `metadata['_interim_send'] = True`. Both egress doors enforce the seal. New doors need the same check.
4. Reconcile a final beside an existing stream by editing its `message_id`. Use a new send only when no editable message exists.

Keep `tests/gateway/test_stream_final_contract.py` mutation-sensitive. Do not trust a truthy mock-created `draft_stream_is_message` attribute.

## Background and login operations

Background-process watcher registration occurs on the gateway loop. Pending watchers are recovery for pre-start or stopping windows, not the normal path. Admission means the internal event was durably accepted. A `None` handler result is refusal. Refused batches refund claimed siblings, while real delivery failures use bounded retry. Raw completion events retain their profile-owned adapter. API-server completions remain durable delivery rows rather than autonomous model turns.

`/login` is allowed only in an authenticated paired DM. Topic, channel, broadcast, and peer transports that label themselves `dm` are not paired conversations. The operation changes the install-wide Nous identity, so shared gateways must restrict admin slash access.

Login polling uses its private single-worker executor and one active identity-stamped attempt per process. Cancellation is cooperative because a thread future cannot interrupt the blocking poll. Shutdown sends no completion message, but a server-completed irreversible transfer still persists. After success, evict cached welcome-route agents and clear stale session overrides. Let normal config and credential resolution rebuild them.

## Lifecycle

`hermes serve` is a Desktop control-plane child and dies with the app. `gateway run` is detached and survives. Do not re-parent the gateway under `serve` or widen process-tree killing to make updates easier. Gateway runtime status stamps code identity for updater verification.

## Routing identity and profile scope

Canonicalize each inbound source once through `gateway/session_identity.py::resolve_identity` before deriving a session key, authorization home, or adapter. Pin the resulting `RoutingIdentity` to source copies with `replace_source`. Under multiplexing, an unresolved routed identity fails closed.

Keep the two adapter questions separate:

- `_intake_adapter_for(source)` is the live transport that received and authorizes the event. Without live provenance it returns `None`.
- `_delivery_adapter_for(source)` is the adapter that answers, using the receiving transport when known and otherwise the unique owner of the platform/runtime profile pair.

Never read `self.adapters[platform]` for a source or add another resolver. Shared-credential routing may have different transport and runtime profiles. Persist `SessionEntry.transport_profile` so restored sessions deliver through their original bot or not at all, never through the default bot by heuristic.

An adapter with a unique credential acquires and releases a scoped token lock in its connect/start and disconnect/stop paths.

Every profile activity binds home, secret, and terminal scope. This includes turns, callbacks, media delivery, eviction, session end, shutdown, notifiers, webhooks, hooks, observers, and thread hops. Resolve ownership from the source or session record, not the launch environment.

- All adapter and authorization credential/config reads use the shared scope-aware helpers in `gateway/platforms/_shared.py`. Under multiplexing, a scoped miss returns the default value and never borrows `os.environ`.
- Adapter YAML goes into `PlatformConfig.extra`. It does not mutate process environment for a secondary profile.
- Native hosted rooms can activate multi-profile hosting even when `gateway.multiplex_profiles` is false. Gate standalone paths through the runtime scope helper, not the config flag.
- Hooks and observers register per served profile, and idempotence keys include the profile.
- A shared-ingress platform enabled for an unserved secondary is reported in runtime status with a remedy. It is not silently ignored.
- Compare routed profile ownership with the pinned routing-process home. Do not introduce a second launch-home identity rule.

The API-server adapter rebuilds an agent per request but retains each session's initialized `MemoryManager` in `platforms/api_server_memory_sessions.py`. Check it out exclusively, return it in `finally`, key it by profile home and agent session, and shut it down under the owning scope.

## Configuration and tests

Gateway runtime config uses `hermes_cli/config_effective.py::load_user_config_effective`, not merged `DEFAULT_CONFIG`. Use the CLI guidance before adding settings or commands.

Run `tests/gateway/` through `scripts/run_tests.sh`. Exercise both message guards, real adapter delivery with fake transport, routing-identity persistence, disconnected fail-closed behavior, and two-profile secret/config conflicts. Test behavior rather than platform or command counts.