# Session Topology Transport Contract v1 (Telegram/OpenClaw)

Date: 2026-03-20  
Status: active (wave-5 slice-2 contract)  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`

## 0) Intent and non-goals

This contract defines bounded **transport-topology** routing semantics for Telegram/OpenClaw:
- lane/topic/agent mapping,
- session isolation boundaries,
- General-topic edge handling,
- deterministic routing invariants.

Non-goals in v1:
- no broad channel plugin refactor,
- no policy-model selection rewrite,
- no mutation of existing non-topic DM session-key shape,
- no model route-class/model-key selection (that lives in `docs/ops/session_topology_contract_v1.md`).

Family pointer map: `docs/ops/SESSION_TOPOLOGY.md`.

## 1) Core doctrine: one topic = one lane = one agent

In topic-enabled contexts:
1. Telegram `message_thread_id` is the canonical lane discriminator.
2. A topic maps to one lane and one owning `agent_id`.
3. Session keys include topic suffix (`:topic:<thread_id>`) for lane isolation.
4. Cross-lane activity is explicit via spawn/handoff (never implicit shared history).

## 2) Topology mapping surface

Canonical mapping artifact (`session.topology.transport_contract.v1`) exposes:
- `telegram.groups.<chat_id>.topics.<thread_id>`
- `telegram.direct.<chat_id>.topics.<thread_id>`

Each chat binding may declare:
- `default_agent_id`
- `default_lane_name`
- `topics_enabled`
- `general_topic_thread_id` (default `1`)
- `reaction_fallback_to_general` (default `true`)
- topic rows (`lane_name`, optional `agent_id` override)

## 3) Session isolation expectations

Session key shape:
- non-topic DM: `agent:<agent_id>:telegram:direct:<chat_id>`
- non-topic group: `agent:<agent_id>:telegram:group:<chat_id>`
- topic route: append `:topic:<thread_id>`

Isolation requirements:
1. Distinct thread ids in same chat MUST resolve to different session keys.
2. Unknown topic ids still resolve to topic-isolated session keys.
3. Lane reset semantics (`/new`, `/reset`) apply only to the resolved session key.

## 4) General-topic quirks (Telegram)

Bounded handling in v1:
1. General topic defaults to thread id `1`.
2. Forum reaction updates missing `thread_id` route to General when `reaction_fallback_to_general=true`.
3. Outbound policy may omit explicit `message_thread_id=1` (`omit_general_topic_thread_id_1=true`) to avoid known General-topic delivery quirks.
4. Topic mode disabled => ignore inbound thread id and emit non-topic session key.

## 5) Deterministic routing invariants

Required invariants:
1. Routing uses transport tuple only (`channel`, `scope`, `chat_id`, normalized `thread_id`).
2. Message content does not affect lane/agent/session resolution.
3. Same tuple + same topology map => same lane and session key.
4. Mapping misses fall back deterministically (never heuristic lane merge).
5. `route_lock` mismatch must fail-closed.

## 6) Structured failure semantics

Transport router fail-close uses structured blocker codes:
- `only_telegram_channel_supported`
- `chat_scope_invalid`
- `chat_id_missing`
- `route_lock_invalid_message_thread_id`
- `route_lock_mismatch`

Operator guidance is emitted in CLI payload (`operator_diagnostics.next_steps`) via:
- `scripts/session_topology_transport_router.py`

## 7) Artifacts

- Schema: `docs/ops/schemas/session_topology_transport_contract.schema.json`
- Template: `docs/ops/templates/session_topology_transport_contract.template.json`
- Router helper: `ops/openclaw/continuity/session_topology_router.py`
- Router CLI: `scripts/session_topology_transport_router.py`
- Dispatcher entrypoint: `ops/openclaw/continuity.sh session-transport-route`
- Request template: `docs/ops/templates/session_topology_transport_route_request.template.json`
- Deterministic tests: `tests/test_session_topology_routing.py`
