# SESSION_TOPOLOGY.md

Pointer map for the **session topology contract family**.

## Contract family split (canonical)

### A) Transport topology (Telegram/OpenClaw lane/session isolation)
Use this when routing by chat/thread/topic identity.

- Contract doc: `docs/ops/session_topology_transport_contract_v1.md`
- Schema: `docs/ops/schemas/session_topology_transport_contract.schema.json`
- Starter template: `docs/ops/templates/session_topology_transport_contract.template.json`
- Request template: `docs/ops/templates/session_topology_transport_route_request.template.json`
- Router helper: `ops/openclaw/continuity/session_topology_router.py`
- Router CLI: `scripts/session_topology_transport_router.py`
- Continuity entrypoint: `ops/openclaw/continuity.sh session-transport-route`
- Tests: `tests/test_session_topology_routing.py`

### B) Route-policy topology (session/task/risk -> route class/model)
Use this when selecting model route class and qualified model.
Low-level CLI is strict/fail-closed by default: provide `--transport-decision` for transport lock.
`continuity.sh session-route` is also strict by default; use `--legacy-allow-missing-transport-decision` only for explicit bounded legacy compatibility.
`continuity.sh session-route` also enforces worker-allocation dispatch fields for `worker_slice` requests by default; use `--legacy-allow-missing-worker-allocation-contract` only for bounded compatibility windows.

- Contract doc: `docs/ops/session_topology_contract_v1.md`
- Schema: `docs/ops/schemas/session_topology_contract.schema.json`
- Starter template: `docs/ops/templates/session_topology_contract.template.json`
- Request template: `docs/ops/templates/session_route_request.template.json`
- Router CLI: `scripts/session_topology_router.py`
- Continuity entrypoint: `ops/openclaw/continuity.sh session-route`
- Tests: `tests/test_session_topology_router.py`

### C) Authority topology (lane classes + leases + ticketed mutation boundaries)
Use this when validating who may govern topology/workflow authority and when mutation must fail-close.

- Contract doc: `docs/ops/lane_topology_authority_contract_v1.md`
- Schema: `docs/ops/schemas/lane_topology_authority_contract.schema.json`
- Starter template: `docs/ops/templates/lane_topology_authority_contract.template.json`

### D) Orchestrator contract bridge topology (snapshot/plan/run/events/replay-resync)
Use this when normalizing XE event/workflow runtime packets into the EX-06 orchestrator contract surfaces.

- Contract doc: `docs/ops/orchestrator_api_contract_v1.md`
- Schemas:
  - `docs/ops/schemas/orchestrator_snapshot_resolve.schema.json`
  - `docs/ops/schemas/orchestrator_plan.schema.json`
  - `docs/ops/schemas/orchestrator_run.schema.json`
  - `docs/ops/schemas/orchestrator_event_stream.schema.json`
  - `docs/ops/schemas/orchestrator_replay_resync.schema.json`
- Templates:
  - `docs/ops/templates/orchestrator_snapshot_resolve.template.json`
  - `docs/ops/templates/orchestrator_plan.template.json`
  - `docs/ops/templates/orchestrator_run.template.json`
  - `docs/ops/templates/orchestrator_event_stream.template.json`
  - `docs/ops/templates/orchestrator_replay_resync.template.json`
  - `docs/ops/templates/orchestrator_contract_bridge_packet.template.json`
- Bridge packet schema:
  - `docs/ops/schemas/orchestrator_contract_bridge_packet.schema.json`
- Bridge evidence packet:
  - `state/continuity/latest/evidence/ex_06_orchestrator_contract_bridge_packet_2026-04-03.json`
- Runtime integration surfaces (bounded EX-06 implementation slice):
  - `ops/openclaw/continuity/orchestrator_contract_v1_surface.py`
  - commands: `plan`, `run`, `emit-event`, `replay-resync`
  - `ops/openclaw/continuity/orchestrator_contract_bridge_packet_v1.py` (integration artifact builder)
- Validation:
  - `tests/test_orchestrator_contract_pack.py`
  - `tests/test_orchestrator_contract_runtime_surface.py`

## Boundary rules (avoid overlap)

1. **Authority topology first**: establish lease/fencing/ticket posture for mutation authority.
2. **Transport topology second**: resolve lane/agent/session-key from transport tuple.
3. **Route-policy topology third**: resolve route class/model for the resulting work context.
4. **Orchestrator bridge topology fourth**: map XE runtime packets to orchestrator contract packets for replay/idempotency/resync semantics.
5. Same basename (`session_topology_router.py`) exists in two locations; always use full path.
6. Do not treat transport topology as a model selector, and do not treat route-policy topology as a topic/thread router.

## Validation

```bash
pytest -q tests/test_session_topology_routing.py
pytest -q tests/test_session_topology_router.py

# optional end-to-end lock run
bash ops/openclaw/continuity.sh session-transport-route \
  --topology docs/ops/templates/session_topology_transport_contract.template.json \
  --request docs/ops/templates/session_topology_transport_route_request.template.json \
  --json > /tmp/transport_decision.json

bash ops/openclaw/continuity.sh session-route \
  --topology docs/ops/templates/session_topology_contract.template.json \
  --request docs/ops/templates/session_route_request.template.json \
  --qualification-decision state/continuity/latest/example_model_gate_decision.json \
  --transport-decision /tmp/transport_decision.json \
  --json
```

## Operator quick diagnostics (disambiguated)

```bash
# Transport/topic routing (chat/thread -> lane/agent/session key)
bash ops/openclaw/continuity.sh session-transport-route \
  --topology docs/ops/templates/session_topology_transport_contract.template.json \
  --request docs/ops/templates/session_topology_transport_route_request.template.json \
  --json

# Route-policy routing (session/task/risk -> route class/model, strict transport conformance default)
bash ops/openclaw/continuity.sh session-route \
  --topology docs/ops/templates/session_topology_contract.template.json \
  --request docs/ops/templates/session_route_request.template.json \
  --qualification-decision state/continuity/latest/example_model_gate_decision.json \
  --transport-decision /tmp/transport_decision.json \
  --json

# Legacy permissive compatibility (bounded opt-in only)
bash ops/openclaw/continuity.sh session-route \
  --legacy-allow-missing-worker-allocation-contract \
  --legacy-allow-missing-transport-decision \
  --topology docs/ops/templates/session_topology_contract.template.json \
  --request docs/ops/templates/session_route_request.template.json \
  --qualification-decision state/continuity/latest/example_model_gate_decision.json \
  --json
```
