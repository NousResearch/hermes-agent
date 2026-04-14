# Marson Pack Graceful Integration Plan

> For Hermes: Use subagent-driven-development skill to implement this plan task-by-task.

Goal: Integrate the imported Marson operating-layer upgrades into Hermes in a graceful, Hermes-native way. Preserve the imported assets as source material, avoid hard-wiring OpenClaw-specific control-plane assumptions, and progressively adapt the most valuable surfaces into real Hermes runtime capabilities.

Architecture: Separate the work into two layers. Layer 1 is asset preservation: the imported Marson files remain available in repo for provenance, reference, and selective reuse. Layer 2 is Hermes-native integration: build thin Hermes-native abstractions that reuse the best governance/evidence/operator ideas while anchoring them to Hermes runtime truth, Hermes storage paths, and Hermes command surfaces.

Tech Stack: Hermes gateway runtime status/evidence, routing governance, knowledge lanes, fallback probing/self-healing, imported docs/ops contracts, imported scripts/, imported ops/openclaw/ reference assets, pytest, git.

---

## Wave 0: Preserve imported Marson assets as a clean checkpoint

Objective: Commit the imported payload as a distinct provenance commit before adapting behavior.

Files:
- `docs/ops/**`
- `ops/openclaw/**`
- imported `scripts/**`

Verification:
- imported files exist and are reviewable
- import remains separable from later Hermes-native code changes

Commit:
- `chore: import marson operating-layer pack assets`

---

## Wave 1: Hermes-native operator surfaces

Objective: Build Hermes-native mission/triage operator surfaces on top of existing Hermes runtime truth.

Inputs:
- `gateway/status.py`
- `gateway/run.py`
- `hermes_cli/status.py`
- imported operator surface docs/contracts as reference only

Deliverables:
- a small Hermes-native operator mission snapshot abstraction
- a Hermes-native triage projection abstraction
- JSON-first outputs and optional CLI exposure

Do not directly reuse:
- `ops/openclaw/continuity/operator_mission_control.sh`
- `ops/openclaw/continuity/operator_triage_console.sh`
- `continuity.sh`

---

## Wave 2: Knowledge queue bridge

Objective: Keep Hermes knowledge lanes as the lightweight substrate, then add a bridge into Marson’s governed promotion workflow.

Inputs:
- `agent/knowledge_lanes.py`
- `scripts/manage_knowledge_lanes.py`
- imported:
  - `scripts/knowledge_ingestion_package.py`
  - `scripts/knowledge_promotion_queue.py`
  - `scripts/production_knowledge_ingestion.py`

Deliverables:
- lane item -> promotion candidate/export bridge
- queue backlink metadata into Hermes provenance
- no duplication of canonical authority between Hermes lanes and promotion queue

---

## Wave 3: Evidence ladder integration

Objective: Extend Hermes runtime evidence into a governed release/evidence ladder.

Inputs:
- existing runtime evidence in `gateway/status.py`
- imported release/evidence contracts and gates

Deliverables:
- Hermes-compatible evidence ladder gate
- evidence refs that point at Hermes runtime artifacts
- activation/release discipline built on Hermes-native artifacts

---

## Wave 4: Routing policy and rollout expansion

Objective: Enrich Hermes routing governance with Marson’s routing/topology/cost-governance ideas without replacing Hermes routing runtime wholesale.

Inputs:
- current Hermes routing governance
- probe-driven self-healing fallback work
- imported:
  - `docs/ops/session_topology_routing_policy_v1.json`
  - `scripts/session_topology_routing_policy_contract.py`
  - `scripts/model_rollout_cost_governance_snapshot.py`

Deliverables:
- policy data import/adaptation
- parity/consistency validation
- incremental rollout/cost-governance extensions

---

## Wave 5: Queue/arbitration continuity concepts

Objective: Re-express the best queue/continuity ideas in Hermes-native terms instead of importing OpenClaw’s shell control plane directly.

Inputs:
- imported queue/arbitration docs and schemas
- imported ops/openclaw continuity scripts as reference only

Deliverables:
- Hermes-native queue state model or handoff model where actually useful
- selective adoption of lock/dependency/artifact ideas
- no direct dependency on OpenClaw wrapper env/action-token semantics

---

## Execution order

1. checkpoint imported assets
2. operator surfaces
3. knowledge queue bridge
4. evidence ladder
5. routing policy expansion
6. queue/arbitration continuity adaptation

## Success criteria

- imported Marson assets remain preserved and attributable
- Hermes-native integrations use Hermes truth sources, not OpenClaw hardcoded runtime assumptions
- each wave lands with tests or validation surfaces
- the repo becomes more operable without becoming a frankenstein of mismatched control planes
