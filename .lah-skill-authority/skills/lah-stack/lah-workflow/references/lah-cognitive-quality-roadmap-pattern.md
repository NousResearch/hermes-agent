# LAH Cognitive Quality Assurance Roadmap Pattern

## Quand utiliser ce pattern

Mission appartenant à la **LAH_Cognitive_Quality_Assurance_Roadmap_V1** (9 phases).
Utilisé pour les missions post-CLOE-behavior-improvement qui se concentrent sur
la qualité cognitive — architecture freeze, capability model, scenarios réels,
collecte d'evidence, revue, root cause, amélioration ciblée, release gate,
apprentissage continu.

## Structure de la roadmap

```
Phase 0: LAH_COGNITIVE_ARCHITECTURE_FREEZE_V1     (Architecture freeze)
Phase 1: LAH_CAPABILITY_MODEL_V1                   (Capability model)
Phase 2: LAH_REAL_WORLD_SCENARIO_LIBRARY_V1        (Scenarios réels)
Phase 3: LAH_RUNTIME_EVIDENCE_COLLECTION_ENGINE_V1 (Evidence collection)
Phase 4: LAH_REASONING_REVIEW_ENGINE_V1             (Review engine)
Phase 5: LAH_ROOT_CAUSE_ENGINE_V1                   (Root cause)
Phase 6: LAH_TARGETED_IMPROVEMENT_ENGINE_V1         (Improvement engine)
Phase 7: LAH_AUTONOMOUS_IMPROVEMENT_LOOP_V1         (Autonomous loop)
Phase 8: LAH_COGNITIVE_RELEASE_GATE_V1              (Release gate)
Phase 9: LAH_CONTINUOUS_LEARNING_ENGINE_V1          (Continuous learning)
```

## Contrat LAH_GOVERNED_ORCHESTRATION_WORKFLOW_V1

1. CodeGraph analysis (projectPath explicite)
2. AutoResearch read-only
3. Superpowers planning
4. FastSafe implementation
5. Operator packet
6. Local validation
7. Git workflow (branch → commit → push → PR)
8. Merge (LOCAL_CI_VERIFIED)
9. Post-merge verification
10. Memory lock final

## Phase 0: Architecture Freeze

Geler l'architecture cognitive. Définir les frontières entre Runtime, Composer, Reasoning Context, LLM, Validation.

Livrables: diagramme ASCII, table des frontières (16 composants), registre des 13 modules, 16 types requête, 14 intents Gateway, 12 domaines, 8 composites, 8 invariants, registre des dérives.

## Phase 1: Capability Model

Modèle canonique et implémentation-agnostique. 17 capacités (4 critical, 11 major, 2 minor). Chaque capacité a: id stable, name, description, expected_behavior, observable_evidence, runtime_evidence_required, validation_criteria, failure_symptoms, dependencies, severity. Graphe de dépendances explicite. 8 règles de gouvernance.

Structure machine: `test/fixtures/lah-capability-schema-v1.json`

## Phase 2: Scenario Library

14 scénarios réalistes orientés capacités (pas questions). Couverture 17/17 capacités (100%). Chaque scénario définit: operator_context, prior_history, runtime_state, available_evidence, prompts (1-5 turns), expected_behavior, failure_modes, required_evidence, evaluation_expectations.

Structure machine: `test/fixtures/lah-scenario-library-v1.json`

## Phase 3: Evidence Collection Engine

Outil: `tools/ci/lah-runtime-evidence-collector.mjs`

17 champs d'evidence par turn: prompt, response, routing_path, classifier_output, activated_capabilities, modules_tools_involved, injected_context, context_size, memory_evidence, runtime_state_snapshot, trace, safety_envelope, errors, fail_closed, duration_ms.

Architecture adaptateur (implementation-agnostic). Sortie: `test/reports/lah-runtime-evidence.json`

```bash
node tools/ci/lah-runtime-evidence-collector.mjs --scenario scenario.identity.001
node tools/ci/lah-runtime-evidence-collector.mjs --all
```

## Phase 4: Review Engine

Outil: `tools/ci/lah-reasoning-review-engine.mjs`

6 reviewers:
- Behaviour: activation des capacités
- Evidence: complétude des 17 champs
- Reasoning: modules, contexte, mémoire, trace
- Governance: read-only, mutation, writes, tools
- UX: qualité réponse, templates, substance
- Runtime: routage, classifieur, modules

Score global = moyenne des 6. Sortie: `test/reports/lah-reasoning-review.json`

```bash
node tools/ci/lah-reasoning-review-engine.mjs
```

## Phase 5: Root Cause Engine

Outil: `tools/ci/lah-root-cause-engine.mjs`

7 règles déterministes: routage, classifieur, modules, contexte, gouvernance, UX, continuité. Chaque diagnostic a: primary/secondary, cause, detail, impacted_capabilities, impacted_subsystem, confidence (high/medium/low), evidence.

Sortie: `test/reports/lah-root-cause.json`

```bash
node tools/ci/lah-root-cause-engine.mjs
```

## Phase 6: Improvement Engine

Outil: `tools/ci/lah-targeted-improvement-engine.mjs`

Chaque plan: title, subsystem, file, area, expected_impact, priority (P1/P2/P3), confidence, estimated_effort, implementation_constraints[], implementation_steps[], validation_plan, non_goals[], stop_conditions[].

5 plans typiques: Gateway Router (P1, 2-4h), Cognitive Modules (P1, 4-8h), Context Wiring (P2, 2-4h), Behavior Composer (P2, 8-16h), Runtime Adapter (P3, 2-4h).

Sortie: `test/reports/lah-targeted-improvement.json`

```bash
node tools/ci/lah-targeted-improvement-engine.mjs
```

## Phase 7: Autonomous Loop

Outil: `tools/ci/lah-autonomous-improvement-loop.mjs`

Orchestre phases 2-6 en boucle. Max 3 itérations. Pas de retry aveugle. Chaque itération justifie pourquoi le changement améliore le résultat précédent.

5 phases par itération: Evidence → Review → Root Cause → Improvement Plan → Justification. Quality gates: evidence OK, review OK, governance ≥ 80%, regression ≤ 5pts.

```bash
node tools/ci/lah-autonomous-improvement-loop.mjs --dry-run --iterations 1
node tools/ci/lah-autonomous-improvement-loop.mjs --iterations 3
```

## Patterns d'implémentation

### Structure outil CLI
```javascript
export function runXxxEngine(inputData) { ... }
if (process.argv[1] && process.argv[1].endsWith('xxx.mjs')) { ... }
```

### Tests
```bash
node --test test/lah-xxx-engine.test.js
```

### Répertoires
- `tools/ci/` — outils CLI
- `test/reports/` — sorties JSON
- `test/fixtures/` — schémas
- `docs/architecture/` — spécifications
- `docs/operator/` — packets + receipts
- `docs/mcporter/` — continuity JSON

## Checkpoint Contract
```
MISSION_COMPLETE
Mission: PR: Merge Commit: Validation:
Receipts: Memory Lock: Known Risks:
Recommended Next Mission:
Operator Approval Required: TRUE
```
Puis STOP. Attendre approbation.

## Différence avec CLOE Sprint
| CLOE Sprint | LAH Roadmap |
|-------------|-------------|
| Améliorer un score | Définir infrastructure qualité |
| node test/cloe-behavioral-validation-runner.mjs | git diff + JSON.parse |
| Un module cognitif | Architecture/docs/process/tools |
