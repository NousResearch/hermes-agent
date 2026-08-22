#!/usr/bin/env node

/**
 * CLOE — Mission Decomposer
 *
 * Parse un texte de mission LAH Stack et produit un plan de phases JSON.
 * Chaque phase est atomique, avec contexte minimal, outils restreints,
 * gate_pass vérifiable, et artefacts propagés.
 *
 * Usage:
 *   node decompose-mission.mjs <mission-file> <repo-path> <mission-type>
 *
 * Mission types: CODE_CHANGE, READ_ONLY_AUDIT, DESIGN_ONLY, MIXED
 *
 * Retourne: JSON plan sur stdout
 * Exit: 0 si plan généré, 1 si erreur
 */

import { readFileSync } from 'node:fs';

const [missionFile, repoPath, missionType] = process.argv.slice(2);

if (!missionFile || !repoPath) {
  console.error(JSON.stringify({ error: 'Usage: decompose-mission.mjs <mission-file> <repo-path> [mission-type]' }));
  process.exit(1);
}

// ──── LIRE LA MISSION ────

let missionText;
try {
  missionText = readFileSync(missionFile, 'utf8');
} catch (err) {
  console.error(JSON.stringify({ error: `Cannot read mission file: ${err.message}` }));
  process.exit(1);
}

const type = (missionType || 'CODE_CHANGE').toUpperCase();

// ──── PATTERNS DE RECONNAISSANCE ────

const patterns = {
  hasRouting:      /Gate 0|repository|canonical|worktree|branch/i.test(missionText),
  hasCodegraph:    /CodeGraph|codegraph_explore|explore|inspect/i.test(missionText),
  hasAutoResearch: /Gate 2|research|contexte|context|web.?search/i.test(missionText),
  hasDesign:       /Gate 3|plan|design|spécification|spec/i.test(missionText),
  hasFastSafe:     /Gate 4|FastSafe|sécurité|safety/i.test(missionText),
  hasImpl:         /Gate 5|implément|implement|code|feature|fix/i.test(missionText),
  hasTests:        /Gate 6|test|vérification|verify/i.test(missionText),
  hasOperator:     /Gate 7|operator|packet|livraison/i.test(missionText),
  hasCommit:       /Gate 8|commit|stage/i.test(missionText),
  hasPR:           /Gate 9|PR|merge|pull.?request/i.test(missionText),
  hasMemory:       /Gate 10|memory.?lock/i.test(missionText),
  hasContinuity:   /Gate 11|continuity|JSON/i.test(missionText),
  hasSubAgents:    /delegate_task|sub.?agent|parallel|batch/i.test(missionText),
  hasLots:         /Lot [A-Z]\d?/i.test(missionText),
  hasTestFirst:    /test.?first|TDD|RED.?GREEN/i.test(missionText),
  hasBR28:         /BR28|preflight|dry.?run/i.test(missionText),
  hasHttpProof:    /runtime.?proof|HTTP.?route|real.?provider/i.test(missionText),
};

// ──── CONSTRUIRE LES PHASES ────

const phases = [];
let phaseId = 0;

function addPhase(overrides) {
  const phase = {
    id: `phase_${phaseId++}`,
    depends_on: [],
    toolsets: ['terminal', 'file'],
    max_retries: 1,
    critical: true,
    context: {},
    ...overrides,
  };

  // Contexte automatique : toujours inclure le repo
  if (!phase.context.repoPath) {
    phase.context.repoPath = repoPath;
  }
  if (!phase.context.missionType) {
    phase.context.missionType = type;
  }

  phases.push(phase);
}

// ──── PHASE 0: ROUTING ────

addPhase({
  id: 'gate0_routing',
  goal: 'Résoudre le repo canonique via lah-repo-router et créer le worktree',
  gate_pass: { type: 'git_clean', params: { ancestor: null } },
  artifacts_out: ['repository_authority', 'worktree_path', 'branch', 'head_sha'],
  max_retries: 1,
});

// ──── PHASE 0.5: INFRASTRUCTURE AUDIT ────

addPhase({
  id: 'gate0.5_infrastructure_audit',
  goal: 'Auditer les modules existants (maintenance-authority, git-workspace, etc.) pour composer plutôt que construire',
  gate_pass: { type: 'artifact_propagation', params: { fields: ['existing_modules', 'compose_decisions'] } },
  artifacts_out: ['existing_modules', 'compose_decisions'],
  depends_on: ['gate0_routing'],
  max_retries: 1,
});

// ──── PHASE 1: CODEGRAPH ────

addPhase({
  id: 'gate1_codegraph',
  goal: 'Explorer les modules impactés avec codegraph_explore',
  toolsets: ['terminal', 'file'],
  gate_pass: { type: 'artifact_propagation', params: { fields: ['explored_modules'] } },
  artifacts_out: ['explored_modules', 'contracts', 'key_symbols'],
  depends_on: ['gate0_routing'],
  max_retries: 1,
});

// ──── PHASE 1b: CONTRAT INSPECTION ────

addPhase({
  id: 'gate1b_contract_inspection',
  goal: 'Inspecter les contrats exacts des lots précédents (symboles, signatures, schémas)',
  toolsets: ['terminal', 'file'],
  gate_pass: { type: 'artifact_propagation', params: { fields: ['contract_details'] } },
  artifacts_out: ['contract_details', 'existing_patterns', 'reuse_candidates'],
  depends_on: ['gate1_codegraph'],
  max_retries: 1,
});

// ──── PHASE 2: AUTORESEARCH ────

if (patterns.hasAutoResearch) {
  addPhase({
    id: 'gate2_autoresearch',
    goal: 'Collecter le contexte interne (sessions passées, assets) et externe (web, niche) sans mutation',
    toolsets: ['terminal', 'file', 'search', 'web'],
    gate_pass: { type: 'artifact_propagation', params: { fields: ['research_findings'] } },
    artifacts_out: ['research_findings', 'risks', 'constraints'],
    depends_on: ['gate0_routing'],
    max_retries: 1,
    critical: false,
  });
}

// ──── PHASE 3: DESIGN ────

addPhase({
  id: 'gate3_design',
  goal: 'Écrire le document de conception dans docs/superpowers/plans/YYYY-MM-DD-MISSION.md',
  toolsets: ['terminal', 'file'],
  gate_pass: { type: 'file_exists', params: { path: null } },  // path sera résolu après
  artifacts_out: ['design_doc_path', 'architecture_decisions'],
  depends_on: ['gate1b_contract_inspection'],
  max_retries: 1,
});

// ──── PHASE 4: FASTSAFE ────

if (patterns.hasFastSafe) {
  addPhase({
    id: 'gate4_fastsafe',
    goal: 'Exécuter les 15 vérifications FastSafe (batch)',
    toolsets: ['terminal', 'file'],
    gate_pass: { type: 'exit_code_zero', params: { command: null } },
    artifacts_out: ['fastsafe_report'],
    depends_on: ['gate3_design'],
    max_retries: 1,
  });
}

// ──── PHASE 5: IMPLEMENTATION ────

if (type === 'CODE_CHANGE' || type === 'MIXED') {
  // BR28 preflight si présent
  if (patterns.hasBR28) {
    addPhase({
      id: 'gate5a_br28_preflight',
      goal: 'Exécuter le dry-run BR28 — vérifier la sécurité, les gates progressifs, l\'evidence locale',
      toolsets: ['terminal', 'file'],
      gate_pass: { type: 'exit_code_zero', params: { command: null } },
      artifacts_out: ['br28_report'],
      depends_on: ['gate3_design'],
      max_retries: 1,
    });
  }

  addPhase({
    id: 'gate5b_implementation',
    goal: 'Implémenter les fichiers source (tests d\'abord si TDD)',
    toolsets: ['terminal', 'file'],
    gate_pass: { type: 'test_count', params: { min: 1, zero_fail: true } },
    artifacts_out: ['files_created', 'source_files', 'test_files'],
    depends_on: patterns.hasBR28 ? ['gate5a_br28_preflight'] : ['gate3_design'],
    max_retries: 3,
  });

  // HTTP runtime proof si présent
  if (patterns.hasHttpProof) {
    addPhase({
      id: 'gate5c_http_runtime_proof',
      goal: 'Exécuter les scénarios runtime réels avec mock serveur',
      toolsets: ['terminal', 'file'],
      gate_pass: { type: 'test_count', params: { min: 4, zero_fail: true } },
      artifacts_out: ['runtime_proof_report'],
      depends_on: ['gate5b_implementation'],
      max_retries: 2,
    });
  }
}

if (type === 'READ_ONLY_AUDIT') {
  addPhase({
    id: 'gate5_audit_execution',
    goal: 'Exécuter l\'audit via sous-agents (max 3 parallèles) — écrire les findings dans docs/superpowers/plans/',
    toolsets: ['terminal', 'file'],
    gate_pass: { type: 'artifact_propagation', params: { fields: ['audit_report'] } },
    artifacts_out: ['audit_report', 'gap_analysis'],
    depends_on: ['gate3_design'],
    max_retries: 1,
  });
}

// ──── PHASE 6: TESTS ────

if (type !== 'DESIGN_ONLY') {
  addPhase({
    id: 'gate6_tests',
    goal: 'Exécuter tous les tests ciblés + vérifier régression',
    toolsets: ['terminal', 'file'],
    gate_pass: { type: 'test_count', params: { min: 1, zero_fail: true } },
    artifacts_out: ['test_results', 'combined_count', 'no_regression'],
    depends_on: ['gate5b_implementation'],
    max_retries: 2,
  });
}

// ──── PHASE 7: OPERATOR PACKET ────

addPhase({
  id: 'gate7_operator_packet',
  goal: 'Produire le operator packet : ce qui est prêt, ce qui est bloqué, ce qui reste',
  toolsets: ['terminal', 'file'],
  gate_pass: { type: 'artifact_propagation', params: { fields: ['operator_packet_path'] } },
  artifacts_out: ['operator_packet_path'],
  depends_on: ['gate6_tests'],
  max_retries: 1,
  critical: false,
});

// ──── PHASE 8: COMMIT ────

if (type === 'CODE_CHANGE' || type === 'MIXED') {
  addPhase({
    id: 'gate8_commit',
    goal: 'Stage ciblé (pas git add .) puis commit avec message tagué',
    toolsets: ['terminal', 'file'],
    gate_pass: { type: 'git_clean', params: { expect_committed: true } },
    artifacts_out: ['commit_sha', 'files_committed', 'diff_stat'],
    depends_on: ['gate6_tests'],
    max_retries: 1,
  });
}

// ──── PHASE 9: PR/MERGE ────

if ((type === 'CODE_CHANGE' || type === 'MIXED') && patterns.hasPR) {
  addPhase({
    id: 'gate9_pr_merge',
    goal: 'Push → PR → merge (squash) → checkout master → pull → tests',
    toolsets: ['terminal', 'file'],
    gate_pass: { type: 'test_count', params: { min: 1, zero_fail: true } },
    artifacts_out: ['pr_url', 'merge_sha'],
    depends_on: ['gate8_commit'],
    max_retries: 1,
  });
}

// ──── PHASE 10: MEMORY LOCK ────

addPhase({
  id: 'gate10_memory_lock',
  goal: 'Mettre à jour la mémoire Hermes avec le résumé de la mission',
  toolsets: ['terminal', 'file'],
  gate_pass: { type: 'exit_code_zero', params: { command: null } },
  artifacts_out: ['memory_entry'],
  depends_on: [type === 'CODE_CHANGE' || type === 'MIXED' ? 'gate8_commit' : 'gate6_tests'],
  max_retries: 1,
});

// ──── PHASE 11: CONTINUITY JSON ────

addPhase({
  id: 'gate11_continuity_json',
  goal: 'Écrire et valider le continuity JSON dans docs/mcporter/',
  toolsets: ['terminal', 'file'],
  gate_pass: { type: 'file_exists', params: { path: null } },
  artifacts_out: ['continuity_json_path'],
  depends_on: ['gate10_memory_lock'],
  max_retries: 1,
});

// ──── GÉNÉRER LE PLAN ────

// Résoudre les dépendances : convertir les noms en IDs
const idMap = {};
for (const p of phases) {
  idMap[p.id] = p;
}

// Valider les dépendances
for (const p of phases) {
  for (const dep of p.depends_on) {
    if (!idMap[dep]) {
      console.error(JSON.stringify({ error: `Phase ${p.id} depends on unknown phase: ${dep}` }));
      process.exit(1);
    }
  }
}

// Vérifier l'ordre topologique (DAG)
const visited = new Set();
const visiting = new Set();

function visit(phaseId) {
  if (visiting.has(phaseId)) {
    console.error(JSON.stringify({ error: `Circular dependency detected involving ${phaseId}` }));
    process.exit(1);
  }
  if (visited.has(phaseId)) return;
  visiting.add(phaseId);
  const phase = idMap[phaseId];
  for (const dep of phase.depends_on) {
    visit(dep);
  }
  visiting.delete(phaseId);
  visited.add(phaseId);
}

for (const p of phases) {
  if (!visited.has(p.id)) visit(p.id);
}

// Ordonner topologiquement
const ordered = [];
const orderedSet = new Set();

function order(phaseId) {
  if (orderedSet.has(phaseId)) return;
  const phase = idMap[phaseId];
  for (const dep of phase.depends_on) {
    order(dep);
  }
  orderedSet.add(phaseId);
  ordered.push(phase);
}

for (const p of phases) {
  if (!orderedSet.has(p.id)) order(p.id);
}

// ──── PRODUCTION ────

const plan = {
  mission: {
    file: missionFile,
    type,
    repo: repoPath,
    patterns,
    estimated_phases: ordered.length,
  },
  phases: ordered.map((p, i) => ({
    ...p,
    sequence: i + 1,
    total: ordered.length,
  })),
  generated_at: new Date().toISOString(),
};

console.log(JSON.stringify(plan, null, 2));

if (ordered.length === 0) {
  console.error(JSON.stringify({ error: 'No phases generated' }));
  process.exit(1);
}

process.exit(0);
