#!/usr/bin/env node
/**
 * LAH Repo Router v4.
 * Canonical model: mission -> roles -> repositories -> explicit write intents.
 * Historical prefixes and aliases are evidence only. They never decide ownership.
 */
const fs = require('node:fs');
const path = require('node:path');

const ROLE_NAMES = [
  'IMPLEMENTATION',
  'EXECUTION_RUNTIME',
  'GOVERNANCE',
  'MEMORY',
  'CONTEXT',
  'SKILL_KNOWLEDGE',
  'BUSINESS_ASSET',
];

const MUTATION_WORDS = [
  'add', 'build', 'change', 'create', 'develop', 'fix', 'implement', 'improve', 'modify', 'repair', 'update', 'write',
  'ajouter', 'corriger', 'créer', 'creer', 'développer', 'developper', 'implémenter', 'implementer',
  'mettre', 'modifier', 'réparer', 'reparer', 'enregistrer', 'publier'
];
const READ_WORDS = ['audit', 'compare', 'consult', 'inspect', 'read', 'reference', 'review', 'analyser', 'analyze', 'lire', 'consulter'];
const ROLE_CUES = {
  MEMORY: ['memory', 'operational memory', 'active operational memory', 'observation', 'observations', 'decision log', 'mémoire', 'memoire', 'context reconstruction', 'historical mission'],
  EXECUTION_RUNTIME: ['runtime', 'provider', 'gateway', 'exoclick', 'supervisor', 'telemetry', 'qdrant'],
  GOVERNANCE: ['decision', 'threshold', 'kill', 'scale', 'bankroll', 'reconciliation', 'profit', 'exploration', 'approval', 'governance', 'checklist', 'economic'],
  SKILL_KNOWLEDGE: ['skill', 'workflow', 'agent instructions', 'knowledge', 'reference pattern', 'codex', 'reusable'],
  BUSINESS_ASSET: ['asset', 'creative', 'niche', 'offer registry', 'business registry', 'content draft', 'business offer'],
  CONTEXT: ['context', 'discovery', 'adr', 'design docs', 'infra specification', 'reference'],
};

function normalize(value) {
  return String(value || '').toLowerCase().replace(/[’]/g, "'").replace(/[_/\\:—–-]+/g, ' ').replace(/[^\p{L}\p{N}]+/gu, ' ').replace(/\s+/g, ' ').trim();
}

function phraseMatches(text, phrase) {
  const haystack = ` ${normalize(text)} `;
  const needle = ` ${normalize(phrase)} `;
  return haystack.includes(needle);
}

function anyPhrase(text, phrases) {
  return phrases.some((phrase) => phraseMatches(text, phrase));
}

function allPhraseWords(text, phrase) {
  const words = normalize(phrase).split(' ').filter(Boolean);
  const haystack = normalize(text);
  return words.every((word) => new RegExp(`\\b${word.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&')}\\b`, 'i').test(haystack));
}

function repoId(repo) {
  return repo.repository_id || repo.repo;
}

function roleOf(repo, role) {
  return (repo.roles || []).includes(role) || repo.repository_role === role.toLowerCase();
}

function createReceipt(mission, mapping) {
  const ontologyStatus = {};
  for (const role of ROLE_NAMES) ontologyStatus[role] = 'CERTIFIED';
  return {
    schema_version: '4',
    schemaVersion: 4,
    mission,
    decision: null,
    decision_source: null,
    reason_code: null,
    primary_role: null,
    repository_authority: null,
    execution_workspace: null,
    implementation_repo: null,
    execution_repo: null,
    governance_repo: null,
    memory_repo: null,
    context_repos: [],
    skill_knowledge_repo: null,
    business_asset_repo: null,
    observability_repo: null,
    write_intents: [],
    write_allowed_repos: [],
    write_forbidden_roots: [],
    role_evidence: {},
    ontology_status: ontologyStatus,
    explicit_target: null,
    conflicts: [],
    confidence: 'low',
    explanation: '',
    evidence: [],
    codegraph_required: false,
    codegraph_reason: null,
    codegraph_used: false,
    codegraph_status: null,
    codegraph_evidence: [],
    codegraph_selected_repo: null,
    codegraph_confidence: null,
    routing_ms: 0,
    total_ms: 0,
    legacy_mapping_schema: mapping.schema_version || mapping.schemaVersion || null,
  };
}

function historicalPrefix(mission, mapping) {
  const token = String(mission).trim().split(/[\s_:—–-]+/)[0].toLowerCase();
  const bare = token.replace(/\d+$/, '');
  const map = mapping.historical_prefixes || mapping.canonicalMissionRepoMap || {};
  const key = Object.keys(map).find((candidate) => bare === candidate || token.startsWith(`${candidate}_`));
  return key ? { prefix: key, repo: map[key] } : null;
}

function explicitTarget(mission, repos) {
  const normalizedMission = normalize(mission);
  const matches = [];
  for (const repo of repos) {
    const id = repoId(repo);
    const checkout = normalize(path.basename(repo.canonical_checkout || ''));
    const aliases = [id, checkout, ...(repo.aliases || [])].filter(Boolean);
    for (const alias of aliases) {
      const normalizedAlias = normalize(alias);
      if (!normalizedAlias || normalizedAlias.length < 4) continue;
      const explicitRepoPattern = new RegExp(`(?:^|\\b)(?:in|dans|within|repo|repository|target|checkout|path)\\s+(?:the\\s+)?${normalizedAlias.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&')}(?:\\b|$)`, 'i');
      if (normalizedMission.includes(normalizedAlias) && (normalizedAlias === normalize(id) || explicitRepoPattern.test(normalizedMission))) {
        matches.push({ repo: id, alias, path: repo.canonical_checkout || null, source: 'operator_mission' });
      }
    }
  }
  const unique = [...new Map(matches.map((match) => [match.repo, match])).values()];
  return unique.length === 1 ? unique[0] : unique.length > 1 ? { conflict: unique } : null;
}

function scoreRepository(repo, mission, role) {
  let score = 0;
  const strong = [];
  const positive = [];
  const negative = [];
  for (const signal of repo.strong_signals || []) {
    if (allPhraseWords(mission, signal)) { score += 60; strong.push(signal); }
  }
  for (const signal of repo.positive_signals || []) {
    if (phraseMatches(mission, signal)) { score += 20; positive.push(signal); }
  }
  for (const signal of repo.negative_signals || []) {
    if (phraseMatches(mission, signal)) { score -= 70; negative.push(signal); }
  }
  const cueHits = (ROLE_CUES[role] || []).filter((cue) => phraseMatches(mission, cue));
  if (cueHits.length && roleOf(repo, role)) score += cueHits.length * 3;
  return { repo: repoId(repo), role, score, strong, positive, negative, cueHits };
}

function mutationRequested(mission) {
  const normalized = normalize(mission);
  return MUTATION_WORDS.some((word) => new RegExp(`\\b${word}\\b`, 'i').test(normalized));
}

function readOnlyRequested(mission) {
  return READ_WORDS.some((word) => new RegExp(`\\b${word}\\b`, 'i').test(normalize(mission))) && !mutationRequested(mission);
}

function intentRoles(mission) {
  const normalized = normalize(mission);
  const roles = [];
  for (const role of ['EXECUTION_RUNTIME', 'GOVERNANCE', 'MEMORY', 'SKILL_KNOWLEDGE', 'BUSINESS_ASSET', 'CONTEXT']) {
    if (role === 'MEMORY') {
      if (anyPhrase(normalized, ['operational memory', 'active operational memory', 'memory authority', 'record observation', 'decision log', 'context reconstruction', 'historical mission lookup', 'update memory'])) roles.push(role);
    } else if (anyPhrase(normalized, ROLE_CUES[role])) roles.push(role);
  }
  return roles;
}

function resolveRole(role, mission, repos, target) {
  const candidates = repos.filter((repo) => roleOf(repo, role)).map((repo) => scoreRepository(repo, mission, role)).filter((item) => item.score > 0 && (item.strong.length || item.positive.length));
  candidates.sort((a, b) => b.score - a.score || a.repo.localeCompare(b.repo));
  if (target) {
    const targetRepo = repos.find((repo) => repoId(repo) === target.repo);
    if (targetRepo && roleOf(targetRepo, role)) {
      const targetScore = candidates.find((candidate) => candidate.repo === target.repo);
      if (targetScore && targetScore.score > 0) return { status: 'RESOLVED', selected: targetScore, candidates, explicit: true };
    }
  }
  if (!candidates.length) return { status: 'UNRESOLVED', selected: null, candidates };
  const top = candidates[0];
  const second = candidates[1];
  if (second && (top.score - second.score < 20 || (top.positive.length && second.positive.length && !top.strong.length))) {
    return { status: 'AMBIGUOUS', selected: null, candidates };
  }
  return { status: 'RESOLVED', selected: top, candidates };
}

function addRoleEvidence(receipt, role, result, repos) {
  receipt.role_evidence[role] = {
    selected_repo: result.selected?.repo || null,
    candidates: result.candidates.map((candidate) => ({
      repo: candidate.repo,
      score: candidate.score,
      strong_signals: candidate.strong,
      positive_signals: candidate.positive,
      negative_signals: candidate.negative,
      cue_hits: candidate.cueHits,
    })),
    evidence_strength: result.selected?.strong?.length ? 'strong' : result.selected ? 'medium' : 'insufficient',
  };
  if (result.selected) receipt.evidence.push(`${role} ownership resolved to ${result.selected.repo}`);
}

function addConflict(receipt, type, details) {
  receipt.conflicts.push({ type, ...details });
}

function writeIntent(receipt, role, repo, scope, source = 'explicit_mission') {
  if (!repo) return;
  if (!receipt.write_intents.some((intent) => intent.role === role && intent.repo === repo)) {
    receipt.write_intents.push({ role, repo, scope, source });
  }
}

function applyWritePolicy(receipt, repos) {
  for (const intent of receipt.write_intents) {
    const repo = repos.find((candidate) => repoId(candidate) === intent.repo);
    if (!repo) continue;
    if (repo.write_policy === 'always_forbidden') {
      receipt.write_forbidden_roots.push(repo.canonical_checkout);
      receipt.decision = 'BLOCKED';
      receipt.reason_code = 'WRITE_POLICY_FORBIDDEN';
      continue;
    }
    if (repo.write_policy === 'requires_explicit_scope' && !intent.source.startsWith('explicit')) {
      receipt.decision = 'BLOCKED';
      receipt.reason_code = 'WRITE_SCOPE_REQUIRED';
      receipt.write_forbidden_roots.push(repo.canonical_checkout);
      continue;
    }
    if (!receipt.write_allowed_repos.includes(intent.repo)) receipt.write_allowed_repos.push(intent.repo);
  }
}

function selectCompatibilityAuthority(receipt) {
  receipt.repository_authority = receipt.implementation_repo ||
    (receipt.primary_role === 'MEMORY' ? receipt.memory_repo : null) ||
    (receipt.primary_role === 'GOVERNANCE' ? receipt.governance_repo : null) ||
    (receipt.primary_role === 'SKILL_KNOWLEDGE' ? receipt.skill_knowledge_repo : null) ||
    (receipt.primary_role === 'BUSINESS_ASSET' ? receipt.business_asset_repo : null) ||
    receipt.execution_repo || null;
}

function routeMission(mission, mapping) {
  const started = Date.now();
  const repos = mapping.repositories || [];
  const receipt = createReceipt(mission, mapping);
  const prefix = historicalPrefix(mission, mapping);
  const target = explicitTarget(mission, repos);
  receipt.explicit_target = target?.conflict ? null : target;
  if (prefix) receipt.evidence.push(`historical prefix ${prefix.prefix.toUpperCase()} is contextual only and cannot resolve ownership`);
  if (target?.conflict) {
    addConflict(receipt, 'MULTIPLE_EXPLICIT_TARGETS', { explicit_targets: target.conflict });
    receipt.decision = 'AMBIGUOUS';
    receipt.reason_code = 'EXPLICIT_TARGET_CONFLICT';
    receipt.confidence = 'low';
    receipt.total_ms = Date.now() - started;
    return receipt;
  }
  if (target) receipt.evidence.push(`explicit operator target preserved: ${target.repo}`);

  const normalized = normalize(mission);
  const isMutation = mutationRequested(mission);
  const requestedRoles = intentRoles(mission);
  const archivedMemory = anyPhrase(normalized, ['archived openclaw agent memory', 'legacy agent memory', 'archived agent memory']);
  if (archivedMemory) {
    receipt.ontology_status.MEMORY = 'CERTIFIED_NON_WRITABLE_ARCHIVE';
    addConflict(receipt, 'ARCHIVED_MEMORY_WRITE', { target: 'openclaw-runtime', reason: 'OpenClaw agent-memory is archived; CarteLogic_v2 is active memory authority.' });
    receipt.decision = 'BLOCKED';
    receipt.reason_code = 'ARCHIVED_MEMORY_NON_WRITABLE';
    receipt.explanation = 'Archived OpenClaw memory cannot receive writes.';
    receipt.total_ms = Date.now() - started;
    return receipt;
  }

  const roleResults = {};
  for (const role of requestedRoles) {
    roleResults[role] = resolveRole(role, mission, repos, target);
    addRoleEvidence(receipt, role, roleResults[role], repos);
  }
  if (isMutation) {
    roleResults.IMPLEMENTATION = resolveRole('IMPLEMENTATION', mission, repos, target);
    addRoleEvidence(receipt, 'IMPLEMENTATION', roleResults.IMPLEMENTATION, repos);
  }

  const execution = roleResults.EXECUTION_RUNTIME?.selected?.repo || null;
  const governance = roleResults.GOVERNANCE?.selected?.repo || null;
  const memory = roleResults.MEMORY?.selected?.repo || null;
  const skills = roleResults.SKILL_KNOWLEDGE?.selected?.repo || null;
  const business = roleResults.BUSINESS_ASSET?.selected?.repo || null;
  let implementation = roleResults.IMPLEMENTATION?.selected?.repo || null;
  if (!implementation) implementation = skills || business || memory || governance || null;

  if (roleResults.IMPLEMENTATION?.status === 'AMBIGUOUS') {
    receipt.decision = 'AMBIGUOUS';
    receipt.reason_code = 'AMBIGUOUS_IMPLEMENTATION_OWNERSHIP';
    addConflict(receipt, 'ROLE_CANDIDATE_CONFLICT', { role: 'IMPLEMENTATION', candidates: roleResults.IMPLEMENTATION.candidates.map((candidate) => candidate.repo) });
  } else if (roleResults.IMPLEMENTATION?.status === 'UNRESOLVED' && isMutation && !implementation) {
    receipt.decision = 'UNRESOLVED';
    receipt.reason_code = 'UNRESOLVED_IMPLEMENTATION_OWNERSHIP';
  }
  for (const role of requestedRoles) {
    if (roleResults[role]?.status === 'AMBIGUOUS') {
      receipt.decision = 'AMBIGUOUS';
      receipt.reason_code = `AMBIGUOUS_${role}_OWNERSHIP`;
      addConflict(receipt, 'ROLE_CANDIDATE_CONFLICT', { role, candidates: roleResults[role].candidates.map((candidate) => candidate.repo) });
    } else if (roleResults[role]?.status === 'UNRESOLVED' && (isMutation || role !== 'CONTEXT')) {
      receipt.decision = 'UNRESOLVED';
      receipt.reason_code = `UNRESOLVED_${role}_OWNERSHIP`;
    }
  }

  if (prefix && !target && receipt.decision === null) {
    const prefixRepo = repos.find((repo) => repoId(repo) === prefix.repo);
    const nonPrefixPositive = repos.flatMap((repo) => (repo.roles || [])
      .filter((role) => ['IMPLEMENTATION', 'BUSINESS_ASSET', 'SKILL_KNOWLEDGE'].includes(role))
      .map((role) => scoreRepository(repo, mission, role)))
      .filter((candidate) => candidate.repo !== prefix.repo && (candidate.positive.length || candidate.strong.length));
    const hermesCoreCurrent = nonPrefixPositive.some((candidate) => candidate.repo === 'hermes-agent' && candidate.strong.length);
    if (prefixRepo && nonPrefixPositive.length && !hermesCoreCurrent) {
      addConflict(receipt, 'HISTORICAL_PREFIX_VS_CURRENT_SIGNALS', { historical_target: prefix.repo, current_candidates: [...new Set(nonPrefixPositive.map((candidate) => candidate.repo))] });
      receipt.decision = 'AMBIGUOUS';
      receipt.reason_code = 'HISTORICAL_SIGNAL_CONFLICT';
    }
  }

  if (target && implementation && implementation !== target.repo) {
    addConflict(receipt, 'EXPLICIT_TARGET_OWNERSHIP_CONFLICT', {
      explicit_target: target,
      ownership_evidence: receipt.role_evidence.IMPLEMENTATION,
      reason: `${target.repo} is not current owner of requested implementation surface`,
    });
    receipt.decision = 'BLOCKED';
    receipt.reason_code = 'EXPLICIT_TARGET_NON_OWNER';
  }

  receipt.implementation_repo = implementation;
  receipt.execution_repo = execution || (implementation && roleOf(repos.find((repo) => repoId(repo) === implementation) || {}, 'EXECUTION_RUNTIME') ? implementation : null);
  receipt.governance_repo = governance;
  receipt.memory_repo = memory;
  receipt.skill_knowledge_repo = skills;
  receipt.business_asset_repo = business;
  receipt.context_repos = (roleResults.CONTEXT?.selected?.repo ? [roleResults.CONTEXT.selected.repo] : []).filter((repo) => repo !== implementation);

  if (isMutation && implementation) writeIntent(receipt, 'IMPLEMENTATION', implementation, 'requested implementation mutation');
  if (memory && isMutation) writeIntent(receipt, 'MEMORY', memory, 'operational memory mutation');
  if (skills && isMutation) writeIntent(receipt, 'SKILL_KNOWLEDGE', skills, 'skill/workflow knowledge mutation');
  if (business && isMutation) writeIntent(receipt, 'BUSINESS_ASSET', business, 'business asset mutation');
  if (governance && isMutation && anyPhrase(normalized, ROLE_CUES.GOVERNANCE)) writeIntent(receipt, 'GOVERNANCE', governance, 'decision/governance mutation');
  applyWritePolicy(receipt, repos);

  const resolvedRoles = [implementation ? 'IMPLEMENTATION' : null, execution ? 'EXECUTION_RUNTIME' : null, governance ? 'GOVERNANCE' : null, memory ? 'MEMORY' : null, skills ? 'SKILL_KNOWLEDGE' : null, business ? 'BUSINESS_ASSET' : null, receipt.context_repos.length ? 'CONTEXT' : null].filter(Boolean);
  if (requestedRoles.length === 1 && ['MEMORY', 'GOVERNANCE', 'SKILL_KNOWLEDGE', 'BUSINESS_ASSET', 'CONTEXT'].includes(requestedRoles[0])) receipt.primary_role = requestedRoles[0];
  else if (resolvedRoles.length === 1) receipt.primary_role = resolvedRoles[0];
  if (resolvedRoles.length > 1 && implementation && requestedRoles.length === 1 && requestedRoles[0] === 'EXECUTION_RUNTIME') receipt.primary_role = 'IMPLEMENTATION';
  if (isMutation && implementation && requestedRoles.length > 0 && requestedRoles.every((role) => ['EXECUTION_RUNTIME', 'CONTEXT'].includes(role))) receipt.primary_role = 'IMPLEMENTATION';
  selectCompatibilityAuthority(receipt);
  if (receipt.decision === null) {
    if (!isMutation && !readOnlyRequested(mission) && !resolvedRoles.length) {
      receipt.decision = 'UNRESOLVED';
      receipt.reason_code = 'INSUFFICIENT_CURRENT_EVIDENCE';
    } else {
      receipt.decision = 'RESOLVED';
      receipt.decision_source = 'ROUTER_V4';
    }
  }
  if (!receipt.decision_source) receipt.decision_source = receipt.decision === 'AMBIGUOUS' || receipt.decision === 'BLOCKED' || receipt.decision === 'UNRESOLVED' ? 'ROUTER_V4' : 'ROUTER_V4';
  if (receipt.decision !== 'RESOLVED') receipt.write_allowed_repos = [];
  receipt.confidence = receipt.decision === 'RESOLVED' ? (receipt.conflicts.length ? 'medium' : 'high') : 'low';
  const authority = repos.find((repo) => repoId(repo) === receipt.repository_authority);
  receipt.execution_workspace = authority?.canonical_checkout || null;
  receipt.total_ms = Date.now() - started;
  receipt.routing_ms = receipt.total_ms;
  return receipt;
}

const mappingPath = process.argv[2] || path.join(process.env.HOME || '/home/deploy', '.hermes/skills/software-development/lah-repo-router/references/repo_mappings.json');
const missionsPath = process.argv[3] || '/dev/stdin';
const mapping = JSON.parse(fs.readFileSync(mappingPath, 'utf8'));
const missions = fs.readFileSync(missionsPath, 'utf8').split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
const counts = { RESOLVED: 0, AMBIGUOUS: 0, UNRESOLVED: 0, BLOCKED: 0 };
for (const mission of missions) {
  const receipt = routeMission(mission, mapping);
  counts[receipt.decision] = (counts[receipt.decision] || 0) + 1;
  const marker = receipt.decision === 'RESOLVED' ? '✅' : receipt.decision === 'AMBIGUOUS' ? '⚠️' : '❌';
  console.log(`${marker} ${mission}`);
  console.log(JSON.stringify(receipt, null, 2));
}
console.log('---');
console.log(`Resolved: ${counts.RESOLVED} | Ambiguous: ${counts.AMBIGUOUS} | Unresolved: ${counts.UNRESOLVED} | Blocked: ${counts.BLOCKED} | Total: ${missions.length}`);
