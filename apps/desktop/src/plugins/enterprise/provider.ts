/**
 * Command Center M1 — in-process synthetic data provider.
 *
 * Per Architect's Path B correction (2026-08-31): `web_server._discover_dashboard_plugins()`
 * only mounts a plugin's `dashboard/manifest.json` API file; a bare `plugins/command-center/`
 * adapter does not mount and cannot be reached via `ctx.rest`. M1 therefore has NO backend
 * adapter, NO server route, and NO network call anywhere in this plugin.
 *
 * This module is the entire M1 "backend": pure, synchronous, deterministic data generation
 * that runs inside the Desktop renderer process. It imports nothing from the Hermes runtime,
 * the gateway client, or any HTTP/fetch/WebSocket surface. `fetch`, `XMLHttpRequest`, and
 * `ctx.rest` do not appear anywhere in this file or in the sibling api module.
 *
 * Every value returned carries the same versioned Envelope<T> contract a future real provider
 * must also satisfy (see ./types.ts), so the 12 screens in ./page.tsx do not need to change
 * shape when a real, separately-authorized provider replaces this one — only the binding in
 * ./api.ts changes.
 *
 * All data is fabricated. `source` is always "synthetic-m1" and `freshness.state` is always
 * "synthetic". Nothing here should ever be mistaken for live Hermes/Syntex/EKV output.
 */

import type {
  AgentObservation,
  AttentionItem,
  Capabilities,
  Envelope,
  FreshnessState,
  KnowledgeStatus,
  ObjectiveSummary,
  OrbState,
  SecurityEvent,
  SummaryData,
  WorkData,
} from './types'

// ---------------------------------------------------------------------------
// Contract versioning
// ---------------------------------------------------------------------------
export const CONTRACT_VERSION = 'm1.0.0'
export const SOURCE = 'synthetic-m1'

/** Injectable clock so tests can assert on a fixed instant instead of wall-clock time.
 *  Defaults to the real clock at call time — still deterministic in shape (ISO string),
 *  just not deterministic in value, which is correct for a running app. */
export type Clock = () => string
const defaultClock: Clock = () => new Date().toISOString()

function envelope<T>(
  projection: string,
  data: T,
  opts: {
    clock?: Clock
    freshness?: FreshnessState
    note?: string
    scopeRef?: string
    status?: Envelope<T>['status']
    warnings?: string[]
  } = {}
): Envelope<T> {
  const clock = opts.clock ?? defaultClock

  return {
    contract_version: CONTRACT_VERSION,
    projection,
    projection_version: '1',
    generated_at: clock(),
    scope_ref: opts.scopeRef ?? 'owner',
    source: SOURCE,
    freshness: { state: opts.freshness ?? 'synthetic', note: opts.note },
    status: opts.status ?? 'available',
    data,
    warnings: opts.warnings ?? [],
  }
}

// ---------------------------------------------------------------------------
// Fixture data — fabricated, stable across calls within a process lifetime.
// ---------------------------------------------------------------------------
const AGENTS: AgentObservation[] = [
  {
    id: 'architect',
    display_label: 'Architect',
    role_label: 'Technical Architecture',
    implementation_status: 'live',
    live_status: 'idle',
    current_focus: 'reviewing the Command Center M1 provider boundary',
    last_observed: null,
  },
  {
    id: 'builder',
    display_label: 'Builder',
    role_label: 'Implementation',
    implementation_status: 'live',
    live_status: 'running',
    current_focus: 'implementing the M1 in-process synthetic provider',
    last_observed: null,
  },
  {
    id: 'picasso',
    display_label: 'Picasso',
    role_label: 'Creative / UX Design',
    implementation_status: 'live',
    live_status: 'idle',
    current_focus: 'awaiting Phase 7 visual fidelity review',
    last_observed: null,
  },
  {
    id: 'antagonist',
    display_label: 'Antagonist',
    role_label: 'Adversarial Review',
    implementation_status: 'live',
    live_status: 'idle',
    current_focus: null,
    last_observed: null,
  },
  {
    id: 'underboss',
    display_label: 'Underboss',
    role_label: 'Executive Orchestration',
    implementation_status: 'live',
    live_status: 'idle',
    current_focus: null,
    last_observed: null,
  },
  {
    id: 'syntex',
    display_label: 'Syntex',
    role_label: 'Knowledge Manager',
    implementation_status: 'live',
    live_status: 'idle',
    current_focus: null,
    last_observed: null,
  },
  {
    id: 'knowledge-agent',
    display_label: 'Knowledge Agent',
    role_label: 'Vault Gatekeeper (proposed)',
    implementation_status: 'proposed',
    live_status: null,
    current_focus: null,
    last_observed: null,
  },
  {
    id: 'recovery-agent',
    display_label: 'Recovery Agent',
    role_label: 'Incident Recovery (proposed)',
    implementation_status: 'proposed',
    live_status: null,
    current_focus: null,
    last_observed: null,
  },
]

const OBJECTIVES: ObjectiveSummary[] = [
  { id: 'obj-cc-m1', name: 'Command Center M1 (synthetic)', status: 'at_risk', progress_pct: 65, note: 'Pending Architect verification of Path B correction.' },
  { id: 'obj-ekv', name: 'Enterprise Knowledge Vault', status: 'on_track', progress_pct: 40, note: 'Phase 2C planning underway.' },
]

const ATTENTION_ITEMS: AttentionItem[] = [
  {
    id: 'att-1',
    classification: 'decision',
    headline: 'Command Center backend adapter approach requires a decision',
    description: 'Path A (server-mounted adapter) does not mount under the current plugin discovery contract. Path B (in-process synthetic provider) is the current direction.',
    raised_by: { agent_id: 'architect', agent_name: 'Architect' },
    raised_at: '',
    evidence_confidence: 'high',
    recommendation_confidence: 'medium',
    primary_action_label: 'Review decision',
  },
  {
    id: 'att-2',
    classification: 'information',
    headline: 'M1 is synthetic-only; no real integration is authorized',
    description: 'All 12 screens render fabricated data. No Hermes, Syntex, EKV, or external service is contacted anywhere in this plugin.',
    raised_by: { agent_id: 'builder', agent_name: 'Builder' },
    raised_at: '',
    primary_action_label: 'Acknowledge',
  },
]

const SECURITY_EVENTS: SecurityEvent[] = [
  {
    id: 'sec-1',
    what: 'Synthetic-only boundary confirmed for Command Center M1.',
    why_it_matters: 'No production credentials, live Vault access, or command execution path exists in M1 — an incident here cannot expose real data.',
    what_is_needed: 'No action needed. Informational only.',
    severity: 'info',
    timestamp: '',
  },
]

// ---------------------------------------------------------------------------
// Public provider surface — pure, synchronous generators, wrapped as Promises
// only to preserve the async call shape ./api.ts already exposes to screens.
// ---------------------------------------------------------------------------
export function getCapabilities(clock: Clock = defaultClock): Envelope<Capabilities> {
  return envelope('capabilities', {
    adapter_version: CONTRACT_VERSION,
    features: {
      command_execution: false,
      real_agent_data: false,
      real_knowledge_data: false,
      real_security_data: false,
      notifications: false,
      recovery_actions: false,
    },
    source_health: [
      { name: 'hermes-gateway', state: 'unknown', note: 'Not contacted in M1 (synthetic-only boundary).' },
      { name: 'syntex', state: 'unknown', note: 'Not contacted in M1.' },
      { name: 'ekv', state: 'unknown', note: 'Not contacted in M1.' },
    ],
    policy_state: {
      command_execution: 'disabled',
      real_integration: 'not_authorized',
    },
  }, { clock })
}

export function getOrbState(clock: Clock = defaultClock): Envelope<OrbState> {
  // Per Architect's G8/G5 direction: orb is ALWAYS unknown in M1. No trigger logic,
  // no count-based inference, no third visual state — unknown is expressed here and
  // in the ARIA label, never as a new orb color/animation.
  return envelope('orb-state', {
    state: 'unknown',
    visual_hint: 'neutral',
    reason: 'No authoritative attention signal is wired in M1 — this is a synthetic placeholder, not a computed state.',
    aria_label: 'Command Center status: unknown. Real-time attention signal is not connected in this build.',
    attention_count: {
      decisions: 0,
      actions: 0,
      source: SOURCE,
      note: 'Not used to drive the orb in M1 — count-based trigger logic is a design placeholder only, per Picasso/Architect.',
    },
  }, { clock, freshness: 'unknown' })
}

export function getAgents(clock: Clock = defaultClock): Envelope<{ agents: AgentObservation[]; observed_at: string; source: string }> {
  const now = clock()

  return envelope('agents', { agents: AGENTS, observed_at: now, source: SOURCE }, { clock })
}

export function getWork(clock: Clock = defaultClock): Envelope<WorkData> {
  const now = clock()
  const items = ATTENTION_ITEMS.map(item => ({ ...item, raised_at: item.raised_at || now }))

  return envelope('work', { observed_at: now, source: SOURCE, objectives: OBJECTIVES, attention_items: items }, { clock })
}

export function getKnowledge(clock: Clock = defaultClock): Envelope<KnowledgeStatus> {
  // Knowledge Agent is documented as PROPOSED, not implemented — Vault is never reached.
  return envelope('knowledge-status', {
    state: 'unavailable',
    reason: 'The Knowledge Agent (Vault gatekeeper) is a proposed role, not an implemented agent.',
    gate_note: 'Knowledge/Vault data is not available. The gatekeeper agent this screen depends on has not been built yet.',
  }, { clock, freshness: 'unavailable', status: 'unavailable' })
}

export function getSecurity(clock: Clock = defaultClock): Envelope<{ events: SecurityEvent[]; summary: Record<string, number>; period_days: number }> {
  const now = clock()
  const events = SECURITY_EVENTS.map(e => ({ ...e, timestamp: e.timestamp || now }))

  return envelope('security-events', { events, summary: { info: events.length }, period_days: 30 }, { clock })
}

export function getSummary(clock: Clock = defaultClock): Envelope<SummaryData> {
  const now = clock()
  const decisions = ATTENTION_ITEMS.filter(i => i.classification === 'decision').length
  const actions = ATTENTION_ITEMS.filter(i => i.classification === 'action').length
  const information = ATTENTION_ITEMS.filter(i => i.classification === 'information').length
  const liveAgents = AGENTS.filter(a => a.implementation_status === 'live')
  const blockedAgents = liveAgents.filter(a => a.live_status === 'blocked')

  return envelope('summary', {
    runtime: { state: 'unknown', source: SOURCE, note: 'Runtime health is not wired in M1.', observed_at: null },
    attention: {
      decision_count: decisions,
      action_count: actions,
      information_count: information,
      source: SOURCE,
      observed_at: now,
      freshness: 'synthetic',
    },
    objectives: {
      total: OBJECTIVES.length,
      on_track: OBJECTIVES.filter(o => o.status === 'on_track').length,
      at_risk: OBJECTIVES.filter(o => o.status === 'at_risk').length,
      source: SOURCE,
      observed_at: now,
    },
    agents: {
      live: liveAgents.length,
      total_known: AGENTS.length,
      blocked: blockedAgents.length,
      source: SOURCE,
      observed_at: now,
    },
    system_health: { state: 'unknown', note: 'System health is not wired in M1.', source: SOURCE, observed_at: now },
  }, { clock })
}
