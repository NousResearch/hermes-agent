/**
 * Command Center M1 — contract types.
 *
 * These types define the stable versioned contract between the Owner Command
 * Center UI and its data provider. In M1 the only implementation is the
 * in-process synthetic provider (./provider.ts) — there is no backend
 * adapter or server route. When a real, separately-authorized provider
 * replaces it, these types must remain compatible or be versioned. The UI
 * must not need to be redesigned for a provider swap.
 *
 * All data carrying source="synthetic-m1" is SIMULATED test data only.
 */

export type FreshnessState =
  | 'synthetic'
  | 'fresh'
  | 'stale'
  | 'unknown'
  | 'unavailable'

export type AvailabilityState =
  | 'available'
  | 'stale'
  | 'unavailable'
  | 'unknown'
  | 'unresolved'
  | 'conflicted'
  | 'no_access'

export interface Envelope<T> {
  contract_version: string
  projection: string
  projection_version: string
  generated_at: string
  scope_ref: string
  source: string
  freshness: { state: FreshnessState; note?: string }
  status: AvailabilityState | 'unknown'
  data: T
  warnings: string[]
}

// Capabilities
export interface Capabilities {
  adapter_version: string
  features: Record<string, boolean>
  source_health: Array<{ name: string; state: string; note?: string }>
  policy_state: Record<string, string>
}

// Orb state
export interface OrbState {
  state: 'unknown' | 'attention' | 'idle'
  visual_hint: 'neutral' | 'attention' | 'idle'
  reason: string
  aria_label: string
  attention_count: {
    decisions: number
    actions: number
    source: string
    note: string
  }
}

// Agents
export type ImplementationStatus = 'live' | 'proposed'
export type LiveStatus = 'running' | 'idle' | 'blocked' | 'unavailable' | null

export interface AgentObservation {
  id: string
  display_label: string
  role_label: string
  implementation_status: ImplementationStatus
  live_status: LiveStatus
  current_focus: string | null
  last_observed: string | null
}

// Attention items
export type Classification = 'decision' | 'action' | 'information'
export type ConfidenceLevel = 'high' | 'medium' | 'low'

export interface AttentionItem {
  id: string
  classification: Classification
  headline: string
  description: string
  raised_by: { agent_id: string; agent_name: string }
  raised_at: string
  evidence_confidence?: ConfidenceLevel
  recommendation_confidence?: ConfidenceLevel
  primary_action_label: string
}

// Objectives
export interface ObjectiveSummary {
  id: string
  name: string
  status: 'on_track' | 'at_risk' | 'blocked' | 'done'
  progress_pct: number
  note: string
}

// Work summary
export interface WorkData {
  observed_at: string
  source: string
  objectives: ObjectiveSummary[]
  attention_items: AttentionItem[]
}

// Security
export interface SecurityEvent {
  id: string
  what: string
  why_it_matters: string
  what_is_needed: string
  severity: string
  timestamp: string
}

// Knowledge status
export interface KnowledgeStatus {
  state: 'unavailable' | 'available'
  reason: string
  gate_note: string
}

// Summary
export interface SummaryData {
  runtime: { state: string; source: string; note: string; observed_at: string | null }
  attention: {
    decision_count: number
    action_count: number
    information_count: number
    source: string
    observed_at: string
    freshness: string
  }
  objectives: { total: number; on_track: number; at_risk: number; source: string; observed_at: string }
  agents: { live: number; total_known: number; blocked: number; source: string; observed_at: string }
  system_health: { state: string; note: string; source: string; observed_at: string }
}
