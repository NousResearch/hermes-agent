/**
 * Command Center M1 — data access layer.
 *
 * Per Architect's Path B correction (2026-08-31): this file makes NO network calls.
 * There is no `ctx.rest`, no `fetch`, no `XMLHttpRequest`, and no backend adapter mounted
 * anywhere in M1 — `web_server._discover_dashboard_plugins()` only mounts a plugin's
 * `dashboard/manifest.json` API file, which this plugin does not have and is
 * not authorized to add in M1. Every function below reads from the in-process, deterministic
 * synthetic provider in ./provider.ts and wraps its (already synchronous) result in a
 * resolved Promise purely to keep the same async call shape react-query expects — no I/O
 * of any kind happens.
 *
 * Query keys are exported so screens can share cache entries.
 */

import * as provider from './provider'
import type {
  AgentObservation,
  Envelope,
  KnowledgeStatus,
  OrbState,
  SecurityEvent,
  SummaryData,
  WorkData,
} from './types'
import type { Capabilities } from './types'

// ---------------------------------------------------------------------------
// Query keys
// ---------------------------------------------------------------------------
export const ccKeys = {
  capabilities: ['cc', 'capabilities'] as const,
  summary: ['cc', 'summary'] as const,
  orbState: ['cc', 'orb-state'] as const,
  agents: ['cc', 'agents'] as const,
  work: ['cc', 'work'] as const,
  knowledge: ['cc', 'knowledge'] as const,
  security: ['cc', 'security'] as const,
}

// ---------------------------------------------------------------------------
// Fetch helpers — in-process only. No network seam.
// ---------------------------------------------------------------------------
export async function fetchCapabilities(): Promise<Envelope<Capabilities>> {
  return provider.getCapabilities()
}

export async function fetchSummary(): Promise<Envelope<SummaryData>> {
  return provider.getSummary()
}

export async function fetchOrbState(): Promise<Envelope<OrbState>> {
  return provider.getOrbState()
}

export async function fetchAgents(): Promise<Envelope<{ agents: AgentObservation[]; observed_at: string; source: string }>> {
  return provider.getAgents()
}

export async function fetchWork(): Promise<Envelope<WorkData>> {
  return provider.getWork()
}

export async function fetchKnowledge(): Promise<Envelope<KnowledgeStatus>> {
  return provider.getKnowledge()
}

export async function fetchSecurity(): Promise<Envelope<{ events: SecurityEvent[]; summary: Record<string, number>; period_days: number }>> {
  return provider.getSecurity()
}

// Polling intervals — retained for shape compatibility with a future real provider.
// Since M1 data is process-local and static, these intervals are inert (no new data
// arrives on refetch) but keep the screens' useQuery calls unchanged for the eventual
// real-provider swap.
export const POLL_SUMMARY_MS = 30_000
export const POLL_AGENTS_MS = 60_000
export const POLL_WORK_MS = 60_000
export const POLL_ORB_MS = 30_000
export const POLL_SECURITY_MS = 120_000
