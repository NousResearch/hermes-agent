import { capabilityScoped, type ProfileScope } from './client'

export type TrustLevel = 'ask' | 'auto' | 'propose' | 'read'

export const TRUST_LEVELS: readonly TrustLevel[] = ['read', 'propose', 'ask', 'auto']

export interface TrustLogEntry {
  action: string
  decision: 'approved' | 'auto' | 'blocked' | 'denied'
  integration: string
  label: string
  label_pl: string
  level: TrustLevel
  preview: string
  ts: number
}

export interface TrustState {
  levels: Record<string, TrustLevel>
  log: TrustLogEntry[]
}

const call = (profile: ProfileScope | undefined, path: string, body?: unknown) =>
  window.hermesDesktop.api<TrustState>({ ...capabilityScoped(profile), body, method: body === undefined ? undefined : 'POST', path })

export const getTrust = (profile?: ProfileScope) => call(profile, '/api/trust')

export const setTrustLevel = (integration: string, level: TrustLevel, profile?: ProfileScope) =>
  call(profile, '/api/trust/level', { integration, level })
