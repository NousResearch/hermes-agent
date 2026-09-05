import { useMemo } from 'react'

import { useContributions } from './react/use-contributions'
import type { Contribution, ContributionSource } from './types'

export const CONNECTION_HEALTH_AREA = 'connections.health'

const CONNECTION_HEALTH_REASONS = [
  'healthy',
  'auth_required',
  'service_unreachable',
  'permission_required',
  'not_installed',
  'not_configured',
  'stale',
  'check_failed'
] as const

export type ConnectionHealthReason = (typeof CONNECTION_HEALTH_REASONS)[number]

const CONNECTION_HEALTH_REASON_SET: ReadonlySet<string> = new Set(CONNECTION_HEALTH_REASONS)

export type ConnectionHealthRepair =
  | { kind: 'message'; message: string }
  | { kind: 'route'; path: string }

export interface ConnectionHealthResult {
  id: string
  name: string
  icon?: string
  status?: string
  reason: ConnectionHealthReason
  detail?: string
  checkedAt: number
  staleAfterMs?: number
  repair?: ConnectionHealthRepair
}

export interface ConnectionHealthProvider {
  name?: string
  icon?: string
  repair?: ConnectionHealthRepair
  load: () => Promise<readonly ConnectionHealthResult[]> | readonly ConnectionHealthResult[]
}

export interface RegisteredConnectionHealthProvider extends ConnectionHealthProvider {
  id: string
  source: ContributionSource
}

function isProvider(value: unknown): value is ConnectionHealthProvider {
  return typeof value === 'object' && value !== null && typeof (value as { load?: unknown }).load === 'function'
}

function safeRepair(value: unknown): ConnectionHealthRepair | undefined {
  if (typeof value !== 'object' || value === null) {
    return undefined
  }

  const repair = value as { kind?: unknown; message?: unknown; path?: unknown }

  if (repair.kind === 'message' && typeof repair.message === 'string') {
    return { kind: 'message', message: repair.message }
  }

  if (
    repair.kind === 'route'
    && typeof repair.path === 'string'
    && repair.path.startsWith('/')
    && !repair.path.startsWith('//')
  ) {
    return { kind: 'route', path: repair.path }
  }

  return undefined
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0
}

function isReason(value: unknown): value is ConnectionHealthReason {
  return typeof value === 'string' && CONNECTION_HEALTH_REASON_SET.has(value)
}

function isTimestamp(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0
}

function isPositiveDuration(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0
}

function safeHealthResults(value: unknown): readonly ConnectionHealthResult[] {
  if (!Array.isArray(value)) {
    return []
  }

  return value.flatMap(item => {
    if (typeof item !== 'object' || item === null) {
      return []
    }

    const result = item as Record<string, unknown>

    if (
      !isNonEmptyString(result.id)
      || !isNonEmptyString(result.name)
      || !isReason(result.reason)
      || !isTimestamp(result.checkedAt)
    ) {
      return []
    }

    const repair = safeRepair(result.repair)

    const safeResult: ConnectionHealthResult = {
      checkedAt: result.checkedAt,
      id: result.id,
      name: result.name,
      reason: result.reason,
      ...(typeof result.icon === 'string' ? { icon: result.icon } : {}),
      ...(typeof result.status === 'string' ? { status: result.status } : {}),
      ...(typeof result.detail === 'string' ? { detail: result.detail } : {}),
      ...(isPositiveDuration(result.staleAfterMs) ? { staleAfterMs: result.staleAfterMs } : {}),
      ...(repair ? { repair } : {})
    }

    return [safeResult]
  })
}

export function connectionHealthProviders(
  contributions: readonly Contribution[]
): RegisteredConnectionHealthProvider[] {
  return contributions.flatMap(contribution => {
    if (!isProvider(contribution.data)) {
      return []
    }

    const provider = contribution.data
    const repair = safeRepair(provider.repair)

    return [{
      ...(typeof provider.icon === 'string' ? { icon: provider.icon } : {}),
      id: contribution.id,
      load: async () => safeHealthResults(await provider.load()),
      ...(typeof provider.name === 'string' ? { name: provider.name } : {}),
      ...(repair ? { repair } : {}),
      source: contribution.source ?? 'core'
    }]
  })
}

export function useConnectionHealthProviders(): RegisteredConnectionHealthProvider[] {
  const contributions = useContributions(CONNECTION_HEALTH_AREA)

  return useMemo(() => connectionHealthProviders(contributions), [contributions])
}
