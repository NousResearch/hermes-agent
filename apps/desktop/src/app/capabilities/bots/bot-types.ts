import type {
  BotCatalogEntryResult,
  BotRoutineListItem,
  BotsInstallResult,
  BotsStatusResult,
  ProfileRow
} from '@hermes/shared'

import type { ProfileScope } from '@/hermes'

export type MarketplaceRequest = <T>(method: string, params: Record<string, unknown>, scope: ProfileScope) => Promise<T>

export type MarketplaceNotify = (notice: {
  kind: 'error' | 'info' | 'success' | 'warning'
  title?: string
  message: string
}) => unknown

export interface InstalledBot {
  entry: BotCatalogEntryResult
  profile: ProfileRow
  scope: ProfileScope
  status: BotsStatusResult
  routines: BotRoutineListItem[]
}

export interface BotInstallReceipt extends BotsInstallResult {}

export function scopeProfile(scope: ProfileScope): string {
  if (scope && typeof scope === 'object') {
    return scope.profile?.trim() || 'default'
  }

  return scope?.trim() || 'default'
}

export function targetBotScope(scope: ProfileScope, profile: string): ProfileScope {
  if (scope && typeof scope === 'object') {
    return { connectionId: scope.connectionId, profile }
  }

  return profile
}

export function supportsLocalModelSetup(scope: ProfileScope): boolean {
  return !(scope && typeof scope === 'object' && scope.connectionId && scope.connectionId !== 'local')
}
