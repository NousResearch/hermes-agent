import { useStore } from '@nanostores/react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useMemo } from 'react'

import { useI18n } from '@/i18n'
import { requestGatewayForProfile } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { $gatewayState } from '@/store/session'

export type VaultSourceName = 'bitwarden' | 'local' | 'onepassword'

/** One login source as reported by `vault.sources` — the backend is authoritative for enabled/unlocked. */
export interface VaultSource {
  name: VaultSourceName
  display_name: string
  enabled: boolean
  needs_unlock: boolean
  unlocked: boolean
  installed: boolean
}

export interface VaultItem {
  id: string
  kind: string
  label: string
  origin: null | string
  created_at: string
  identifier?: null | string
  identifier_type?: null | string
  backend?: VaultSourceName
  has_otp?: boolean
}

export function useVaultData(owner: string, scopeProfile: string) {
  const { t } = useI18n()
  const v = t.settings.vault
  const gatewayState = useStore($gatewayState)
  const queryClient = useQueryClient()

  const requestGateway = useCallback(
    <T>(method: string, params: Record<string, unknown> = {}) =>
      requestGatewayForProfile<T>(scopeProfile, method, params),
    [scopeProfile]
  )

  const VAULT_QUERY_KEY = useMemo(() => ['vault-items', owner] as const, [owner])
  const VAULT_SOURCES_QUERY_KEY = useMemo(() => ['vault-sources', owner] as const, [owner])

  const { data: sourcesData } = useQuery({
    enabled: gatewayState === 'open',
    // Reconnecting revokes Settings unlocks even when the connection/profile owner is unchanged.
    staleTime: 0,
    queryKey: VAULT_SOURCES_QUERY_KEY,
    queryFn: async () => {
      const result = await requestGateway<{ sources: VaultSource[] }>('vault.sources', {})

      return result.sources
    }
  })

  const externalSources = useMemo(() => (sourcesData ?? []).filter(s => s.needs_unlock), [sourcesData])

  const invalidateVault = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: VAULT_QUERY_KEY })
    void queryClient.invalidateQueries({ queryKey: VAULT_SOURCES_QUERY_KEY })
  }, [queryClient, VAULT_QUERY_KEY, VAULT_SOURCES_QUERY_KEY])

  const { data, error, isPending } = useQuery({
    enabled: gatewayState === 'open',
    staleTime: 0,
    queryKey: VAULT_QUERY_KEY,
    queryFn: async () => {
      const result = await requestGateway<{ items: VaultItem[] }>('vault.list', {})

      return result.items
    }
  })

  useEffect(() => {
    if (error) {
      notifyError(error, v.loadFailed)
    }
  }, [error, v.loadFailed])

  const items = useMemo(() => data ?? [], [data])

  const invalidate = useCallback(
    () => queryClient.invalidateQueries({ queryKey: VAULT_QUERY_KEY }),
    [queryClient, VAULT_QUERY_KEY]
  )

  return { requestGateway, externalSources, items, isPending, invalidate, invalidateVault }
}

export type VaultData = ReturnType<typeof useVaultData>
