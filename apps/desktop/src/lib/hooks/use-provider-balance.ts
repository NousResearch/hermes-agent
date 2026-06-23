import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef } from 'react'

import {
  $providerBalance,
  setProviderBalance,
  setProviderBalanceError,
  setProviderBalanceLoading,
} from '@/store/provider-balance'
import type { BalanceViewResponse, ProviderBalance } from '@/types/hermes'

const POLL_MS = 120_000  // 2 minutes

type GatewayRequester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

export function useProviderBalance(
  requestGateway: GatewayRequester,
  gatewayState: string | undefined,
): { balance: ProviderBalance | null; fetchBalance: (force?: boolean) => Promise<void> } {
  const state = useStore($providerBalance)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchBalance = useCallback(
    async (force = false) => {
      setProviderBalanceLoading(true)
      try {
        const res = await requestGateway<BalanceViewResponse>('balance.view', { force })
        if (res.ok && res.balance) {
          setProviderBalance(res.balance)
        } else {
          setProviderBalanceError(res.error ?? 'unknown error')
        }
      } catch (err) {
        setProviderBalanceError(err instanceof Error ? err.message : String(err))
      }
    },
    [requestGateway],
  )

  // Fetch on mount and when gateway opens.
  useEffect(() => {
    if (gatewayState === 'open') {
      void fetchBalance()
    }
  }, [gatewayState, fetchBalance])

  // Poll while gateway is open.
  useEffect(() => {
    if (gatewayState !== 'open') {
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
      return
    }

    pollRef.current = setInterval(() => void fetchBalance(), POLL_MS)

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }
  }, [gatewayState, fetchBalance])

  return { balance: state.balance, fetchBalance }
}