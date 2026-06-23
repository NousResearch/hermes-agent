import { atom } from 'nanostores'

import type { ProviderBalance } from '@/types/hermes'

export interface ProviderBalanceState {
  balance: ProviderBalance | null
  loading: boolean
  error: string | null
  fetchedAtMs: number | null
}

export const $providerBalance = atom<ProviderBalanceState>({
  balance: null,
  loading: false,
  error: null,
  fetchedAtMs: null,
})

export function setProviderBalance(balance: ProviderBalance): void {
  $providerBalance.set({
    balance,
    loading: false,
    error: null,
    fetchedAtMs: Date.now(),
  })
}

export function setProviderBalanceError(error: string): void {
  const current = $providerBalance.get()
  $providerBalance.set({
    ...current,
    loading: false,
    error,
    fetchedAtMs: Date.now(),
  })
}

export function setProviderBalanceLoading(loading: boolean): void {
  $providerBalance.set({
    ...$providerBalance.get(),
    loading,
  })
}