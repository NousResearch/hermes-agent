import { atom } from 'nanostores'

import { getUsageMonth, type ProfileScope } from '@/hermes'
import type { UsageMonthResponse } from '@/types/hermes'

/** This calendar month's usage per billing provider, with each budget evaluated (the backend owns it). */
export const $usageMonth = atom<null | UsageMonthResponse>(null)

/** `startedAt` (epoch ms) lets a loading view count elapsed seconds. */
export const $usageMonthState = atom<{ error: string; loading: boolean; startedAt: number }>({
  error: '',
  loading: false,
  startedAt: 0
})

let latestRequest = 0

/** Reload this month's usage. A response older than the newest request is dropped, and a failure
 *  keeps the last good month on screen beside the error. */
export async function refreshUsageMonth(profile?: ProfileScope): Promise<void> {
  const request = ++latestRequest

  $usageMonthState.set({ error: '', loading: true, startedAt: Date.now() })

  try {
    const month = await getUsageMonth(profile)

    if (request === latestRequest) {
      $usageMonth.set(month)
      $usageMonthState.set({ error: '', loading: false, startedAt: 0 })
    }
  } catch (error) {
    if (request === latestRequest) {
      $usageMonthState.set({
        error: error instanceof Error ? error.message : String(error),
        loading: false,
        startedAt: 0
      })
    }
  }
}
