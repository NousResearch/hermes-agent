import { atom } from 'nanostores'

import { getCronJobs } from '@/hermes'
import type { CronJob } from '@/types/hermes'

// Cron jobs surfaced in the sidebar "Cron jobs" section. The cron page owns
// its own richer fetch/refresh cycle; this store is a lightweight projection
// refreshed alongside the session list so the sidebar can render without
// mounting the cron view.
export const $cronJobs = atom<CronJob[]>([])

export const setCronJobs = (jobs: CronJob[]) => $cronJobs.set(jobs)

export async function refreshCronJobs(): Promise<void> {
  try {
    $cronJobs.set(await getCronJobs())
  } catch {
    // Gateway not ready or RPC unavailable — keep the last known list; the
    // sidebar section simply stays hidden until a refresh succeeds.
  }
}
