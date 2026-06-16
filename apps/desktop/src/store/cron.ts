import { atom } from 'nanostores'

import { getCronJobs } from '@/hermes'
import type { CronJob } from '@/types/hermes'

// Cron jobs surfaced in the sidebar "Cron jobs" section. The cron page owns
// its own richer fetch/refresh cycle; this store is a lightweight projection
// refreshed alongside the session list so the sidebar can render without
// mounting the cron view.
export const $cronJobs = atom<CronJob[]>([])

export const setCronJobs = (jobs: CronJob[]) => $cronJobs.set(jobs)

// In-place edit so cron overlay mutations land in the same projection the
// sidebar renders, without waiting for the next gateway poll.
export const updateCronJobs = (fn: (jobs: CronJob[]) => CronJob[]) => $cronJobs.set(fn($cronJobs.get()))

// One-shot focus target: clicking Manage on a cron job opens the overlay,
// selects that job, then clears the target after consumption.
export const $cronFocusJobId = atom<null | string>(null)
export const setCronFocusJobId = (id: null | string) => $cronFocusJobId.set(id)

export async function refreshCronJobs(): Promise<void> {
  try {
    $cronJobs.set(await getCronJobs())
  } catch {
    // Gateway not ready or RPC unavailable — keep the last known list; the
    // sidebar section simply stays hidden until a refresh succeeds.
  }
}
