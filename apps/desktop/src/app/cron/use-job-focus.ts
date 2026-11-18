import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { asText } from '@/lib/text'
import { $cronFocusJobId, setCronFocusJobId } from '@/store/cron'
import type { CronJob } from '@/types/cron'

/** Sidebar → "open this job": resolve the one-shot focus id (or name) to a job, hand it to `open`,
 *  then clear the focus so reopening the page does not re-trigger it. Before `settled` (the jobs have
 *  loaded) an unmatched focus waits for them; after, it is cleared, so it never hijacks a later
 *  selection. */
export function useJobFocus(jobs: readonly CronJob[], settled: boolean, open: (job: CronJob) => void) {
  const focusJobId = useStore($cronFocusJobId)

  useEffect(() => {
    if (!focusJobId) {
      return
    }

    const match = jobs.find(job => job.id === focusJobId || asText(job.name).trim() === focusJobId)

    if (match) {
      open(match)
    }

    if (match || settled) {
      setCronFocusJobId(null)
    }
  }, [focusJobId, jobs, open, settled])
}
