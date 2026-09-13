import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { pauseLocalDownload, resumeLocalDownload } from '@/hermes'
import { useI18n } from '@/i18n'
import { Loader2, Pause, Play } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { watchLocalRuntimeJobs } from '@/store/local-runtime-jobs'
import { notifyError } from '@/store/notifications'
import type { LocalRuntimeJob } from '@/types/hermes'

import { Pill } from './primitives'

interface ProgressBarProps {
  percent: number | undefined
  paused?: boolean
}

interface LocalModelDownloadProps {
  job: LocalRuntimeJob
}

export function ProgressBar({ percent, paused = false }: ProgressBarProps) {
  const unknown = typeof percent !== 'number'

  return (
    <div
      aria-valuemax={100}
      aria-valuemin={0}
      {...(unknown ? {} : { 'aria-valuenow': percent })}
      className="h-1.5 w-full overflow-hidden rounded-full bg-(--ui-bg-tertiary)"
      role="progressbar"
    >
      <div
        className={cn(
          'h-full rounded-full',
          paused ? 'bg-muted-foreground/60' : 'bg-primary transition-[width] duration-300'
        )}
        style={{ width: `${Math.max(0, Math.min(100, percent ?? 0))}%` }}
      />
    </div>
  )
}

export function gbLabel(bytes: number | null | undefined): string {
  if (bytes == null) {
    return '—'
  }

  return `${(bytes / (1 << 30)).toFixed(1)} GB`
}

// Phases where bytes are actually moving (or parked mid-move). Quickstart
// recomputes percent against EACH stage's own download plan — the counter
// resets between stages by design, so it is only shown during a genuine
// download phase. Gate of last resort only: the backend's can_pause flag
// is the primary control gate.
const QUICKSTART_DOWNLOAD_PHASES = new Set([
  'downloading',
  'downloading-runtime',
  'unpacking-runtime',
  'verifying-runtime'
])

export function isDownloadPhase(job: LocalRuntimeJob): boolean {
  if (job.kind === 'model-download' || job.kind === 'runtime-install') {
    return true
  }

  return job.kind === 'quickstart' && QUICKSTART_DOWNLOAD_PHASES.has(job.phase)
}

// Progress bar + honest byte counter. The counter is suppressed outside
// download phases (a stage hand-off would otherwise read as progress loss);
// paused rows keep the frozen counter they parked with.
export function LocalModelDownloadProgress({ job }: LocalModelDownloadProps) {
  const { t } = useI18n()
  const copy = t.settings.localModels
  const showCounter = isDownloadPhase(job)

  return (
    <div className="grid gap-1">
      <ProgressBar paused={job.status === 'paused'} percent={job.percent} />

      <p className="text-[0.68rem] text-muted-foreground">
        {!showCounter || (!job.done_bytes && job.detail)
          ? job.detail
          : copy.downloadProgress(gbLabel(job.done_bytes), gbLabel(job.total_bytes))}
      </p>
    </div>
  )
}

export function LocalModelDownloadActions({ job }: LocalModelDownloadProps) {
  const { t } = useI18n()
  const copy = t.settings.localModels
  const [busy, setBusy] = useState<boolean>(false)

  const send = async (kind: 'pause' | 'resume'): Promise<void> => {
    setBusy(true)

    try {
      // A false paused/resumed flag is usually a benign race (the job
      // settled between render and click) — the authoritative refresh
      // below decides what the row shows; only a real transport failure
      // surfaces as an error.
      if (kind === 'pause') {
        await pauseLocalDownload(job.job_id)
      } else {
        await resumeLocalDownload(job.job_id)
      }

      watchLocalRuntimeJobs()
    } catch (err) {
      notifyError(err, copy.downloadFailed(job.target))
    } finally {
      setBusy(false)
    }
  }

  if (job.status === 'paused') {
    return (
      <div className="flex items-center justify-end gap-2">
        <Pill tone="warn">
          <Pause className="mr-1 size-3" />
          {copy.downloadPausedLabel}
        </Pill>

        {job.can_resume === true && (
          <Button
            className={cn(busy && '[&_svg]:animate-spin')}
            disabled={busy}
            onClick={() => void send('resume')}
            size="sm"
            variant="outline"
          >
            {busy ? <Loader2 /> : <Play />}
            {copy.downloadResumeAction}
          </Button>
        )}
      </div>
    )
  }

  // No gate invented client-side: can_pause is the backend's explicit
  // verdict; pause_requested keeps the control visible but disabled so a
  // pending request never reads as "gone".
  if (job.status !== 'running' || (!job.pause_requested && job.can_pause !== true)) {
    return null
  }

  if (job.pause_requested) {
    return (
      <Button className={cn('[&_svg]:animate-spin')} disabled size="sm" variant="outline">
        <Loader2 />
        {copy.downloadPauseAction}
      </Button>
    )
  }

  return (
    <Button
      className={cn(busy && '[&_svg]:animate-spin')}
      disabled={busy}
      onClick={() => void send('pause')}
      size="sm"
      variant="outline"
    >
      {busy ? <Loader2 /> : <Pause />}
      {copy.downloadPauseAction}
    </Button>
  )
}
