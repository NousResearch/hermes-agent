import { type ReactElement, useState } from 'react'

import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { cancelLocalDownload, pauseLocalDownload, resumeLocalDownload } from '@/hermes'
import { useI18n } from '@/i18n'
import { Loader2, Pause, Play, X } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  isCurrentLocalModelsOwner,
  type LocalModelsOwner,
  localModelsRequestScope,
  watchLocalRuntimeJobs
} from '@/store/local-runtime-jobs'
import { notifyError, readableError } from '@/store/notifications'
import type { LocalRuntimeJob } from '@/types/hermes'

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

// The status line's copy, as the pieces it composes — a locale changes the
// words around each number without the caller knowing the grammar.
export interface DownloadStatusCopy {
  downloadPausedLabel: string
  downloadStatusRunning: string
  downloadProgress: (done: string, total: string) => string
  downloadSpeed: (rate: string) => string
  downloadEta: (time: string) => string
  downloadEtaSeconds: (count: number) => string
  downloadEtaMinutes: (count: number) => string
  downloadEtaHours: (hours: number, minutes: number) => string
}

export type DownloadEtaCopy = Pick<DownloadStatusCopy, 'downloadEtaHours' | 'downloadEtaMinutes' | 'downloadEtaSeconds'>

export function formatSpeed(bytesPerSec: number | null | undefined): string {
  if (!bytesPerSec || bytesPerSec <= 0 || !Number.isFinite(bytesPerSec)) {
    return ''
  }

  const mb = bytesPerSec / (1 << 20)

  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB/s` : mb >= 10 ? `${Math.round(mb)} MB/s` : `${mb.toFixed(1)} MB/s`
}

// Byte-rate units (MB/s, GB/s) are SI symbols and stay as-is; the duration
// words are the locale's, so the ETA only picks the magnitude here.
export function formatEta(seconds: number | null | undefined, copy: DownloadEtaCopy): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 1) {
    return ''
  }

  if (seconds < 60) {
    return copy.downloadEtaSeconds(Math.max(1, Math.round(seconds)))
  }

  const minutes: number = Math.round(seconds / 60)

  if (minutes < 60) {
    return copy.downloadEtaMinutes(minutes)
  }

  return copy.downloadEtaHours(Math.floor(minutes / 60), minutes % 60)
}

// One status line for every download surface: what it is doing, how far, how
// fast, how long left — each part dropped rather than guessed when it isn't
// honestly known yet. Shared by the model rows, the setup hero and the
// browser tiles so the three can't drift apart.
export function downloadStatusText(job: LocalRuntimeJob, copy: DownloadStatusCopy): string {
  if (!isDownloadPhase(job)) {
    return job.detail
  }

  const parts: string[] = [job.status === 'paused' ? copy.downloadPausedLabel : copy.downloadStatusRunning]

  const size = job.total_bytes
    ? copy.downloadProgress(gbLabel(job.done_bytes), gbLabel(job.total_bytes))
    : job.done_bytes
      ? gbLabel(job.done_bytes)
      : ''

  if (size) {
    parts.push(size)
  }

  // Speed and ETA only while bytes move: a parked transfer keeps its frozen
  // counter but has no live rate, so the backend sends none and we render none.
  if (job.status === 'running') {
    const speed = formatSpeed(job.bytes_per_sec)

    if (speed) {
      parts.push(copy.downloadSpeed(speed))

      const eta: string = formatEta(job.eta_seconds, copy)

      if (eta) {
        parts.push(copy.downloadEta(eta))
      }
    }
  }

  return parts.length > 1 ? parts.join(' · ') : job.detail || parts[0]
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

// Progress bar + honest status line. The line is the shared composer, so the
// state word, byte counter, speed and ETA read the same here as on every
// other download surface.
export function LocalModelDownloadProgress({ job }: LocalModelDownloadProps) {
  const { t } = useI18n()
  const copy = t.settings.localModels

  return (
    <div className="grid gap-1">
      <ProgressBar paused={job.status === 'paused'} percent={job.percent} />

      <p className="text-[0.68rem] text-muted-foreground">{downloadStatusText(job, copy)}</p>
    </div>
  )
}

interface DownloadControlProps extends LocalModelDownloadProps {
  owner?: LocalModelsOwner
}

// Pause / resume / cancel for one download job. Every control is the
// backend's explicit verdict (can_pause, can_resume, can_cancel) — nothing is
// gated client-side, so an older backend that never sends a flag shows no
// control for it rather than a button that 404s.
//
// One layout for every state: [Pause ⇄ Resume] [Cancel]. Pausing flips the
// first button in place instead of swapping in a different row — the parked
// state is already labelled by the shared status line and the grey bar — so
// the controls never grow past a narrow browse tile, and the row wraps inside
// its container rather than spill out when a locale's labels run long.
export function LocalModelDownloadActions({ job, owner }: DownloadControlProps): ReactElement | null {
  const paused = job.status === 'paused'

  const showPauseResume = paused ? job.can_resume === true : job.pause_requested === true || job.can_pause === true
  const showCancel = job.cancel_requested === true || job.can_cancel === true

  if ((!paused && job.status !== 'running') || (!showPauseResume && !showCancel)) {
    return null
  }

  return (
    <div className="flex min-w-0 max-w-full flex-wrap items-center justify-end gap-2">
      {showPauseResume && <PauseResumeButton job={job} owner={owner} />}

      {showCancel && <CancelDownloadButton job={job} owner={owner} />}
    </div>
  )
}

function PauseResumeButton({ job, owner }: DownloadControlProps): ReactElement {
  const { t } = useI18n()
  const copy = t.settings.localModels
  const [busy, setBusy] = useState<boolean>(false)
  const kind: 'pause' | 'resume' = job.status === 'paused' ? 'resume' : 'pause'

  const send = async (): Promise<void> => {
    setBusy(true)

    try {
      // A false paused/resumed flag is usually a benign race (the job
      // settled between render and click) — the authoritative refresh
      // below decides what the row shows; only a real transport failure
      // surfaces as an error.
      if (kind === 'pause') {
        await pauseLocalDownload(job.job_id, owner ? localModelsRequestScope(owner) : undefined)
      } else {
        await resumeLocalDownload(job.job_id, owner ? localModelsRequestScope(owner) : undefined)
      }

      watchLocalRuntimeJobs(owner)
    } catch (err) {
      if (!owner || isCurrentLocalModelsOwner(owner)) {
        notifyError(
          err,
          kind === 'pause' ? copy.downloadPauseFailed(job.target) : copy.downloadResumeFailed(job.target)
        )
      }
    } finally {
      setBusy(false)
    }
  }

  // pause_requested keeps the control visible but disabled so a pending
  // request never reads as "gone".
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
      onClick={() => void send()}
      size="sm"
      variant="outline"
    >
      {busy ? <Loader2 /> : kind === 'pause' ? <Pause /> : <Play />}
      {kind === 'pause' ? copy.downloadPauseAction : copy.downloadResumeAction}
    </Button>
  )
}

// Cancel discards progress that can be tens of gigabytes, so it always asks
// first. A failure keeps the dialog open with the reason inline; a
// cancelled:false answer is the benign race where the job settled first —
// the refresh shows whatever it became.
function CancelDownloadButton({ job, owner }: DownloadControlProps): ReactElement {
  const { t } = useI18n()
  const copy = t.settings.localModels
  const [confirming, setConfirming] = useState<boolean>(false)

  const cancel = async (): Promise<void> => {
    try {
      await cancelLocalDownload(job.job_id, owner ? localModelsRequestScope(owner) : undefined)
    } catch (err) {
      throw new Error(readableError(err, copy.downloadCancelFailed(job.target)).message)
    }

    watchLocalRuntimeJobs(owner)
  }

  if (job.cancel_requested) {
    return (
      <Button className={cn('[&_svg]:animate-spin')} disabled size="sm" variant="ghost">
        <Loader2 />
        {copy.downloadCancelAction}
      </Button>
    )
  }

  return (
    <>
      <Button onClick={() => setConfirming(true)} size="sm" variant="ghost">
        <X />
        {copy.downloadCancelAction}
      </Button>

      <ConfirmDialog
        cancelLabel={copy.downloadCancelKeep}
        confirmLabel={copy.downloadCancelConfirm}
        description={copy.downloadCancelBody(job.target)}
        destructive
        onClose={() => setConfirming(false)}
        onConfirm={cancel}
        open={confirming}
        title={copy.downloadCancelTitle}
      />
    </>
  )
}
