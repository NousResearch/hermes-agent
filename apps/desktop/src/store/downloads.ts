import { atom } from 'nanostores'

import type { HermesDownloadProgress } from '@/global'

/**
 * Live download rows for the tray.
 *
 * Fed by the main process's `hermes:download:progress` events — the renderer
 * never polls, and a row is identified by the task id the manager assigned.
 * Rows stay after completion so the tray can report "done, open folder"; the
 * user dismisses them (or `dismissFinished` sweeps them on app start).
 */
export const $downloads = atom<HermesDownloadProgress[]>([])

/** Statuses where the transfer is still doing work. */
const ACTIVE = new Set<HermesDownloadProgress['status']>(['queued', 'probing', 'downloading', 'pausing'])

export function isDownloadActive(progress: HermesDownloadProgress) {
  return ACTIVE.has(progress.status)
}

/** Upsert one row, newest first, keeping the list stable for untouched rows. */
export function upsertDownload(rows: HermesDownloadProgress[], progress: HermesDownloadProgress) {
  const index = rows.findIndex(row => row.id === progress.id)

  if (index === -1) {
    return [progress, ...rows]
  }

  const next = rows.slice()

  next[index] = progress

  return next
}

let watching = false

/**
 * Subscribe to main-process progress once per renderer. Older shells have no
 * `downloads` bridge — the tray simply stays empty rather than erroring.
 */
export function ensureDownloadWatch() {
  if (watching) {
    return
  }

  const bridge = window.hermesDesktop?.downloads

  if (!bridge) {
    return
  }

  watching = true
  void bridge
    .list()
    .then(rows => {
      if (rows.length > 0) {
        $downloads.set(rows)
      }
    })
    .catch(() => undefined)
  bridge.onProgress(progress => {
    $downloads.set(upsertDownload($downloads.get(), progress))
  })
}

export async function startDownload(payload: {
  url: string
  filename?: string
  savePath?: string
  connections?: number
  headers?: Record<string, string>
}) {
  const bridge = window.hermesDesktop?.downloads

  if (!bridge) {
    throw new Error('Downloads are unavailable in this build')
  }

  ensureDownloadWatch()
  const progress = await bridge.start(payload)

  $downloads.set(upsertDownload($downloads.get(), progress))

  return progress
}

export async function pauseDownload(id: string) {
  const progress = await window.hermesDesktop?.downloads?.pause(id)

  if (progress) {
    $downloads.set(upsertDownload($downloads.get(), progress))
  }
}

export async function resumeDownload(id: string) {
  const progress = await window.hermesDesktop?.downloads?.resume(id)

  if (progress) {
    $downloads.set(upsertDownload($downloads.get(), progress))
  }
}

export async function cancelDownload(id: string) {
  const progress = await window.hermesDesktop?.downloads?.cancel(id)

  if (progress) {
    $downloads.set(upsertDownload($downloads.get(), progress))
  }
}

export async function dismissDownload(id: string) {
  await window.hermesDesktop?.downloads?.dismiss(id)
  $downloads.set($downloads.get().filter(row => row.id !== id))
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return '0 B'
  }

  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const exponent = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)))
  const value = bytes / 1024 ** exponent

  return `${value >= 100 || exponent === 0 ? Math.round(value) : value.toFixed(1)} ${units[exponent]}`
}

export function formatSpeed(bytesPerSecond: number): string {
  return bytesPerSecond > 0 ? `${formatBytes(bytesPerSecond)}/s` : '—'
}

export function formatEta(etaMs: number | null): string {
  if (etaMs === null || !Number.isFinite(etaMs) || etaMs <= 0) {
    return '—'
  }

  const seconds = Math.round(etaMs / 1000)

  if (seconds < 60) {
    return `${seconds}s`
  }

  const minutes = Math.floor(seconds / 60)

  if (minutes < 60) {
    return `${minutes}m ${seconds % 60}s`
  }

  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`
}
