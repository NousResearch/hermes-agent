import { atom } from 'nanostores'

import type { DesktopBootProgress } from '@/global'
import { translateNow } from '@/i18n'

export type DesktopBootFailureKind = 'connection-lost' | 'startup'

export interface DesktopBootState extends DesktopBootProgress {
  failureKind?: DesktopBootFailureKind | null
  visible: boolean
}

const INITIAL_BOOT_STATE: DesktopBootState = {
  error: null,
  failureKind: null,
  fakeMode: false,
  message: translateNow('boot.steps.startingHermesDesktop'),
  phase: 'renderer.init',
  progress: 2,
  running: true,
  timestamp: Date.now(),
  visible: true
}

export const $desktopBoot = atom<DesktopBootState>(INITIAL_BOOT_STATE)

function clampProgress(value: number) {
  if (!Number.isFinite(value)) {
    return 0
  }

  return Math.max(0, Math.min(100, Math.round(value)))
}

export function applyDesktopBootProgress(progress: DesktopBootProgress & { failureKind?: DesktopBootFailureKind | null }) {
  const current = $desktopBoot.get()
  const nextProgress = clampProgress(progress.progress)
  const mergedProgress = progress.running ? Math.max(current.progress, nextProgress) : nextProgress

  $desktopBoot.set({
    ...current,
    ...progress,
    error: progress.error ?? null,
    failureKind: progress.error ? (progress.failureKind ?? 'startup') : null,
    progress: mergedProgress,
    visible: progress.running || mergedProgress < 100 || Boolean(progress.error)
  })
}

export function setDesktopBootStep(step: {
  phase: string
  message: string
  progress: number
  running?: boolean
  fakeMode?: boolean
  error?: string | null
  failureKind?: DesktopBootFailureKind | null
}) {
  const current = $desktopBoot.get()
  applyDesktopBootProgress({
    error: step.error ?? null,
    fakeMode: step.fakeMode ?? current.fakeMode,
    failureKind: step.error ? (step.failureKind ?? 'startup') : null,
    message: step.message,
    phase: step.phase,
    progress: step.progress,
    running: step.running ?? true,
    timestamp: Date.now()
  })
}

export function completeDesktopBoot(message = translateNow('boot.ready')) {
  const current = $desktopBoot.get()
  $desktopBoot.set({
    ...current,
    error: null,
    failureKind: null,
    message,
    phase: 'renderer.ready',
    progress: 100,
    running: false,
    timestamp: Date.now(),
    visible: false
  })
}

export function failDesktopBoot(message: string, failureKind: DesktopBootFailureKind = 'startup') {
  const current = $desktopBoot.get()
  $desktopBoot.set({
    ...current,
    error: message,
    failureKind,
    message: translateNow('boot.desktopBootFailedWithMessage', message),
    phase: 'renderer.error',
    progress: clampProgress(current.progress),
    running: false,
    timestamp: Date.now(),
    visible: true
  })
}
