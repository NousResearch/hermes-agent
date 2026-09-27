/**
 * Linux GPU-child software fallback for #124843.
 *
 * On Mesa/Wayland (Intel iGPU, no NVIDIA driver) the Chromium GPU child can
 * fail initialization (`GPU process launch failed: error_code=1002`) and —
 * instead of crashing — retry inside a sub-zygote forever: ~20 spinning
 * threads, repeated `/dev/dri/renderD128` opens, no `--type=gpu-process`,
 * ~350% CPU from launch with no recovery. Chromium never surfaces a fatal,
 * so the app must bound the retry itself: one relaunch into software
 * rendering, then a sticky per-version marker so the next boot goes straight
 * to software. An app update re-probes the GPU once instead of degrading
 * forever.
 *
 * Deliberately reactive, not proactive: a healthy Mesa/Wayland stack (Brave
 * renders the same node at ~1% CPU) keeps full acceleration. The fallback
 * engages only on a witnessed GPU launch failure/crash, two consecutive
 * mid-boot aborts, or a sticky marker from a prior recovery.
 *
 * Skipped when software rendering is already on (remote-display path,
 * NVIDIA SwiftShader path, `HERMES_DESKTOP_DISABLE_GPU=1`, `--disable-gpu`),
 * and never when `HERMES_DESKTOP_DISABLE_GPU=0` keeps the GPU on.
 *
 * Pure + dependency-free so it can be unit-tested and called before app ready.
 * `--disable-gpu` argv helpers are reused from windows-stack-cookie-fallback
 * (one spelling per concept) — they are platform-agnostic despite the name.
 */

import fs from 'node:fs'
import path from 'node:path'

import { alreadyHasDisableGpu, isHermesDesktopGpuOverrideOff } from './windows-stack-cookie-fallback'

export const LINUX_GPU_FALLBACK_MARKER_FILENAME = 'linux-gpu-fallback.json'

/** Consecutive mid-boot aborts required before falling back to software. */
export const BOOT_ABORTS_BEFORE_LINUX_GPU_FALLBACK = 2

/** `child-process-gone` reasons that witness a broken GPU child. */
const GPU_FAILURE_REASONS = new Set(['crashed', 'launch-failure'])

export type LinuxGpuMarkerState = 'booting' | 'fallback' | 'ok'

export type LinuxGpuFallbackReason = 'gpu-launch-failure' | 'gpu-crash' | 'boot-loop'

export interface LinuxGpuMarker {
  state: LinuxGpuMarkerState
  /** Why the fallback engaged (state === 'fallback'). */
  reason?: LinuxGpuFallbackReason
  /** App version that entered fallback — a version change triggers a re-probe. */
  version?: string
  /** Consecutive aborted boots observed so far (state === 'booting'). */
  bootAborts?: number
  /** This boot is a GPU re-probe after an app update; an abort returns
   *  straight to fallback instead of restarting the two-strike count. */
  reprobe?: boolean
}

export function linuxGpuMarkerPath(userDataDir: string): string {
  return path.join(String(userDataDir || ''), LINUX_GPU_FALLBACK_MARKER_FILENAME)
}

const FALLBACK_REASONS: readonly string[] = ['gpu-launch-failure', 'gpu-crash', 'boot-loop']

export function parseLinuxGpuMarker(raw: unknown): LinuxGpuMarker | null {
  if (!raw || typeof raw !== 'object') {
    return null
  }

  const record = raw as Record<string, unknown>
  const state = record.state

  if (state !== 'booting' && state !== 'fallback' && state !== 'ok') {
    return null
  }

  const marker: LinuxGpuMarker = { state }

  if (typeof record.reason === 'string' && FALLBACK_REASONS.includes(record.reason)) {
    marker.reason = record.reason as LinuxGpuFallbackReason
  }

  if (typeof record.version === 'string' && record.version) {
    marker.version = record.version
  }

  const aborts = Number(record.bootAborts)

  if (Number.isInteger(aborts) && aborts > 0) {
    marker.bootAborts = aborts
  }

  if (record.reprobe === true) {
    marker.reprobe = true
  }

  return marker
}

export function readLinuxGpuMarker(
  userDataDir: string,
  { readFileSync = fs.readFileSync } = {}
): LinuxGpuMarker | null {
  try {
    const raw = JSON.parse(readFileSync(linuxGpuMarkerPath(userDataDir), 'utf8'))

    return parseLinuxGpuMarker(raw)
  } catch {
    return null
  }
}

export function writeLinuxGpuMarker(
  userDataDir: string,
  marker: LinuxGpuMarker,
  {
    mkdirSync = fs.mkdirSync,
    writeFileSync = fs.writeFileSync
  }: {
    mkdirSync?: typeof fs.mkdirSync
    writeFileSync?: typeof fs.writeFileSync
  } = {}
): void {
  const dir = String(userDataDir || '')

  if (!dir) {
    return
  }

  mkdirSync(dir, { recursive: true })
  writeFileSync(linuxGpuMarkerPath(dir), `${JSON.stringify(marker)}\n`, 'utf8')
}

export function linuxGpuFallbackMarker(reason: LinuxGpuFallbackReason, appVersion?: string): LinuxGpuMarker {
  const marker: LinuxGpuMarker = { state: 'fallback', reason }

  if (appVersion) {
    marker.version = appVersion
  }

  return marker
}

/**
 * After the main window is revealed: keep the sticky fallback when we
 * launched with software rendering, otherwise mark a clean boot so future
 * launches trust the GPU again.
 */
export function linuxGpuMarkerAfterSuccessfulBoot(options: {
  fallbackActive: boolean
  appVersion?: string
}): LinuxGpuMarker {
  if (!options.fallbackActive) {
    return { state: 'ok' }
  }

  return linuxGpuFallbackMarker('gpu-launch-failure', options.appVersion)
}

export interface LinuxGpuLaunchDecision {
  enable: boolean
  reason: string | null
  /** Marker to persist immediately, before GPU children start. */
  nextMarker: LinuxGpuMarker
}

/**
 * Single launch-time transition: decide whether this Linux launch disables
 * GPU hardware acceleration AND what the marker becomes for crash-detection
 * on the next launch. Same lifecycle as the Windows sandbox fallback:
 * sticky per app version, two-strike boot-loop detection, one post-update
 * re-probe.
 */
export function decideLinuxGpuLaunch(
  options: {
    platform?: NodeJS.Platform | string
    argv?: readonly string[]
    env?: NodeJS.ProcessEnv
    marker?: LinuxGpuMarker | null
    appVersion?: string
    remoteDisplayReason?: string | null
    nvidiaFallbackActive?: boolean
  } = {}
): LinuxGpuLaunchDecision {
  const appVersion = String(options.appVersion || '')

  if ((options.platform ?? process.platform) !== 'linux') {
    return { enable: false, reason: null, nextMarker: { state: 'booting' } }
  }

  const argv = options.argv ?? process.argv
  const env = options.env ?? process.env
  const marker = options.marker ?? null

  // A user who forced GPU back on owns that call — even over a sticky marker. That boot is
  // the recovery path this file documents, so it must not leave the stale fallback marker
  // behind: keeping it makes `main.ts` sticky again, the next override-free launch re-engages
  // software rendering (`sticky-fallback`) and only an app-version change would clear it.
  if (isHermesDesktopGpuOverrideOff(env)) {
    return { enable: false, reason: null, nextMarker: { state: 'booting' } }
  }

  if (alreadyHasDisableGpu(argv, env)) {
    // Honor the explicit flag; keep the marker lifecycle unchanged. When the
    // relaunch path set the flag, the fallback marker it wrote is preserved.
    const nextMarker: LinuxGpuMarker = marker?.state === 'fallback' ? marker : { state: 'booting' }

    return { enable: true, reason: 'already-enabled', nextMarker }
  }

  // The remote-display and NVIDIA paths already force software rendering;
  // don't pile a second fallback on top of theirs.
  if (options.remoteDisplayReason || options.nvidiaFallbackActive) {
    return { enable: false, reason: null, nextMarker: { state: 'booting' } }
  }

  if (marker?.state === 'fallback') {
    if (marker.version && appVersion && marker.version !== appVersion) {
      // App updated since the fallback engaged — re-probe the GPU once.
      return {
        enable: false,
        reason: null,
        nextMarker: { state: 'booting', reprobe: true, bootAborts: 0 }
      }
    }

    return {
      enable: true,
      reason: `sticky-fallback (${marker.reason ?? 'gpu-launch-failure'})`,
      nextMarker: { ...marker, version: marker.version || appVersion || undefined }
    }
  }

  if (marker?.state === 'booting') {
    const abortsObserved = (marker.bootAborts ?? 0) + 1

    if (marker.reprobe) {
      // The one post-update GPU re-probe aborted → back to fallback.
      return {
        enable: true,
        reason: 'reprobe-failed (boot-loop)',
        nextMarker: linuxGpuFallbackMarker('boot-loop', appVersion)
      }
    }

    if (abortsObserved >= BOOT_ABORTS_BEFORE_LINUX_GPU_FALLBACK) {
      return {
        enable: true,
        reason: 'boot-loop',
        nextMarker: linuxGpuFallbackMarker('boot-loop', appVersion)
      }
    }

    return {
      enable: false,
      reason: null,
      nextMarker: { state: 'booting', bootAborts: abortsObserved }
    }
  }

  // No marker, or a clean `ok` from the previous run.
  return { enable: false, reason: null, nextMarker: { state: 'booting' } }
}

/**
 * True when a Linux GPU child died with launch-failure/crash evidence and we
 * should one-shot relaunch with `--disable-gpu` before Chromium's retry loop
 * burns the machine (#124843). Bounded: once per process (caller tracks
 * `relaunchAttempted`), sticky across boots via the marker the caller writes.
 */
export function shouldRelaunchForLinuxGpuCrash(options: {
  platform?: NodeJS.Platform | string
  details?: { type?: string; reason?: string } | null
  alreadySoftware?: boolean
  relaunchAttempted?: boolean
}): boolean {
  if ((options.platform ?? process.platform) !== 'linux') {
    return false
  }

  if (options.alreadySoftware || options.relaunchAttempted) {
    return false
  }

  const type = String(options.details?.type || '').toLowerCase()

  if (type !== 'gpu') {
    return false
  }

  return GPU_FAILURE_REASONS.has(String(options.details?.reason || '').toLowerCase())
}
