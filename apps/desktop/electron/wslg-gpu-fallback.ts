/**
 * WSLg GPU crash-loop recovery.
 *
 * Chromium blocklists the WSLg Mesa/D3D12 vGPU, so by default it composites in
 * software (steady, but typing lags). When `/dev/dxg` is present the app
 * un-blocklists the GPU for a snappier UI — but on some WSLg driver stacks the
 * un-blocklisted GPU process can't initialize (e.g. `samplerYcbcrConversion is
 * not supported`) and segfaults (`exit_code=139`) in a tight loop, taking the
 * whole app down.
 *
 * The un-blocklist is decided PRE-LAUNCH (before app `ready`), but the crash is
 * a RUNTIME event, so a simple try/catch can't guard it. Instead we mirror the
 * Windows sandbox-fallback pattern (windows-sandbox-fallback.ts): persist a
 * marker across launches. When the un-blocklisted GPU crashes repeatedly in a
 * session we record a sticky `fallback`, and the next launch skips the
 * un-blocklist and rides Chromium's stable software path instead.
 *
 * The fallback is sticky per app version: an app update re-probes the GPU once
 * (a new Electron / Mesa may have fixed the host) before degrading again.
 *
 * Pure helpers stay injectable so tests never boot Electron or touch real files.
 */

import fs from 'node:fs'
import path from 'node:path'

export const WSLG_GPU_MARKER_FILENAME = 'wslg-gpu-fallback.json'

/** GPU-process crashes in one session before the sticky fallback engages. */
export const GPU_CRASHES_BEFORE_FALLBACK = 3

export type WslgGpuMarkerState = 'fallback' | 'ok' | 'probing'

export interface WslgGpuMarker {
  state: WslgGpuMarkerState
  /** App version that entered fallback — a version change triggers a re-probe. */
  version?: string
  /** This launch is a post-update GPU re-probe after a prior fallback. */
  reprobe?: boolean
  /** GPU crashes observed so far while `probing`. Persisted each crash so a
   *  force-quit mid crash-loop still carries progress into the next launch. */
  crashes?: number
}

export function wslgGpuMarkerPath(userDataDir: string): string {
  return path.join(String(userDataDir || ''), WSLG_GPU_MARKER_FILENAME)
}

export function parseWslgGpuMarker(raw: unknown): WslgGpuMarker | null {
  if (!raw || typeof raw !== 'object') {
    return null
  }

  const record = raw as Record<string, unknown>
  const state = record.state

  if (state !== 'probing' && state !== 'fallback' && state !== 'ok') {
    return null
  }

  const marker: WslgGpuMarker = { state }

  if (typeof record.version === 'string' && record.version) {
    marker.version = record.version
  }

  if (record.reprobe === true) {
    marker.reprobe = true
  }

  const crashes = Number(record.crashes)

  if (Number.isInteger(crashes) && crashes > 0) {
    marker.crashes = crashes
  }

  return marker
}

export function readWslgGpuMarker(
  userDataDir: string,
  { readFileSync = fs.readFileSync } = {}
): WslgGpuMarker | null {
  try {
    return parseWslgGpuMarker(JSON.parse(readFileSync(wslgGpuMarkerPath(userDataDir), 'utf8')))
  } catch {
    return null
  }
}

export function writeWslgGpuMarker(
  userDataDir: string,
  marker: WslgGpuMarker,
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

  try {
    mkdirSync(dir, { recursive: true })
    writeFileSync(wslgGpuMarkerPath(dir), `${JSON.stringify(marker)}\n`, 'utf8')
  } catch {
    // Best-effort: a marker we can't persist just means the next launch
    // re-probes the GPU. Never let it break boot.
  }
}

export function wslgGpuFallbackMarker(appVersion?: string): WslgGpuMarker {
  const marker: WslgGpuMarker = { state: 'fallback' }

  if (appVersion) {
    marker.version = appVersion
  }

  return marker
}

export interface WslgGpuLaunchDecision {
  /** Whether to un-blocklist the WSLg vGPU this launch. */
  enableGpu: boolean
  /** Short reason for the decision, for logging (null when GPU stays on). */
  reason: string | null
  /** Marker to persist immediately, before the GPU process starts. */
  nextMarker: WslgGpuMarker
}

/**
 * Single launch-time transition: decide whether this WSLg launch un-blocklists
 * the vGPU AND what the marker becomes for crash detection on the next launch.
 *
 * - No marker / clean `ok` → probe the GPU (`probing`).
 * - `probing` left behind → the last launch crashed before it could mark `ok`.
 *   Its persisted `crashes` count carries forward: if it already reached the
 *   threshold, engage the sticky fallback now (a force-quit mid crash-loop still
 *   self-heals on the next launch); otherwise probe again, seeding the runtime
 *   counter with the carried crashes so two half-loops still add up.
 * - `fallback` is sticky within one app version. A version change re-probes the
 *   GPU once (`reprobe`) so a fixed host returns to acceleration; if that
 *   re-probe launch also crashes, the runtime counter re-arms the fallback.
 */
export function decideWslgGpuLaunch(
  options: { marker?: WslgGpuMarker | null; appVersion?: string; threshold?: number } = {}
): WslgGpuLaunchDecision {
  const appVersion = String(options.appVersion || '')
  const threshold = options.threshold ?? GPU_CRASHES_BEFORE_FALLBACK
  const marker = options.marker ?? null

  if (marker?.state === 'fallback') {
    if (marker.version && appVersion && marker.version !== appVersion) {
      // App updated since the fallback engaged — re-probe the GPU once.
      return { enableGpu: true, reason: 'reprobe-after-update', nextMarker: { state: 'probing', reprobe: true } }
    }

    return {
      enableGpu: false,
      reason: 'sticky-fallback',
      nextMarker: { ...marker, version: marker.version || appVersion || undefined }
    }
  }

  // A prior probing session that carried enough crashes across a force-quit →
  // engage the fallback now instead of re-entering the crash loop.
  const carriedCrashes = marker?.state === 'probing' ? (marker.crashes ?? 0) : 0

  if (carriedCrashes >= threshold) {
    return { enableGpu: false, reason: 'carried-crash-loop', nextMarker: wslgGpuFallbackMarker(appVersion) }
  }

  // No marker, a clean `ok`, or a probing marker under the threshold — try the
  // GPU, seeding the counter with any carried crashes. Runtime crash detection
  // (see gpuCrashEngagesFallback) trips the sticky fallback.
  return {
    enableGpu: true,
    reason: null,
    nextMarker: carriedCrashes > 0 ? { state: 'probing', crashes: carriedCrashes } : { state: 'probing' }
  }
}

/**
 * Persist an incremented crash count on the `probing` marker so progress toward
 * the fallback survives a force-quit mid crash-loop. Returns the marker to write.
 */
export function recordGpuCrash(previousCrashes: number, appVersion?: string): WslgGpuMarker {
  const crashes = (Number.isFinite(previousCrashes) ? previousCrashes : 0) + 1
  const marker: WslgGpuMarker = { state: 'probing', crashes }

  if (appVersion) {
    marker.version = appVersion
  }

  return marker
}

/**
 * After enough GPU-process crashes in one session, engage the sticky fallback
 * so the NEXT launch skips the un-blocklist. Returns the marker to persist, or
 * null when the crash count is still under the threshold.
 */
export function gpuCrashEngagesFallback(options: {
  crashCount: number
  appVersion?: string
  threshold?: number
}): WslgGpuMarker | null {
  const threshold = options.threshold ?? GPU_CRASHES_BEFORE_FALLBACK

  if (!Number.isFinite(options.crashCount) || options.crashCount < threshold) {
    return null
  }

  return wslgGpuFallbackMarker(options.appVersion)
}

/**
 * A GPU child died. Only crashes (segfault / abnormal exit) count toward the
 * fallback threshold — a clean GPU shutdown (`clean-exit`, exit 0) does not.
 */
export function isGpuChildCrash(details: { type?: string; reason?: string; exitCode?: number | string } | null): boolean {
  if (!details) {
    return false
  }

  if (String(details.type || '').toLowerCase() !== 'gpu') {
    return false
  }

  const reason = String(details.reason || '').toLowerCase()

  // Electron's child-process-gone reasons: 'crashed' | 'killed' |
  // 'oom' | 'abnormal-exit' | 'clean-exit' | 'launch-failed' | ...
  // A clean exit (code 0) is a normal GPU teardown, not a crash.
  if (reason === 'clean-exit' && Number(details.exitCode) === 0) {
    return false
  }

  return true
}

/**
 * After the main window reaches ready-to-show without a GPU crash loop: mark a
 * clean boot so future launches keep trusting the GPU. If the fallback engaged
 * this session, keep it sticky.
 */
export function wslgGpuMarkerAfterSuccessfulBoot(options: {
  fallbackActive: boolean
  appVersion?: string
}): WslgGpuMarker {
  return options.fallbackActive ? wslgGpuFallbackMarker(options.appVersion) : { state: 'ok' }
}
