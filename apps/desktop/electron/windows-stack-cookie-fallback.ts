/**
 * Windows Chromium renderer recovery for #108047.
 *
 * On some local Windows hosts the renderer dies with
 * STATUS_STACK_BUFFER_OVERRUN (`0xC0000409` / signed exit `-1073740791`,
 * FAST_FAIL_STACK_COOKIE_CHECK_FAILURE). After the shared crash-loop budget
 * (3 / 60s) is exhausted, the window would otherwise stay blank: the
 * existing #38216 sandbox fallback only matches STATUS_BREAKPOINT
 * (`0x80000003`) and must not drop the sandbox for this signature.
 *
 * Recovery, all scoped to win32 renderer `crashed` + this exit code:
 *
 * 1. One-shot relaunch with GPU disabled (sticky per app version). Next
 *    launch, before app `ready`, reads the marker and applies the same
 *    switches as the remote-display path (`disableHardwareAcceleration` +
 *    `disable-gpu-compositing`).
 * 2. If GPU is already off, the override forbids disabling GPU, or this
 *    process already attempted the relaunch — do not loop; the caller
 *    surfaces the visible renderer error page instead of a blank window.
 *
 * An app update re-probes GPU once (a new Electron may have fixed the host).
 * Successful boot keeps the sticky fallback within the same version.
 *
 * Pure helpers stay injectable so tests never boot Electron.
 */

import fs from 'node:fs'
import path from 'node:path'

export const WINDOWS_GPU_STACK_COOKIE_MARKER_FILENAME = 'windows-gpu-stack-cookie-fallback.json'

/** STATUS_STACK_BUFFER_OVERRUN as a signed Win32 exit code (WER / Chromium). */
export const WINDOWS_STACK_COOKIE_EXIT = -1073740791

const GPU_OVERRIDE_ON = new Set(['1', 'true', 'yes', 'on'])
const GPU_OVERRIDE_OFF = new Set(['0', 'false', 'no', 'off'])
const DISABLE_GPU_SWITCH = '--disable-gpu'
const DISABLE_GPU_COMPOSITING_SWITCH = '--disable-gpu-compositing'

export type GpuStackCookieMarkerState = 'booting' | 'fallback' | 'ok'

export type GpuStackCookieFallbackReason = 'renderer-crash-loop'

export interface GpuStackCookieMarker {
  state: GpuStackCookieMarkerState
  /** Why the fallback engaged (state === 'fallback'). */
  reason?: GpuStackCookieFallbackReason
  /** App version that entered fallback — a version change triggers a re-probe. */
  version?: string
  /** This boot is a GPU re-probe after an app update. */
  reprobe?: boolean
}

export interface GpuStackCookieLaunchDecision {
  enable: boolean
  reason: string | null
  nextMarker: GpuStackCookieMarker
}

export interface RendererStackCookieCrashLoopOptions {
  platform?: NodeJS.Platform | string
  reason?: string
  exitCode?: number | string
  alreadyGpuDisabled?: boolean
  relaunchAttempted?: boolean
  gpuOverrideOff?: boolean
}

export function gpuStackCookieMarkerPath(userDataDir: string): string {
  return path.join(String(userDataDir || ''), WINDOWS_GPU_STACK_COOKIE_MARKER_FILENAME)
}

export function isWindowsStackCookieExit(exitCode: unknown): boolean {
  const n = Number(exitCode)

  if (!Number.isFinite(n)) {
    return false
  }

  // Signed STATUS_STACK_BUFFER_OVERRUN, or the same 32-bit pattern as unsigned.
  return n === WINDOWS_STACK_COOKIE_EXIT || n >>> 0 === 0xc0000409
}

export function isHermesDesktopGpuOverrideOff(env: NodeJS.ProcessEnv = process.env): boolean {
  const override = String(env.HERMES_DESKTOP_DISABLE_GPU || '')
    .trim()
    .toLowerCase()

  return GPU_OVERRIDE_OFF.has(override)
}

/**
 * True when this process already launched with GPU off — Chromium argv
 * (`--disable-gpu`) or HERMES_DESKTOP_DISABLE_GPU on. Mirrors
 * `alreadyHasNoSandbox`: the crash-loop relaunch MUST pass these switches so
 * the next process is protected even if the sticky marker write failed.
 */
export function alreadyHasDisableGpu(argv: readonly string[] = [], env: NodeJS.ProcessEnv = process.env): boolean {
  if (Array.isArray(argv) && argv.some(arg => arg === DISABLE_GPU_SWITCH)) {
    return true
  }

  const override = String(env.HERMES_DESKTOP_DISABLE_GPU || '')
    .trim()
    .toLowerCase()

  return GPU_OVERRIDE_ON.has(override)
}

/** Relaunch argv with a single `--disable-gpu` + compositing switch. */
export function buildDisableGpuRelaunchArgs(argv: readonly string[]): string[] {
  const args = (Array.isArray(argv) ? argv : []).filter(
    arg => arg !== DISABLE_GPU_SWITCH && arg !== DISABLE_GPU_COMPOSITING_SWITCH
  )

  args.push(DISABLE_GPU_SWITCH, DISABLE_GPU_COMPOSITING_SWITCH)

  return args
}

export function parseGpuStackCookieMarker(raw: unknown): GpuStackCookieMarker | null {
  if (!raw || typeof raw !== 'object') {
    return null
  }

  const record = raw as Record<string, unknown>
  const state = record.state

  if (state !== 'booting' && state !== 'fallback' && state !== 'ok') {
    return null
  }

  const marker: GpuStackCookieMarker = { state }

  if (record.reason === 'renderer-crash-loop') {
    marker.reason = 'renderer-crash-loop'
  }

  if (typeof record.version === 'string' && record.version) {
    marker.version = record.version
  }

  if (record.reprobe === true) {
    marker.reprobe = true
  }

  return marker
}

export function readGpuStackCookieMarker(
  userDataDir: string,
  { readFileSync = fs.readFileSync } = {}
): GpuStackCookieMarker | null {
  try {
    const raw = JSON.parse(readFileSync(gpuStackCookieMarkerPath(userDataDir), 'utf8'))

    return parseGpuStackCookieMarker(raw)
  } catch {
    return null
  }
}

export function writeGpuStackCookieMarker(
  userDataDir: string,
  marker: GpuStackCookieMarker,
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
  writeFileSync(gpuStackCookieMarkerPath(dir), `${JSON.stringify(marker)}\n`, 'utf8')
}

export function gpuStackCookieFallbackMarker(
  reason: GpuStackCookieFallbackReason,
  appVersion?: string
): GpuStackCookieMarker {
  const marker: GpuStackCookieMarker = { state: 'fallback', reason }

  if (appVersion) {
    marker.version = appVersion
  }

  return marker
}

/**
 * After the main window reaches ready-to-show: keep the sticky GPU fallback
 * when this launch disabled GPU for #108047, otherwise mark a clean boot so
 * a later app update can re-probe hardware acceleration.
 */
export function markerAfterSuccessfulGpuStackCookieBoot(options: {
  fallbackActive: boolean
  appVersion?: string
}): GpuStackCookieMarker {
  if (!options.fallbackActive) {
    return { state: 'ok' }
  }

  return gpuStackCookieFallbackMarker('renderer-crash-loop', options.appVersion)
}

/**
 * True when a renderer crash loop carries the #108047 stack-cookie signature
 * and a one-shot GPU-disable relaunch should replace the dead window.
 * Gated on the exit code so unrelated crash loops (sandbox breakpoint, OOM)
 * don't silently disable GPU, and so an explicit GPU-off override is honored.
 */
export function shouldRelaunchForRendererStackCookieCrashLoop(options: RendererStackCookieCrashLoopOptions): boolean {
  if ((options.platform ?? process.platform) !== 'win32') {
    return false
  }

  if (options.alreadyGpuDisabled || options.relaunchAttempted || options.gpuOverrideOff) {
    return false
  }

  if (String(options.reason || '') !== 'crashed') {
    return false
  }

  return isWindowsStackCookieExit(options.exitCode)
}

/**
 * True when the #108047 signature matched but GPU fallback cannot run
 * (already disabled, override off, or this process already relaunched).
 * The caller should load the visible renderer error page instead of a blank
 * window. Does not drop the Chromium sandbox.
 */
export function shouldSurfaceErrorForRendererStackCookieCrashLoop(
  options: RendererStackCookieCrashLoopOptions
): boolean {
  if ((options.platform ?? process.platform) !== 'win32') {
    return false
  }

  if (String(options.reason || '') !== 'crashed') {
    return false
  }

  if (!isWindowsStackCookieExit(options.exitCode)) {
    return false
  }

  return Boolean(options.alreadyGpuDisabled || options.relaunchAttempted || options.gpuOverrideOff)
}

/**
 * Launch-time transition: decide whether this Windows launch disables GPU
 * for the sticky #108047 stack-cookie fallback AND what the marker becomes.
 *
 * - `fallback` is sticky within one app version. A version change re-probes
 *   GPU once so a fixed host returns to hardware acceleration.
 * - A marker without a version stays sticky (legacy / no-version contract).
 * - `--disable-gpu` already in argv (crash-loop relaunch / Chromium switch)
 *   is honored even if the sticky marker write failed, but is NOT made sticky
 *   from argv alone.
 * - `HERMES_DESKTOP_DISABLE_GPU` explicitly off (`0`/`false`/`no`/`off`)
 *   fail-opens: do not disable GPU.
 */
export function decideWindowsGpuStackCookieLaunch(
  options: {
    platform?: NodeJS.Platform | string
    argv?: readonly string[]
    env?: NodeJS.ProcessEnv
    marker?: GpuStackCookieMarker | null
    appVersion?: string
  } = {}
): GpuStackCookieLaunchDecision {
  const appVersion = String(options.appVersion || '')
  const argv = options.argv ?? process.argv
  const env = options.env ?? process.env
  const marker = options.marker ?? null

  if ((options.platform ?? process.platform) !== 'win32') {
    return { enable: false, reason: null, nextMarker: { state: 'booting' } }
  }

  if (isHermesDesktopGpuOverrideOff(env)) {
    const nextMarker: GpuStackCookieMarker = marker?.state === 'fallback' ? marker : { state: 'booting' }

    return { enable: false, reason: null, nextMarker }
  }

  // Honor an in-process `--disable-gpu` relaunch (or env override) even when
  // the sticky marker write failed. Not made sticky from argv alone — the
  // marker lifecycle stays unchanged, matching sandbox's already-enabled path.
  if (alreadyHasDisableGpu(argv, env)) {
    const nextMarker: GpuStackCookieMarker = marker?.state === 'fallback' ? marker : { state: 'booting' }

    return { enable: true, reason: 'already-enabled', nextMarker }
  }

  if (marker?.state === 'fallback') {
    if (marker.version && appVersion && marker.version !== appVersion) {
      return {
        enable: false,
        reason: null,
        nextMarker: { state: 'booting', reprobe: true }
      }
    }

    return {
      enable: true,
      reason: 'sticky-fallback',
      nextMarker: { ...marker, version: marker.version || appVersion || undefined }
    }
  }

  return { enable: false, reason: null, nextMarker: { state: 'booting' } }
}
