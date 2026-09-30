/**
 * Linux NVIDIA EGL fallback for #40077 / #124255.
 *
 * History: NVIDIA driver 580.x breaks ANGLE's EGL probing on X11/Wayland
 * ("Invalid visual ID requested"), the GPU process dies, and the app goes down
 * with it. Rendering ANGLE through SwiftShader (`--use-angle=swiftshader`)
 * skips the NVIDIA EGL probe entirely, so the app launches and stays up — at
 * the cost of CPU compositing.
 *
 * The gate used to be a driver-major set (first `>= 580`, then the closed
 * `{580}` from #123213). But the same driver point release breaks one host and
 * not another — 580.178.04 kills the GPU process on a desktop where NVIDIA
 * drives the display (#40077, #124032) and renders fine on hybrid hosts whose
 * session EGL lands on the iGPU (#124255, psukez on #123203). A driver-series
 * bound cannot separate them, and the all-CPU SwiftShader fallback converts a
 * known Chromium stuck-Button-observer leak (~0 cost with GPU compositing)
 * into a 4–9 core burn on every healthy 580 host.
 *
 * So the gate is now behavioral (#124255 Option A): a Linux + NVIDIA host
 * boots with hardware GL and a `booting` marker; if the GPU process dies
 * before the first window is revealed, the marker turns `fallback` (sticky
 * per app version + full driver version) and the app relaunches once with
 * `--use-angle=swiftshader`. An app update re-probes once instead of degrading
 * forever.
 *
 * The `booting` witness now outlives first paint (#129174): field reports show
 * the "GPU process isn't usable" FATAL abort landing 1–2.5 minutes into the
 * run — long after the window is on screen. Marking `ok` at reveal erased the
 * only witness the abort leaves behind, so every launch re-probed the broken
 * driver and died in a crash loop. Instead, `ok` is only written once the GPU
 * has survived a grace window (`NVIDIA_EGL_GPU_GRACE_MS`) past first paint, or
 * on a clean quit (`nvidiaEglMarkerAfterCleanExit`) — a user-initiated exit
 * proves the run ended without a GPU abort. Any unclean death inside the grace
 * window leaves `booting` behind and the next launch engages the fallback.
 *
 * If Chromium's "GPU process isn't usable" FATAL abort wins the race and the
 * process dies before the handler runs, the leftover `booting` marker engages
 * the fallback on the next launch.
 *
 * Deliberately NOT `app.disableHardwareAcceleration()`: on 580.173.02 +
 * Electron 40 that path SIGKILLs the renderer (see #40077 discussion, and the
 * closed #40119 which was rejected for exactly this). We only reroute ANGLE;
 * we never disable the GPU pipeline wholesale.
 *
 * Skipped when a remote display already forced software rendering, under WSLg
 * (vGPU is healthy there), or when `HERMES_DESKTOP_DISABLE_GPU=0` keeps the
 * GPU on. `HERMES_DESKTOP_NVIDIA_SWIFTSHADER=1` forces the fallback on;
 * `=0` opts out entirely.
 *
 * Pure + dependency-free so it can be unit-tested and called before app ready.
 */

import fs from 'node:fs'
import path from 'node:path'

const OVERRIDE_ON = new Set(['1', 'true', 'yes', 'on'])
const OVERRIDE_OFF = new Set(['0', 'false', 'no', 'off'])

export const NVIDIA_EGL_FALLBACK_MARKER_FILENAME = 'nvidia-egl-fallback.json'

/**
 * How long the GPU must keep running past first paint before the probe counts
 * as healthy. Sized past the observed "GPU process isn't usable" FATAL aborts
 * (1–2.5 min into the run, #129174) with margin. Until it elapses — or a clean
 * quit proves the run ended safely — the `booting` witness stays on disk so an
 * unclean death is still attributable to the GPU probe on the next launch.
 */
export const NVIDIA_EGL_GPU_GRACE_MS = 5 * 60 * 1000

/**
 * `child-process-gone` reasons that witness a broken GPU child. `killed` is
 * included on purpose: in #40077 the GPU process exited with exit_code=15 —
 * Chromium's own GPU health check SIGTERM — not a crash reason.
 */
export const NVIDIA_GPU_DEATH_REASONS: ReadonlySet<string> = new Set(['crashed', 'launch-failure', 'killed'])

export type NvidiaEglMarkerState = 'booting' | 'fallback' | 'ok'

export interface NvidiaEglMarker {
  state: NvidiaEglMarkerState
  /** App version that entered fallback — a version change triggers a re-probe. */
  version?: string
  /** Full driver version (e.g. "580.178.04") the fallback was witnessed on. */
  driverVersion?: string
}

export function nvidiaEglMarkerPath(userDataDir: string): string {
  return path.join(String(userDataDir || ''), NVIDIA_EGL_FALLBACK_MARKER_FILENAME)
}

export function parseNvidiaEglMarker(raw: unknown): NvidiaEglMarker | null {
  if (!raw || typeof raw !== 'object') {
    return null
  }

  const record = raw as Record<string, unknown>
  const state = record.state

  if (state !== 'booting' && state !== 'fallback' && state !== 'ok') {
    return null
  }

  const marker: NvidiaEglMarker = { state }

  if (typeof record.version === 'string' && record.version) {
    marker.version = record.version
  }

  if (typeof record.driverVersion === 'string' && record.driverVersion) {
    marker.driverVersion = record.driverVersion
  }

  return marker
}

export function readNvidiaEglMarker(
  userDataDir: string,
  { readFileSync = fs.readFileSync } = {}
): NvidiaEglMarker | null {
  try {
    return parseNvidiaEglMarker(JSON.parse(readFileSync(nvidiaEglMarkerPath(userDataDir), 'utf8')))
  } catch {
    return null
  }
}

export function writeNvidiaEglMarker(
  userDataDir: string,
  marker: NvidiaEglMarker,
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
  writeFileSync(nvidiaEglMarkerPath(dir), `${JSON.stringify(marker)}\n`, 'utf8')
}

export function nvidiaEglFallbackMarker(appVersion: string, driverVersion: string): NvidiaEglMarker {
  return { state: 'fallback', version: appVersion, driverVersion }
}

/**
 * Extract the driver major version from /proc/driver/nvidia/version content.
 * Format: "NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.82.09 ..."
 */
export function parseNvidiaDriverMajor(procVersion: string): number | null {
  const match = /\b(\d{3,})\.\d+\.\d+\b/.exec(String(procVersion || ''))

  if (!match) {
    return null
  }

  const major = Number.parseInt(match[1], 10)

  return Number.isFinite(major) ? major : null
}

/**
 * Extract the FULL driver version ("580.178.04") from /proc/driver/nvidia/version.
 * The marker is keyed on it so a driver update re-probes like an app update.
 */
export function parseNvidiaDriverVersion(procVersion: string): string | null {
  const match = /\b(\d{3,}\.\d+\.\d+)\b/.exec(String(procVersion || ''))

  return match ? match[1] : null
}

export interface NvidiaEglFallbackDecision {
  enable: boolean
  reason: string | null
  /** Marker to persist immediately, before GPU children start. */
  nextMarker: NvidiaEglMarker
}

/**
 * Single launch-time transition: whether this Linux launch routes ANGLE
 * through SwiftShader, and what the marker becomes. The probe shape:
 *
 * - `fallback` marker matching this app version AND driver version → enable,
 *   keep the marker (sticky until either version changes).
 * - `fallback` marker for a different app or driver version → re-probe once:
 *   boot with hardware GL, write `booting`.
 * - `booting` marker left behind by a launch that died before it could mark
 *   `ok` (the "GPU process isn't usable" FATAL abort) → enable; the previous
 *   boot witnessed a GPU death by not surviving.
 * - `ok` or no marker → boot with hardware GL, write `booting` so a death
 *   mid-boot is attributable.
 *
 * The env override `HERMES_DESKTOP_NVIDIA_SWIFTSHADER=1` forces the fallback
 * on regardless of the marker; `=0` keeps it off and clears the marker.
 */
export function decideNvidiaEglFallback(options: {
  driverMajor: number | null
  driverVersion?: string | null
  marker?: NvidiaEglMarker | null
  appVersion?: string
  env?: NodeJS.ProcessEnv
  platform?: NodeJS.Platform
  isWsl?: boolean
  remoteDisplayReason?: string | null
}): NvidiaEglFallbackDecision {
  const env = options.env ?? process.env
  const platform = options.platform ?? process.platform
  const isWsl = options.isWsl ?? false
  const remoteDisplayReason = options.remoteDisplayReason ?? null
  const driverMajor = options.driverMajor
  const driverVersion = options.driverVersion ?? null
  const appVersion = options.appVersion ?? ''
  const marker = options.marker ?? null

  const bootMarker: NvidiaEglMarker = { state: 'booting' }

  const nvidiaOverride = String(env.HERMES_DESKTOP_NVIDIA_SWIFTSHADER || '')
    .trim()
    .toLowerCase()

  if (OVERRIDE_OFF.has(nvidiaOverride)) {
    return { enable: false, reason: null, nextMarker: bootMarker }
  }

  if (platform !== 'linux' || driverMajor === null) {
    // No NVIDIA driver to probe — nothing to persist either.
    return { enable: false, reason: null, nextMarker: bootMarker }
  }

  // A user who forced GPU back on (HERMES_DESKTOP_DISABLE_GPU=0) owns that call.
  const gpuOverride = String(env.HERMES_DESKTOP_DISABLE_GPU || '')
    .trim()
    .toLowerCase()

  if (OVERRIDE_OFF.has(gpuOverride)) {
    return { enable: false, reason: null, nextMarker: bootMarker }
  }

  // The remote-display path already forces full software rendering; don't pile
  // ANGLE switches on top of `disableHardwareAcceleration`.
  if (remoteDisplayReason) {
    return { enable: false, reason: null, nextMarker: bootMarker }
  }

  // WSLg reports a healthy vGPU; NVIDIA EGL probing hasn't been reported
  // broken there. Keep the WSL GPU passthrough path untouched.
  if (isWsl) {
    return { enable: false, reason: null, nextMarker: bootMarker }
  }

  if (OVERRIDE_ON.has(nvidiaOverride)) {
    return { enable: true, reason: 'override (HERMES_DESKTOP_NVIDIA_SWIFTSHADER)', nextMarker: bootMarker }
  }

  // Witnessed brokenness: sticky only for the same app AND driver version.
  if (
    marker?.state === 'fallback' &&
    marker.version === (appVersion || marker.version) &&
    marker.driverVersion === (driverVersion ?? marker.driverVersion)
  ) {
    return {
      enable: true,
      reason: `witnessed GPU-process death (app ${marker.version ?? '?'}, driver ${marker.driverVersion ?? '?'})`,
      nextMarker: marker
    }
  }

  // A booting marker that never resolved: the previous launch died before it
  // could mark a healthy boot — on this driver that means the GPU process
  // took the app down (the "GPU process isn't usable" FATAL abort wins the
  // race against our relaunch handler). Trust it and engage.
  if (marker?.state === 'booting') {
    return {
      enable: true,
      reason: 'previous launch aborted mid-boot with the GPU probe live',
      nextMarker: nvidiaEglFallbackMarker(appVersion, driverVersion ?? String(driverMajor))
    }
  }

  // ok, stale fallback (app or driver updated), or first boot: probe the GPU.
  return { enable: false, reason: null, nextMarker: bootMarker }
}

export interface NvidiaGpuDeathOptions {
  platform?: NodeJS.Platform | string
  details: { type?: string; reason?: string } | null
  /** The fallback is already active this launch (sticky marker or override). */
  fallbackActive: boolean
  /** This process already attempted the one-shot relaunch. */
  relaunchAttempted: boolean
}

/**
 * Whether a `child-process-gone` event witnesses the #40077 GPU death and
 * should trigger the one-shot SwiftShader relaunch. `killed` counts: the
 * #40077 GPU process exited with exit_code=15, Chromium's health-check
 * SIGTERM, not a crash reason.
 */
export function shouldRelaunchForNvidiaGpuDeath({
  platform = process.platform,
  details,
  fallbackActive,
  relaunchAttempted
}: NvidiaGpuDeathOptions): boolean {
  if (platform !== 'linux' || fallbackActive || relaunchAttempted) {
    return false
  }

  const type = String(details?.type ?? '').toLowerCase()
  const reason = String(details?.reason ?? '').toLowerCase()

  return (type === 'gpu' || type === 'gpu-process') && NVIDIA_GPU_DEATH_REASONS.has(reason)
}

/**
 * After the GPU has survived the grace window past first paint: the probe
 * passed. Keep a sticky fallback when we launched with SwiftShader, otherwise
 * mark a clean boot so future launches trust the GPU again.
 */
export function nvidiaEglMarkerAfterSuccessfulBoot(options: {
  fallbackActive: boolean
  appVersion?: string
  driverVersion?: string | null
}): NvidiaEglMarker {
  if (options.fallbackActive) {
    return nvidiaEglFallbackMarker(options.appVersion ?? '', options.driverVersion ?? '')
  }

  return { state: 'ok' }
}

/**
 * On a clean quit: the run ended without a GPU abort, so an unresolved `booting`
 * witness is resolved to `ok`. Returns `null` when there is nothing to write —
 * an on-disk `fallback` or `ok` marker is never rewritten here (the SwiftShader
 * relaunch path exits through the same exit plumbing with a fresh `fallback`
 * marker that must survive untouched). A fallback launch that exits cleanly
 * keeps its sticky marker, matching the grace-timer write. (#129174)
 */
export function nvidiaEglMarkerAfterCleanExit(options: {
  marker: NvidiaEglMarker | null
  fallbackActive: boolean
  appVersion?: string
  driverVersion?: string | null
}): NvidiaEglMarker | null {
  if (options.fallbackActive) {
    return nvidiaEglMarkerAfterSuccessfulBoot(options)
  }

  if (options.marker?.state === 'booting') {
    return { state: 'ok' }
  }

  return null
}
