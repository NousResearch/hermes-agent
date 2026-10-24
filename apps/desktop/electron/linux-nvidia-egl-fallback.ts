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
 * `--use-angle=swiftshader`. A healthy boot marks `ok` and keeps full
 * acceleration. An app update re-probes once instead of degrading forever.
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

import type { Details } from 'electron'

import { launchMarkerNeedsReprobe } from './launch-build-identity'

const OVERRIDE_ON = new Set(['1', 'true', 'yes', 'on'])
const OVERRIDE_OFF = new Set(['0', 'false', 'no', 'off'])

export const NVIDIA_EGL_FALLBACK_MARKER_FILENAME = 'nvidia-egl-fallback.json'

/**
 * `child-process-gone` reasons that witness a broken GPU child. `killed` is
 * included on purpose: in #40077 the GPU process exited with exit_code=15 —
 * Chromium's own GPU health check SIGTERM — not a crash reason.
 */
export const NVIDIA_GPU_DEATH_REASONS: ReadonlySet<Details['reason']> = new Set<Details['reason']>([
  'crashed',
  'launch-failed',
  'killed'
])

export type NvidiaEglMarkerState = 'booting' | 'fallback' | 'ok'

export interface NvidiaEglMarker {
  state: NvidiaEglMarkerState
  /** App version that entered fallback — a version change triggers a re-probe. */
  version?: string
  /** Full driver version (e.g. "580.178.04") the fallback was witnessed on. */
  driverVersion?: string
  /** Source builds can share 0.0.0, so retain their immutable build identity. */
  build?: string
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

  if (typeof record.build === 'string' && record.build) {
    marker.build = record.build
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

export function nvidiaEglFallbackMarker(
  appVersion: string,
  driverVersion: string,
  buildIdentity?: string
): NvidiaEglMarker {
  return {
    state: 'fallback',
    version: appVersion,
    driverVersion,
    ...(buildIdentity && buildIdentity !== appVersion ? { build: buildIdentity } : {})
  }
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
  /** Marker to persist only after acquiring the single-instance lock. */
  nextMarker: NvidiaEglMarker
}

/**
 * Single launch-time transition: whether this Linux launch routes ANGLE
 * through SwiftShader, and what the marker becomes. The probe shape:
 *
 * - `fallback` marker matching this artifact build AND driver version → enable,
 *   keep the marker until the artifact build or driver changes.
 * - `fallback` or `booting` marker for an older build/driver → re-probe once:
 *   boot with hardware GL, write `booting`.
 * - `booting` marker for this same build/driver left by a launch that died before it could mark
 *   `ok` (the "GPU process isn't usable" FATAL abort) → enable; the previous
 *   boot witnessed a GPU death by not surviving.
 * - `ok` or no marker → boot with hardware GL, write `booting` so a death
 *   mid-boot is attributable.
 *
 * The env override `HERMES_DESKTOP_NVIDIA_SWIFTSHADER=1` forces the fallback
 * on regardless of the marker; `=0` keeps it off during startup and GPU death.
 * The lock owner records a healthy boot or clean exit before removing the override.
 */
export interface NvidiaEglFallbackOptions {
  driverMajor: number | null
  driverVersion?: string | null
  marker?: NvidiaEglMarker | null
  appVersion?: string
  buildIdentity?: string
  env?: NodeJS.ProcessEnv
  platform?: NodeJS.Platform
  isWsl?: boolean
  remoteDisplayReason?: string | null
}

/** The same exclusions govern startup and later GPU-death recovery. */
export function nvidiaEglFallbackAllowed(options: NvidiaEglFallbackOptions): boolean {
  const env = options.env ?? process.env

  const off = (key: string): boolean =>
    OVERRIDE_OFF.has(
      String(env[key] || '')
        .trim()
        .toLowerCase()
    )

  return (
    (options.platform ?? process.platform) === 'linux' &&
    options.driverMajor !== null &&
    !options.isWsl &&
    !options.remoteDisplayReason &&
    !off('HERMES_DESKTOP_NVIDIA_SWIFTSHADER') &&
    !off('HERMES_DESKTOP_DISABLE_GPU')
  )
}

export function decideNvidiaEglFallback(options: NvidiaEglFallbackOptions): NvidiaEglFallbackDecision {
  const env = options.env ?? process.env
  const appVersion = options.appVersion ?? ''
  const driverVersion = options.driverVersion ?? String(options.driverMajor ?? '')
  const buildIdentity = options.buildIdentity ?? ''
  const marker = options.marker ?? null

  const bootMarker: NvidiaEglMarker = {
    ...nvidiaEglFallbackMarker(appVersion, driverVersion, buildIdentity),
    state: 'booting'
  }

  if (!nvidiaEglFallbackAllowed(options)) {
    return { enable: false, reason: null, nextMarker: bootMarker }
  }

  const nvidiaOverride = String(env.HERMES_DESKTOP_NVIDIA_SWIFTSHADER || '')
    .trim()
    .toLowerCase()

  if (OVERRIDE_ON.has(nvidiaOverride)) {
    return { enable: true, reason: 'override (HERMES_DESKTOP_NVIDIA_SWIFTSHADER)', nextMarker: bootMarker }
  }

  const sameBuild =
    marker?.version === appVersion &&
    marker.driverVersion === driverVersion &&
    !launchMarkerNeedsReprobe(marker, { appVersion, buildIdentity })

  if (marker?.state === 'fallback' && sameBuild) {
    return {
      enable: true,
      reason: `witnessed GPU-process death (app ${marker.version}, driver ${marker.driverVersion})`,
      nextMarker: marker
    }
  }

  // An abort is evidence only for the build and driver that actually attempted it.
  // Legacy unqualified booting records get one fresh probe, as do stale fallbacks.
  if (marker?.state === 'booting' && sameBuild) {
    return {
      enable: true,
      reason: 'previous launch aborted mid-boot with the GPU probe live',
      nextMarker: nvidiaEglFallbackMarker(appVersion, driverVersion, buildIdentity)
    }
  }

  return { enable: false, reason: null, nextMarker: bootMarker }
}

export interface NvidiaGpuDeathOptions {
  platform?: NodeJS.Platform | string
  details: { type?: string; reason?: Details['reason'] } | null
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
  const reason = details?.reason

  return (type === 'gpu' || type === 'gpu-process') && Boolean(reason && NVIDIA_GPU_DEATH_REASONS.has(reason))
}

/**
 * After the first window is revealed: the GPU survived this boot. Keep a
 * sticky fallback when we launched with SwiftShader, otherwise mark a clean
 * boot so future launches trust the GPU again.
 */
export function nvidiaEglMarkerAfterSuccessfulBoot(options: {
  fallbackActive: boolean
  appVersion?: string
  driverVersion?: string | null
  buildIdentity?: string
}): NvidiaEglMarker {
  if (options.fallbackActive) {
    return nvidiaEglFallbackMarker(options.appVersion ?? '', options.driverVersion ?? '', options.buildIdentity)
  }

  return { state: 'ok' }
}
