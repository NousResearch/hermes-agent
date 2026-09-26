/**
 * Linux NVIDIA 580+ EGL fallback for #40077.
 *
 * NVIDIA driver 580+ breaks ANGLE's EGL probing on X11/Wayland: the bundled
 * ANGLE `libEGL.so` probes the driver's EGL implementation, hits the 580
 * EGL/X11 bug ("Invalid visual ID requested"), and the GPU process dies —
 * taking the app with it. Rendering through ANGLE's SwiftShader backend
 * (`--use-angle=swiftshader`) skips the NVIDIA EGL probe entirely, so the app
 * launches and stays up; the cost is CPU rendering (slow but stable).
 *
 * Deliberately NOT `app.disableHardwareAcceleration()`: on 580.173.02 +
 * Electron 40 that path SIGKILLs the renderer (see #40077 discussion, and the
 * closed #40119 which was rejected for exactly this). We only reroute ANGLE;
 * we never disable the GPU pipeline wholesale.
 *
 * Skipped when a remote display already forced software rendering (the
 * `--disable-gpu-compositing` path covers it), under WSLg (vGPU is healthy
 * there), or when `HERMES_DESKTOP_DISABLE_GPU=0` keeps the GPU on.
 * `HERMES_DESKTOP_NVIDIA_SWIFTSHADER` overrides detection both ways.
 *
 * Pure + dependency-free so it can be unit-tested and called before app ready.
 */

const OVERRIDE_ON = new Set(['1', 'true', 'yes', 'on'])
const OVERRIDE_OFF = new Set(['0', 'false', 'no', 'off'])

/** First driver major with the broken EGL/X11 probing (580.x and newer). */
export const NVIDIA_BROKEN_EGL_MAJOR = 580

/**
 * First Electron major whose bundled ANGLE/Chromium fixed the NVIDIA 580 EGL
 * probe (#124032: 42.11.8 renders 580.178.04 hardware-accelerated; Electron 40
 * still dies in EGL init without the fallback). Older or unknown runtimes keep
 * the safe legacy default; only a positively-identified newer runtime skips it.
 */
export const ELECTRON_FIXED_EGL_MAJOR = 42

export interface NvidiaEglFallbackDecision {
  enable: boolean
  reason: string | null
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
 * Extract the Electron major from a `process.versions.electron` value
 * (format: "42.11.8"). Returns null when absent or unparsable so callers
 * fall back to the safe legacy default.
 */
export function parseElectronMajor(electronVersion: string): number | null {
  const match = /^(\d+)\./.exec(String(electronVersion || '').trim())

  if (!match) {
    return null
  }

  const major = Number.parseInt(match[1], 10)

  return Number.isFinite(major) ? major : null
}

export function decideNvidiaEglFallback(options: {
  driverMajor: number | null
  env?: NodeJS.ProcessEnv
  platform?: NodeJS.Platform
  isWsl?: boolean
  remoteDisplayReason?: string | null
  electronMajor?: number | null
}): NvidiaEglFallbackDecision {
  const env = options.env ?? process.env
  const platform = options.platform ?? process.platform
  const isWsl = options.isWsl ?? false
  const remoteDisplayReason = options.remoteDisplayReason ?? null
  const driverMajor = options.driverMajor
  const electronMajor = options.electronMajor ?? null

  const nvidiaOverride = String(env.HERMES_DESKTOP_NVIDIA_SWIFTSHADER || '')
    .trim()
    .toLowerCase()

  if (OVERRIDE_OFF.has(nvidiaOverride)) {
    return { enable: false, reason: null }
  }

  if (platform !== 'linux') {
    return { enable: false, reason: null }
  }

  // A user who forced GPU back on (HERMES_DESKTOP_DISABLE_GPU=0) owns that call.
  const gpuOverride = String(env.HERMES_DESKTOP_DISABLE_GPU || '')
    .trim()
    .toLowerCase()

  if (OVERRIDE_OFF.has(gpuOverride)) {
    return { enable: false, reason: null }
  }

  // The remote-display path already forces full software rendering; don't pile
  // ANGLE switches on top of `disableHardwareAcceleration`.
  if (remoteDisplayReason) {
    return { enable: false, reason: null }
  }

  // WSLg reports a healthy vGPU; NVIDIA 580 EGL probing hasn't been reported
  // broken there. Keep the WSL GPU passthrough path untouched.
  if (isWsl) {
    return { enable: false, reason: null }
  }

  const runtimeFixed = electronMajor !== null && electronMajor >= ELECTRON_FIXED_EGL_MAJOR
  // #124032: on runtimes with the fixed EGL probe the fallback is pure CPU
  // cost, so detection stays off there; the HERMES_DESKTOP_NVIDIA_SWIFTSHADER
  // override below still forces it on for systems that still fail.
  // ponytail: fixed-major threshold, not a probe; lower it only with a new
  // verified-broken runtime report, raise/extend it when newer majors regress.
  const detected = driverMajor !== null && driverMajor >= NVIDIA_BROKEN_EGL_MAJOR && !runtimeFixed

  if (!detected && !OVERRIDE_ON.has(nvidiaOverride)) {
    return { enable: false, reason: null }
  }

  const reason = detected
    ? `NVIDIA driver ${driverMajor} (>= ${NVIDIA_BROKEN_EGL_MAJOR})`
    : 'override (HERMES_DESKTOP_NVIDIA_SWIFTSHADER)'

  return { enable: true, reason }
}
