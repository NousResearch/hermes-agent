/**
 * SwiftShader stuck-animation watchdog for #124255.
 *
 * The NVIDIA 580 EGL fallback routes all compositing through SwiftShader
 * (CPU rasterization). A known Chromium-on-Wayland/NVIDIA defect occasionally
 * leaves a native `views::Button` animation observer stuck, so the compositor
 * produces frames forever — harmless with GPU compositing, but a silent
 * multi-core CPU burn under SwiftShader. The stuck observer already logs:
 *
 *   CompositorAnimationObserver is active for too long (182.893s)
 *   location=Button@ui/views/controls/button/button.cc:667
 *
 * (ERROR severity, so it lands in desktop-chromium.log next to desktop.log.)
 * This module is the pure decision surface for that watchdog: parse the line,
 * decide whether to warn, format the warning. The poll loop in main.ts
 * (`startChromiumLogWatcher`) applies it. Pure + dependency-free so it can be
 * unit-tested and warn exactly once per process.
 */

/** Warn once a stuck observer outlives this; below it the animation may still settle. */
export const STUCK_ANIMATION_WARN_SECONDS = 60

const STUCK_OBSERVER_RE =
  /CompositorAnimationObserver is active for too long \(([\d.]+)s\)\s+location=(\S+)/

export interface StuckAnimationObservation {
  activeSeconds: number
  location: string
}

export function parseStuckAnimationObserverLine(line: string): StuckAnimationObservation | null {
  const match = STUCK_OBSERVER_RE.exec(String(line || ''))

  if (!match) {
    return null
  }

  const activeSeconds = Number.parseFloat(match[1])

  if (!Number.isFinite(activeSeconds)) {
    return null
  }

  return { activeSeconds, location: match[2] }
}

export interface StuckAnimationWarning {
  warn: boolean
  activeSeconds: number | null
  location: string | null
}

export function decideStuckAnimationWarning(options: {
  /** Recently-appended chromium log text (a tail, not the whole file). */
  tail: string
  /** Whether the NVIDIA EGL SwiftShader fallback engaged this launch. */
  fallbackActive: boolean
  thresholdSeconds?: number
  alreadyWarned?: boolean
}): StuckAnimationWarning {
  const thresholdSeconds = options.thresholdSeconds ?? STUCK_ANIMATION_WARN_SECONDS
  const silent: StuckAnimationWarning = { warn: false, activeSeconds: null, location: null }

  if (!options.fallbackActive || options.alreadyWarned) {
    return silent
  }

  let latest: StuckAnimationObservation | null = null

  for (const line of String(options.tail || '').split('\n')) {
    const observation = parseStuckAnimationObserverLine(line)

    if (observation) {
      latest = observation
    }
  }

  if (!latest || latest.activeSeconds < thresholdSeconds) {
    return silent
  }

  return { warn: true, activeSeconds: latest.activeSeconds, location: latest.location }
}

export function formatStuckAnimationWarning(observation: StuckAnimationObservation): string {
  return (
    `[hermes] stuck compositor animation (${observation.location} active ` +
    `${observation.activeSeconds}s) while the NVIDIA EGL SwiftShader fallback is engaged: ` +
    'every frame is CPU-rasterized, so this burns CPU until the animation settles. ' +
    'HERMES_DESKTOP_NVIDIA_SWIFTSHADER=0 to opt out of the fallback (#124255).'
  )
}
