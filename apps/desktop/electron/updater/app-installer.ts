// updater/app-installer.ts — the win32 out-of-store MSIX strategy.
//
// The OS App Installer owns the apply. The package was installed from an
// .appinstaller, which registered the feed URI as the package's update source;
// the OS checks it and swaps the package wholesale. The app's only jobs:
//   check()  ask the OS whether an update is available (via the bundled
//            payload python's winrt), surfacing UNKNOWN honestly;
//   apply()  run graceful teardown, trigger ms-appinstaller:, quit — and
//            a pending-relaunch marker so Hermes comes back by itself.
//
// Pure-injectable: the impure pieces (python runner, shell, quit, relaunch
// marker) are injected, so vitest covers the arm without a payload.

import {
  type AppInstallerCheck,
  type PayloadPythonRunner,
  triggerAppInstallerUpdate,
  win32AppInstallerFeedPath
} from '../app-updater'

import type { UpdaterApplyResultWire, UpdaterStatusWire } from './index'

export interface AppInstallerStrategyDeps {
  /** Absolute path to the bundled payload python (tools/<entry>/python.exe). */
  python: string
  /** The checker script's absolute path (payload repo snapshot). */
  script: string
  run: PayloadPythonRunner['run']
  /** Channel + variant from the baked install stamp. */
  channel: 'stable' | 'canary'
  light: boolean
  /** The App Installer feed base URL; empty when nothing configured it. */
  feedBaseUrl: string
  shell: { openExternal: (url: string) => Promise<void> }
  /** Graceful backend teardown before the package swap. */
  teardownBundledBackend: () => void | Promise<void>
  /** Progress emitter for the updates overlay. */
  emitUpdateProgress: (payload: { stage: string; message: string; percent: number | null }) => void
  /** App version label for the status wire. */
  appVersion: string
  /** Quit the app (after handing the swap to the OS). */
  quit: () => void
  /**
   * Register a one-shot post-update relaunch (run-key entry or equivalent)
   * before quitting, so Hermes reopens on the new version with no user
   * action. The new version's first run detects + deletes the marker.
   * Returns false when registration failed (relaunch stays manual).
   */
  registerPendingRelaunch: (targetVersion: string) => boolean
}

export interface CheckOutcome {
  status: UpdaterStatusWire
}

/**
 * Ask the OS whether an App Installer update is available. `available: null`
 * (checker unavailable) is surfaced as an honest unknown on the wire —
 * NEVER as "no update".
 */
export function appInstallerCheckToStatus(check: AppInstallerCheck, appVersion: string): UpdaterStatusWire {
  return {
    supported: true,
    mechanism: 'app-installer',
    currentVersion: appVersion,
    updateAvailable: check.available === true,
    // null = unknown (checker unavailable) — surface honestly, never "no update".
    error: check.available === null ? check.error || 'update check unavailable' : undefined,
    fetchedAt: Date.now()
  }
}

export class AppInstallerStrategy {
  readonly mechanism = 'app-installer' as const

  constructor(private readonly deps: AppInstallerStrategyDeps) {}

  async check(): Promise<UpdaterStatusWire> {
    const { code, stdout } = await this.deps.run(this.deps.python, this.deps.script)
    const check = parseCheckOutput(code, stdout)

    return appInstallerCheckToStatus(check, this.deps.appVersion)
  }

  async apply(_opts: { stopSafeBlockers?: boolean }): Promise<UpdaterApplyResultWire> {
    const feedBaseUrl = this.deps.feedBaseUrl

    if (!feedBaseUrl) {
      this.deps.emitUpdateProgress({
        stage: 'manual',
        message: 'bundled install: update by installing the new app release',
        percent: null
      })

      return { ok: true, manual: true, bundled: true, mechanism: this.mechanism }
    }

    this.deps.emitUpdateProgress({
      stage: 'restart',
      message: 'Applying the Hermes update — the window will close and the App Installer will finish.',
      percent: 100
    })

    // Unconditional relaunch: register the one-shot marker BEFORE the swap so
    // Hermes comes back on the new version with no user action.
    this.deps.registerPendingRelaunch(this.deps.appVersion)

    await triggerAppInstallerUpdate(
      feedBaseUrl,
      this.deps.channel,
      this.deps.light,
      this.deps.shell,
      this.deps.teardownBundledBackend
    )

    this.deps.quit()

    return { ok: true, manual: false, bundled: true, mechanism: this.mechanism }
  }
}

/** Parse the payload-python checker's output into an AppInstallerCheck. */
export function parseCheckOutput(code: number, stdout: string): AppInstallerCheck {
  const text = stdout.trim()
  let parsed: { available?: boolean | null; availability?: string; error?: string; reason?: string } | null = null

  try {
    parsed = text ? JSON.parse(text) : null
  } catch {
    parsed = null
  }

  if (parsed && typeof parsed.available === 'boolean') {
    return { available: parsed.available, availability: parsed.availability, error: parsed.error }
  }

  if (parsed && parsed.available === null) {
    return { available: null, error: parsed.error || 'checker returned unknown' }
  }

  if (code !== 0) {
    return { available: null, error: parsed?.error || `checker exited ${code}` }
  }

  return { available: null, error: 'checker returned no availability' }
}

export { win32AppInstallerFeedPath }
export type { AppInstallerCheck, PayloadPythonRunner }
