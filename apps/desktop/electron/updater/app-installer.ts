// updater/app-installer.ts — the win32 out-of-store MSIX strategy.
//
// The OS App Installer owns the apply. The package was installed from an
// .appinstaller, which registered the feed URI as the package's update source;
// the OS checks it and swaps the package wholesale. The app's only jobs:
//   check()  ask the OS whether an update is available (via the bundled
//            payload python's winrt), surfacing UNKNOWN honestly;
//   apply()  stage the descriptor, tear down, open it, and quit with
//            a pending-relaunch marker so Hermes comes back by itself.
//
// Pure-injectable: the impure pieces (python runner, shell, quit, relaunch
// marker) are injected, so vitest covers the arm without a payload.

import {
  type AppInstallerCheck,
  parseCheckOutput,
  type PayloadPythonRunner,
  triggerAppInstallerUpdate,
  win32AppInstallerFeedPath
} from '../app-updater'

import { applyPackagedHandoff } from './packaged-handoff'
import { channelPublicBase } from './channel-protocol'
import type { ChannelTarget } from './channel'
import type { RelaunchRegistration } from './relaunch'

import type { UpdaterApplyResultWire, UpdaterStatusWire } from './index'

export interface AppInstallerStrategyDeps {
  /** Absolute path to the bundled payload python (tools/<entry>/python.exe). */
  python: string
  /** The checker script's absolute path (payload repo snapshot). */
  script: string
  run: PayloadPythonRunner['run']
  /** Channel + variant from the baked install stamp. */
  channel: string
  /** Verification must bind descriptor, artifact digest, native identity and signer. */
  feed?: { url: string; version: string; verifyPrepared: (file: string) => Promise<void> }
  light: boolean
  /** The App Installer feed base URL; empty when nothing configured it. */
  feedBaseUrl: string
  installer: { prepare: (url: string) => Promise<string>; open: (file: string) => Promise<string> }
  /** Graceful backend teardown before the package swap. */
  teardownBundledBackend: () => void | Promise<void>
  restoreBundledBackend: () => Promise<void>
  /** Progress emitter for the updates overlay. */
  emitUpdateProgress: (payload: { stage: string; message: string; percent: number | null }) => void
  /** App version label for the status wire. */
  appVersion: string
  /** Quit the app (after handing the swap to the OS). */
  quit: () => void
  /** Retain marker and waiter ownership until the OS accepts the handoff. */
  registerPendingRelaunch: (fromVersion: string) => Promise<RelaunchRegistration>
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
    updateAvailable: check.available ?? undefined,
    // null = unknown (checker unavailable) — surface honestly, never "no update".
    error: check.available === null ? check.error || 'update check unavailable' : undefined,
    fetchedAt: Date.now()
  }
}

export function createChannelAppInstallerStrategy(
  deps: AppInstallerStrategyDeps,
  target: ChannelTarget,
  verifyPrepared: (file: string, target: ChannelTarget) => Promise<void>
): AppInstallerStrategy {
  if (target.package.platform !== 'win32') { throw new Error('Expected Windows channel target') }
  return new AppInstallerStrategy({
    ...deps, channel: target.channel.name, feedBaseUrl: target.manifest.request.publicBase,
    feed: { url: target.feedUrl, version: target.package.version,
      verifyPrepared: (file: string): Promise<void> => verifyPrepared(file, target) }
  })
}

export class AppInstallerStrategy {
  readonly mechanism = 'app-installer' as const

  constructor(private readonly deps: AppInstallerStrategyDeps) {}

  async check(): Promise<UpdaterStatusWire> {
    if (this.deps.feed) {
      return { supported: true, mechanism: this.mechanism, currentVersion: this.deps.appVersion,
        updateAvailable: newerWindowsVersion(this.deps.feed.version, this.deps.appVersion), fetchedAt: Date.now() }
    }
    const { code, stdout } = await this.deps.run(this.deps.python, this.deps.script)
    const check = parseCheckOutput(code, stdout)

    return appInstallerCheckToStatus(check, this.deps.appVersion)
  }

  async apply(): Promise<UpdaterApplyResultWire> {
    const feedBaseUrl = this.deps.feedBaseUrl
    let sourceUri: string | undefined = this.deps.feed?.url
    if (sourceUri) {
      channelPublicBase(sourceUri)
      const base = channelPublicBase(feedBaseUrl)
      if (!sourceUri.startsWith(`${base}/`) || new URL(sourceUri).origin !== new URL(base).origin) { throw new Error('Native feed authority mismatch') }
      if (!newerWindowsVersion(this.deps.feed!.version, this.deps.appVersion)) { return { ok: true, mechanism: this.mechanism } }
    }

    if (!feedBaseUrl && !sourceUri) {
      const { code, stdout } = await this.deps.run(this.deps.python, this.deps.script)
      sourceUri = parseCheckOutput(code, stdout).sourceUri
    }

    if (!feedBaseUrl && !sourceUri) {
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

    return applyPackagedHandoff(
      {
        teardown: this.deps.teardownBundledBackend,
        restore: this.deps.restoreBundledBackend,
        emitProgress: this.deps.emitUpdateProgress,
        relaunch: {
          register: (): Promise<RelaunchRegistration> => this.deps.registerPendingRelaunch(this.deps.appVersion),
          onManual: (): void =>
            this.deps.emitUpdateProgress({
              stage: 'restart',
              percent: 100,
              message: 'Automatic relaunch could not be registered. Reopen Hermes after App Installer finishes.'
            })
        }
      },
      async (stop: () => Promise<void>): Promise<UpdaterApplyResultWire> => {
        await triggerAppInstallerUpdate(
          feedBaseUrl,
          this.deps.channel,
          this.deps.light,
          {
            prepare: async (url: string): Promise<string> => {
              const file = await this.deps.installer.prepare(url)
              await this.deps.feed?.verifyPrepared(file)
              return file
            },
            open: this.deps.installer.open
          },
          stop,
          sourceUri
        )
        this.deps.quit()

        return { ok: true, manual: false, bundled: true, handedOff: true, mechanism: this.mechanism }
      }
    )
  }
}

function newerWindowsVersion(target: string, current: string): boolean {
  const parse = (version: string): number[] => {
    if (!/^\d+\.\d+\.\d+\.\d+$/.test(version)) { throw new Error('Windows channel updates require native numeric versions') }
    const parts = version.split('.').map(Number)
    if (parts.some((part: number): boolean => part > 65535)) { throw new Error('Invalid Windows native version') }
    return parts
  }
  const left = parse(target)
  const right = parse(current)
  for (let index = 0; index < 4; index += 1) {
    if (left[index] !== right[index]) { return left[index] > right[index] }
  }
  return false
}

export { parseCheckOutput }

export { win32AppInstallerFeedPath }
export type { AppInstallerCheck, PayloadPythonRunner }
