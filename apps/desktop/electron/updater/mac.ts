import fs from 'node:fs'
import path from 'node:path'

import type { AppUpdater } from 'electron-updater'

import { applyPackagedHandoff } from './packaged-handoff'

import type { UpdaterApplyResultWire, UpdaterStatusWire, UpdaterStrategy } from './index'

export interface MacStrategyDeps {
  updater: Pick<AppUpdater, 'checkForUpdates' | 'downloadUpdate' | 'quitAndInstall' | 'on' | 'removeListener'>
  channel: string
  expectedVersion?: string
  verifyDownload?: (files: string[]) => Promise<void>
  appVersion: string
  /** Squirrel verifies the signed app before any backend is stopped. */
  prepareInstall: () => Promise<void>
  beforeInstall: () => Promise<void>
  onInstallFailure: () => Promise<void>
  emitProgress: (payload: { stage: string; message: string; percent: number | null }) => void
  /** The running Hermes.app; Squirrel must be able to replace it in place. */
  bundlePath?: string
  /** fs.accessSync stand-in for tests. */
  access?: (file: string, mode: number) => void
}

const READ_ONLY_LOCATION =
  'Hermes can not install updates while it runs from this location (its disk image, or a temporary copy macOS made). ' +
  'Quit Hermes, drag Hermes into the Applications folder in Finder (if it is already there, drag it out and back in), ' +
  'then open it and install the update again.'

/**
 * Squirrel.Mac can not replace a bundle on a read-only volume, and finds out only
 * after quitAndInstall: Hermes quit, nothing installed, and the next start offered
 * the same update. App Translocation (a quarantined app run from Downloads) and a
 * mounted disk image are both read-only. Returns the user-facing reason, or null.
 */
export function macReadOnlyBundleReason(
  bundlePath: string,
  access: (file: string, mode: number) => void = fs.accessSync
): null | string {
  if (bundlePath.includes('/AppTranslocation/')) {
    return READ_ONLY_LOCATION
  }

  try {
    access(path.dirname(bundlePath), fs.constants.W_OK)
  } catch (error) {
    // EACCES (a root-owned folder) is Squirrel's to handle: it asks for an admin password.
    if ((error as NodeJS.ErrnoException).code === 'EROFS') {
      return READ_ONLY_LOCATION
    }
  }

  return null
}

export class MacStrategy implements UpdaterStrategy {
  readonly mechanism = 'electron-updater' as const
  private applying = false

  constructor(private readonly deps: MacStrategyDeps) {}

  async check(): Promise<UpdaterStatusWire> {
    if (this.applying) {
      throw new Error('An update is already in progress.')
    }

    return this.checkRelease()
  }

  private async checkRelease(): Promise<UpdaterStatusWire> {
    const result = await this.deps.updater.checkForUpdates()

    if (!result) {
      throw new Error('The macOS updater is not active for this app.')
    }

    if (this.deps.expectedVersion && result.updateInfo.version !== this.deps.expectedVersion) {
      throw new Error('Native macOS feed does not match the pinned channel version')
    }

    return {
      supported: true,
      mechanism: this.mechanism,
      currentVersion: this.deps.appVersion,
      channel: this.deps.channel,
      latestTag: `v${result.updateInfo.version}`,
      updateAvailable: result.isUpdateAvailable,
      fetchedAt: Date.now()
    }
  }

  async apply(): Promise<UpdaterApplyResultWire> {
    if (this.applying) {
      throw new Error('An update is already in progress.')
    }

    this.applying = true

    const progress = ({ percent }: { percent: number }): void => {
      this.deps.emitProgress({ stage: 'fetch', message: 'Downloading the Hermes update.', percent })
    }

    this.deps.updater.on('download-progress', progress)

    try {
      return await applyPackagedHandoff(
        {
          teardown: this.deps.beforeInstall,
          restore: this.deps.onInstallFailure,
          emitProgress: this.deps.emitProgress
        },
        async (stop: () => Promise<void>): Promise<UpdaterApplyResultWire> => {
          const readOnly = this.deps.bundlePath ? macReadOnlyBundleReason(this.deps.bundlePath, this.deps.access) : null

          if (readOnly) {
            throw new Error(readOnly)
          }

          const status = await this.checkRelease()

          if (!status.updateAvailable) {
            return { ok: true, mechanism: this.mechanism }
          }

          const files = await this.deps.updater.downloadUpdate()
          await this.deps.verifyDownload?.(files)
          this.deps.emitProgress({ stage: 'prepare', message: 'Verifying the signed macOS update.', percent: null })
          await this.deps.prepareInstall()
          await stop()
          this.deps.emitProgress({
            stage: 'restart',
            message: 'Restarting Hermes to install the update.',
            percent: 100
          })
          this.deps.updater.quitAndInstall()

          return { ok: true, bundled: true, handedOff: true, mechanism: this.mechanism }
        }
      )
    } finally {
      this.deps.updater.removeListener('download-progress', progress)
      this.applying = false
    }
  }
}

export interface NativeMacUpdater {
  once(event: 'update-downloaded', listener: () => void): unknown
  once(event: 'error', listener: (error: Error) => void): unknown
  removeListener(event: 'update-downloaded', listener: () => void): unknown
  removeListener(event: 'error', listener: (error: Error) => void): unknown
  checkForUpdates(): void
}

/** Download completion alone does not mean Squirrel accepted the signature. */
export function prepareMacInstall(native: NativeMacUpdater, timeoutMs: number = 120_000): Promise<void> {
  return new Promise<void>((resolve: () => void, reject: (error: Error) => void): void => {
    const cleanup = (): void => {
      clearTimeout(timer)
      native.removeListener('error', failed)
      native.removeListener('update-downloaded', ready)
    }

    const failed = (error: Error): void => {
      cleanup()
      reject(error)
    }

    const ready = (): void => {
      cleanup()
      resolve()
    }

    const timer = setTimeout((): void => failed(new Error('macOS update verification timed out.')), timeoutMs)
    native.once('error', failed)
    native.once('update-downloaded', ready)

    try {
      native.checkForUpdates()
    } catch (error) {
      failed(error as Error)
    }
  })
}
