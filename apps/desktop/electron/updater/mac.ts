import { stat } from 'node:fs/promises'

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
  /** Smallest zip the bundled Squirrel.Mac cannot install; unset when it streams to disk. */
  zipLimitBytes?: number
  beforeInstall: () => Promise<void>
  onInstallFailure: () => Promise<void>
  emitProgress: (payload: { stage: string; message: string; percent: number | null }) => void
}

/**
 * Squirrel.Mac before Electron 44.7.0 reads the whole update zip into one
 * buffer that CoreFoundation grows to the next power of two. From 1 GiB on that
 * asks for 2 GiB, past Chromium's allocation cap, and the app crashes while
 * installing. Squirrel.Mac #327, shipped in Electron 44.7.0, streams to disk.
 */
export function squirrelZipLimit(electronVersion: string): number | undefined {
  const [major, minor] = electronVersion.split('.').map(Number)

  return major < 44 || (major === 44 && minor < 7) ? 2 ** 30 : undefined
}

const ZIP_TOO_LARGE =
  'This update is too large for the built-in macOS installer, which would crash while installing it. ' +
  'Install the latest Hermes Desktop from https://hermes-agent.nousresearch.com/ instead.'

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
          const status = await this.checkRelease()

          if (!status.updateAvailable) {
            return { ok: true, mechanism: this.mechanism }
          }

          const files = await this.deps.updater.downloadUpdate()
          await this.deps.verifyDownload?.(files)
          await this.refuseOversizedZip(files)
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

  private async refuseOversizedZip(files: string[]): Promise<void> {
    const limit = this.deps.zipLimitBytes

    if (limit === undefined) {
      return
    }

    // electron-updater may cache the zip under any feed basename, so check every file it returns.
    for (const file of files) {
      if ((await stat(file)).size >= limit) {
        throw new Error(ZIP_TOO_LARGE)
      }
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
