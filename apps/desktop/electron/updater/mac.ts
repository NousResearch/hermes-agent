import type { AppUpdater } from 'electron-updater'

import type { UpdaterApplyResultWire, UpdaterStatusWire, UpdaterStrategy } from './index'

export interface MacStrategyDeps {
  updater: Pick<AppUpdater, 'checkForUpdates' | 'downloadUpdate' | 'quitAndInstall' | 'on' | 'removeListener'>
  channel: 'stable' | 'canary'
  appVersion: string
  /** Squirrel verifies the signed app before any backend is stopped. */
  prepareInstall: () => Promise<void>
  beforeInstall: () => Promise<void>
  onInstallFailure: () => Promise<void>
  emitProgress: (payload: { stage: string; message: string; percent: number | null }) => void
}

export class MacStrategy implements UpdaterStrategy {
  readonly mechanism = 'electron-updater' as const
  private applying = false

  constructor(private readonly deps: MacStrategyDeps) {}

  async check(): Promise<UpdaterStatusWire> {
    if (this.applying) { throw new Error('An update is already in progress.') }

    return this.checkRelease()
  }

  private async checkRelease(): Promise<UpdaterStatusWire> {
    const result = await this.deps.updater.checkForUpdates()

    if (!result) { throw new Error('The macOS updater is not active for this app.') }

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
    if (this.applying) { throw new Error('An update is already in progress.') }
    this.applying = true
    let stopped = false

    const progress = ({ percent }: { percent: number }): void => {
      this.deps.emitProgress({ stage: 'fetch', message: 'Downloading the Hermes update.', percent })
    }

    this.deps.updater.on('download-progress', progress)

    try {
      const status = await this.checkRelease()

      if (!status.updateAvailable) { return { ok: true, mechanism: this.mechanism } }
      await this.deps.updater.downloadUpdate()
      this.deps.emitProgress({ stage: 'prepare', message: 'Verifying the signed macOS update.', percent: null })
      await this.deps.prepareInstall()
      stopped = true
      await this.deps.beforeInstall()
      this.deps.emitProgress({ stage: 'restart', message: 'Restarting Hermes to install the update.', percent: 100 })
      this.deps.updater.quitAndInstall()

      return { ok: true, bundled: true, handedOff: true, mechanism: this.mechanism }
    } catch (error) {
      if (stopped) { await this.deps.onInstallFailure() }
      throw error
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
export function prepareMacInstall(native: NativeMacUpdater, timeoutMs = 120_000): Promise<void> {
  return new Promise((resolve, reject) => {
    const cleanup = (): void => {
      clearTimeout(timer)
      native.removeListener('error', failed)
      native.removeListener('update-downloaded', ready)
    }

    const failed = (error: Error): void => { cleanup(); reject(error) }

    const ready = (): void => { cleanup(); resolve() }
    const timer = setTimeout(() => failed(new Error('macOS update verification timed out.')), timeoutMs)
    native.once('error', failed)
    native.once('update-downloaded', ready)

    try { native.checkForUpdates() } catch (error) { failed(error as Error) }
  })
}
