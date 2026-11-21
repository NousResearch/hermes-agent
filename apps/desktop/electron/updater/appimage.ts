import fs from 'node:fs'
import path from 'node:path'

import type { AppUpdater, UpdateInfo } from 'electron-updater'

import { applyPackagedHandoff } from './packaged-handoff'

import type { UpdaterApplyResultWire, UpdaterStatusWire, UpdaterStrategy } from './index'

export interface AppImageStrategyDeps {
  updater: Pick<AppUpdater, 'checkForUpdates' | 'downloadUpdate' | 'quitAndInstall' | 'on' | 'removeListener'>
  channel: string
  appVersion: string
  /** The AppImage file this launch runs from, which the update replaces. */
  appImage: string
  /** The feed directory, to name the new AppImage when this one cannot be replaced. */
  feedUrl: string | null
  canReplace?: (appImage: string) => boolean
  beforeInstall: () => Promise<void>
  onInstallFailure: () => Promise<void>
  emitProgress: (payload: { stage: string; message: string; percent: number | null }) => void
}

/** The install unlinks the AppImage and moves the new one in beside it, so its folder must be writable. */
export function appImageReplaceable(appImage: string): boolean {
  try {
    fs.accessSync(path.dirname(appImage), fs.constants.W_OK | fs.constants.X_OK)

    return true
  } catch {
    return false
  }
}

export function appImageDownloadUrl(info: UpdateInfo, feedUrl: string | null): string | null {
  const file = info.files?.find(entry => entry.url.endsWith('.AppImage'))

  if (!file || !feedUrl) {
    return null
  }

  try {
    return new URL(file.url, feedUrl).href
  } catch {
    return null
  }
}

export class AppImageStrategy implements UpdaterStrategy {
  readonly mechanism = 'electron-updater' as const
  private applying = false

  constructor(private readonly deps: AppImageStrategyDeps) {}

  async check(): Promise<UpdaterStatusWire> {
    if (this.applying) {
      throw new Error('An update is already in progress.')
    }

    return (await this.checkRelease()).status
  }

  private async checkRelease(): Promise<{ status: UpdaterStatusWire; info: UpdateInfo }> {
    const result = await this.deps.updater.checkForUpdates()

    if (!result) {
      throw new Error('The AppImage updater is not active for this app.')
    }

    return {
      info: result.updateInfo,
      status: {
        supported: true,
        mechanism: this.mechanism,
        currentVersion: this.deps.appVersion,
        channel: this.deps.channel,
        latestTag: `v${result.updateInfo.version}`,
        updateAvailable: result.isUpdateAvailable,
        fetchedAt: Date.now()
      }
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
          const { status, info } = await this.checkRelease()

          if (!status.updateAvailable) {
            return { ok: true, mechanism: this.mechanism }
          }

          if (!(this.deps.canReplace ?? appImageReplaceable)(this.deps.appImage)) {
            return this.notReplaceable(info)
          }

          await this.deps.updater.downloadUpdate()
          await stop()
          this.deps.emitProgress({
            stage: 'restart',
            message: 'Restarting Hermes to install the update.',
            percent: 100
          })
          this.install()

          return { ok: true, bundled: true, handedOff: true, mechanism: this.mechanism }
        }
      )
    } finally {
      this.deps.updater.removeListener('download-progress', progress)
      this.applying = false
    }
  }

  /** quitAndInstall reports a failed replacement only as an 'error' event, and then does not quit. */
  private install(): void {
    let failure: Error | undefined

    const failed = (error: Error): void => {
      failure ??= error
    }

    this.deps.updater.on('error', failed)

    try {
      this.deps.updater.quitAndInstall()
    } finally {
      this.deps.updater.removeListener('error', failed)
    }

    if (failure) {
      throw failure
    }
  }

  private notReplaceable(info: UpdateInfo): UpdaterApplyResultWire {
    const folder = path.dirname(this.deps.appImage)
    const downloadUrl = appImageDownloadUrl(info, this.deps.feedUrl)
    const download = downloadUrl ? `Download it from ${downloadUrl}` : 'Download the new AppImage'

    return {
      ok: false,
      mechanism: this.mechanism,
      error: 'appimage-not-writable',
      message:
        `Hermes ${info.version} can't be installed here because ${folder} is not writable. ` +
        `${download} and replace ${this.deps.appImage} with it, or move the AppImage to a folder you own ` +
        'and update again.',
      downloadUrl
    }
  }
}
