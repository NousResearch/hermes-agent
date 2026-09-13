import { autoUpdater as nativeUpdater } from 'electron'
import electronUpdater from 'electron-updater'

import feedContract from '../../update-feed.cjs'

import { MacStrategy, type MacStrategyDeps, prepareMacInstall } from './mac'

export interface MacClientDeps extends Omit<MacStrategyDeps, 'updater' | 'prepareInstall'> {
  light: boolean
  feedBaseUrl: string
  log: (message: string) => void
}

export function createMacStrategy(deps: MacClientDeps): MacStrategy {
  const feed = feedContract.darwinFeed(deps.channel, deps.light)
  const updater = new electronUpdater.MacUpdater()
  updater.autoDownload = false
  updater.autoInstallOnAppQuit = false
  updater.autoRunAppAfterInstall = true
  updater.channel = feed.channel
  updater.allowPrerelease = feed.allowPrerelease
  // Setting channel enables downgrades in electron-updater. This app never does.
  updater.allowDowngrade = false
  updater.on('error', error => deps.log(`macOS updater: ${error.message}`))

  if (deps.feedBaseUrl) {
    const base = new URL(deps.feedBaseUrl)

    if (base.protocol !== 'https:' && !(base.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(base.hostname))) {
      throw new Error('The update feed must use HTTPS or a loopback HTTP address.')
    }

    updater.setFeedURL({ provider: 'generic', url: `${base.href.replace(/\/+$/, '')}/${feed.directory}/`, channel: feed.channel })
  }

  return new MacStrategy({ ...deps, updater, prepareInstall: () => prepareMacInstall(nativeUpdater) })
}
