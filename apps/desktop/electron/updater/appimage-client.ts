import fs from 'node:fs'
import path from 'node:path'

import electronUpdater from 'electron-updater'
import type { AppAdapter } from 'electron-updater/out/AppAdapter'
import { load } from 'js-yaml'

import feedContract from '../../update-feed.cjs'
import { LAUNCHER_READY_FD_ENV } from '../linux-launcher-ready'
import { restoreLaunchTmpdir } from '../linux-sandbox-relaunch'
import { canRelaunch, mountFreeEnv, spawnRelaunch } from '../relaunch-target'
import { collectRelaunchArgs } from '../updater-process'

import { AppImageStrategy } from './appimage'
import { channelPublicBase } from './channel-protocol'
import type { MacClientDeps } from './mac-client'

export interface RelaunchHost {
  argv: string[]
  execPath: string
  spawn: typeof spawnRelaunch
}

/**
 * Start the new AppImage once this process is gone, like every other self-relaunch on Linux
 * (app-relaunch.ts): started at once, as electron-updater does, it would lose the single-instance
 * lock to this process and quit, and it would inherit this image's mount entries.
 */
export function relaunchUpdatedAppImage(
  target: string,
  env: NodeJS.ProcessEnv,
  host: RelaunchHost = { argv: process.argv, execPath: process.execPath, spawn: spawnRelaunch }
): void {
  const childEnv: NodeJS.ProcessEnv = mountFreeEnv({ ...env }, target, host.execPath)
  delete childEnv[LAUNCHER_READY_FD_ENV]
  restoreLaunchTmpdir(childEnv)

  // A launch without FUSE (--appimage-extract-and-run) must start the new image the same way.
  if (env.APPDIR && path.basename(env.APPDIR).startsWith('appimage_extracted_')) {
    childEnv.APPIMAGE_EXTRACT_AND_RUN = '1'
  }

  if (!canRelaunch(target, 'not restarting', childEnv)) {
    throw new Error(`Hermes was updated, but ${target} cannot be started. Start it yourself.`)
  }

  host.spawn(target, collectRelaunchArgs(host.argv.slice(1)), childEnv, host.execPath)
}

export class RelaunchingAppImageUpdater extends electronUpdater.AppImageUpdater {
  constructor(
    private readonly relaunch: (target: string, env: NodeJS.ProcessEnv) => void,
    app?: AppAdapter
  ) {
    super(undefined, app)
  }

  protected override spawnLog(command: string, _args?: string[], env?: NodeJS.ProcessEnv): Promise<boolean> {
    this.relaunch(command, env ?? process.env)

    return Promise.resolve(true)
  }
}

/** The feed electron-builder baked into app-update.yml, which the updater uses without an override. */
function bakedFeedUrl(resourcesPath: string): string | null {
  try {
    const config: unknown = load(fs.readFileSync(path.join(resourcesPath, 'app-update.yml'), 'utf8'))
    const url: unknown = config && typeof config === 'object' && 'url' in config ? config.url : null

    return typeof url === 'string' ? url : null
  } catch {
    return null
  }
}

export function createAppImageStrategy(
  deps: MacClientDeps,
  appImage: string,
  arch: string = process.arch,
  updater: InstanceType<typeof electronUpdater.AppImageUpdater> = new RelaunchingAppImageUpdater(
    relaunchUpdatedAppImage
  )
): AppImageStrategy {
  const feed = feedContract.linuxFeed(deps.channel, arch)

  updater.autoDownload = false
  updater.autoInstallOnAppQuit = false
  updater.autoRunAppAfterInstall = true
  updater.channel = feed.channel
  updater.allowPrerelease = feed.allowPrerelease
  // Setting channel enables downgrades in electron-updater. This app never does.
  updater.allowDowngrade = false
  updater.on('error', error => deps.log(`AppImage updater: ${error.message}`))

  const override: string | null = deps.feedBaseUrl ? `${channelPublicBase(deps.feedBaseUrl)}/${feed.directory}/` : null

  if (override) {
    updater.setFeedURL({ provider: 'generic', url: override, channel: feed.channel })
  }

  return new AppImageStrategy({ ...deps, updater, appImage, feedUrl: override ?? bakedFeedUrl(process.resourcesPath) })
}
