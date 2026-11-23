import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

import electronUpdater from 'electron-updater'
import type { AppAdapter } from 'electron-updater/out/AppAdapter'
import type { InstallOptions } from 'electron-updater/out/BaseUpdater'
import { load } from 'js-yaml'

import feedContract from '../../update-feed.cjs'
import { LAUNCHER_READY_FD_ENV } from '../linux-launcher-ready'
import { restoreLaunchTmpdir } from '../linux-sandbox-relaunch'
import { canRelaunch, mountFreeEnv, spawnRelaunch } from '../relaunch-target'
import { collectRelaunchArgs } from '../updater-process'

import { AppImageStrategy } from './appimage'
import { channelPublicBase } from './channel-protocol'
import { ExternalStrategy } from './external'
import type { MacClientDeps } from './mac-client'

import type { UpdaterStrategy } from './index'

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

/**
 * Put the download at destination in one rename, so a failure leaves whatever was there.
 * A download on another filesystem is first copied beside destination for the same reason.
 */
function moveIntoPlace(download: string, destination: string): void {
  try {
    fs.renameSync(download, destination)

    return
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'EXDEV') {
      throw error
    }
  }

  const staging: string = fs.mkdtempSync(path.join(path.dirname(destination), '.hermes-update-'))

  try {
    const staged: string = path.join(staging, path.basename(destination))
    fs.copyFileSync(download, staged, fs.constants.COPYFILE_FICLONE)
    fs.renameSync(staged, destination)
  } finally {
    fs.rmSync(staging, { recursive: true, force: true })
  }

  fs.rmSync(download, { force: true })
}

export class RelaunchingAppImageUpdater extends electronUpdater.AppImageUpdater {
  constructor(
    private readonly relaunch: (target: string, env: NodeJS.ProcessEnv) => void,
    app?: AppAdapter
  ) {
    super(undefined, app)
  }

  /**
   * electron-updater's install, except that it unlinks the AppImage before moving the download in,
   * so a failed move left no Hermes to start. Here the old file goes only once the new one is in place.
   */
  protected override doInstall(options: InstallOptions): boolean {
    const current: string | undefined = process.env.APPIMAGE
    const download: string | null = this.installerPath

    if (!current || !path.isAbsolute(current) || current.includes('\0') || !download) {
      throw new Error(`Cannot install the update: APPIMAGE (${current}) or the download (${download}) is missing.`)
    }

    // Upstream's naming: a name without a version is the user's own, so the update keeps it.
    const versioned: boolean = /\d+\.\d+\.\d+/.test(path.basename(current))
    const destination: string = versioned ? path.join(path.dirname(current), path.basename(download)) : current

    moveIntoPlace(download, destination)

    if (destination !== current) {
      fs.unlinkSync(current)
      this.emit('appimage-filename-updated', destination)
    }

    const env: NodeJS.ProcessEnv = { ...process.env, APPIMAGE_SILENT_INSTALL: 'true' }

    if (options.isForceRunAfter) {
      void this.spawnLog(destination, [], env)
    } else {
      execFileSync(destination, [], { env: { ...env, APPIMAGE_EXIT_AFTER_INSTALL: 'true' } })
    }

    return true
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
): UpdaterStrategy {
  const feed = feedContract.linuxFeed(deps.channel, arch)
  const override: string | null = deps.feedBaseUrl ? `${channelPublicBase(deps.feedBaseUrl)}/${feed.directory}/` : null
  const feedUrl: string | null = override ?? bakedFeedUrl(process.resourcesPath)

  // A build made without the feed URL (local, or a fork without the R2 vars) has no app-update.yml to check.
  if (!feedUrl) {
    return new ExternalStrategy()
  }

  updater.autoDownload = false
  updater.autoInstallOnAppQuit = false
  updater.autoRunAppAfterInstall = true
  updater.channel = feed.channel
  updater.allowPrerelease = feed.allowPrerelease
  // Setting channel enables downgrades in electron-updater. This app never does.
  updater.allowDowngrade = false
  updater.on('error', error => deps.log(`AppImage updater: ${error.message}`))

  if (override) {
    updater.setFeedURL({ provider: 'generic', url: override, channel: feed.channel })
  }

  return new AppImageStrategy({ ...deps, updater, appImage, feedUrl })
}
