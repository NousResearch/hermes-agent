import { ownAppImage } from '../relaunch-target'

import { type AppImageClientDeps, createAppImageStrategy } from './appimage-client'
import type { ChannelTarget } from './channel'
import { ExternalStrategy } from './external'
import { createChannelMacStrategy, createMacStrategy, type MacClientDeps } from './mac-client'

import type { UpdaterStrategy } from './index'

/**
 * What main.ts gives both platform clients. The macOS channel-build options (feed, expectedVersion, verifyDownload)
 * are filled in by createChannelMacStrategy, so they never reach the AppImage client.
 */
export type ElectronUpdaterClientDeps = AppImageClientDeps & Pick<MacClientDeps, 'light'>

export interface ElectronUpdaterHost {
  platform: NodeJS.Platform
  execPath: string
  env: NodeJS.ProcessEnv
}

/**
 * The stamp names electron-updater for every Linux bundle, but only an AppImage can replace itself:
 * the same build run unpacked, or repackaged, keeps its package owner.
 */
export function createElectronUpdaterStrategy(
  deps: ElectronUpdaterClientDeps,
  target?: ChannelTarget,
  host: ElectronUpdaterHost = { platform: process.platform, execPath: process.execPath, env: process.env },
  createAppImage: typeof createAppImageStrategy = createAppImageStrategy
): UpdaterStrategy {
  if (host.platform !== 'linux') {
    return target ? createChannelMacStrategy(deps, target) : createMacStrategy(deps)
  }

  const appImage: string | undefined = ownAppImage(host.execPath, host.env)

  return appImage ? createAppImage(deps, appImage) : new ExternalStrategy()
}
