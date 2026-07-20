import { spawn, type SpawnOptions } from 'node:child_process'

import { hiddenWindowsChildOptions } from './windows-child-options'

export interface UpdaterChild {
  pid?: number
  unref: () => void
}

export interface SpawnUpdaterProcessDeps {
  isWindows?: boolean
  spawnProcess?: (command: string, args: string[], options: SpawnOptions) => UpdaterChild
}

/**
 * Spawn the detached installer used for update and bootstrap-recovery handoffs.
 * The helper owns both hidden-console selection and unref semantics so every
 * updater handoff follows the same behavior and can be tested without Electron.
 */
export function spawnUpdaterProcess(
  updater: string,
  updaterArgs: string[],
  options: SpawnOptions,
  deps: SpawnUpdaterProcessDeps = {}
): UpdaterChild {
  const isWindows = deps.isWindows ?? process.platform === 'win32'
  const spawnOptions = hiddenWindowsChildOptions(options, isWindows) as SpawnOptions

  const child = deps.spawnProcess
    ? deps.spawnProcess(updater, updaterArgs, spawnOptions)
    : spawn(updater, updaterArgs, spawnOptions)

  child.unref()

  return child
}
