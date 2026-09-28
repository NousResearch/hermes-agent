import { readFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { app } from 'electron'

import { resolveDesktopHermesHome } from './data-paths'
import { readDesktopLaunchConfig } from './renderer-heap-flags'
import { wslgLaunchArgs } from './wslg-launch'
import { spawnWslgLaunch } from './wslg-launch-process'

function configuredElectronFlags(env: NodeJS.ProcessEnv): string[] {
  // Resolve the home exactly like main.ts does, through the shared resolver:
  // HERMES_DATA_DIR_SUFFIX channel installs and profiles/-rooted HERMES_HOME
  // values must pick the same config.yaml before the relaunch and inside the
  // app, or desktop.electron_flags silently never reaches the relaunch.
  const home = resolveDesktopHermesHome({
    home: os.homedir(),
    env,
    // Linux-only pre-launch path; the win32 legacy-migration probe is never
    // consulted on posix, so its directoryExists callback is not needed here.
    directoryExists: () => false,
    readWindowsHome: () => null
  })

  try {
    return readDesktopLaunchConfig(readFileSync(path.join(home, 'config.yaml'), 'utf8')).electronFlags
  } catch {
    return []
  }
}

const electronFlags = process.platform === 'linux' ? configuredElectronFlags(process.env) : []
const args = wslgLaunchArgs(process.argv.slice(1), process.env, process.platform, electronFlags)

if (args) {
  // Keep the launcher alive until the child exits: npm's concurrently must not
  // tear down Vite during this handoff. No backend, windows or single-instance
  // lock are created in this parent. The child has an explicit platform flag,
  // so it goes straight into main on its first pass.
  const child = spawnWslgLaunch(args)

  child.once('error', error => {
    console.error('[hermes] Wayland ozone launch failed:', error)
    app.exit(1)
  })
  child.once('exit', code => app.exit(code ?? 1))

  for (const signal of ['SIGINT', 'SIGTERM'] as const) {
    process.once(signal, () => child.kill(signal))
  }
} else {
  await import('./main')
}
