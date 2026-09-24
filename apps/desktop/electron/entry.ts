import { readFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { app } from 'electron'

import { readDesktopLaunchConfig } from './renderer-heap-flags'
import { wslgLaunchArgs } from './wslg-launch'
import { spawnWslgLaunch } from './wslg-launch-process'

function configuredElectronFlags(env: NodeJS.ProcessEnv): string[] {
  const raw = env.HERMES_HOME

  const home = raw
    ? path.resolve(raw === '~' || raw.startsWith('~/') ? path.join(os.homedir(), raw.slice(1)) : raw)
    : env.HERMES_DESKTOP_USER_DATA_DIR
      ? path.join(path.resolve(env.HERMES_DESKTOP_USER_DATA_DIR), 'hermes-home')
      : path.join(os.homedir(), '.hermes')

  try {
    return readDesktopLaunchConfig(readFileSync(path.join(home, 'config.yaml'), 'utf8')).electronFlags
  } catch {
    return []
  }
}

const electronFlags = process.platform === 'linux' ? configuredElectronFlags(process.env) : []
const args = wslgLaunchArgs(process.argv.slice(1), process.env, process.platform, undefined, electronFlags)

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
