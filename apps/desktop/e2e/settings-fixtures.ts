import * as fs from 'node:fs'
import * as path from 'node:path'

import { _electron } from '@playwright/test'

import { buildAppEnv, findElectron, type Sandbox, writeEnvFile, writeMockProviderConfig } from './fixtures'
import { installErrorBannerGuard } from './test'

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')

export interface SettingsDesktopWindow extends Window {
  hermesDesktop: {
    getConnectionFor: (payload: { connectionId: string; profile: string; priority: 'background' }) => Promise<unknown>
    api: <T>(request: { connectionId: string; profile: string; path: string }) => Promise<T>
  }
}

export function seedSettingsProfile(home: string, mockUrl: string, maxTurns: number) {
  fs.mkdirSync(home, { recursive: true })
  writeMockProviderConfig(
    home,
    mockUrl,
    undefined,
    `agent:\n  max_turns: ${maxTurns}\nmemory:\n  memory_enabled: false\n  user_profile_enabled: false`
  )
  writeEnvFile(home)
}

export function settingsEnv(sandbox: Sandbox, home = sandbox.root) {
  const env = buildAppEnv(sandbox)

  const ownKeys = new Set([
    'HERMES_HOME',
    'HERMES_DESKTOP_USER_DATA_DIR',
    'HERMES_DESKTOP_IGNORE_EXISTING',
    'HERMES_DESKTOP_HERMES_ROOT',
    'HERMES_DESKTOP_APP_NAME',
    'HERMES_DESKTOP_SKIP_QUIT_CONFIRM'
  ])

  for (const key of Object.keys(env)) {
    if (key.startsWith('HERMES_') && !ownKeys.has(key)) {
      delete env[key]
    }
  }

  return {
    ...env,
    HOME: home,
    XDG_CONFIG_HOME: path.join(home, '.config'),
    XDG_CACHE_HOME: path.join(home, '.cache'),
    XDG_DATA_HOME: path.join(home, '.local', 'share'),
    XDG_STATE_HOME: path.join(home, '.local', 'state'),
    HERMES_DESKTOP_DEV_SERVER: ''
  }
}

export async function launchSettingsDesktop(sandbox: Sandbox) {
  const app = await _electron.launch({
    executablePath: findElectron(),
    chromiumSandbox: true,
    args: [
      DESKTOP_ROOT,
      '--disable-gpu',
      ...(process.platform === 'linux' && process.env.DISPLAY ? ['--ozone-platform=x11'] : [])
    ],
    env: settingsEnv(sandbox),
    cwd: DESKTOP_ROOT
  })

  const page = await app.firstWindow()
  installErrorBannerGuard(page)

  return { app, page }
}
