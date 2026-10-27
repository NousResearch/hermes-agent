import { expect, test } from 'vitest'

import { relaunchApp } from './app-relaunch'
import { LAUNCHER_READY_FD_ENV } from './linux-launcher-ready'

test('a self-relaunch never hands the launcher ready-pipe fd number to the new process', () => {
  const env: NodeJS.ProcessEnv = { [LAUNCHER_READY_FD_ENV]: '4', KEEP: '1' }
  const seen: Array<{ args?: string[]; carried: string | undefined }> = []

  relaunchApp({ relaunch: options => seen.push({ args: options.args, carried: env[LAUNCHER_READY_FD_ENV] }) }, { args: ['--disable-gpu'] }, env)

  expect(seen).toEqual([{ args: ['--disable-gpu'], carried: undefined }])
  expect(env.KEEP).toBe('1')
})
