import { describe, expect, it } from 'vitest'

import type { InstallStamp } from '../install-stamp'

import { ExternalStrategy } from './external'

import { resolveUpdaterMechanism, type UpdaterStrategy } from './index'

const message: string = "This build doesn't get updates. Ask the developer who gave it to you for a new build."

describe('one-commit artifacts', (): void => {
  it.each(['win32', 'darwin', 'linux'] as const)('cannot select an active updater on %s', (platform: NodeJS.Platform): void => {
    for (const updateMechanism of ['self', 'app-installer', 'electron-updater', 'microsoft-store', 'external'] as const) {
      expect(resolveUpdaterMechanism({ platform, updateMechanism, source: 'commit-build' })).toBe('external')
    }
  })

  it('refuses both checks and apply with no manual-update escape', async (): Promise<void> => {
    const stamp: InstallStamp = { source: 'commit-build' } as InstallStamp
    const strategy: UpdaterStrategy = new ExternalStrategy(stamp)
    expect(await strategy.check({ force: true })).toMatchObject({
      supported: false, mechanism: 'external', reason: 'commit-build', message
    })
    expect(await strategy.apply({ stopSafeBlockers: true })).toMatchObject({
      ok: false, mechanism: 'external', error: 'commit-build', message
    })
    expect(await strategy.apply({})).not.toHaveProperty('command')
  })
})
