import { describe, expect, it } from 'vitest'

import type { InstallStamp } from '../install-stamp'

import { ExternalStrategy } from './external'

import { resolveUpdaterMechanism, type UpdaterStrategy } from './index'

const message: string = "This build doesn't get updates. Ask the developer who gave it to you for a new build."

describe('one-commit artifacts', (): void => {
  it.each(['win32', 'darwin', 'linux'] as const)(
    'cannot select an active updater on %s',
    (platform: NodeJS.Platform): void => {
      for (const updateMechanism of [
        'self',
        'app-installer',
        'electron-updater',
        'microsoft-store',
        'external'
      ] as const) {
        expect(resolveUpdaterMechanism({ platform, updateMechanism, source: 'commit-build' })).toBe('external')
      }
    }
  )

  it('refuses both checks and apply with no manual-update escape', async (): Promise<void> => {
    const stamp: InstallStamp = { source: 'commit-build' } as InstallStamp
    const strategy: UpdaterStrategy = new ExternalStrategy(stamp)
    expect(await strategy.check({ force: true })).toMatchObject({
      supported: false,
      mechanism: 'external',
      reason: 'commit-build',
      message
    })
    expect(await strategy.apply()).toMatchObject({
      ok: false,
      mechanism: 'external',
      error: 'commit-build',
      message
    })
    expect(await strategy.apply()).not.toHaveProperty('command')
  })
})

describe('a packaged Linux install (.deb / AppImage / rpm)', (): void => {
  // #86987: the .deb lands via dpkg/apt, so the package manager owns the
  // update loop. The stamp declares 'external' (write-build-stamp.mjs) and
  // the strategy must never offer an in-place swap the package cannot
  // honor — check reports the package owner, apply stays manual-only.
  it('declares an external owner and never self-updates in place', async (): Promise<void> => {
    const stamp: InstallStamp = { payload: 'bundled', updateMechanism: 'external' } as InstallStamp
    expect(resolveUpdaterMechanism({ platform: 'linux', updateMechanism: stamp.updateMechanism })).toBe('external')

    const strategy: UpdaterStrategy = new ExternalStrategy(stamp)
    const status = await strategy.check({ force: true })
    expect(status).toMatchObject({
      supported: false,
      mechanism: 'external',
      reason: 'bundled-not-appinstaller'
    })

    const apply = await strategy.apply()
    expect(apply).toMatchObject({ ok: true, manual: true, mechanism: 'external' })
    // No handed-off relaunch: the running package is never swapped by the app.
    expect(apply.handedOff).not.toBe(true)
  })
})
