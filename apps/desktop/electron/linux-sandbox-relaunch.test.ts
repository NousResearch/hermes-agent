import { describe, expect, it, vi } from 'vitest'

import { linuxNoSandboxRelaunch } from './linux-sandbox-relaunch'

const argv = ['/opt/Hermes/hermes', '--ozone-platform=x11', 'hermes://session/abc']

describe('Linux runtime --no-sandbox relaunch', () => {
  it('moves a runtime-only --no-sandbox onto argv exactly once', () => {
    const relaunch = linuxNoSandboxRelaunch({
      platform: 'linux',
      runtimeNoSandbox: true,
      argv,
      env: {},
      launchedWithoutSandbox: false
    })

    expect(relaunch).toEqual({ args: ['--ozone-platform=x11', 'hermes://session/abc', '--no-sandbox'] })

    // The relaunched process started unsandboxed, so it never restarts again.
    expect(
      linuxNoSandboxRelaunch({
        platform: 'linux',
        runtimeNoSandbox: true,
        argv: ['/opt/Hermes/hermes', ...relaunch!.args],
        env: {},
        launchedWithoutSandbox: true
      })
    ).toBeNull()
  })

  it('leaves sandboxed launches, launch-time opt-outs and other platforms alone', () => {
    expect(
      linuxNoSandboxRelaunch({
        platform: 'linux',
        runtimeNoSandbox: false,
        argv,
        env: {},
        launchedWithoutSandbox: false
      })
    ).toBeNull()
    expect(
      linuxNoSandboxRelaunch({ platform: 'linux', runtimeNoSandbox: true, argv, env: {}, launchedWithoutSandbox: true })
    ).toBeNull()
    // Windows reads the switch per child launch, so its runtime fallback works in place.
    expect(
      linuxNoSandboxRelaunch({
        platform: 'win32',
        runtimeNoSandbox: true,
        argv,
        env: {},
        launchedWithoutSandbox: false
      })
    ).toBeNull()
  })

  // The relaunch skips `hermes desktop`, so it must not inherit the long TMPDIR
  // main.ts restored after the lock: that path hangs requestSingleInstanceLock().
  it.skipIf(process.platform !== 'linux')('relaunches with the short socket TMPDIR the launcher chose', async () => {
    const saved = { ...process.env }
    const shortDir = '/run/user/1000/hermes-desktop'
    const longDir = `/home/u/.hermes/profiles/${'p'.repeat(80)}/cache/scratch`

    try {
      process.env.TMPDIR = shortDir
      process.env.HERMES_DESKTOP_TMPDIR = longDir
      delete process.env.ELECTRON_DISABLE_SANDBOX
      vi.resetModules()
      const { relaunchIfNoSandboxCameTooLate } = await import('./linux-sandbox-relaunch')

      // main.ts: the lock is held, the backend's directory comes back.
      process.env.TMPDIR = longDir
      delete process.env.HERMES_DESKTOP_TMPDIR

      let relaunchEnv: NodeJS.ProcessEnv = {}

      const relaunched = relaunchIfNoSandboxCameTooLate({
        commandLine: { hasSwitch: name => name === 'no-sandbox' },
        relaunch: () => {
          relaunchEnv = { ...process.env }
        },
        exit: () => undefined
      })

      expect(relaunched).toBe(true)
      expect(relaunchEnv.TMPDIR).toBe(shortDir)
      expect(relaunchEnv.HERMES_DESKTOP_TMPDIR).toBe(longDir)
    } finally {
      for (const key of Object.keys(process.env)) {
        if (!(key in saved)) {
          delete process.env[key]
        }
      }

      Object.assign(process.env, saved)
    }
  })

  it('relaunches the AppImage itself rather than its vanishing mount', () => {
    const relaunch = linuxNoSandboxRelaunch({
      platform: 'linux',
      runtimeNoSandbox: true,
      argv: ['/tmp/.mount_HermesX/hermes'],
      execPath: '/tmp/.mount_HermesX/hermes',
      env: { APPIMAGE: '/home/u/Apps/Hermes.AppImage', APPDIR: '/tmp/.mount_HermesX' },
      launchedWithoutSandbox: false
    })

    expect(relaunch).toEqual({ args: ['--no-sandbox'], execPath: '/home/u/Apps/Hermes.AppImage' })
  })

  // The type-2 runtime mounts under $TMPDIR/.mount_<name><random>; a launch that
  // lost APPDIR still runs from that mount, so APPIMAGE is still ours.
  it('relaunches the AppImage when only APPIMAGE survives but this runs from its mount', () => {
    const relaunch = linuxNoSandboxRelaunch({
      platform: 'linux',
      runtimeNoSandbox: true,
      argv: ['/run/user/1000/.mount_HermesQ7x2Lp/hermes'],
      execPath: '/run/user/1000/.mount_HermesQ7x2Lp/hermes',
      env: { APPIMAGE: '/home/u/Apps/Hermes.AppImage' },
      launchedWithoutSandbox: false
    })

    expect(relaunch).toEqual({ args: ['--no-sandbox'], execPath: '/home/u/Apps/Hermes.AppImage' })
  })

  // AppImage apps (terminals, IDEs) export APPIMAGE/APPDIR to everything they
  // start; Hermes launched from one must relaunch itself, not that app.
  it('relaunches this executable when APPIMAGE was inherited from another AppImage', () => {
    for (const env of [
      { APPIMAGE: '/home/u/Apps/Terminal.AppImage' },
      { APPIMAGE: '/home/u/Apps/Terminal.AppImage', APPDIR: '/tmp/.mount_TermY' },
      { APPIMAGE: '/home/u/Apps/Terminal.AppImage', APPDIR: '/opt/Herm' }
    ]) {
      expect(
        linuxNoSandboxRelaunch({
          platform: 'linux',
          runtimeNoSandbox: true,
          argv,
          execPath: '/opt/Hermes/hermes',
          env,
          launchedWithoutSandbox: false
        })
      ).toEqual({ args: ['--ozone-platform=x11', 'hermes://session/abc', '--no-sandbox'] })
    }
  })
})
