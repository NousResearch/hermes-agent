import { describe, expect, it } from 'vitest'

import { linuxNoSandboxRelaunch } from './linux-sandbox-relaunch'

const argv = ['/opt/Hermes/hermes', '--ozone-platform=x11', 'hermes://session/abc']

describe('Linux runtime --no-sandbox relaunch', () => {
  it('moves a runtime-only --no-sandbox onto argv exactly once', () => {
    const relaunch = linuxNoSandboxRelaunch({ platform: 'linux', runtimeNoSandbox: true, argv, env: {}, launchedWithoutSandbox: false })

    expect(relaunch).toEqual({ args: ['--ozone-platform=x11', 'hermes://session/abc', '--no-sandbox'] })

    // The relaunched process started unsandboxed, so it never restarts again.
    expect(
      linuxNoSandboxRelaunch({ platform: 'linux', runtimeNoSandbox: true, argv: ['/opt/Hermes/hermes', ...relaunch!.args], env: {}, launchedWithoutSandbox: true })
    ).toBeNull()
  })

  it('leaves sandboxed launches, launch-time opt-outs and other platforms alone', () => {
    expect(linuxNoSandboxRelaunch({ platform: 'linux', runtimeNoSandbox: false, argv, env: {}, launchedWithoutSandbox: false })).toBeNull()
    expect(linuxNoSandboxRelaunch({ platform: 'linux', runtimeNoSandbox: true, argv, env: {}, launchedWithoutSandbox: true })).toBeNull()
    // Windows reads the switch per child launch, so its runtime fallback works in place.
    expect(linuxNoSandboxRelaunch({ platform: 'win32', runtimeNoSandbox: true, argv, env: {}, launchedWithoutSandbox: false })).toBeNull()
  })

  it('relaunches the AppImage itself rather than its vanishing mount', () => {
    const relaunch = linuxNoSandboxRelaunch({
      platform: 'linux',
      runtimeNoSandbox: true,
      argv: ['/tmp/.mount_HermesX/hermes'],
      env: { APPIMAGE: '/home/u/Apps/Hermes.AppImage' },
      launchedWithoutSandbox: false
    })

    expect(relaunch).toEqual({ args: ['--no-sandbox'], execPath: '/home/u/Apps/Hermes.AppImage' })
  })
})
