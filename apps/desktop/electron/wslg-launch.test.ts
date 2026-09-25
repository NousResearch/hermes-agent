import { describe, expect, it } from 'vitest'

import { wslgLaunchArgs } from './wslg-launch'

const env = { WSL_DISTRO_NAME: 'Ubuntu', WAYLAND_DISPLAY: 'wayland-0', DISPLAY: ':0' }

describe('WSLg launch arguments', () => {
  it('preserves the invocation and restarts only once with native Wayland', () => {
    const args = ['.', '--inspect=9229', 'hermes://session/example']
    const next = wslgLaunchArgs(args, env, 'linux')!

    expect(next).toEqual([...args, '--ozone-platform=wayland'])
    expect(wslgLaunchArgs(next, env, 'linux')).toBeNull()
    expect(args).toEqual(['.', '--inspect=9229', 'hermes://session/example'])
  })

  it('respects explicit backends and the existing config-bridged X11 hint', () => {
    for (const args of [['--ozone-platform=x11'], ['--ozone-platform', 'x11']]) {
      expect(wslgLaunchArgs(args, env, 'linux')).toBeNull()
    }

    expect(wslgLaunchArgs(['--ozone-platform-hint=x11'], env, 'linux')).toEqual([
      '--ozone-platform-hint=x11',
      '--ozone-platform=x11'
    ])
    expect(wslgLaunchArgs([], { ...env, HERMES_DESKTOP_DISABLE_GPU: 'true' }, 'linux')).toEqual([
      '--ozone-platform=wayland'
    ])
    expect(wslgLaunchArgs([], { ...env, ELECTRON_OZONE_PLATFORM_HINT: 'x11' }, 'linux')).toEqual([
      '--ozone-platform=x11'
    ])
  })

  it('leaves other platforms, missing Wayland displays and forwarded sessions alone', () => {
    expect(wslgLaunchArgs([], env, 'win32')).toBeNull()
    expect(wslgLaunchArgs([], env, 'darwin')).toBeNull()
    expect(wslgLaunchArgs([], { DISPLAY: ':0' }, 'linux')).toBeNull()
    expect(wslgLaunchArgs([], { WSL_DISTRO_NAME: 'Ubuntu' }, 'linux')).toBeNull()
    expect(wslgLaunchArgs([], { ...env, SSH_CONNECTION: 'remote' }, 'linux')).toBeNull()
    expect(wslgLaunchArgs([], { ...env, DISPLAY: 'localhost:10.0' }, 'linux')).toBeNull()
  })
})

const nativeWayland = { XDG_SESSION_TYPE: 'wayland', WAYLAND_DISPLAY: 'wayland-0', DISPLAY: ':0' }

describe('native Wayland launch arguments', () => {
  it('appends wayland on a native Linux Wayland session when the user did not choose a platform', () => {
    expect(wslgLaunchArgs(['.'], nativeWayland, 'linux')).toEqual(['.', '--ozone-platform=wayland'])
    expect(wslgLaunchArgs([], { WAYLAND_DISPLAY: 'wayland-0', DISPLAY: ':0' }, 'linux')).toEqual([
      '--ozone-platform=wayland'
    ])
    expect(wslgLaunchArgs([], { XDG_SESSION_TYPE: 'wayland', DISPLAY: ':0' }, 'linux')).toEqual([
      '--ozone-platform=wayland'
    ])
    expect(wslgLaunchArgs(['--ozone-platform=wayland'], nativeWayland, 'linux')).toBeNull()
  })

  it('keeps an explicit x11 platform already on the command line', () => {
    expect(wslgLaunchArgs(['--ozone-platform=x11'], nativeWayland, 'linux')).toBeNull()
    expect(wslgLaunchArgs(['--ozone-platform', 'x11'], nativeWayland, 'linux')).toBeNull()
  })

  it('lets an x11 hint win over the automatic wayland platform', () => {
    expect(wslgLaunchArgs([], { ...nativeWayland, ELECTRON_OZONE_PLATFORM_HINT: 'x11' }, 'linux')).toEqual([
      '--ozone-platform=x11'
    ])
    expect(wslgLaunchArgs(['--ozone-platform-hint=x11'], nativeWayland, 'linux')).toEqual([
      '--ozone-platform-hint=x11',
      '--ozone-platform=x11'
    ])
  })

  it('lets desktop.electron_flags choose the ozone platform instead of forcing wayland', () => {
    expect(wslgLaunchArgs([], nativeWayland, 'linux', ['--ozone-platform=x11'])).toEqual([
      '--ozone-platform=x11'
    ])
    expect(wslgLaunchArgs(['.'], nativeWayland, 'linux', ['--disable-gpu'])).toEqual([
      '.',
      '--ozone-platform=wayland'
    ])
    expect(wslgLaunchArgs([], { XDG_SESSION_TYPE: 'x11', DISPLAY: ':0' }, 'linux')).toBeNull()
    expect(wslgLaunchArgs([], { ...nativeWayland, SSH_CONNECTION: 'remote' }, 'linux')).toBeNull()
  })
})
