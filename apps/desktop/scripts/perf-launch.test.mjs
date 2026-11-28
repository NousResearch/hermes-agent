import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, describe, expect, it, vi } from 'vitest'

import { wslgLaunchArgs } from '../electron/wslg-launch.ts'
import { startIsolatedInstance } from './perf/lib/launch.mjs'

const launch = vi.hoisted(() => ({ spawns: [], nvidia: false }))
vi.mock('node:module', () => ({ createRequire: () => () => 'fake-electron' }))
vi.mock('node:child_process', () => ({
  spawn: (file, args, options) => {
    if (file === 'fake-electron') launch.spawns.push({ args, env: options.env })
    return {
      on: (event, listener) => {
        if (event === 'exit') queueMicrotask(() => listener(0))
      },
      kill: () => {}
    }
  }
}))
vi.mock('node:fs', async importOriginal => {
  const actual = await importOriginal()
  return {
    ...actual,
    readFileSync: (file, ...args) => {
      if (file !== '/proc/driver/nvidia/version') return actual.readFileSync(file, ...args)
      if (launch.nvidia) return 'NVRM version: NVIDIA UNIX Kernel Module 615.0'
      throw Object.assign(new Error('No NVIDIA driver'), { code: 'ENOENT' })
    }
  }
})
vi.mock('./perf/lib/cdp.mjs', () => ({
  CDP: { connect: async () => ({ eval: async () => true, send: async () => {}, close: () => {} }) },
  requireDriver: async () => {},
  sleep: async () => {}
}))

afterEach(() => vi.unstubAllEnvs())

const wayland = { XDG_SESSION_TYPE: 'wayland', WAYLAND_DISPLAY: 'wayland-0', DISPLAY: ':0' }
describe.skipIf(process.platform !== 'linux')('perf first-spawn display selection', () => {
  it.each([
    ['native Wayland', wayland, false, '', 'wayland'],
    ['proprietary NVIDIA', wayland, true, '', 'x11'],
    ['X11 environment hint', { ...wayland, ELECTRON_OZONE_PLATFORM_HINT: 'x11' }, false, '', 'x11'],
    ['configured X11', wayland, false, 'desktop:\n  electron_flags: [--ozone-platform=x11]\n', 'x11'],
    ['forwarded X11', { ...wayland, DISPLAY: 'localhost:10.0' }, false, '', null],
    ['local X11', { DISPLAY: ':0' }, false, '', null]
  ])('honors %s without a CDP-port relaunch', async (_label, env, nvidia, config, ozone) => {
    launch.nvidia = nvidia
    launch.spawns.length = 0
    for (const key of [
      ...Object.keys(wayland),
      'ELECTRON_OZONE_PLATFORM_HINT',
      'SSH_CONNECTION',
      'SSH_CLIENT',
      'SSH_TTY'
    ]) {
      vi.stubEnv(key, env[key])
    }
    const home = mkdtempSync(join(tmpdir(), 'perf-launch-'))
    writeFileSync(join(home, 'config.yaml'), config)
    let instance
    try {
      instance = await startIsolatedInstance({ prod: true, coldStart: true, hermesHome: home, port: 41123 })
      expect(launch.spawns).toHaveLength(1)
      const [{ args, env: childEnv }] = launch.spawns
      expect(args).toContain('--remote-debugging-port=41123')
      expect(childEnv.HERMES_DESKTOP_CDP_PORT).toBe('41123')
      expect(childEnv.HERMES_HOME).toBe(home)
      expect(args.filter(arg => arg.startsWith('--ozone-platform='))).toEqual(
        ozone ? [`--ozone-platform=${ozone}`] : []
      )
      // Desktop's entrypoint must not need a second process holding the same port.
      expect(wslgLaunchArgs(args, childEnv, process.platform, [], nvidia)).toBeNull()
    } finally {
      instance?.teardown()
      rmSync(home, { recursive: true, force: true })
    }
  })
})
