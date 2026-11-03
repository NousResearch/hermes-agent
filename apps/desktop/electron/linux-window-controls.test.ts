import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test, vi } from 'vitest'

import {
  isTilingSession,
  linuxWindowControlsHidden,
  readLinuxWindowControlsMode,
  registerLinuxWindowControlsIpc,
  setupLinuxWindowControls
} from './linux-window-controls'

const TILING_ENVS = [
  { XDG_CURRENT_DESKTOP: 'niri' },
  { XDG_CURRENT_DESKTOP: 'Hyprland' },
  { XDG_CURRENT_DESKTOP: 'niri:GNOME' },
  { XDG_SESSION_DESKTOP: 'sway' },
  { NIRI_SOCKET: '/run/user/1000/niri.sock' },
  { SWAYSOCK: '/run/user/1000/sway.sock' },
  { I3SOCK: '/run/user/1000/i3.sock' },
  { HYPRLAND_INSTANCE_SIGNATURE: 'abc' }
]

const FLOATING_ENVS = [{ XDG_CURRENT_DESKTOP: 'KDE' }, { XDG_CURRENT_DESKTOP: 'ubuntu:GNOME' }, {}]

test('tiling sessions are recognised from desktop names and compositor sockets', () => {
  for (const env of TILING_ENVS) {
    assert.equal(isTilingSession(env), true, JSON.stringify(env))
  }

  for (const env of FLOATING_ENVS) {
    assert.equal(isTilingSession(env), false, JSON.stringify(env))
  }
})

test('auto follows the session, while show and hide override it', () => {
  const hidden = (mode: 'auto' | 'hide' | 'show', env: NodeJS.ProcessEnv) =>
    linuxWindowControlsHidden({ env, isWsl: false, mode, platform: 'linux' })

  assert.equal(hidden('auto', { XDG_CURRENT_DESKTOP: 'niri' }), true)
  assert.equal(hidden('auto', { XDG_CURRENT_DESKTOP: 'KDE' }), false)
  assert.equal(hidden('show', { XDG_CURRENT_DESKTOP: 'niri' }), false)
  assert.equal(hidden('hide', { XDG_CURRENT_DESKTOP: 'KDE' }), true)
})

test('controls are never hidden outside native Linux', () => {
  const env = { XDG_CURRENT_DESKTOP: 'niri' }

  assert.equal(linuxWindowControlsHidden({ env, isWsl: true, mode: 'hide', platform: 'linux' }), false)
  assert.equal(linuxWindowControlsHidden({ env, isWsl: false, mode: 'hide', platform: 'win32' }), false)
  assert.equal(linuxWindowControlsHidden({ env, isWsl: false, mode: 'hide', platform: 'darwin' }), false)
})

test('the saved mode round-trips, and a bad file or value falls back safely', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-window-controls-'))
  const configPath = path.join(dir, 'nested', 'window-controls.json')
  const handlers = new Map<string, (event: unknown, mode: unknown) => unknown>()

  registerLinuxWindowControlsIpc(
    { handle: (channel, listener) => handlers.set(channel, listener) },
    configPath,
    mode => ({
      mode,
      restartNeeded: mode === 'hide'
    })
  )

  const get = handlers.get('hermes:window-controls:mode:get')!
  const set = handlers.get('hermes:window-controls:mode:set')!

  assert.deepEqual(get(null, undefined), { mode: 'auto', restartNeeded: false })

  assert.deepEqual(set(null, 'hide'), { mode: 'hide', restartNeeded: true })
  assert.equal(readLinuxWindowControlsMode(configPath), 'hide')

  assert.throws(() => set(null, 'sideways'), /Unknown window controls mode: sideways/)
  assert.equal(readLinuxWindowControlsMode(configPath), 'hide')

  fs.writeFileSync(configPath, '{not json')
  assert.equal(readLinuxWindowControlsMode(configPath), 'auto')

  fs.rmSync(dir, { force: true, recursive: true })
})

afterEach(() => {
  vi.unstubAllEnvs()
})

test.runIf(process.platform === 'linux')('a restart is asked for only when the buttons would really change', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-window-controls-'))
  const handlers = new Map<string, (event: unknown, mode: unknown) => unknown>()

  for (const name of ['HYPRLAND_INSTANCE_SIGNATURE', 'I3SOCK', 'NIRI_SOCKET', 'SWAYSOCK', 'XDG_SESSION_DESKTOP']) {
    vi.stubEnv(name, '')
  }

  vi.stubEnv('XDG_CURRENT_DESKTOP', 'KDE')

  const hidden = setupLinuxWindowControls(
    { handle: (channel, listener) => handlers.set(channel, listener) },
    dir,
    false
  )

  const set = handlers.get('hermes:window-controls:mode:set')!

  assert.equal(hidden, false)
  assert.deepEqual(set(null, 'show'), { mode: 'show', restartNeeded: false })
  assert.deepEqual(set(null, 'hide'), { mode: 'hide', restartNeeded: true })
  assert.deepEqual(set(null, 'auto'), { mode: 'auto', restartNeeded: false })

  fs.rmSync(dir, { force: true, recursive: true })
})
