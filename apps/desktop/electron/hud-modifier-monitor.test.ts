import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { test, vi } from 'vitest'

import { hudModifierMonitorSupported, resolveHudModifierMonitorPath } from './hud-modifier-monitor'
import { NativeGestureMonitor, type NativeGestureStatus } from './native-gesture-monitor'

class Child extends EventEmitter {
  stdin = new PassThrough()
  stdout = new PassThrough()
  kill = vi.fn((_signal?: NodeJS.Signals) => true)
}

test('native transport gates gestures on ready, bounds framing and isolates replacement children', () => {
  vi.useFakeTimers()

  try {
    const first = new Child()
    const second = new Child()
    const spawn = vi.fn().mockReturnValueOnce(first).mockReturnValueOnce(second)
    const statuses: NativeGestureStatus[] = []
    let summons = 0

    const monitor = new NativeGestureMonitor({
      path: '/helper',
      parseGesture: value => (value.type === 'summon' ? true : null),
      spawn,
      startupTimeoutMs: 100,
      stopTimeoutMs: 50
    })

    monitor.start(
      () => summons++,
      status => statuses.push(status)
    )
    assert.deepEqual(spawn.mock.calls[0][1], [])
    first.stdout.write('{"type":"summon"}\n{"type":"rea')
    assert.equal(summons, 0)
    first.stdout.write('dy"}\n{"type":"key","value":"discard"}\n{"type":"summon"}\n')
    assert.equal(summons, 1)
    monitor.start(
      () => summons++,
      status => statuses.push(status),
      true
    )
    assert.deepEqual(spawn.mock.calls[1][1], ['--request-permission'])
    first.stdout.write('{"type":"summon"}\n')
    first.emit('close', 0, null)
    second.stdout.write('x'.repeat(4097))
    assert.deepEqual(statuses.at(-1), { type: 'error', code: 'unavailable' })
    vi.advanceTimersByTime(50)
    assert.deepEqual(second.kill.mock.calls, [['SIGTERM'], ['SIGKILL']])
    second.emit('close', 0, null)
    assert.equal(summons, 1)
    assert.equal(vi.getTimerCount(), 0)
  } finally {
    vi.useRealTimers()
  }
})

test('support policy excludes Wayland even when Xwayland supplies DISPLAY', () => {
  assert.equal(hudModifierMonitorSupported('linux', { DISPLAY: ':0', WAYLAND_DISPLAY: 'wayland-0' }), false)
  assert.equal(hudModifierMonitorSupported('linux', { DISPLAY: ':0', XDG_SESSION_TYPE: 'wayland' }), false)
  assert.equal(hudModifierMonitorSupported('linux', { DISPLAY: ':0', XDG_SESSION_TYPE: 'x11' }), true)
  assert.equal(hudModifierMonitorSupported('linux', {}), false)
  assert.equal(hudModifierMonitorSupported('darwin', {}), true)
  assert.equal(hudModifierMonitorSupported('win32', {}), true)
  assert.equal(hudModifierMonitorSupported('freebsd', {}), false)
  const path = resolveHudModifierMonitorPath('/apps/app.asar')
  assert.ok(path.includes('app.asar.unpacked'))
  assert.ok(path.endsWith(process.platform === 'win32' ? 'hud-modifier-monitor.exe' : 'hud-modifier-monitor'))
})

test('a missing helper is distinguished from a helper that cannot start', () => {
  for (const code of ['ENOENT', 'EACCES']) {
    const child = new Child()
    const statuses: NativeGestureStatus[] = []
    const monitor = new NativeGestureMonitor({ path: '/helper', parseGesture: () => null, spawn: () => child })
    monitor.start(() => {}, status => statuses.push(status))
    child.emit('error', Object.assign(new Error('spawn failed'), { code }))
    child.emit('close', -1, null)
    assert.deepEqual(statuses.at(-1), {
      type: 'error',
      code: 'unavailable',
      ...(code === 'ENOENT' ? { reason: 'missing-helper' } : {})
    })
  }
})
