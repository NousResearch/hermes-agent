import assert from 'node:assert/strict'
import { type SpawnOptions } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { test, vi } from 'vitest'

import {
  type CommandScreenshotCapture,
  CommandScreenshotMonitor,
  type CommandScreenshotStatus,
  resolveCommandScreenshotMonitorPath
} from './command-screenshot-monitor'

class FakeChild extends EventEmitter {
  stdin = new PassThrough()
  stdout = new PassThrough()
  stderr = null
  kill = vi.fn((_signal?: NodeJS.Signals) => true)
}

test('launches the unpacked helper without prompting and delivers only validated capture messages', () => {
  const child = new FakeChild()
  const spawn = vi.fn((_command: string, _args: string[], _options: SpawnOptions) => child)
  const captures: CommandScreenshotCapture[] = []
  const statuses: CommandScreenshotStatus[] = []

  const monitor = new CommandScreenshotMonitor({
    platform: 'darwin',
    appPath: '/Applications/Hermes.app/Contents/Resources/app.asar',
    spawn
  })

  monitor.start(
    value => captures.push(value),
    value => statuses.push(value)
  )
  assert.equal(spawn.mock.calls.length, 1)
  assert.deepEqual(spawn.mock.calls[0], [
    '/Applications/Hermes.app/Contents/Resources/app.asar.unpacked/dist/native/command-screenshot-monitor',
    [],
    { stdio: ['pipe', 'pipe', 'ignore'], shell: false, detached: false, windowsHide: true }
  ])
  child.stdout.write('{"type":"capture","windowId":2,"width":100,"height":200}\n')
  assert.deepEqual(captures, []) // No capture until readiness is established.
  child.stdout.write('{"type":"rea')
  child.stdout.write('dy"}\nnot json\n{"type":"key","key":"private"}\n')
  child.stdout.write('{"type":"capture","windowId":0,"width":100,"height":200}\n')
  child.stdout.write('{"type":"capture","windowId":2,"width":-1,"height":200}\n')
  child.stdout.write('{"type":"capture","windowId":2,"width":100,"height":200,"private":"discard"}\n')
  assert.deepEqual(captures, [{ type: 'capture', windowId: 2, width: 100, height: 200 }])
  assert.deepEqual(statuses, [{ type: 'starting' }, { type: 'ready' }])
  monitor.stop()
  child.emit('close', 0, null)
  assert.equal(child.kill.mock.calls[0]?.[0], 'SIGTERM')
  assert.equal(child.stdout.listenerCount('data'), 0)
  assert.deepEqual(statuses.at(-1), { type: 'stopped' })
  assert.equal(resolveCommandScreenshotMonitorPath('/tmp/dev'), '/tmp/dev/dist/native/command-screenshot-monitor')
})

test('bounds startup and termination, preserves permission failures, and isolates restarts', () => {
  vi.useFakeTimers()

  try {
    const first = new FakeChild()
    const second = new FakeChild()
    const spawn = vi.fn().mockReturnValueOnce(first).mockReturnValueOnce(second)
    const statuses: CommandScreenshotStatus[] = []
    const captures: CommandScreenshotCapture[] = []

    const monitor = new CommandScreenshotMonitor({
      platform: 'darwin',
      spawn,
      startupTimeoutMs: 100,
      stopTimeoutMs: 50
    })

    monitor.start(
      value => captures.push(value),
      value => statuses.push(value),
      true
    )
    assert.deepEqual(spawn.mock.calls[0][1], ['--request-permission'])
    first.stdout.write('{"type":"error","code":"permission-required"}\n')
    assert.deepEqual(statuses.at(-1), { type: 'error', code: 'permission-required' })
    assert.equal(first.stdin.writableEnded, true)
    monitor.start(
      value => captures.push(value),
      value => statuses.push(value)
    )
    first.stdout.write('{"type":"ready"}\n{"type":"capture","windowId":1,"width":1,"height":1}\n')
    assert.deepEqual(captures, [])
    vi.advanceTimersByTime(50)
    assert.deepEqual(first.kill.mock.calls, [['SIGTERM'], ['SIGKILL']])
    first.emit('close', null, 'SIGKILL')
    assert.deepEqual(statuses.at(-1), { type: 'starting' })
    vi.advanceTimersByTime(50)
    assert.deepEqual(statuses.at(-1), { type: 'error', code: 'unavailable' })
    assert.equal(second.kill.mock.calls[0]?.[0], 'SIGTERM')
    second.emit('close', 0, null)
    assert.equal(vi.getTimerCount(), 0)
    assert.equal(second.listenerCount('error'), 0)
  } finally {
    vi.useRealTimers()
  }
})

test('stopping from the starting callback cancels the child before it can become ready', () => {
  const child = new FakeChild()
  const statuses: CommandScreenshotStatus[] = []
  const monitor = new CommandScreenshotMonitor({ platform: 'darwin', spawn: () => child })
  monitor.start(
    () => assert.fail('stopped monitor delivered a capture'),
    status => {
      statuses.push(status)

      if (status.type === 'starting') {
        monitor.stop()
      }
    }
  )
  assert.equal(child.kill.mock.calls[0]?.[0], 'SIGTERM')
  child.stdout.write('{"type":"ready"}\n')
  child.emit('close', 0, null)
  assert.deepEqual(statuses, [{ type: 'starting' }, { type: 'stopped' }])
})
