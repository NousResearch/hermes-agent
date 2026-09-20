import assert from 'node:assert/strict'
import { execFileSync, spawn as nodeSpawn, type SpawnOptions, spawnSync } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { resolve } from 'node:path'
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

test.skipIf(process.platform !== 'darwin')(
  'native state machine requires distinct keys, a clean chord and full release',
  () => {
    const dir = mkdtempSync(resolve(tmpdir(), 'hermes-command-monitor-test-'))

    try {
      const fixture = resolve(dir, 'gesture.m')
      const binary = resolve(dir, 'gesture')
      // Compile the real state machine; no posted input events, screen pixels or TCC prompts.
      writeFileSync(
        fixture,
        `
#define COMMAND_SCREENSHOT_MONITOR_TEST 1
#include "${resolve(import.meta.dirname, 'native/command-screenshot-monitor.m')}"
#include <assert.h>
static const CGEventFlags L = kCGEventFlagMaskCommand | NX_DEVICELCMDKEYMASK;
static const CGEventFlags R = kCGEventFlagMaskCommand | NX_DEVICERCMDKEYMASK;
static const CGEventFlags B = kCGEventFlagMaskCommand | NX_DEVICELCMDKEYMASK | NX_DEVICERCMDKEYMASK;
static bool flags(CommandGesture *s, CGKeyCode key, CGEventFlags f) {
  return CommandGestureUpdate(s, kCGEventFlagsChanged, key, f, false);
}
int main(void) { @autoreleasepool {
  CommandGesture s = {0};
  assert(!flags(&s, 55, L));
  assert(!flags(&s, 55, L)); // duplicate left Command is not the right Command
  assert(flags(&s, 54, B)); // capture immediately on the second physical press
  assert(!flags(&s, 54, L));
  assert(!flags(&s, 54, B));
  assert(!flags(&s, 54, L)); // one release does not re-arm
  assert(!flags(&s, 55, 0));
  assert(!flags(&s, 54, R));
  assert(flags(&s, 55, B));
  assert(!flags(&s, 55, R));
  assert(!flags(&s, 54, 0));
  for (int phase = 0; phase < 2; phase++) {
    s = (CommandGesture){0};
    if (phase == 0) CommandGestureUpdate(&s, kCGEventKeyDown, 0, 0, false);
    flags(&s, 55, L);
    if (phase == 1) CommandGestureUpdate(&s, kCGEventKeyDown, 0, L, false);
    assert(!flags(&s, 54, B));
    CommandGestureUpdate(&s, kCGEventKeyUp, 0, B, false);
    assert(!flags(&s, 54, L));
    assert(!flags(&s, 55, 0));
    flags(&s, 55, L);
    assert(flags(&s, 54, B));
    assert(!flags(&s, 54, L));
    flags(&s, 55, 0);
  }
  s = (CommandGesture){0};
  flags(&s, 55, L);
  CommandGestureUpdate(&s, kCGEventKeyUp, 0, L, false); // key was held before startup
  assert(!flags(&s, 54, B));
  s = (CommandGesture){0};
  flags(&s, 55, L);
  assert(flags(&s, 54, B));
  assert(!flags(&s, 54, B)); // repeat while held never triggers again
  assert(!CommandGestureUpdate(&s, kCGEventKeyDown, 0, B, true));
  assert(!flags(&s, 54, L)); // later keys cannot retroactively cancel the capture
  CGEventFlags modifiers[] = { kCGEventFlagMaskShift, kCGEventFlagMaskControl,
    kCGEventFlagMaskAlternate, kCGEventFlagMaskSecondaryFn, kCGEventFlagMaskAlphaShift };
  for (unsigned i = 0; i < sizeof(modifiers) / sizeof(*modifiers); i++) {
    s = (CommandGesture){0};
    flags(&s, 55, L | modifiers[i]);
    assert(!flags(&s, 54, B | modifiers[i]));
    flags(&s, 56, B);
    assert(!flags(&s, 54, L));
    assert(!flags(&s, 55, 0));
  }
  s = (CommandGesture){0};
  flags(&s, 55, kCGEventFlagMaskCommand);
  flags(&s, 54, kCGEventFlagMaskCommand);
  assert(!flags(&s, 54, 0)); // aggregate flags cannot prove two physical keys
  s = (CommandGesture){0};
  flags(&s, 54, B); // starting in the middle of a held chord must not arm
  assert(!flags(&s, 54, L));
  NSDictionary *(^window)(int, int, int, bool, double) = ^NSDictionary *(int wid, int pid, int layer, bool visible, double width) {
    return @{ (id)kCGWindowNumber: @(wid), (id)kCGWindowOwnerPID: @(pid),
      (id)kCGWindowLayer: @(layer), (id)kCGWindowIsOnscreen: @(visible), (id)kCGWindowAlpha: @1,
      (id)kCGWindowBounds: @{ @"X": @0, @"Y": @0, @"Width": @(width), @"Height": @200 } };
  };
  NSArray *windows = @[window(1, 42, 20, true, 100), window(2, 90, 0, true, 100),
    window(3, 42, 0, false, 100), window(4, 42, 0, true, 0),
    window(5, 42, 0, true, 300), window(6, 42, 0, true, 400)];
  NSDictionary *capture = CommandCaptureWindow(windows, 42);
  assert([capture[@"windowId"] intValue] == 5);
  assert([capture[@"width"] intValue] == 300 && [capture[@"height"] intValue] == 200);
  assert(CommandCaptureWindow(windows, 123) == nil); // never capture another app or a display
  assert(CommandCaptureWindow(windows, 0) == nil);
  puts("native gesture assertions passed");
} return 0; }
`
      )
      execFileSync(
        'xcrun',
        ['clang', '-fobjc-arc', '-fblocks', '-framework', 'Cocoa', '-framework', 'CoreGraphics', fixture, '-o', binary],
        { timeout: 30_000 }
      )
      assert.equal(
        execFileSync(binary, [], { encoding: 'utf8', timeout: 5_000 }).trim(),
        'native gesture assertions passed'
      )
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  },
  40_000
)

test.skipIf(process.platform !== 'darwin')(
  'builds a universal helper with a read-only permission check and real controller lifecycle',
  async () => {
    const dir = mkdtempSync(resolve(tmpdir(), 'hermes-command-monitor-build-'))

    try {
      const script = resolve(import.meta.dirname, '../scripts/build-command-screenshot-monitor.mjs')
      execFileSync(process.execPath, [script, '--out-dir', resolve(dir, 'dist')], { timeout: 60_000 })
      const binary = resolveCommandScreenshotMonitorPath(dir)
      const architectures = execFileSync('xcrun', ['lipo', '-archs', binary], { encoding: 'utf8' })
        .trim()
        .split(/\s+/)
        .sort()
      assert.deepEqual(architectures, ['arm64', 'x86_64'])
      const result = spawnSync(binary, ['--check'], { encoding: 'utf8', timeout: 5_000 })
      assert.equal(result.error, undefined)
      assert.equal(result.stderr, '')
      const permission = JSON.parse(result.stdout)
      assert.deepEqual(
        permission,
        result.status === 0 ? { type: 'ready' } : { type: 'error', code: 'permission-required' }
      )
      assert.ok(result.status === 0 || result.status === 2)
      const invalid = spawnSync(binary, ['--check', '--request-permission'], { encoding: 'utf8', timeout: 5_000 })
      assert.equal(invalid.status, 64)
      assert.deepEqual(JSON.parse(invalid.stdout), { type: 'error', code: 'unavailable' })
      let closed: Promise<NodeJS.Signals | null> | undefined

      const monitor = new CommandScreenshotMonitor({
        appPath: dir,
        spawn: (command, args, options) => {
          const child = nodeSpawn(command, args, options)
          closed = new Promise(settle => child.once('close', (_code, signal) => settle(signal)))

          return child
        }
      })

      const statuses: CommandScreenshotStatus[] = []

      const terminal = await new Promise<CommandScreenshotStatus>((settle, reject) => {
        const timeout = setTimeout(() => {
          monitor.stop()
          reject(new Error('real monitor did not settle'))
        }, 10_000)
        monitor.start(
          () => {},
          status => {
            statuses.push(status)

            if (status.type === 'ready' || status.type === 'error') {
              clearTimeout(timeout)
              settle(status)
            }
          }
        )
      })

      assert.equal(statuses[0].type, 'starting')

      if (permission.type === 'error') {
        assert.deepEqual(terminal, permission)
      }

      monitor.stop()
      assert.ok(closed)
      assert.notEqual(await closed, 'SIGKILL') // the real helper exits without escalation
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  },
  75_000
)
