import assert from 'node:assert/strict'
import test from 'node:test'

import { teardownOrder } from './run-short-session-hang-repro.mjs'

test('teardown signals the browser process alone, never its children first', () => {
  // A realistic Electron tree: the browser plus the children whose death it
  // treats as fatal (zygote, GPU process).
  const captured = [
    { command: 'electron .', pid: 100, ppid: 1 },
    { command: 'electron --type=zygote', pid: 101, ppid: 100 },
    { command: 'electron --type=gpu-process', pid: 102, ppid: 101 },
    { command: 'electron --type=renderer', pid: 103, ppid: 101 }
  ]

  const { root, children } = teardownOrder(captured, 100)

  // The first signal must reach the browser and nothing else. Killing the
  // zygote or GPU process while the browser lives trips
  // "GPU process isn't usable. Goodbye." — a SIGTRAP core dump manufactured
  // out of an ordinary shutdown.
  assert.deepEqual(root.map(row => row.pid), [100])
  assert.deepEqual(children.map(row => row.pid), [101, 102, 103])
})

test('teardown still names a root that process discovery never observed', () => {
  // processTree() synthesises the root row when `ps` cannot see it. The
  // synthetic root must still be signalled first rather than dropped, or the
  // sweep becomes children-first again.
  const captured = [
    { command: '<synthetic-root>', pid: 200, ppid: 0, synthetic: true },
    { command: 'electron --type=zygote', pid: 201, ppid: 200 }
  ]

  const { root, children } = teardownOrder(captured, 200)

  assert.deepEqual(root.map(row => row.pid), [200])
  assert.deepEqual(children.map(row => row.pid), [201])
})
