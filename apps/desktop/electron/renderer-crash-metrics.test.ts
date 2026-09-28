import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, beforeEach, describe, expect, test } from 'vitest'

import {
  MAX_PENDING_RENDERER_CRASHES,
  PENDING_RENDERER_CRASH_FILE,
  registerRendererCrashIpc,
  rendererCrashReason,
  RendererCrashRecorder
} from './renderer-crash-metrics'

let dir: string
let file: string

beforeEach(() => {
  dir = fs.mkdtempSync(path.join(process.env.TMPDIR || os.tmpdir(), 'hermes-renderer-crash-'))
  file = path.join(dir, PENDING_RENDERER_CRASH_FILE)
})

afterEach(() => fs.rmSync(dir, { recursive: true, force: true }))

function pending(): unknown {
  return JSON.parse(fs.readFileSync(file, 'utf8'))
}

function enabledRecorder(): RendererCrashRecorder {
  const recorder = new RendererCrashRecorder({ dir })
  recorder.setEnabled(true)

  return recorder
}

test('rendererCrashReason buckets Electron reasons', () => {
  expect(rendererCrashReason('crashed')).toBe('crash')
  expect(rendererCrashReason('oom')).toBe('oom')
  expect(rendererCrashReason('killed')).toBe('killed')

  for (const other of ['launch-failed', 'integrity-failure', 'abnormal-exit', undefined, 7, 'toString']) {
    expect(rendererCrashReason(other)).toBe('other')
  }
})

describe('RendererCrashRecorder', () => {
  test('records nothing until the user opts in', () => {
    const recorder = new RendererCrashRecorder({ dir })

    recorder.record('crashed')

    expect(fs.existsSync(file)).toBe(false)
    expect(recorder.take()).toBeNull()
  })

  test('persists only bucketed reasons, ignores clean-exit, and caps the list', () => {
    const recorder = enabledRecorder()

    recorder.record('crashed')
    recorder.record('clean-exit')
    recorder.record('oom')
    recorder.record('killed')
    recorder.record('launch-failed')

    expect(pending()).toEqual({ v: 1, reasons: ['crash', 'oom', 'killed', 'other'] })

    for (let i = 0; i < 30; i++) {
      recorder.record('killed')
    }

    const reasons = (pending() as { reasons: string[] }).reasons
    expect(reasons).toHaveLength(MAX_PENDING_RENDERER_CRASHES)
    expect(reasons.slice(0, 4)).toEqual(['crash', 'oom', 'killed', 'other'])
  })

  test('revoking consent deletes the pending file and stops recording', () => {
    const recorder = enabledRecorder()

    recorder.record('crashed')
    expect(fs.existsSync(file)).toBe(true)

    recorder.setEnabled(false)
    expect(fs.existsSync(file)).toBe(false)

    recorder.record('crashed')
    expect(fs.existsSync(file)).toBe(false)
  })

  test('take/ack: one claim at a time, a failed send keeps it, a sent ack keeps later crashes', () => {
    const recorder = enabledRecorder()

    recorder.record('crashed')
    recorder.record('oom')

    expect(recorder.take()).toEqual({ reasons: ['crash', 'oom'] })
    expect(recorder.take()).toBeNull()

    recorder.ack(false)
    expect(recorder.take()).toEqual({ reasons: ['crash', 'oom'] })

    recorder.record('killed')
    recorder.ack(true)

    expect(pending()).toEqual({ v: 1, reasons: ['killed'] })
    expect(recorder.take()).toEqual({ reasons: ['killed'] })

    recorder.ack(true)
    expect(fs.existsSync(file)).toBe(false)
    expect(recorder.take()).toBeNull()
  })

  test('a garbage or foreign file is treated as empty; unknown entries are dropped', () => {
    const recorder = enabledRecorder()

    fs.writeFileSync(file, 'not json {')
    expect(recorder.take()).toBeNull()

    fs.writeFileSync(file, JSON.stringify({ v: 2, reasons: ['crash'] }))
    expect(recorder.take()).toBeNull()

    fs.writeFileSync(file, JSON.stringify({ v: 1, reasons: ['crash', '/home/me/secret', 3, 'oom'] }))
    expect(recorder.take()).toEqual({ reasons: ['crash', 'oom'] })
  })

  test('fs errors never escape', () => {
    const boom = () => {
      throw new Error('EACCES')
    }

    const recorder = new RendererCrashRecorder({
      dir,
      fs: { mkdirSync: boom, readFileSync: boom, renameSync: boom, rmSync: boom, writeFileSync: boom } as never
    })

    recorder.setEnabled(true)
    expect(() => recorder.record('crashed')).not.toThrow()
    expect(recorder.take()).toBeNull()
    expect(() => recorder.setEnabled(false)).not.toThrow()
  })
})

test('registerRendererCrashIpc wires consent, take and ack', () => {
  const handlers = new Map<string, (event: unknown, ...args: unknown[]) => unknown>()
  const recorder = new RendererCrashRecorder({ dir })

  registerRendererCrashIpc({ handle: (channel, fn) => handlers.set(channel, fn) }, recorder)

  handlers.get('hermes:desktop-metrics:set-enabled')!({}, 'yes')
  recorder.record('crashed')
  expect(fs.existsSync(file)).toBe(false)

  handlers.get('hermes:desktop-metrics:set-enabled')!({}, true)
  recorder.record('crashed')
  expect(handlers.get('hermes:desktop-metrics:crash:take')!({})).toEqual({ reasons: ['crash'] })

  handlers.get('hermes:desktop-metrics:crash:ack')!({}, true)
  expect(fs.existsSync(file)).toBe(false)
})
