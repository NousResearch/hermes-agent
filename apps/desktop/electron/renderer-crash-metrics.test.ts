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

const W = 1

function enabledRecorder(): RendererCrashRecorder {
  const recorder = new RendererCrashRecorder({ dir })
  recorder.setEnabled(W, 'local|default', true)

  return recorder
}

const reasonsOnDisk = () => (pending() as { entries: { reason: string }[] }).entries.map(entry => entry.reason)

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

    recorder.record(W, 'crashed')

    expect(fs.existsSync(file)).toBe(false)
    expect(recorder.take(W)).toBeNull()
  })

  test('persists only bucketed reasons, ignores clean-exit, and caps the list', () => {
    const recorder = enabledRecorder()

    for (const reason of ['crashed', 'clean-exit', 'oom', 'killed', 'launch-failed']) {
      recorder.record(W, reason)
    }

    expect(reasonsOnDisk()).toEqual(['crash', 'oom', 'killed', 'other'])
    expect(JSON.stringify(pending())).not.toContain('default')

    for (let i = 0; i < 30; i++) {
      recorder.record(W, 'killed')
    }

    const reasons = reasonsOnDisk()
    expect(reasons).toHaveLength(MAX_PENDING_RENDERER_CRASHES)
    expect(reasons.slice(0, 4)).toEqual(['crash', 'oom', 'killed', 'other'])
  })

  test('revoking consent deletes the pending file and stops recording', () => {
    const recorder = enabledRecorder()

    recorder.record(W, 'crashed')
    expect(fs.existsSync(file)).toBe(true)

    recorder.setEnabled(W, 'local|default', false)
    expect(fs.existsSync(file)).toBe(false)

    recorder.record(W, 'crashed')
    expect(fs.existsSync(file)).toBe(false)
  })

  test('consent is per window and entries per profile: another profile neither records, drains nor purges them', () => {
    const recorder = new RendererCrashRecorder({ dir })
    const [a, b] = [1, 2]

    recorder.setEnabled(a, 'local|alpha', false) // window a: profile alpha, collection off
    recorder.setEnabled(b, 'local|beta', true) // window b: profile beta, on
    recorder.record(a, 'crashed')
    expect(fs.existsSync(file)).toBe(false)

    recorder.setEnabled(a, 'local|alpha', true)
    recorder.record(a, 'oom')
    expect(recorder.take(b)).toBeNull()

    recorder.setEnabled(b, 'local|beta', false)
    expect(recorder.take(a)).toEqual({ reasons: ['oom'] })
  })

  test('take/ack: one claim at a time, a failed send keeps it, a sent ack keeps later crashes', () => {
    const recorder = enabledRecorder()

    const peer = 2

    recorder.setEnabled(peer, 'local|default', true)
    recorder.record(W, 'crashed')
    recorder.record(W, 'oom')

    expect(recorder.take(W)).toEqual({ reasons: ['crash', 'oom'] })
    expect(recorder.take(peer)).toBeNull()

    recorder.ack(W, false)
    expect(recorder.take(W)).toEqual({ reasons: ['crash', 'oom'] })

    recorder.record(W, 'killed')
    recorder.ack(W, true)

    expect(reasonsOnDisk()).toEqual(['killed'])
    expect(recorder.take(W)).toEqual({ reasons: ['killed'] })

    recorder.ack(W, true)
    expect(fs.existsSync(file)).toBe(false)
    expect(recorder.take(W)).toBeNull()
  })

  test('a garbage or foreign file is treated as empty; unknown entries are dropped', () => {
    const recorder = enabledRecorder()

    fs.writeFileSync(file, 'not json {')
    expect(recorder.take(W)).toBeNull()

    fs.writeFileSync(file, JSON.stringify({ v: 1, reasons: ['crash'] })) // untagged: whose it was is unknown
    expect(recorder.take(W)).toBeNull()

    recorder.record(W, 'crashed')
    const [{ profile }] = (pending() as { entries: { profile: string }[] }).entries
    const junk = [{ profile, reason: '/home/me/secret' }, { profile, reason: 3 }, null, { profile, reason: 'oom' }]
    fs.writeFileSync(file, JSON.stringify({ v: 2, entries: [{ profile, reason: 'crash' }, ...junk] }))
    expect(recorder.take(W)).toEqual({ reasons: ['crash', 'oom'] })
  })

  test('fs errors never escape', () => {
    const boom = () => {
      throw new Error('EACCES')
    }

    const recorder = new RendererCrashRecorder({
      dir,
      fs: { mkdirSync: boom, readFileSync: boom, renameSync: boom, rmSync: boom, writeFileSync: boom } as never
    })

    recorder.setEnabled(W, 'p', true)
    expect(() => recorder.record(W, 'crashed')).not.toThrow()
    expect(recorder.take(W)).toBeNull()
    expect(() => recorder.setEnabled(W, 'p', false)).not.toThrow()
  })
})

test('registerRendererCrashIpc wires consent, take and ack to the calling window', () => {
  const handlers = new Map<string, (event: unknown, ...args: unknown[]) => unknown>()
  const recorder = new RendererCrashRecorder({ dir })
  const from = { sender: { id: 7 } }

  registerRendererCrashIpc({ handle: (channel, fn) => handlers.set(channel, fn) }, recorder)

  handlers.get('hermes:desktop-metrics:set-enabled')!(from, 'yes', 'local|default')
  recorder.record(7, 'crashed')
  expect(fs.existsSync(file)).toBe(false)

  handlers.get('hermes:desktop-metrics:set-enabled')!(from, true, 'local|default')
  recorder.record(7, 'crashed')
  expect(handlers.get('hermes:desktop-metrics:crash:take')!({ sender: { id: 8 } })).toBeNull()
  expect(handlers.get('hermes:desktop-metrics:crash:take')!(from)).toEqual({ reasons: ['crash'] })

  handlers.get('hermes:desktop-metrics:crash:ack')!(from, true)
  expect(fs.existsSync(file)).toBe(false)
})
