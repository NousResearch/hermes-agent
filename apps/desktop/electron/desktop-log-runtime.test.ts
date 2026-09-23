import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test, vi } from 'vitest'

import { createDesktopLogRuntime } from './desktop-log-runtime'

const directories: string[] = []

function newLogPath(): string {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-log-'))
  directories.push(directory)

  return path.join(directory, 'logs', 'desktop.log')
}

afterEach(() => {
  vi.useRealTimers()

  for (const directory of directories.splice(0)) {
    fs.rmSync(directory, { recursive: true, force: true })
  }
})

test('shutdown cancels the pending timer and flushes the same buffered lines synchronously', async () => {
  vi.useFakeTimers()
  const logPath = newLogPath()
  const runtime = createDesktopLogRuntime(logPath)
  runtime.rememberLog('first line\nsecond line')

  assert.equal(runtime.hermesLog.length, 2)
  assert.equal(fs.existsSync(logPath), false)
  runtime.stopDesktopLogFlushTimer()
  await vi.advanceTimersByTimeAsync(500)
  assert.equal(fs.existsSync(logPath), false)

  runtime.flushDesktopLogBufferSync()
  const saved = fs.readFileSync(logPath, 'utf8')
  assert.match(saved, /first line/)
  assert.match(saved, /second line/)
  assert.equal(saved.split('\n').filter(Boolean).length, 2)
})

test('scheduled async flush and a later synchronous flush retain append order', async () => {
  const logPath = newLogPath()
  const runtime = createDesktopLogRuntime(logPath)
  runtime.rememberLog('from timer')

  await vi.waitFor(
    () => {
      assert.match(fs.readFileSync(logPath, 'utf8'), /from timer/)
    },
    { timeout: 1500, interval: 20 }
  )

  runtime.rememberLog('from shutdown')
  runtime.stopDesktopLogFlushTimer()
  runtime.flushDesktopLogBufferSync()
  const saved = fs.readFileSync(logPath, 'utf8')
  assert.ok(saved.indexOf('from timer') < saved.indexOf('from shutdown'))
})
