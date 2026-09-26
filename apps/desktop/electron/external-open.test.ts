/**
 * Tests for electron/external-open.ts — the single route every external URL
 * open funnels through. All I/O is injected, so the "open failed → notify"
 * behavior is asserted without loading electron. Run with the electron test
 * script that runs electron/*.test.ts (same pattern as native-oauth-login).
 */

import assert from 'node:assert/strict'
import type { ChildProcess } from 'node:child_process'
import { EventEmitter } from 'node:events'

import { test } from 'vitest'

import { type ExternalOpenDeps, openExternalUrl } from './external-open'

function makeDeps(overrides: Partial<ExternalOpenDeps> = {}) {
  const calls = {
    opened: [] as string[],
    fileOpened: [] as string[],
    notified: [] as Array<[string, string]>,
    logged: [] as string[]
  }

  const deps: ExternalOpenDeps = {
    isWsl: false,
    spawn: () => {
      throw new Error('not used')
    },
    openExternal: async url => {
      calls.opened.push(url)
    },
    openFile: async raw => {
      calls.fileOpened.push(raw)
    },
    notifyFailure: (url, message) => calls.notified.push([url, message]),
    log: line => calls.logged.push(line),
    ...overrides
  }

  return { deps, calls }
}

test('resolves ok and opens when openExternal succeeds', async () => {
  const { deps, calls } = makeDeps()

  const result = await openExternalUrl('https://example.com/x', deps)

  assert.deepEqual(result, { ok: true })
  assert.deepEqual(calls.opened, ['https://example.com/x'])
  assert.equal(calls.notified.length, 0)
})

test('notifies and resolves failed when openExternal rejects', async () => {
  const { deps, calls } = makeDeps({
    openExternal: async () => {
      throw new Error('no method available for opening')
    }
  })

  const result = await openExternalUrl('https://example.com', deps)

  assert.deepEqual(result, {
    ok: false,
    reason: 'failed',
    message: 'no method available for opening'
  })
  assert.deepEqual(calls.notified, [['https://example.com/', 'no method available for opening']])
  assert.ok(calls.logged.some(line => line.includes('openExternal failed')))
})

test('resolves invalid for a URL the route does not open, with no notify', async () => {
  const { deps, calls } = makeDeps()

  for (const url of ['', 'not a url', 'ftp://x.com']) {
    const result = await openExternalUrl(url, deps)
    assert.equal(result.ok, false)

    if (result.ok === false) {
      assert.equal(result.reason, 'invalid')
    }
  }

  assert.equal(calls.opened.length, 0)
  assert.equal(calls.notified.length, 0)
})

test('dispatches file:// URLs to openFile', async () => {
  const { deps, calls } = makeDeps()

  const result = await openExternalUrl('file:///C:/x.html', deps)

  assert.deepEqual(result, { ok: true })
  assert.deepEqual(calls.fileOpened, ['file:///C:/x.html'])
})

test('wsl: spawns cmd.exe and resolves ok on the happy path', async () => {
  const spawned: string[] = []
  const proc = new EventEmitter() as unknown as ChildProcess

  const { deps } = makeDeps({
    isWsl: true,
    spawn: (cmd, args) => {
      spawned.push(cmd, ...args)

      return proc
    }
  })

  const result = await openExternalUrl('https://example.com', deps)

  assert.deepEqual(result, { ok: true })
  assert.equal(spawned[0], 'cmd.exe')
  assert.ok(spawned.some(arg => arg === 'https://example.com/'))
})

test('wsl: falls back to openExternal and notifies when cmd.exe fails to spawn', async () => {
  const proc = new EventEmitter() as unknown as ChildProcess
  const { deps, calls } = makeDeps({ isWsl: true, spawn: () => proc })

  deps.openExternal = async url => {
    calls.opened.push(url)
    throw new Error('xdg-open missing')
  }

  const result = await openExternalUrl('https://example.com', deps)
  assert.deepEqual(result, { ok: true })

  // openExternalUrl's fire-and-forget WSL path resolves before the spawn
  // error can arrive; drive the error to exercise the fallback.
  ;(proc as unknown as EventEmitter).emit('error', new Error('ENOENT'))

  await new Promise(resolve => setTimeout(resolve, 20))

  assert.deepEqual(calls.opened, ['https://example.com/'])
  assert.deepEqual(calls.notified, [['https://example.com/', 'xdg-open missing']])
})
