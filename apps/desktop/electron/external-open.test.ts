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

import { type ExternalOpenDeps, openExternalUrl, reportPreOpenStatFailure } from './external-open'

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
  // The URL reaches cmd.exe double-quoted: Node's Windows escaping covers
  // whitespace and quotes only, so an unquoted argument's punctuation would
  // read as line syntax.
  assert.ok(spawned.includes('"https://example.com/"'))
})

test('wsl: a URL with cmd.exe metacharacters is refused before any spawn (#126939)', async () => {
  let spawnCalls = 0
  const proc = new EventEmitter() as unknown as ChildProcess

  const { deps, calls } = makeDeps({
    isWsl: true,
    spawn: () => {
      spawnCalls += 1

      return proc
    }
  })

  // & is the reported vector (cmd.exe command separator — calc launches);
  // the rest of the class cmd.exe acts on is refused with it: separators,
  // redirection, the escape character, variable expansion (which survives
  // double quotes), line breaks, and the quotes themselves.
  for (const url of [
    'https://example.com/x&calc',
    'https://example.com/?a=1&b=2',
    'https://example.com/a|b',
    'https://example.com/a>b',
    'https://example.com/a<b',
    'https://example.com/a^b',
    'https://example.com/%PATH%',
    'https://example.com/a"b',
    "https://example.com/a'b",
    'https://example.com/a\r\nb'
  ]) {
    const result = await openExternalUrl(url, deps)

    assert.deepEqual(result, { ok: false, reason: 'invalid' }, url)
  }

  assert.equal(spawnCalls, 0)
  assert.deepEqual(calls.opened, [])
  assert.equal(calls.notified.length, 0)
  assert.ok(calls.logged.some(line => line.includes('refusing WSL open')))
})

test('wsl: mailto stays spawnable when it carries no cmd.exe metacharacters', async () => {
  const spawned: string[] = []
  const proc = new EventEmitter() as unknown as ChildProcess

  const { deps } = makeDeps({
    isWsl: true,
    spawn: (cmd, args) => {
      spawned.push(cmd, ...args)

      return proc
    }
  })

  const result = await openExternalUrl('mailto:someone@example.com?subject=hi', deps)

  assert.deepEqual(result, { ok: true })
  assert.equal(spawned[0], 'cmd.exe')
  assert.ok(spawned.includes('"mailto:someone@example.com?subject=hi"'))
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

test('guard: missing-file error is reported once and classified as a miss', () => {
  const reported: Array<[string, string]> = []
  const logged: string[] = []

  const error = Object.assign(new Error('This file does not exist: /tmp/gone.html'), { code: 'missing-file' })

  const isMiss = reportPreOpenStatFailure(error, 'file:///tmp/gone.html', {
    log: line => logged.push(line),
    reportMissing: (url, message) => reported.push([url, message])
  })

  assert.equal(isMiss, true)
  assert.deepEqual(reported, [['file:///tmp/gone.html', 'This file does not exist: /tmp/gone.html']])
  assert.equal(logged.length, 0)
})

test('guard: a non-missing stat failure is logged and classified as proceed-to-OS', () => {
  const reported: Array<[string, string]> = []
  const logged: string[] = []

  for (const code of ['EACCES', 'ELOOP', 'EPERM', 'ENAMETOOLONG']) {
    const error = Object.assign(new Error(`${code}: stat failed`), { code })

    const isMiss = reportPreOpenStatFailure(error, 'file:///srv/locked/report.html', {
      log: line => logged.push(line),
      reportMissing: (url, message) => reported.push([url, message])
    })

    assert.equal(isMiss, false, code)
  }

  // Nothing was fabricated as a miss, and every failure left a log line.
  assert.equal(reported.length, 0)
  assert.equal(logged.length, 4)
  assert.ok(logged.every(line => line.includes('[file] pre-open stat failed')))
})
