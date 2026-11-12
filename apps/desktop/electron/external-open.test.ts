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

import {
  type ExternalOpenDeps,
  isUnsafeWslLaunchArgument,
  openExternalUrl,
  reportPreOpenStatFailure
} from './external-open'

function makeDeps(overrides: Partial<ExternalOpenDeps> = {}) {
  const calls = {
    opened: [] as string[],
    fileOpened: [] as string[],
    localOpened: [] as string[],
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
    openLocalPath: async raw => {
      calls.localOpened.push(raw)

      return true
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

test('opens bare local paths through openLocalPath instead of rejecting them', async () => {
  const { deps, calls } = makeDeps()

  // Every shape `new URL()` cannot express as a web/file URL: a Windows drive
  // letter parses as the bogus `c:` scheme, POSIX/UNC/`~` make the parser
  // throw. All previously landed in the "Invalid external URL" reject.
  for (const raw of [
    'C:\\Users\\x\\a.md',
    'C:/Work/report.html',
    '/tmp/report.pdf',
    '~/logs/desktop.log',
    '\\\\server\\share\\a.md'
  ]) {
    const result = await openExternalUrl(raw, deps)

    assert.deepEqual(result, { ok: true }, `expected ${raw} to open as a local path`)
  }

  assert.deepEqual(calls.localOpened, [
    'C:\\Users\\x\\a.md',
    'C:/Work/report.html',
    '/tmp/report.pdf',
    '~/logs/desktop.log',
    '\\\\server\\share\\a.md'
  ])
  assert.equal(calls.fileOpened.length, 0)
  assert.equal(calls.opened.length, 0)
})

test('resolves invalid, with the path logged, when openLocalPath cannot resolve the path', async () => {
  const { deps, calls } = makeDeps({
    openLocalPath: async () => false
  })

  const result = await openExternalUrl('C:\\Users\\x\\a.md', deps)

  assert.deepEqual(result, { ok: false, reason: 'invalid' })
  assert.ok(
    calls.logged.some(line => line.includes('openPath resolve rejected') && line.includes('C:\\Users\\x\\a.md'))
  )
  assert.equal(calls.notified.length, 0)
})

test('notifies failed when openLocalPath throws', async () => {
  const { deps, calls } = makeDeps({
    openLocalPath: async () => {
      throw new Error('stat failed')
    }
  })

  const result = await openExternalUrl('/tmp/report.pdf', deps)

  assert.deepEqual(result, { ok: false, reason: 'failed', message: 'stat failed' })
  assert.deepEqual(calls.notified, [['/tmp/report.pdf', 'stat failed']])
})

test('opens protocol-relative URLs as https, never as local paths', async () => {
  const { deps, calls } = makeDeps()

  const result = await openExternalUrl('//cdn.example.com/img.png', deps)

  assert.deepEqual(result, { ok: true })
  assert.deepEqual(calls.opened, ['https://cdn.example.com/img.png'])
  assert.equal(calls.localOpened.length, 0)
})

test('wsl: hands the URL to rundll32 and resolves ok on the happy path', async () => {
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
  assert.equal(spawned[0], 'rundll32.exe')
  assert.deepEqual(spawned.slice(1), ['url.dll,FileProtocolHandler', 'https://example.com/'])
})

test('wsl: a URL with shell metacharacters stays one argument — no cmd.exe to interpret it (#126939)', async () => {
  const spawns: Array<{ cmd: string; args: string[] }> = []
  const proc = new EventEmitter() as unknown as ChildProcess

  const { deps } = makeDeps({
    isWsl: true,
    spawn: (cmd, args) => {
      spawns.push({ cmd, args: [...args] })

      return proc
    }
  })

  // `&` is what cmd.exe would have parsed as a command separator; `%2F` and
  // `^` are the other characters a quoting-only fix cannot make safe there.
  const url = 'https://example.com/x&calc?^q=1%2F2'
  const result = await openExternalUrl(url, deps)

  assert.deepEqual(result, { ok: true })
  assert.equal(spawns.length, 1)
  assert.equal(spawns[0].cmd, 'rundll32.exe')
  // The URL travels as a single argv element to a non-shell sink, so nothing
  // re-parses it — there is no cmd.exe command line to escape from.
  assert.deepEqual(spawns[0].args, ['url.dll,FileProtocolHandler', 'https://example.com/x&calc?^q=1%2F2'])
  assert.ok(!spawns[0].args.includes('cmd.exe'))
  assert.ok(!spawns[0].args.includes('start'))
})

test('wsl guard: quotes, whitespace, and control characters make a launch argument unsafe', () => {
  for (const url of [
    'https://example.com/a"b',
    'https://example.com/a b',
    'https://example.com/a\nb',
    'https://example.com/a\rb',
    'https://example.com/a\tb',
    'https://example.com/a\x00b',
    'https://example.com/a\x1fb',
    'https://example.com/a\x7fb'
  ]) {
    assert.equal(isUnsafeWslLaunchArgument(url), true, url)
  }
})

test('wsl guard: URL metacharacters that rundll32 never interprets stay allowed', () => {
  for (const url of [
    'https://example.com/x&calc',
    'https://example.com/?a=1&b=2',
    'https://example.com/a|b',
    'https://example.com/a^b',
    'https://example.com/a%2Fb',
    'mailto:user@example.com?subject=hi%20there'
  ]) {
    assert.equal(isUnsafeWslLaunchArgument(url), false, url)
  }
})

test('wsl: raw quotes and line breaks are normalized away and no shell is requested', async () => {
  const spawns: Array<{ cmd: string; args: string[]; opts: unknown }> = []
  const proc = new EventEmitter() as unknown as ChildProcess

  const { deps } = makeDeps({
    isWsl: true,
    spawn: (cmd, args, opts) => {
      spawns.push({ cmd, args: [...args], opts })

      return proc
    }
  })

  // Raw input with every character class cmd.exe acts on plus a quote and a
  // CRLF. URL normalization percent-encodes the quote and drops CR/LF before
  // the guard; `&`, `|`, `^` and `%PATH%` stay as URL data. Newer WHATWG
  // parsers also encode `^` in paths (older Node releases do not), so the
  // expected argument comes from `new URL()` rather than a literal.
  const raw = 'https://example.com/a&b|c^d%PATH%"g\r\n?h=i'
  const normalized = new URL(raw).toString()
  const result = await openExternalUrl(raw, deps)

  assert.deepEqual(result, { ok: true })
  assert.match(normalized, /^https:\/\/example\.com\/a&b\|c(\^|%5E)d%PATH%%22g\?h=i$/)
  // Pin the options too: under WSL the Electron process is Linux, so a
  // `shell: true` here would hand the URL to /bin/sh, which also splits on `&`.
  assert.deepEqual(spawns, [
    {
      cmd: 'rundll32.exe',
      args: ['url.dll,FileProtocolHandler', normalized],
      opts: { detached: true, stdio: 'ignore', windowsHide: true }
    }
  ])
})

test('wsl: mailto metacharacters in the query reach rundll32 unchanged', async () => {
  const spawned: string[][] = []
  const proc = new EventEmitter() as unknown as ChildProcess

  const { deps } = makeDeps({
    isWsl: true,
    spawn: (cmd, args) => {
      spawned.push([cmd, ...args])

      return proc
    }
  })

  const result = await openExternalUrl('mailto:user@example.com?subject=a&body=b|c', deps)

  assert.deepEqual(result, { ok: true })
  assert.deepEqual(spawned, [
    ['rundll32.exe', 'url.dll,FileProtocolHandler', 'mailto:user@example.com?subject=a&body=b|c']
  ])
})

test('wsl: the launch guard is enforced end to end for input that survives normalization', async () => {
  // http(s) serialization percent-encodes quotes and spaces, but a mailto
  // address is an opaque path that keeps both. This is the reachable case of
  // the guard: nothing is spawned and nothing falls back to xdg-open.
  for (const url of ['mailto:"a b"@example.com', 'mailto:a b@example.com']) {
    let spawnCalls = 0

    const { deps, calls } = makeDeps({
      isWsl: true,
      spawn: () => {
        spawnCalls += 1

        return new EventEmitter() as unknown as ChildProcess
      }
    })

    const result = await openExternalUrl(url, deps)

    assert.deepEqual(result, { ok: false, reason: 'invalid' }, url)
    assert.equal(spawnCalls, 0, url)
    assert.deepEqual(calls.opened, [], url)
    assert.deepEqual(calls.notified, [], url)
  }
})

test('wsl: falls back to openExternal and notifies when rundll32 fails to spawn', async () => {
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
