/**
 * The renderer builds media open URLs with shared/src/local-file-url.ts; the
 * main process turns them back into the path it hands the OS
 * (openExternalFile -> resolveRequestedPathForIpc -> shell.showItemInFolder).
 * These assert that round trip lands on the exact file, so a `#`, `?`, `%`,
 * space or non-ASCII name no longer opens a truncated/wrong path.
 */

import assert from 'node:assert/strict'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { test } from 'vitest'

import { localFileUrl } from '../../shared/src/local-file-url'

import { type ExternalOpenDeps, openExternalUrl } from './external-open'
import { resolveRequestedPathForIpc } from './hardening'

const POSIX_PATHS = [
  '/tmp/a.png',
  '/Users/c/Library/Application Support/Hermes/x.png',
  '/tmp/weird#name?.png',
  '/tmp/100%/caf\u00e9 \u56fe.png',
  '/tmp/back\\slash.png'
]

const WINDOWS_PATHS = [
  'C:\\Users\\me\\a b.png',
  'C:\\Users\\me\\weird#name 100%.png',
  'D:\\caf\u00e9\\\u56fe.png',
  '\\\\server\\share\\dir\\a#b.png'
]

test('POSIX paths survive the file URL round trip (platform-independent parser)', () => {
  for (const p of POSIX_PATHS) {
    const url = localFileUrl(p)
    assert.ok(url, p)
    assert.equal(fileURLToPath(url, { windows: false }), p)
  }
})

test('Windows drive and UNC paths survive the file URL round trip (platform-independent parser)', () => {
  for (const p of WINDOWS_PATHS) {
    const url = localFileUrl(p)
    assert.ok(url, p)
    assert.equal(fileURLToPath(url, { windows: true }), p)
  }
})

test('the path handed to the OS is the original file (resolveRequestedPathForIpc on this platform)', () => {
  const paths = process.platform === 'win32' ? WINDOWS_PATHS : POSIX_PATHS

  for (const p of paths) {
    assert.equal(resolveRequestedPathForIpc(localFileUrl(p), { purpose: 'Open external file' }), path.resolve(p))
  }
})

test('the old string concatenation truncated at # (regression reference)', () => {
  const p = process.platform === 'win32' ? 'C:\\tmp\\a#b.png' : '/tmp/a#b.png'
  const naive = `file://${process.platform === 'win32' ? '/' : ''}${p.replace(/\\/g, '/')}`

  assert.notEqual(resolveRequestedPathForIpc(naive, { purpose: 'Open external file' }), path.resolve(p))
  assert.equal(resolveRequestedPathForIpc(localFileUrl(p), { purpose: 'Open external file' }), path.resolve(p))
})

test('relative and ~ paths have no file URL form (resolved against the workspace, not here)', () => {
  for (const p of ['out.png', 'out/report.png', './a.png', '../a.png', '~/Desktop/a.png', '~']) {
    assert.equal(localFileUrl(p), null, p)
  }
})

test('the open route passes an encoded file URL to the file opener unmodified', async () => {
  const opened: string[] = []

  const deps: ExternalOpenDeps = {
    isWsl: false,
    spawn: () => {
      throw new Error('not used')
    },
    openExternal: async () => {
      throw new Error('not used')
    },
    openFile: async raw => {
      opened.push(raw)
    },
    notifyFailure: () => {},
    log: () => {}
  }

  const url = localFileUrl('/tmp/weird#name?.png')

  assert.deepEqual(await openExternalUrl(url, deps), { ok: true })
  assert.deepEqual(opened, [url])
})
