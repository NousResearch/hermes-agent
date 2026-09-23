import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import { test, vi } from 'vitest'

import { createDesktopExternalOpenRuntime } from './desktop-external-open-runtime'

function setup(isWsl = false) {
  const child = Object.assign(new EventEmitter(), { unref: vi.fn() })
  const shell = {
    openPath: vi.fn(async () => ''),
    showItemInFolder: vi.fn(),
    openExternal: vi.fn(async () => undefined)
  }
  const deps = {
    IS_WSL: isWsl,
    shell,
    spawn: vi.fn(() => child),
    resolveRequestedPathForIpc: vi.fn(() => 'C:/safe/report.txt'),
    pathToFileURL: vi.fn(() => new URL('file:///safe/report.txt')),
    rememberLog: vi.fn()
  }

  return { child, deps, shell, runtime: createDesktopExternalOpenRuntime(deps) }
}

test('rejects malformed and unsupported external URLs before touching the OS', () => {
  const { deps, runtime, shell } = setup()

  assert.equal(runtime.openExternalUrl(''), false)
  assert.equal(runtime.openExternalUrl('not a url'), false)
  assert.equal(runtime.openExternalUrl('javascript:alert(1)'), false)
  assert.equal(deps.resolveRequestedPathForIpc.mock.calls.length, 0)
  assert.equal(shell.openPath.mock.calls.length, 0)
  assert.equal(shell.openExternal.mock.calls.length, 0)
  assert.equal(deps.spawn.mock.calls.length, 0)
})

test('file URLs pass the path guard and reveal the file when openPath reports an association failure', async () => {
  const { deps, runtime, shell } = setup()
  shell.openPath.mockResolvedValue('no file association')

  assert.equal(runtime.openExternalUrl('file:///C:/safe/report.txt'), true)
  await Promise.resolve()

  assert.deepEqual(deps.resolveRequestedPathForIpc.mock.calls, [
    ['file:///C:/safe/report.txt', { purpose: 'Open external file' }]
  ])
  assert.deepEqual(shell.openPath.mock.calls, [['C:/safe/report.txt']])
  assert.deepEqual(shell.showItemInFolder.mock.calls, [['C:/safe/report.txt']])
})

test('a rejected local path never reaches shell.openPath', () => {
  const { deps, runtime, shell } = setup()
  deps.resolveRequestedPathForIpc.mockImplementation(() => {
    throw new Error('outside allowed path')
  })

  assert.equal(runtime.openExternalUrl('file:///C:/blocked.txt'), false)
  assert.equal(shell.openPath.mock.calls.length, 0)
})

test('ordinary HTTPS and mail links use the OS external handler', () => {
  const { runtime, shell } = setup()

  assert.equal(runtime.openExternalUrl(' https://example.test/guide '), true)
  assert.equal(runtime.openExternalUrl('mailto:person@example.test'), true)
  assert.deepEqual(shell.openExternal.mock.calls, [['https://example.test/guide'], ['mailto:person@example.test']])
})

test('WSL opens allowed links through hidden cmd.exe and falls back to xdg-open on spawn error', () => {
  const { child, deps, runtime, shell } = setup(true)

  assert.equal(runtime.openExternalUrl('https://example.test/guide'), true)
  assert.deepEqual(deps.spawn.mock.calls, [
    [
      'cmd.exe',
      ['/c', 'start', '""', 'https://example.test/guide'],
      { detached: true, stdio: 'ignore', windowsHide: true }
    ]
  ])
  assert.equal(child.unref.mock.calls.length, 1)
  child.emit('error', new Error('cmd unavailable'))
  assert.deepEqual(shell.openExternal.mock.calls, [['https://example.test/guide']])
})

test('preview file URLs use the external browser path after validation', async () => {
  const { deps, runtime, shell } = setup()

  assert.equal(await runtime.openPreviewInBrowser('file:///C:/safe/report.txt'), true)
  assert.deepEqual(deps.resolveRequestedPathForIpc.mock.calls, [
    ['file:///C:/safe/report.txt', { purpose: 'Open preview in browser' }]
  ])
  assert.deepEqual(deps.pathToFileURL.mock.calls, [['C:/safe/report.txt']])
  assert.deepEqual(shell.openExternal.mock.calls, [['file:///safe/report.txt']])
  assert.equal(shell.openPath.mock.calls.length, 0)
})
