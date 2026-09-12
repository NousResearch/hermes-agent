import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test, vi } from 'vitest'

const electronMock = vi.hoisted(() => {
  const handlers = new Map<string, (...args: any[]) => any>()

  return {
    handlers,
    ipcMain: {
      handle: vi.fn((channel: string, handler: (...args: any[]) => any) => {
        handlers.set(channel, handler)
      })
    },
    shell: {
      openPath: vi.fn(async (_path: string) => ''),
      showItemInFolder: vi.fn(),
      trashItem: vi.fn()
    }
  }
})

vi.mock('electron', () => ({
  ipcMain: electronMock.ipcMain,
  shell: electronMock.shell
}))

import { registerFsIpc } from './fs-ipc'

afterEach(() => {
  electronMock.handlers.clear()
  vi.clearAllMocks()
})

test('openDir validates and uses the resolved path before creating or opening it', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-fs-opendir-'))
  const resolved = path.join(root, 'plugins')
  const expandUserPath = vi.fn((value: string) => `/expanded/${value}`)
  const resolveRequestedPathForIpc = vi.fn(() => resolved)

  try {
    registerFsIpc({
      hermesHome: root,
      readActiveDesktopProfile: () => null,
      expandUserPath,
      resolveRequestedPathForIpc,
      directoryExists: () => true,
      resolveGitBinary: () => 'git'
    })

    const handler = electronMock.handlers.get('hermes:fs:openDir')
    assert.ok(handler)

    const result = await handler(undefined, '~/plugins')

    assert.deepEqual(result, { ok: true })
    assert.equal(expandUserPath.mock.calls[0]?.[0], '~/plugins')
    assert.deepEqual(resolveRequestedPathForIpc.mock.calls[0], [
      '/expanded/~/plugins',
      { purpose: 'Open directory' }
    ])
    assert.equal(fs.existsSync(resolved), true)
    assert.equal(electronMock.shell.openPath.mock.calls[0]?.[0], path.normalize(resolved))
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})


test('openDir stops before opening when path validation rejects the input', async () => {
  const resolveRequestedPathForIpc = vi.fn(() => {
    throw new Error('blocked path')
  })

  registerFsIpc({
    hermesHome: '/tmp/hermes',
    readActiveDesktopProfile: () => null,
    expandUserPath: (value: string) => value,
    resolveRequestedPathForIpc,
    directoryExists: () => true,
    resolveGitBinary: () => 'git'
  })

  const handler = electronMock.handlers.get('hermes:fs:openDir')
  assert.ok(handler)

  const result = await handler(undefined, '\\\\?\\C:\\secret')

  assert.deepEqual(result, { ok: false, error: 'blocked path' })
  assert.equal(resolveRequestedPathForIpc.mock.calls.length, 1)
  assert.equal(electronMock.shell.openPath.mock.calls.length, 0)
})
