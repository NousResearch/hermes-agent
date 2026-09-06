import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it, vi } from 'vitest'

import type { PreviewUblockExtension } from './preview-ublock'
import { createPreviewUblockRuntime, type PreviewUblockRuntimeSession } from './preview-ublock-runtime'

const temporaryPaths: string[] = []

function extensionFixture(): string {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-preview-ublock-runtime-'))
  temporaryPaths.push(fixture)
  fs.writeFileSync(
    path.join(fixture, 'manifest.json'),
    JSON.stringify({ declarative_net_request: { rule_resources: [{ enabled: true, path: 'rules.json' }] } })
  )
  fs.writeFileSync(path.join(fixture, 'rules.json'), JSON.stringify([]))

  return fixture
}

function session(): PreviewUblockRuntimeSession {
  const extensions = new Map<string, PreviewUblockExtension>()
  let nextId = 0
  const webRequest = {
    onBeforeRequest: vi.fn(),
    onCompleted: vi.fn(),
    onErrorOccurred: vi.fn(),
    removeCompletedListener: vi.fn(),
    removeListener: vi.fn()
  }

  return {
    extensions: {
      getAllExtensions: () => [...extensions.values()],
      getExtension: (id: string) => extensions.get(id) ?? null,
      loadExtension: async (extensionPath: string) => {
        const extension = {
          id: `ublock-${++nextId}`,
          manifest: { version: '2026.825.1619' },
          path: extensionPath,
          url: `chrome-extension://ublock-${nextId}`
        }
        extensions.set(extension.id, extension)

        return extension
      },
      removeExtension: (id: string) => extensions.delete(id)
    },
    webRequest
  }
}

function runtime(
  settingsPath: string,
  installerResolve: unknown = { path: extensionFixture(), version: '2026.825.1619' }
) {
  const logs: string[] = []
  const previewSession = session()
  const installer = { resolve: vi.fn().mockResolvedValue(installerResolve) }
  const runtime = createPreviewUblockRuntime({
    createBootstrapWindow: () => ({
      destroy: vi.fn(),
      executeJavaScript: vi.fn().mockResolvedValue(true),
      loadURL: vi.fn().mockResolvedValue(undefined)
    }),
    createInstaller: () => installer,
    createPopupWindow: () => {
      throw new Error('popup not used by this test')
    },
    getOwnerWebContents: () => null,
    getStateWindows: () => [],
    log: message => logs.push(message),
    openExternal: () => true,
    previewSession,
    settingsPath,
    userDataPath: path.dirname(settingsPath)
  })

  return { installer, logs, runtime }
}

afterEach(() => {
  for (const temporaryPath of temporaryPaths.splice(0)) {
    fs.rmSync(temporaryPath, { force: true, recursive: true })
  }
})

describe('preview uBlock runtime', () => {
  it('composes lifecycle helpers and preserves enable persistence rules', async () => {
    const settingsDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-preview-ublock-settings-'))
    temporaryPaths.push(settingsDirectory)
    const settingsPath = path.join(settingsDirectory, 'preview-ublock.json')
    const owner = runtime(settingsPath)

    await expect(owner.runtime.initialize()).resolves.toMatchObject({ enabled: false, available: false })
    await expect(owner.runtime.setEnabled(true)).resolves.toMatchObject({ enabled: true, available: true })
    expect(owner.installer.resolve).toHaveBeenCalledWith('pinned', expect.any(Object))
    await expect(owner.runtime.setEnabled(false)).resolves.toMatchObject({ enabled: false, available: false })
    expect(JSON.parse(fs.readFileSync(settingsPath, 'utf8'))).toEqual({ enabled: false })

    await owner.runtime.dispose()
    await owner.runtime.dispose()
  })

  it('resets a stale enabled preference when cached activation fails', async () => {
    const settingsDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-preview-ublock-settings-'))
    temporaryPaths.push(settingsDirectory)
    const settingsPath = path.join(settingsDirectory, 'preview-ublock.json')
    fs.writeFileSync(settingsPath, JSON.stringify({ enabled: true }))
    const owner = runtime(settingsPath, null)

    await expect(owner.runtime.initialize()).resolves.toMatchObject({ enabled: false, available: false })
    expect(JSON.parse(fs.readFileSync(settingsPath, 'utf8'))).toEqual({ enabled: false })
    expect(owner.logs).toContain('[preview] uBlock Origin Lite cache unavailable; content blocking remains disabled')
  })
})
