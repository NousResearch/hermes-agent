import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test } from 'vitest'

import { registerDesktopFileIpc } from './desktop-file-ipc'

const fixtureRoots: string[] = []

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-file-ipc-'))
  fixtureRoots.push(root)

  const handlers = new Map<string, (...args: any[]) => any>()
  const windows: unknown[] = []
  let currentWindow: any = { id: 'first' }
  let clipboardText = ''

  registerDesktopFileIpc({
    app: { getPath: () => root } as any,
    clipboard: {
      writeText: (text: string) => {
        clipboardText = text
      },
      readText: () => clipboardText,
      readImage: () => ({ isEmpty: () => true })
    } as any,
    dialog: {
      showOpenDialog: async (window: unknown) => {
        windows.push(window)

        return { canceled: false, filePaths: [root] }
      },
      showSaveDialog: async (window: unknown) => {
        windows.push(window)

        return { canceled: false, filePath: path.join(root, 'save.txt') }
      }
    } as any,
    electronWebContents: { fromId: () => null },
    getMainWindow: () => currentWindow,
    HERMES_HOME: root,
    ipcMain: { handle: (name: string, handler: (...args: any[]) => any) => handlers.set(name, handler) } as any,
    IS_WINDOWS: false,
    IS_WSL: false,
    lastContextMenuPoint: new Map(),
    mimeTypeForPath: () => 'text/plain',
    rememberLog: () => {},
    saveGatewayFile: () => false,
    saveImageFromUrl: async () => false,
    writeComposerImage: async () => ''
  })

  function invoke(name: string, ...args: unknown[]) {
    const handler = handlers.get(name)
    assert.ok(handler, `${name} registered`)

    return handler({}, ...args)
  }

  return {
    root,
    handlers,
    invoke,
    windows,
    setWindow: (window: unknown) => {
      currentWindow = window
    }
  }
}

afterEach(() => {
  for (const root of fixtureRoots.splice(0)) {
    assert.equal(path.dirname(root), os.tmpdir())
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('file preview setting persists and is applied to later reads', async () => {
  const { root, invoke } = fixture()
  assert.equal(invoke('hermes:data-url-read-max:get').maxMb, 16)
  assert.equal(invoke('hermes:data-url-read-max:set', 1).maxMb, 1)
  assert.equal(JSON.parse(fs.readFileSync(path.join(root, 'data-url-read-max.json'), 'utf8')).maxMb, 1)

  const tooLarge = path.join(root, 'large.txt')
  fs.writeFileSync(tooLarge, Buffer.alloc(1024 * 1024 + 1))
  await assert.rejects(invoke('hermes:readFileDataUrl', tooLarge), /too large|exceeds|size|limit/i)
  assert.equal(invoke('hermes:data-url-read-max:set', 2).maxMb, 2)
  assert.match(await invoke('hermes:readFileDataUrl', tooLarge), /^data:text\/plain;base64,/)
})

test('native pickers use the live primary window and clipboard handlers remain registered', async () => {
  const { invoke, windows, setWindow } = fixture()
  const nextWindow = { id: 'replacement' }
  setWindow(nextWindow)

  assert.equal((await invoke('hermes:selectPaths', {})).length, 1)
  assert.match(await invoke('hermes:selectSavePath', {}), /save\.txt$/)
  assert.deepEqual(windows, [nextWindow, nextWindow])
  assert.equal(invoke('hermes:writeClipboard', 'hello'), true)
  assert.equal(invoke('hermes:readClipboard'), 'hello')
})
