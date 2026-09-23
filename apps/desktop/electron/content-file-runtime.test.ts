import assert from 'node:assert/strict'
import fs from 'node:fs'
import { createServer } from 'node:http'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { afterEach, test } from 'vitest'

import { createContentFileRuntime } from './content-file-runtime'
import { resolveReadableFileForIpc } from './hardening'

const directories: string[] = []

function fixtureDirectory() {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-content-file-'))
  directories.push(directory)

  return directory
}

afterEach(() => {
  for (const directory of directories.splice(0)) {
    assert.equal(path.dirname(directory), os.tmpdir())
    fs.rmSync(directory, { recursive: true, force: true })
  }
})

test('image file authorization and loopback download reach the live save window and chosen destination', async () => {
  const directory = fixtureDirectory()
  const imageBytes = Buffer.from('image bytes')
  const localImage = path.join(directory, 'local.png')
  const secretFile = path.join(directory, '.env')
  const destination = path.join(directory, 'saved.png')
  fs.writeFileSync(localImage, imageBytes)
  fs.writeFileSync(secretFile, 'private')

  const oldWindow = { id: 'old' }
  const currentWindow = { id: 'current' }
  let activeWindow = oldWindow
  const dialogCalls: Array<{ window: unknown; options: { defaultPath: string } }> = []

  const runtime = createContentFileRuntime({
    app: { getPath: () => directory },
    dialog: {
      showSaveDialog: async (window, options) => {
        dialogCalls.push({ window, options })

        return { canceled: false, filePath: destination }
      }
    },
    getMainWindow: () => activeWindow,
    resolveReadableFileForIpc
  })

  assert.deepEqual(await runtime.resourceBufferFromUrl(pathToFileURL(localImage).toString()), {
    buffer: imageBytes,
    mimeType: 'image/png'
  })
  await assert.rejects(runtime.resourceBufferFromUrl(pathToFileURL(secretFile).toString()), /sensitive/i)

  const server = createServer((_request, response) => {
    response.writeHead(200, { 'content-type': 'image/png' })
    response.end(imageBytes)
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  assert.ok(address && typeof address !== 'string')

  try {
    activeWindow = currentWindow
    assert.equal(await runtime.saveImageFromUrl(`http://127.0.0.1:${address.port}/image-hash`), true)
    assert.deepEqual(fs.readFileSync(destination), imageBytes)
    assert.equal(dialogCalls.length, 1)
    assert.equal(dialogCalls[0].window, currentWindow)
    assert.equal(dialogCalls[0].options.defaultPath, path.join(directory, 'image.png'))
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})

test('composer image writing reads userData on each call and reduces an untrusted name to a local basename', async () => {
  const firstDirectory = fixtureDirectory()
  const secondDirectory = fixtureDirectory()
  let userData = firstDirectory

  const runtime = createContentFileRuntime({
    app: { getPath: () => userData },
    dialog: { showSaveDialog: async () => ({ canceled: true }) },
    getMainWindow: () => null,
    resolveReadableFileForIpc
  })

  userData = secondDirectory
  const bytes = Buffer.from('composer image')
  const savedPath = await runtime.writeComposerImage(bytes, 'JPG', '..\\nested\\a<>b.png')
  assert.equal(path.dirname(savedPath), path.join(secondDirectory, 'composer-images'))
  assert.match(path.basename(savedPath), /^a_b_[0-9a-f]{6}\.jpg$/)
  assert.deepEqual(fs.readFileSync(savedPath), bytes)
  assert.equal(fs.existsSync(path.join(firstDirectory, 'composer-images')), false)
})
