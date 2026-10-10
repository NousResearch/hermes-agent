import assert from 'node:assert/strict'

import { test } from 'vitest'

import { readClipboardImage, registerClipboardImageIpc } from './clipboard-image-ipc'

const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

test('clipboard image IPC saves the first available image item as PNG', async () => {
  const source = Buffer.concat([PNG_SIGNATURE, Buffer.from('source')])
  const savedPng = Buffer.concat([PNG_SIGNATURE, Buffer.from('normalized')])
  let written: Buffer | undefined
  let registeredChannel = ''
  let handler: (() => Promise<string>) | undefined

  registerClipboardImageIpc({
    ipcMain: {
      handle(channel, listener) {
        registeredChannel = channel
        handler = listener as () => Promise<string>
      }
    } as never,
    clipboard: {
      read: async () => [
        { types: ['text/plain'], getType: async () => 'ignored' },
        { types: ['image/jpeg', 'image/png'], getType: async type => {
          assert.equal(type, 'image/jpeg')

          return new Blob([source])
        } }
      ]
    },
    nativeImage: {
      createFromBuffer: buffer => {
        assert.deepEqual(buffer, source)

        return { isEmpty: () => false, toPNG: () => savedPng }
      }
    },
    isWsl: false,
    readWslWindowsClipboardImage: () => null,
    writeComposerImage: async (buffer, ext) => {
      assert.equal(ext, '.png')
      written = buffer

      return 'composer-image.png'
    }
  })

  assert.equal(registeredChannel, 'hermes:saveClipboardImage')
  assert.equal(await handler?.(), 'composer-image.png')
  assert.deepEqual(written, savedPng)
})

test('clipboard image reading uses the Windows clipboard fallback in WSL', async () => {
  const fallback = Buffer.concat([PNG_SIGNATURE, Buffer.from('fallback')])
  let written: Buffer | undefined

  const result = await readClipboardImage({
    clipboard: { read: async () => [] },
    nativeImage: { createFromBuffer: () => ({ isEmpty: () => true, toPNG: () => Buffer.alloc(0) }) },
    isWsl: true,
    readWslWindowsClipboardImage: () => fallback,
    writeComposerImage: async (buffer, ext) => {
      assert.equal(ext, '.png')
      written = buffer

      return 'fallback-image.png'
    }
  })

  assert.equal(result, 'fallback-image.png')
  assert.deepEqual(written, fallback)
})

test('clipboard image reading returns empty when neither clipboard path has an image', async () => {
  let writes = 0

  const result = await readClipboardImage({
    clipboard: { read: async () => [{ types: ['text/plain'], getType: async () => 'text' }] },
    nativeImage: { createFromBuffer: () => ({ isEmpty: () => true, toPNG: () => Buffer.alloc(0) }) },
    isWsl: false,
    readWslWindowsClipboardImage: () => null,
    writeComposerImage: async () => {
      writes += 1

      return 'unexpected.png'
    }
  })

  assert.equal(result, '')
  assert.equal(writes, 0)
})
