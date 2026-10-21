import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, test, vi } from 'vitest'

import { previewFileMetadata } from './preview-file-metadata'

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'preview-metadata-'))
const document = path.join(root, 'report.md')
fs.writeFileSync(document, 'report')
afterEach(() => vi.restoreAllMocks())

test('automatic classification validates paths without opening document contents', async () => {
  const open = vi.spyOn(fs.promises, 'open')

  const result = await previewFileMetadata(path.join(root, 'unused') + '/../report.md', 'text/markdown', {
    metadataOnly: true,
    maxPreviewBytes: 4096
  })

  expect(result.byteSize).toBe(6)
  expect(result.resolvedPath).toBe(document)
  expect(open).not.toHaveBeenCalled()
  const secret = path.join(root, '.env')
  fs.writeFileSync(secret, 'synthetic')
  await expect(
    previewFileMetadata(secret, 'text/plain', { metadataOnly: true, maxPreviewBytes: 4096 })
  ).rejects.toThrow()
  const link = path.join(root, 'innocent.md')
  fs.symlinkSync(secret, link)
  await expect(previewFileMetadata(link, 'text/plain', { metadataOnly: true, maxPreviewBytes: 4096 })).rejects.toThrow()
  expect(open).not.toHaveBeenCalled()
})

test('a stalled explicit preview read yields to the event loop and closes its file', async () => {
  let finishRead!: () => void
  let readStarted!: () => void

  const started = new Promise<void>(resolve => {
    readStarted = resolve
  })

  const close = vi.fn(async () => undefined)

  const read = vi.fn(
    (buffer: Buffer) =>
      new Promise(resolve => {
        finishRead = () => {
          buffer[0] = 0
          resolve({ bytesRead: 1, buffer })
        }

        readStarted()
      })
  )

  vi.spyOn(fs.promises, 'open').mockResolvedValue({ read, close } as never)
  const pending = previewFileMetadata(document, 'text/markdown', { maxPreviewBytes: 4096 })
  await started
  await new Promise<void>(resolve => setTimeout(resolve, 0))
  expect(close).not.toHaveBeenCalled()
  finishRead()
  expect((await pending).binary).toBe(true)
  expect(close).toHaveBeenCalledOnce()
})
