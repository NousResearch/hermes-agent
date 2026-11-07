import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, it, vi } from 'vitest'

import { saveTextFileAs } from './save-text-file'

const scratch: string[] = []
afterEach(() => {
  for (const dir of scratch.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

it('rejects malformed binary content before opening the native picker', async () => {
  const showSaveDialog = vi.fn(async () => ({ canceled: true }))
  await expect(
    saveTextFileAs({ content: 'not!base64', encoding: 'base64' }, { defaultDir: () => '/unused', showSaveDialog })
  ).rejects.toThrow('Invalid base64')
  expect(showSaveDialog).not.toHaveBeenCalled()
})

it.each([
  { content: '4142', encoding: 'hex', error: 'Invalid encoding' },
  { content: 'AB==', encoding: 'base64', error: 'Invalid base64' }
])('rejects invalid encoding or ambiguous base64 before the picker: $encoding / $content', async request => {
  const showSaveDialog = vi.fn(async () => ({ canceled: true }))
  await expect(saveTextFileAs(request, { defaultDir: () => '/unused', showSaveDialog })).rejects.toThrow(request.error)
  expect(showSaveDialog).not.toHaveBeenCalled()
})

it('writes decoded office-package bytes only to the native picker-selected path', async () => {
  const dir = fs.mkdtempSync(path.join(process.env.TMPDIR || os.tmpdir(), 'transcript-binary-'))
  scratch.push(dir)
  const target = path.join(dir, 'chosen.docx')
  const bytes = Buffer.from([0x50, 0x4b, 0x03, 0x04, 0, 0xff, 0x80, 0x0a])

  const result = await saveTextFileAs(
    { content: bytes.toString('base64'), encoding: 'base64', defaultPath: '../../not-chosen.docx' },
    { defaultDir: () => dir, showSaveDialog: async () => ({ canceled: false, filePath: target }) }
  )

  expect(result).toEqual({ saved: true, path: target })
  expect(fs.readFileSync(target)).toEqual(bytes)
  expect(fs.existsSync(path.join(dir, 'not-chosen.docx'))).toBe(false)
})
