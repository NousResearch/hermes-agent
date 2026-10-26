import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it, vi } from 'vitest'

import { registerFileDataUrlRead } from './file-data-url-ipc'

const roots: string[] = []

afterEach(() => {
  for (const root of roots.splice(0)) {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'markdown-ipc-'))
  roots.push(root)
  const source = path.join(root, 'report.md')
  const image = path.join(root, 'chart.svg')
  fs.writeFileSync(source, '# Report')
  fs.writeFileSync(image, '<svg/>')
  const handle = vi.fn()
  const resolveReadPath = vi.fn((value: unknown) => String(value).replace('/backend/', `${root}${path.sep}`))
  registerFileDataUrlRead(
    { handle },
    {
      maxBytes: () => 1024,
      mimeTypeForPath: () => 'image/svg+xml',
      resolveReadPath
    }
  )
  expect(handle).toHaveBeenCalledWith('hermes:readFileDataUrl', expect.any(Function))

  return { image, read: handle.mock.calls[0][1], resolveReadPath, root, source }
}

describe('file data URL IPC', () => {
  it('bridges both image and source paths before enforcing their directory scope', async () => {
    const { read, resolveReadPath } = fixture()
    const result = await read(null, '/backend/chart.svg', '/backend/report.md', 128)
    expect(result).toBe(`data:image/svg+xml;base64,${Buffer.from('<svg/>').toString('base64')}`)
    expect(resolveReadPath.mock.calls.map(call => call[0])).toEqual(['/backend/chart.svg', '/backend/report.md'])
  })

  it('keeps the configured ceiling when a caller requests a larger limit', async () => {
    const { image, read } = fixture()
    fs.writeFileSync(image, 'x'.repeat(1025))
    await expect(read(null, image, undefined, 4096)).rejects.toMatchObject({ code: 'EFBIG' })
  })

  it('preserves the structured missing-file response', async () => {
    const { read, root } = fixture()
    const missing = path.join(root, 'missing.svg')
    await expect(read(null, missing)).resolves.toMatchObject({ ok: false, path: missing })
  })
})
