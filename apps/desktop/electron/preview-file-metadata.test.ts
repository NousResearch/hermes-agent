import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { previewFileMetadata, previewFileStat, previewTargetNeedsBase } from './preview-file-metadata'

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void

  const promise = new Promise<T>((done, fail) => {
    resolve = done
    reject = fail
  })

  return { promise, reject, resolve }
}

let scratch: string

beforeEach(async () => {
  scratch = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hermes-preview-metadata-'))
})

afterEach(async () => {
  vi.restoreAllMocks()
  await fs.promises.rm(scratch, { force: true, recursive: true })
})

async function fixture(name: string, contents: string | Buffer) {
  const filePath = path.join(scratch, name)
  await fs.promises.writeFile(filePath, contents)

  return filePath
}

describe('previewTargetNeedsBase', () => {
  it.each([
    ['/cloud/report.pdf', 'linux', false],
    ['notes/report.pdf', 'linux', true],
    ['file:///cloud/report.pdf', 'linux', false],
    ['C:/cloud/report.pdf', 'win32', false],
    ['C:\\cloud\\report.pdf', 'win32', false],
    ['\\\\server\\share\\report.pdf', 'win32', false],
    ['//server/share/report.pdf', 'win32', false],
    ['\\report.pdf', 'win32', true],
    ['/report.pdf', 'win32', true],
    ['C:report.pdf', 'win32', true],
    ['report.pdf', 'win32', true],
    ['file:///C:/cloud/report.pdf', 'win32', false]
  ] as const)('preserves the base-directory need for %s on %s', (filePath, platform, needsBase) => {
    expect(previewTargetNeedsBase(filePath, platform)).toBe(needsBase)
  })
})

describe('previewFileMetadata', () => {
  it.each([
    ['cloud.pdf', 'application/pdf'],
    ['cloud.pdf', 'application/octet-stream'],
    ['cloud.png', 'image/png'],
    ['cloud.html', 'text/html']
  ])('classifies %s without opening content that could recall a cloud file', async (name, mimeType) => {
    const filePath = await fixture(name, Buffer.from([37, 80, 68, 70, 0, 1]))
    const syncOpen = vi.spyOn(fs, 'openSync')
    const asyncOpen = vi.spyOn(fs.promises, 'open')

    await expect(previewFileMetadata(filePath, mimeType)).resolves.toEqual({
      binary: false,
      byteSize: 6,
      large: false
    })
    expect(syncOpen).not.toHaveBeenCalled()
    expect(asyncOpen).not.toHaveBeenCalled()
  })

  it('keeps text and binary classification with real local files and size limits', async () => {
    const text = await fixture('notes.txt', 'hello\tworld\n汉字')
    const binary = await fixture('unknown.dat', Buffer.from([0, 1, 2, 3]))

    await expect(previewFileMetadata(text, 'text/plain', { maxTextBytes: 4 })).resolves.toEqual({
      binary: false,
      byteSize: Buffer.byteLength('hello\tworld\n汉字'),
      large: true
    })
    await expect(previewFileMetadata(binary, 'application/octet-stream')).resolves.toEqual({
      binary: true,
      byteSize: 4,
      large: false
    })
  })

  it('keeps empty and missing files best-effort without opening an empty file', async () => {
    const empty = await fixture('empty.txt', '')
    const open = vi.spyOn(fs.promises, 'open')

    await expect(previewFileMetadata(empty, 'text/plain')).resolves.toEqual({
      binary: false,
      byteSize: 0,
      large: false
    })
    await expect(previewFileMetadata(path.join(scratch, 'missing.txt'), 'text/plain')).resolves.toEqual({
      binary: false,
      byteSize: 0,
      large: false
    })
    expect(open).not.toHaveBeenCalled()
  })

  it('returns on the deadline and closes a late-opened handle without starting a read', async () => {
    const filePath = await fixture('late-open.dat', Buffer.from([0, 1, 2, 3]))
    const handle = await fs.promises.open(filePath, 'r')
    const lateOpen = deferred<typeof handle>()
    const read = vi.spyOn(handle, 'read')
    const close = vi.spyOn(handle, 'close')
    vi.spyOn(fs.promises, 'open').mockReturnValueOnce(lateOpen.promise)

    await expect(previewFileMetadata(filePath, 'application/octet-stream', { timeoutMs: 25 })).resolves.toEqual({
      binary: false,
      byteSize: 4,
      large: false
    })
    lateOpen.resolve(handle)
    await vi.waitFor(() => expect(close).toHaveBeenCalledOnce())
    expect(read).not.toHaveBeenCalled()
  })

  it.each(['success', 'failure'] as const)(
    'stays responsive during a delayed read and closes after late %s',
    async outcome => {
      const filePath = await fixture('late-read.dat', Buffer.from([0, 1, 2, 3]))
      const handle = await fs.promises.open(filePath, 'r')
      const lateRead = deferred<{ bytesRead: number; buffer: Buffer }>()
      const readStarted = deferred<Buffer>()
      const close = vi.spyOn(handle, 'close')
      vi.spyOn(fs.promises, 'open').mockResolvedValueOnce(handle)
      vi.spyOn(handle, 'read').mockImplementationOnce(options => {
        const buffer = Buffer.isBuffer(options) ? options : options?.buffer

        if (!Buffer.isBuffer(buffer)) {
          throw new Error('Expected a preview sample buffer')
        }

        readStarted.resolve(buffer)

        return lateRead.promise
      })

      const metadata = previewFileMetadata(filePath, 'application/octet-stream', { stat: { size: 4 }, timeoutMs: 25 })
      let heartbeat = false

      const heartbeatTimer = setTimeout(() => {
        heartbeat = true
      }, 0)

      await expect(metadata).resolves.toEqual({ binary: false, byteSize: 4, large: false })
      clearTimeout(heartbeatTimer)
      expect(heartbeat).toBe(true)
      const buffer = await readStarted.promise

      if (outcome === 'failure') {
        lateRead.reject(new Error('cloud provider unavailable'))
      } else {
        buffer[0] = 0
        lateRead.resolve({ buffer, bytesRead: 4 })
      }

      await vi.waitFor(() => expect(close).toHaveBeenCalledOnce())
      expect(await metadata).toEqual({ binary: false, byteSize: 4, large: false })
    }
  )

  it('bounds a stalled metadata stat without opening file content', async () => {
    const filePath = await fixture('late-stat.dat', Buffer.from([0, 1, 2, 3]))
    const lateStat = deferred<fs.Stats>()
    vi.spyOn(fs.promises, 'stat').mockReturnValueOnce(lateStat.promise as ReturnType<typeof fs.promises.stat>)
    const open = vi.spyOn(fs.promises, 'open')

    await expect(previewFileMetadata(filePath, 'application/octet-stream', { timeoutMs: 25 })).resolves.toEqual({
      binary: false,
      byteSize: 0,
      large: false
    })
    lateStat.reject(new Error('metadata unavailable'))
    await new Promise(resolve => setImmediate(resolve))
    expect(open).not.toHaveBeenCalled()
  })
})

describe('previewFileStat', () => {
  it('returns a found verdict with real metadata', async () => {
    const filePath = await fixture('found.txt', 'text')

    await expect(previewFileStat(filePath)).resolves.toMatchObject({ kind: 'found', stat: { size: 4 } })
  })

  it('returns missing only for a path that does not exist or crosses a non-directory', async () => {
    const filePath = await fixture('file.txt', 'text')

    await expect(previewFileStat(path.join(scratch, 'missing.txt'))).resolves.toEqual({ kind: 'missing' })
    await expect(previewFileStat(path.join(filePath, 'child.txt'))).resolves.toEqual({ kind: 'missing' })
  })

  it('keeps a permissions failure unavailable so the card can retry', async () => {
    const filePath = await fixture('denied.txt', 'text')
    vi.spyOn(fs.promises, 'stat').mockRejectedValueOnce(Object.assign(new Error('access denied'), { code: 'EACCES' }))

    await expect(previewFileStat(filePath)).resolves.toEqual({ kind: 'unavailable' })
  })

  it('keeps a timeout unavailable even when a late stat fails as missing', async () => {
    const filePath = await fixture('cloud.txt', 'text')
    const lateStat = deferred<fs.Stats>()
    vi.spyOn(fs.promises, 'stat').mockReturnValueOnce(lateStat.promise as ReturnType<typeof fs.promises.stat>)
    const result = previewFileStat(filePath, 25)

    await expect(result).resolves.toEqual({ kind: 'unavailable' })
    lateStat.reject(Object.assign(new Error('late provider failure'), { code: 'ENOENT' }))
    await new Promise(resolve => setImmediate(resolve))
    await expect(result).resolves.toEqual({ kind: 'unavailable' })
  })
})
