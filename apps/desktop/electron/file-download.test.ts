import { EventEmitter } from 'node:events'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { Readable } from 'node:stream'

import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  copyFileAtomically,
  downloadFilename,
  streamDownloadRequest,
  streamDownloadWithDataUrlFallback
} from './file-download'

const temporaryDirectories: string[] = []

function temporaryDirectory(): string {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-download-test-'))
  temporaryDirectories.push(directory)

  return directory
}

function downloadResponse(statusCode: number, body: string, statusMessage?: string) {
  return Object.assign(Readable.from([Buffer.from(body)]), { statusCode, statusMessage })
}

class FakeDownloadRequest extends EventEmitter {
  aborted = false

  constructor(private readonly response: NodeJS.ReadableStream) {
    super()
  }

  abort() {
    this.aborted = true
  }

  end() {
    queueMicrotask(() => this.emit('response', this.response))
  }
}

afterEach(async () => {
  vi.restoreAllMocks()
  await Promise.all(
    temporaryDirectories.splice(0).map(directory => fs.promises.rm(directory, { force: true, recursive: true }))
  )
})

describe('downloadFilename', () => {
  it('extracts safe basenames from POSIX and Windows paths', () => {
    expect(downloadFilename('/srv/reports/summary.pdf')).toBe('summary.pdf')
    expect(downloadFilename('C:\\Users\\alice\\notes.txt')).toBe('notes.txt')
    expect(downloadFilename('/srv/report\u0000.txt')).toBe('report_.txt')
    expect(downloadFilename('/')).toBe('download')
  })
})

describe('file download writes', () => {
  it('copies a local file through a sibling temporary file', async () => {
    const directory = temporaryDirectory()
    const source = path.join(directory, 'source.txt')
    const destination = path.join(directory, 'saved.txt')
    const temporary = path.join(directory, '.saved.tmp')

    await fs.promises.writeFile(source, 'new contents')
    await fs.promises.writeFile(destination, 'old contents')
    await copyFileAtomically(source, destination, { tempPath: temporary })

    await expect(fs.promises.readFile(destination, 'utf8')).resolves.toBe('new contents')
    await expect(fs.promises.stat(temporary)).rejects.toMatchObject({ code: 'ENOENT' })
  })

  it('restores an existing destination when the Windows replacement retry fails', async () => {
    const directory = temporaryDirectory()
    const source = path.join(directory, 'source.txt')
    const destination = path.join(directory, 'saved.txt')
    const temporary = path.join(directory, '.saved.tmp')
    const rename = fs.promises.rename.bind(fs.promises)
    let renameCall = 0

    await fs.promises.writeFile(source, 'new contents')
    await fs.promises.writeFile(destination, 'old contents')
    vi.spyOn(fs.promises, 'rename').mockImplementation(async (from, to) => {
      renameCall += 1

      if (renameCall === 1) {
        throw Object.assign(new Error('destination exists'), { code: 'EEXIST' })
      }

      if (renameCall === 3) {
        throw Object.assign(new Error('replacement failed'), { code: 'EIO' })
      }

      return rename(from, to)
    })

    await expect(copyFileAtomically(source, destination, { tempPath: temporary })).rejects.toThrow(
      'replacement failed'
    )

    await expect(fs.promises.readFile(destination, 'utf8')).resolves.toBe('old contents')
    await expect(fs.promises.stat(temporary)).rejects.toMatchObject({ code: 'ENOENT' })
    await expect(fs.promises.readdir(directory)).resolves.toEqual(['saved.txt', 'source.txt'])
  })

  it('streams a successful remote response to the selected destination', async () => {
    const directory = temporaryDirectory()
    const destination = path.join(directory, 'report.txt')
    const temporary = path.join(directory, '.report.tmp')
    const request = new FakeDownloadRequest(downloadResponse(200, 'streamed contents'))

    await streamDownloadRequest(request as never, destination, { inactivityTimeoutMs: 5_000, tempPath: temporary })

    await expect(fs.promises.readFile(destination, 'utf8')).resolves.toBe('streamed contents')
    expect(request.aborted).toBe(false)
  })

  it('surfaces gateway errors without leaving a partial file', async () => {
    const directory = temporaryDirectory()
    const destination = path.join(directory, 'missing.txt')
    const temporary = path.join(directory, '.missing.tmp')
    const request = new FakeDownloadRequest(downloadResponse(404, '{"detail":"File not found"}', 'Not Found'))

    await expect(
      streamDownloadRequest(request as never, destination, { inactivityTimeoutMs: 5_000, tempPath: temporary })
    ).rejects.toThrow('404: File not found')
    await expect(fs.promises.stat(destination)).rejects.toMatchObject({ code: 'ENOENT' })
    await expect(fs.promises.stat(temporary)).rejects.toMatchObject({ code: 'ENOENT' })
  })

  it('falls back to the capped data-url route only when the streaming route is missing', async () => {
    const directory = temporaryDirectory()
    const destination = path.join(directory, 'legacy.txt')
    const fallback = vi.fn(async () => `data:text/plain;base64,${Buffer.from('legacy contents').toString('base64')}`)
    const request = new FakeDownloadRequest(downloadResponse(404, '{"detail":"Not Found"}', 'Not Found'))

    await streamDownloadWithDataUrlFallback(request as never, destination, fallback, { inactivityTimeoutMs: 5_000 })

    expect(fallback).toHaveBeenCalledOnce()
    await expect(fs.promises.readFile(destination, 'utf8')).resolves.toBe('legacy contents')
  })

  it('does not hide permission failures behind the compatibility fallback', async () => {
    const directory = temporaryDirectory()
    const destination = path.join(directory, 'forbidden.txt')
    const fallback = vi.fn(async () => 'data:text/plain;base64,bm90IHVzZWQ=')
    const request = new FakeDownloadRequest(downloadResponse(403, '{"detail":"Sensitive path"}', 'Forbidden'))

    await expect(
      streamDownloadWithDataUrlFallback(request as never, destination, fallback, { inactivityTimeoutMs: 5_000 })
    ).rejects.toThrow('403: Sensitive path')
    expect(fallback).not.toHaveBeenCalled()
  })
})
