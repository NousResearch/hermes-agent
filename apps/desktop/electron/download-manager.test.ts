import { createHash } from 'node:crypto'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import http from 'node:http'
import os from 'node:os'
import path from 'node:path'

import { afterAll, beforeAll, describe, expect, test } from 'vitest'

import {
  DownloadManager,
  filenameFromHeaders,
  isRehostableDownload,
  planSegments
} from './download-manager'

// An 8 MiB body with a per-range handler that honours Range and can be told to
// ignore it, which is the two shapes planSegments has to serve.
const BODY = Buffer.alloc(8 * 1024 * 1024)

for (let index = 0; index < BODY.length; index += 1) {
  BODY[index] = index % 251
}

const DIGEST = createHash('sha256').update(BODY).digest('hex')

let server: http.Server
let baseUrl = ''
let ignoreRanges = false
let rangeRequests = 0

beforeAll(async () => {
  server = http.createServer((req, res) => {
    const headers: Record<string, string | number> = {
      'accept-ranges': ignoreRanges ? 'none' : 'bytes',
      'content-disposition': 'attachment; filename="fixture.bin"',
      'content-length': BODY.length
    }

    const range = ignoreRanges ? undefined : (req.headers.range as string | undefined)

    if (req.method === 'HEAD') {
      res.writeHead(200, headers)
      res.end()

      return
    }

    if (range) {
      rangeRequests += 1
      const match = /bytes=(\d+)-(\d*)/.exec(range)

      if (match) {
        const start = Number(match[1])
        const end = match[2] ? Number(match[2]) : BODY.length - 1
        const slice = BODY.subarray(start, end + 1)

        res.writeHead(206, {
          ...headers,
          'content-length': slice.length,
          'content-range': `bytes ${start}-${end}/${BODY.length}`
        })
        res.end(slice)

        return
      }
    }

    res.writeHead(200, headers)
    res.end(BODY)
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()

  baseUrl = `http://127.0.0.1:${typeof address === 'object' && address ? address.port : 0}/fixture.bin`
})

afterAll(async () => {
  await new Promise<void>(resolve => server.close(() => resolve()))
})

async function tmpDir() {
  return fsp.mkdtemp(path.join(os.tmpdir(), 'hermes-dl-'))
}

async function waitFor(
  manager: DownloadManager,
  id: string,
  predicate: (progress: ReturnType<DownloadManager['get']>) => boolean,
  timeoutMs = 20_000
) {
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    const progress = manager.get(id)

    if (progress && predicate(progress)) {
      return progress
    }

    await new Promise(resolve => setTimeout(resolve, 25))
  }

  throw new Error(`timed out; last state ${JSON.stringify(manager.get(id))}`)
}

describe('planSegments', () => {
  test('splits a large body into the requested connection count', () => {
    const segments = planSegments(8 * 1024 * 1024, 4)

    expect(segments).toHaveLength(4)
    expect(segments[0].start).toBe(0)
    expect(segments.at(-1)?.end).toBe(8 * 1024 * 1024 - 1)

    // Contiguous, no gaps or overlaps — a missing byte here is a corrupt file.
    for (let index = 1; index < segments.length; index += 1) {
      expect(segments[index].start).toBe(segments[index - 1].end + 1)
    }
  })

  test('never asks for more connections than the body can fill', () => {
    expect(planSegments(300 * 1024, 8)).toHaveLength(1)
  })

  test('keeps a single segment when the size is unknown', () => {
    expect(planSegments(0, 4)).toEqual([{ done: 0, end: -1, start: 0 }])
  })
})

describe('filenameFromHeaders', () => {
  test('prefers the RFC 5987 encoded filename', () => {
    expect(
      filenameFromHeaders('https://example.com/a.bin', {
        'content-disposition': "attachment; filename*=UTF-8''%E6%8A%A5%E5%91%8A.docx"
      })
    ).toBe('报告.docx')
  })

  test('falls back to the URL basename', () => {
    expect(filenameFromHeaders('https://example.com/deep/file.tar.gz', {})).toBe('file.tar.gz')
  })
})

describe('isRehostableDownload', () => {
  test('accepts http(s)', () => {
    expect(isRehostableDownload('https://example.com/a')).toBe(true)
  })

  test('rejects blob and data URLs, which only Chromium can resolve', () => {
    expect(isRehostableDownload('blob:http://localhost/abc')).toBe(false)
    expect(isRehostableDownload('data:text/plain,hi')).toBe(false)
  })
})

describe('DownloadManager', () => {
  test('downloads over several Range connections and never corrupts the body', async () => {
    ignoreRanges = false
    const directory = await tmpDir()
    const manager = new DownloadManager({ defaultDir: () => directory })
    const started = await manager.start(baseUrl, { connections: 4 })

    const done = await waitFor(manager, started.id, progress => progress.status === 'completed')

    expect(done.ranged).toBe(true)
    expect(done.receivedBytes).toBe(BODY.length)
    expect(rangeRequests).toBeGreaterThan(1)

    const written = await fsp.readFile(done.savePath)

    expect(written.length).toBe(BODY.length)
    expect(createHash('sha256').update(written).digest('hex')).toBe(DIGEST)
    // The resume sidecar must not outlive a completed download.
    expect(fs.existsSync(`${done.savePath}.hermesdl.json`)).toBe(false)
  })

  test('falls back to a single stream when the server ignores Range', async () => {
    ignoreRanges = true
    const directory = await tmpDir()
    const manager = new DownloadManager({ defaultDir: () => directory })
    const started = await manager.start(baseUrl, { connections: 4 })

    const done = await waitFor(manager, started.id, progress => progress.status === 'completed')

    expect(done.ranged).toBe(false)
    expect(done.connections).toBe(0)
    const written = await fsp.readFile(done.savePath)

    expect(createHash('sha256').update(written).digest('hex')).toBe(DIGEST)
    ignoreRanges = false
  })

  test('reports a rate and an ETA while it is moving', async () => {
    ignoreRanges = false
    const directory = await tmpDir()
    const manager = new DownloadManager({ defaultDir: () => directory })
    const started = await manager.start(baseUrl, { connections: 2 })

    const moving = await waitFor(
      manager,
      started.id,
      progress => progress.status === 'downloading' && progress.receivedBytes > 0,
      20_000
    )

    expect(moving.totalBytes).toBe(BODY.length)
    expect(moving.connections).toBeGreaterThanOrEqual(1)
    await waitFor(manager, started.id, progress => progress.status === 'completed')
  })

  test('pause keeps the partial file plus a resume sidecar, and resume finishes it', async () => {
    ignoreRanges = false
    const directory = await tmpDir()
    const manager = new DownloadManager({ defaultDir: () => directory })
    const started = await manager.start(baseUrl, { connections: 2 })

    // Pause as soon as bytes land, so a real prefix is on disk.
    await waitFor(manager, started.id, progress => progress.receivedBytes > 512 * 1024)
    manager.pause(started.id)
    const paused = await waitFor(manager, started.id, progress => progress.status === 'paused')

    expect(paused.receivedBytes).toBeGreaterThan(0)
    expect(paused.receivedBytes).toBeLessThan(BODY.length)
    expect(fs.existsSync(`${paused.savePath}.hermesdl.json`)).toBe(true)

    await manager.resume(started.id, { connections: 2 })
    const done = await waitFor(manager, started.id, progress => progress.status === 'completed')
    const written = await fsp.readFile(done.savePath)

    expect(createHash('sha256').update(written).digest('hex')).toBe(DIGEST)
  })

  test('cancel removes the partial file and its sidecar', async () => {
    ignoreRanges = false
    const directory = await tmpDir()
    const manager = new DownloadManager({ defaultDir: () => directory })
    const started = await manager.start(baseUrl, { connections: 2 })

    await waitFor(manager, started.id, progress => progress.receivedBytes > 256 * 1024)
    const target = manager.get(started.id)!.savePath
    manager.cancel(started.id)
    await waitFor(manager, started.id, progress => progress.status === 'canceled')

    expect(fs.existsSync(target)).toBe(false)
    expect(fs.existsSync(`${target}.hermesdl.json`)).toBe(false)
  })
})
