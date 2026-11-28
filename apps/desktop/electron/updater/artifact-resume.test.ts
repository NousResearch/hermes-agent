import { createHash } from 'node:crypto'
import { once } from 'node:events'
import { mkdtemp, readFile, rm, stat } from 'node:fs/promises'
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import type { AddressInfo } from 'node:net'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, test } from 'vitest'

import { downloadPinnedArtifact, type PinnedArtifact, PinnedDownloadStalledError } from './artifact'

// A real loopback server whose pace each test scripts per request.
type Handler = (request: IncomingMessage, response: ServerResponse, attempt: number) => void

const BYTES: Buffer = Buffer.alloc(64 * 1024, 7).map((_value: number, index: number): number => index % 251) as Buffer
const ETAG = '"v1"'
const cleanups: (() => Promise<void>)[] = []

afterEach(async (): Promise<void> => {
  for (const cleanup of cleanups.splice(0)) {
    await cleanup()
  }
})

const sleep = (ms: number): Promise<void> => new Promise(resolve => setTimeout(resolve, ms))

async function serve(
  handler: Handler
): Promise<{ artifact: PinnedArtifact; directory: string; requests: IncomingMessage[] }> {
  const requests: IncomingMessage[] = []

  const server: Server = createServer((request, response): void => {
    requests.push(request)
    handler(request, response, requests.length)
  })

  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  const directory = await mkdtemp(path.join(os.tmpdir(), 'hermes-artifact-resume-'))

  cleanups.push(async (): Promise<void> => {
    server.closeAllConnections()
    server.close()
    await rm(directory, { recursive: true, force: true })
  })

  const { port } = server.address() as AddressInfo

  return {
    artifact: {
      url: `http://127.0.0.1:${port}/stable.msixbundle`,
      sha256: createHash('sha256').update(BYTES).digest('hex'),
      size: BYTES.length,
      format: 'msixbundle'
    },
    directory,
    requests
  }
}

/** Write `body` in `chunks` pieces, `gapMs` apart; `hold` keeps the socket open after. */
async function trickle(response: ServerResponse, body: Buffer, chunks: number, gapMs: number): Promise<void> {
  const step = Math.ceil(body.length / chunks)

  for (let at = 0; at < body.length && !response.destroyed; at += step) {
    response.write(body.subarray(at, at + step))
    await sleep(gapMs)
  }
}

function rangeStart(request: IncomingMessage): number | null {
  const match = /^bytes=(\d+)-$/.exec(String(request.headers.range ?? ''))

  return match && request.headers['if-range'] === ETAG ? Number(match[1]) : null
}

test('a slow transfer that keeps moving outlives any total budget and verifies', async (): Promise<void> => {
  const { artifact, directory } = await serve((_request, response): void => {
    response.writeHead(200, { 'content-length': BYTES.length, etag: ETAG })
    void trickle(response, BYTES, 16, 60).then((): void => {
      response.end()
    })
  })

  // ~1 s in total, never idle for 300 ms: only a stall may fail a download.
  const file = await downloadPinnedArtifact(directory, artifact, { stallMs: 300, attemptsWithoutProgress: 1 })

  expect(await readFile(file)).toEqual(BYTES)
})

test('a transfer that stops moving fails as stalled; the next call resumes its partial bytes', async (): Promise<void> => {
  const half = BYTES.length / 2
  let healthy = false

  const { artifact, directory, requests } = await serve((request, response): void => {
    const start = rangeStart(request) ?? 0
    response.writeHead(start ? 206 : 200, {
      'content-length': BYTES.length - start,
      etag: ETAG,
      ...(start ? { 'content-range': `bytes ${start}-${BYTES.length - 1}/${BYTES.length}` } : {})
    })

    if (healthy) {
      response.end(BYTES.subarray(start))

      return
    }

    // Never more than the first half, then silence with the socket held open.
    if (start < half) {
      response.write(BYTES.subarray(start, half))
    }
  })

  await expect(
    downloadPinnedArtifact(directory, artifact, { stallMs: 200, attemptsWithoutProgress: 2 })
  ).rejects.toBeInstanceOf(PinnedDownloadStalledError)
  // Attempt 1 delivered half; attempt 2 resumed at half and added nothing; attempt 3 is never made.
  expect(requests.map(request => request.headers.range ?? null)).toEqual([null, `bytes=${half}-`, `bytes=${half}-`])
  expect((await stat(path.join(directory, 'destination.msixbundle.partial'))).size).toBe(half)

  // A later update attempt (the link recovered) continues from the kept bytes.
  healthy = true
  const file = await downloadPinnedArtifact(directory, artifact, { stallMs: 200, attemptsWithoutProgress: 2 })

  expect(await readFile(file)).toEqual(BYTES)
  expect(requests.at(-1)?.headers.range).toBe(`bytes=${half}-`)
})

test('a dropped connection resumes with Range/If-Range and verifies', async (): Promise<void> => {
  const third = Math.floor(BYTES.length / 3)

  const { artifact, directory, requests } = await serve((request, response, attempt): void => {
    const start = rangeStart(request) ?? 0
    response.writeHead(start ? 206 : 200, {
      'content-length': BYTES.length - start,
      etag: ETAG,
      ...(start ? { 'content-range': `bytes ${start}-${BYTES.length - 1}/${BYTES.length}` } : {})
    })

    if (attempt < 3) {
      // Killed mid-body: one more third, then the connection dies.
      response.write(BYTES.subarray(start, start + third))
      setTimeout((): void => {
        response.socket?.destroy()
      }, 150)

      return
    }

    response.end(BYTES.subarray(start))
  })

  const file = await downloadPinnedArtifact(directory, artifact, { stallMs: 2_000, attemptsWithoutProgress: 2 })

  expect(await readFile(file)).toEqual(BYTES)
  expect(requests.map(request => request.headers.range ?? null)).toEqual([
    null,
    `bytes=${third}-`,
    `bytes=${2 * third}-`
  ])
  expect(requests.slice(1).every(request => request.headers['if-range'] === ETAG)).toBe(true)
})

test('a changed object restarts from zero instead of splicing, and the digest still decides', async (): Promise<void> => {
  const half = BYTES.length / 2
  let calls = 0

  const { artifact, directory, requests } = await serve((request, response): void => {
    calls += 1

    if (calls === 1) {
      response.writeHead(200, { 'content-length': BYTES.length, etag: '"old"' })
      response.write(BYTES.subarray(0, half))
      setTimeout((): void => {
        response.socket?.destroy()
      }, 150)

      return
    }

    // If-Range names "old"; the entity is now "v1": the whole body, 200.
    expect(request.headers['if-range']).toBe('"old"')
    response.writeHead(200, { 'content-length': BYTES.length, etag: ETAG })
    response.end(BYTES)
  })

  const file = await downloadPinnedArtifact(directory, artifact, { stallMs: 2_000, attemptsWithoutProgress: 2 })

  expect(await readFile(file)).toEqual(BYTES)
  expect(requests).toHaveLength(2)
})

test('a server that ignores ranges cannot reset the retry budget by repeating bytes it already sent', async (): Promise<void> => {
  const half = BYTES.length / 2

  const { artifact, directory, requests } = await serve((_request, response, attempt): void => {
    response.writeHead(200, { 'content-length': BYTES.length, etag: ETAG })

    if (attempt > 6) {
      response.end(BYTES)

      return
    }

    // Always from zero; dropped alternately after half and after a quarter.
    response.write(BYTES.subarray(0, attempt % 2 ? half : half / 2))
    setTimeout((): void => {
      response.socket?.destroy()
    }, 150)
  })

  await expect(
    downloadPinnedArtifact(directory, artifact, { stallMs: 2_000, attemptsWithoutProgress: 2 })
  ).rejects.toThrow()
  // Attempt 1 reached half; attempts 2 and 3 never passed it.
  expect(requests).toHaveLength(3)
})

test('an attempt refused before its body is read closes that connection', async (): Promise<void> => {
  const third = Math.floor(BYTES.length / 3)
  let refusedClosed: Promise<unknown> | null = null

  const { artifact, directory } = await serve((_request, response, attempt): void => {
    if (attempt === 1) {
      response.writeHead(200, { 'content-length': BYTES.length, etag: ETAG })
      response.write(BYTES.subarray(0, third))
      setTimeout((): void => {
        response.socket?.destroy()
      }, 150)

      return
    }

    // A resume answered with the wrong range, then a body that keeps coming.
    refusedClosed = once(response, 'close')
    response.writeHead(206, { 'content-range': `bytes 0-${BYTES.length - 1}/${BYTES.length}`, etag: ETAG })
    void trickle(response, BYTES, 64, 100)
  })

  await expect(
    downloadPinnedArtifact(directory, artifact, { stallMs: 5_000, attemptsWithoutProgress: 3 })
  ).rejects.toThrow('different range')
  expect(refusedClosed).not.toBeNull()
  await expect(Promise.race([refusedClosed, sleep(1_500).then((): string => 'still open')])).resolves.not.toBe(
    'still open'
  )
})
