import { createHash } from 'node:crypto'
import { createServer, type Server } from 'node:http'
import type { AddressInfo } from 'node:net'

import { afterAll, beforeAll, describe, expect, it } from 'vitest'

import { apiJsonFetch, type ApiJsonFetchDeps } from './api-json-fetch'

// The main-process read seam (C3 desktop half, t_5c4b52bf) against a REAL local
// HTTP server: the 200 → 304 revalidation pair, zero body on the 304, the
// changed-content 200, and — the load-bearing compatibility guarantee — the
// non-conditional path behaving EXACTLY as before (a 304 there is still a
// failure, 2xx still resolves the raw parsed body).

const deps: ApiJsonFetchDeps = {
  defaultTimeoutMs: 5_000,
  resolveTimeout: (timeoutMs, fallback) => timeoutMs ?? fallback,
  remoteHeaders: () => ({}),
  multipart: () => {
    throw new Error('multipart is not exercised here')
  }
}

interface RecordedRequest {
  ifNoneMatch: string | null
  url: string
}

let server: Server
let baseUrl: string
let sessionRows: Array<Record<string, unknown>>
const recorded: RecordedRequest[] = []
// Bytes of response body actually written per status, so a 304's "no body"
// claim is measured on the wire, not asserted in prose.
const bodyBytesByStatus = new Map<number, number>()

beforeAll(async () => {
  sessionRows = [
    { id: 's1', title: 'one' },
    { id: 's2', title: 'two' }
  ]

  server = createServer((req, res) => {
    recorded.push({ ifNoneMatch: req.headers['if-none-match'] ?? null, url: req.url || '' })

    const etag = `"${createHash('sha256').update(JSON.stringify(sessionRows)).digest('hex')}"`

    if (req.url?.startsWith('/api/sessions')) {
      if (req.headers['if-none-match'] === etag) {
        bodyBytesByStatus.set(304, (bodyBytesByStatus.get(304) ?? 0) + 0)
        res.writeHead(304, { ETag: etag })
        res.end()

        return
      }

      const body = JSON.stringify({ sessions: sessionRows, total: sessionRows.length })

      bodyBytesByStatus.set(200, (bodyBytesByStatus.get(200) ?? 0) + body.length)
      res.writeHead(200, { 'Content-Type': 'application/json', ETag: etag })
      res.end(body)

      return
    }

    if (req.url?.startsWith('/always-304')) {
      bodyBytesByStatus.set(304, (bodyBytesByStatus.get(304) ?? 0) + 0)
      res.writeHead(304, { ETag: '"fixed"' })
      res.end()

      return
    }

    if (req.url?.startsWith('/bad')) {
      const body = JSON.stringify({ detail: 'nope' })

      bodyBytesByStatus.set(400, (bodyBytesByStatus.get(400) ?? 0) + body.length)
      res.writeHead(400, { 'Content-Type': 'application/json' })
      res.end(body)

      return
    }

    res.writeHead(404)
    res.end()
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`
})

afterAll(async () => {
  await new Promise(resolve => server.close(resolve))
})

describe('apiJsonFetch conditional seam', () => {
  it('resolves a conditional 200 as { data, status, etag } with the parsed body', async () => {
    const result = await apiJsonFetch(`${baseUrl}/api/sessions`, 'token', { conditional: true }, deps)

    expect(result).toEqual({
      data: { sessions: sessionRows, total: 2 },
      status: 200,
      etag: expect.stringMatching(/^"/)
    })
    expect(recorded.at(-1)?.ifNoneMatch).toBeNull()
  })

  it('answers a matching If-None-Match with a 304 envelope and transfers no body', async () => {
    const first = (await apiJsonFetch(`${baseUrl}/api/sessions`, 'token', { conditional: true }, deps)) as {
      data: unknown
      status: number
      etag: string | null
    }

    const before = bodyBytesByStatus.get(304) ?? 0
    const second = await apiJsonFetch(`${baseUrl}/api/sessions`, 'token', { conditional: true, ifNoneMatch: first.etag ?? undefined }, deps)

    expect(second).toEqual({ data: null, status: 304, etag: first.etag })
    expect(recorded.at(-1)?.ifNoneMatch).toBe(first.etag)
    // The 304 wrote zero body bytes: nothing crossed the wire to parse.
    expect(bodyBytesByStatus.get(304) ?? 0).toBe(before)
  })

  it('returns a fresh 200 body when the list changed', async () => {
    const first = (await apiJsonFetch(`${baseUrl}/api/sessions`, 'token', { conditional: true }, deps)) as { etag: string | null }

    sessionRows.push({ id: 's3', title: 'three' })

    const second = await apiJsonFetch(`${baseUrl}/api/sessions`, 'token', { conditional: true, ifNoneMatch: first.etag ?? undefined }, deps)

    expect(second).toMatchObject({ status: 200, data: { sessions: sessionRows, total: 3 } })
    expect((second as { etag: string }).etag).not.toBe(first.etag)
  })

  it('rejects a 304 on the NON-conditional path exactly as before', async () => {
    await expect(apiJsonFetch(`${baseUrl}/always-304`, 'token', {}, deps)).rejects.toThrow(/redirected \(status 304\)/)
  })

  it('resolves the raw parsed body on the non-conditional 200 path (byte-identical contract)', async () => {
    const result = await apiJsonFetch(`${baseUrl}/api/sessions`, 'token', {}, deps)

    expect(result).toEqual({ sessions: sessionRows, total: sessionRows.length })
    expect(recorded.at(-1)?.ifNoneMatch).toBeNull()
  })

  it('keeps the 4xx error shape untouched for conditional reads', async () => {
    await expect(apiJsonFetch(`${baseUrl}/bad`, 'token', { conditional: true }, deps)).rejects.toMatchObject({
      statusCode: 400,
      message: expect.stringContaining('400:')
    })
  })

  it('sends no If-None-Match header when the option is absent', async () => {
    await apiJsonFetch(`${baseUrl}/api/sessions`, 'token', { conditional: true }, deps)

    expect(recorded.at(-1)?.ifNoneMatch).toBeNull()
  })
})
