import { createHash } from 'node:crypto'
import { createServer, type Server } from 'node:http'
import type { AddressInfo } from 'node:net'

import { afterAll, beforeAll, describe, expect, it } from 'vitest'

import { conditionalResultForRenderer, isConditionalEnvelope } from './api-conditional'
import { apiJsonFetch, type ApiJsonFetchDeps } from './api-json-fetch'

// The main side of the hermes:api conditional seam (C3 desktop half,
// t_5c4b52bf), driven with the PRODUCER'S real output: the transport
// (api-json-fetch against a real HTTP server) resolves a plain
// `{ data, status, etag }`; the handler's post-processing
// (conditionalResultForRenderer — the exact step both hermes:api handler
// paths now run) must brand it into the envelope the renderer
// capability-gates on. Review round 1 found the two halves had been tested
// against two different contracts (the renderer suite mocked the IPC with
// its own constructor); this suite exists so the producer's real output must
// satisfy the consumer contract. The crossing test (branded envelope into
// the renderer's revalidation store) lives in
// src/lib/session-list-revalidation.test.ts — the ui project, where the
// src→electron import direction is established (the electron project's
// tsconfig excludes src).

const deps: ApiJsonFetchDeps = {
  defaultTimeoutMs: 5_000,
  resolveTimeout: (timeoutMs, fallback) => timeoutMs ?? fallback,
  remoteHeaders: () => ({}),
  multipart: () => {
    throw new Error('multipart is not exercised here')
  }
}

let server: Server
let baseUrl: string
let sessionRows: Array<Record<string, unknown>>

beforeAll(async () => {
  sessionRows = [
    { id: 's1', title: 'one' },
    { id: 's2', title: 'two' }
  ]

  server = createServer((req, res) => {
    const etag = `"${createHash('sha256').update(JSON.stringify(sessionRows)).digest('hex')}"`

    if (req.url?.startsWith('/api/sessions')) {
      if (req.headers['if-none-match'] === etag) {
        res.writeHead(304, { ETag: etag })
        res.end()

        return
      }

      res.writeHead(200, { 'Content-Type': 'application/json', ETag: etag })
      res.end(JSON.stringify({ sessions: sessionRows, total: sessionRows.length }))

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

describe('hermes:api conditional seam (producer real output → renderer contract)', () => {
  it('brands the transport 200 result: the renderer sees an envelope with sessions intact', async () => {
    const raw = await apiJsonFetch(`${baseUrl}/api/sessions`, 'token', { conditional: true }, deps)
    const envelope = conditionalResultForRenderer(raw)

    expect(isConditionalEnvelope(envelope)).toBe(true)
    expect(envelope.status).toBe(200)
    expect(envelope.etag).toEqual(expect.stringMatching(/^"/))
    expect(envelope.data).toEqual({ sessions: sessionRows, total: sessionRows.length })
  })

  it('brands the transport 304 result: status 304, null data, the stored validator', async () => {
    const first = await apiJsonFetch(`${baseUrl}/api/sessions`, 'token', { conditional: true }, deps) as { etag: string | null }
    const raw = await apiJsonFetch(`${baseUrl}/api/sessions`, 'token', { conditional: true, ifNoneMatch: first.etag ?? undefined }, deps)
    const envelope = conditionalResultForRenderer(raw)

    expect(isConditionalEnvelope(envelope)).toBe(true)
    expect(envelope.status).toBe(304)
    expect(envelope.etag).toBe(first.etag)
    expect(envelope.data).toBeNull()
  })

  it('brands the plain body the OAuth cookie leg resolves (200, no validator)', () => {
    const envelope = conditionalResultForRenderer({ sessions: sessionRows, total: sessionRows.length })

    expect(isConditionalEnvelope(envelope)).toBe(true)
    expect(envelope).toMatchObject({ status: 200, etag: null })
    expect(envelope.data).toEqual({ sessions: sessionRows, total: sessionRows.length })
  })

  it('runs the per-route post-processing on the payload data, not the transport envelope', async () => {
    const raw = await apiJsonFetch(`${baseUrl}/api/sessions`, 'token', { conditional: true }, deps)
    const seen: unknown[] = []

    const envelope = conditionalResultForRenderer(raw, data => {
      seen.push(data)

      return data
    })

    expect(isConditionalEnvelope(envelope)).toBe(true)
    // The registry tag (and preference recording) must see { sessions, ... } —
    // never { data, status, etag }.
    expect(seen).toHaveLength(1)
    expect(seen[0]).toEqual({ sessions: sessionRows, total: sessionRows.length })
    expect(seen[0]).not.toHaveProperty('status')
    expect(seen[0]).not.toHaveProperty('etag')
  })
})
