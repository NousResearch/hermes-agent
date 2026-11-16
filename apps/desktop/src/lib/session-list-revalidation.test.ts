import { createHash } from 'node:crypto'
import { createServer, type Server } from 'node:http'
import type { AddressInfo } from 'node:net'

import { afterAll, beforeAll, describe, expect, it } from 'vitest'

import { conditionalResultForRenderer } from '../../electron/api-conditional'
import { apiJsonFetch, type ApiJsonFetchDeps } from '../../electron/api-json-fetch'

import { createSessionListRevalidation } from './session-list-revalidation'

// The renderer side of the hermes:api conditional seam (C3 desktop half,
// t_5c4b52bf). The crossing test below drives the store with the PRODUCER'S
// real output — the main-process transport (api-json-fetch against a real
// HTTP server) through the handler's exact post-processing
// (conditionalResultForRenderer) — so the producer and the consumer are
// proven against ONE contract. Review round 1: the two halves had been
// tested separately against two different contracts (this store's suite
// mocked the IPC with the consumer's own constructor), which let an
// unbranded object reach the renderer. The main-side branding contract
// itself is covered by electron/api-conditional.test.ts (the electron
// project's tsconfig excludes src, so the crossing lives here where the
// src→electron import direction is established).

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

describe("session-list revalidation against the producer's real output", () => {
  // What the NEW main process puts on the IPC wire for one renderer list
  // read: the real transport result through the real handler post-processing.
  const fromMain = async (ifNoneMatch?: string) =>
    conditionalResultForRenderer(
      await apiJsonFetch(`${baseUrl}/api/sessions`, 'token', { conditional: true, ...(ifNoneMatch ? { ifNoneMatch } : {}) }, deps)
    )

  it('crosses the seam: 200 stores the validator, the 304 serves the cached value with referential identity', async () => {
    const revalidation = createSessionListRevalidation()
    const key = 'conn\u0000default\u0000/api/sessions?limit=40'

    const first = await revalidation.read(key, fromMain)

    expect(first.notModified).toBe(false)
    expect((first.raw as { sessions: unknown[] }).sessions).toEqual(sessionRows)
    expect(typeof first.etag).toBe('string')
    const derived = { sessions: (first.raw as { sessions: unknown[] }).sessions, total: 2, offset: 0 }

    revalidation.remember(key, first.etag as string, derived)

    const second = await revalidation.read(key, fromMain)

    // The real server answered 304 to the stored validator: the SAME derived
    // value comes back — zero re-parse, referential identity intact for the
    // downstream store no-op.
    expect(second.notModified).toBe(true)
    expect(second.value).toBe(derived)
  })

  it('serves the fresh body and the new validator when the list changed', async () => {
    const revalidation = createSessionListRevalidation()
    const key = 'conn\u0000default\u0000/api/sessions?limit=40'

    const first = await revalidation.read(key, fromMain)

    revalidation.remember(key, first.etag as string, { sessions: [] })

    sessionRows.push({ id: 's3', title: 'three' })

    const second = await revalidation.read(key, fromMain)

    expect(second.notModified).toBe(false)
    expect((second.raw as { sessions: unknown[] }).sessions).toEqual(sessionRows)
    expect(second.etag).not.toBe(first.etag)

    sessionRows.pop()
  })

  it('treats an unbranded result (older main) as a plain body and stores no validator', async () => {
    const revalidation = createSessionListRevalidation()
    const key = 'conn\u0000default\u0000/api/sessions?limit=40'
    const sent: Array<string | undefined> = []

    const plain = async (ifNoneMatch?: string) => {
      sent.push(ifNoneMatch)

      return { sessions: sessionRows, total: sessionRows.length }
    }

    const first = await revalidation.read(key, plain)
    const second = await revalidation.read(key, plain)

    expect(first).toMatchObject({ notModified: false, etag: null })
    expect(second).toMatchObject({ notModified: false, etag: null })
    // No validator was ever stored: neither read sent If-None-Match.
    expect(sent).toEqual([undefined, undefined])
  })

  it('re-reads unconditionally when a 304 arrives without a stored validator', async () => {
    const revalidation = createSessionListRevalidation()
    const key = 'conn\u0000default\u0000/api/sessions?limit=40'

    const envelope304 = conditionalResultForRenderer({ data: null, status: 304, etag: '"surprise"' })
    const envelope200 = conditionalResultForRenderer(await apiJsonFetch(`${baseUrl}/api/sessions`, 'token', { conditional: true }, deps))

    let calls = 0

    const result = await revalidation.read(key, async () => {
      calls += 1

      return calls === 1 ? envelope304 : envelope200
    })

    // The unasked-for 304 was not trusted: one unconditional re-read, and the
    // fresh body is what comes back.
    expect(calls).toBe(2)
    expect(result.notModified).toBe(false)
    expect((result.raw as { sessions: unknown[] }).sessions).toEqual(sessionRows)
  })
})
