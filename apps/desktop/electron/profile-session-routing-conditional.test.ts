// Split out of profile-session-routing.test.ts, which is at the file-length
// cap: the C3 conditional-revalidation cases live here so the routing file
// goes back to its size on main and the ratchet holds.

import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'

import { test } from 'vitest'

import { fetchRegistrySessionRows, fetchRemoteProfileSessions, mergeRegistrySessionRows } from './profile-session-routing'
import { createConditionalReadCache } from './session-conditional-reads'

// ── Conditional revalidation of the splice reads (C3 desktop half) ──────────
//
// The mock below stands in for main's conditional `getJsonForBackend`: it
// serves /api/sessions pages stamped with a content ETag, answers a matching
// If-None-Match with the 304 envelope (no body), and records every request so
// the tests can prove WHAT crossed the wire.

function makeEtaggingBackend(initialRows: Array<Record<string, unknown>>) {
  const state = {
    rows: [...initialRows],
    requests: [] as Array<{ path: string; ifNoneMatch: string | null; outcome: '200' | '304' }>
  }

  // Like the real server (sessions.py `_conditional_list_response`), the
  // validator covers the EXACT response bytes for THAT request — per page —
  // so an unchanged page keeps its validator while a changed one breaks.
  const etagFor = (body: unknown) => `"${createHash('sha256').update(JSON.stringify(body)).digest('hex').slice(0, 24)}"`

  const getJson = async (_descriptor: unknown, path: string, ifNoneMatch?: string) => {
    const params = new URL(path, 'http://x').searchParams
    const offset = Number(params.get('offset') || '0')
    const limit = Number(params.get('limit'))
    // Serve COPIES, like a fresh JSON parse: the caller (row tagging) may
    // mutate what it receives without touching the store the etag covers.
    const body = { sessions: state.rows.slice(offset, offset + limit).map(row => ({ ...row })), total: state.rows.length }
    const etag = etagFor(body)

    if (ifNoneMatch && ifNoneMatch === etag) {
      state.requests.push({ path, ifNoneMatch, outcome: '304' })

      return { data: null, status: 304, etag }
    }

    state.requests.push({ path, ifNoneMatch: ifNoneMatch ?? null, outcome: '200' })

    return { data: body, status: 200, etag }
  }

  const mutate = (rowIndex: number, patch: Record<string, unknown>) => {
    state.rows[rowIndex] = { ...state.rows[rowIndex], ...patch }
  }

  return { state, getJson, mutate }
}

test('a conditional registry scan revalidates every page and serves the cached join on all-304', async () => {
  const backend = makeEtaggingBackend(Array.from({ length: 150 }, (_, index) => ({ id: `s${index}`, profile: 'work', last_active: 150 - index })))
  const sources = [{ connectionId: 'gw', kind: 'local' as const, backends: [{ descriptor: { baseUrl: 'http://gw' }, profileLabel: 'work' as const }] }]
  const cache = createConditionalReadCache()
  const conditional = { cache, keyOf: (descriptor: unknown) => String((descriptor as { baseUrl: string }).baseUrl) }
  const params = new URLSearchParams({ limit: '150', offset: '0' })

  const first = await fetchRegistrySessionRows(sources, params, backend.getJson, undefined, conditional)

  // First scan: two pages (100 + 50), both fresh 200s, no validators sent.
  assert.equal(first.length, 150)
  assert.deepEqual(backend.state.requests.map(r => r.outcome), ['200', '200'])
  assert.ok(backend.state.requests.every(r => r.ifNoneMatch === null))

  const second = await fetchRegistrySessionRows(sources, params, backend.getJson, undefined, conditional)

  // Second scan: both pages revalidate as 304 — validators sent, ZERO 200
  // bodies transferred (so zero body parses), and the rows are the cached
  // join: same objects as the first scan, not a re-derivation.
  assert.deepEqual(backend.state.requests.slice(2).map(r => r.outcome), ['304', '304'])
  assert.ok(backend.state.requests.slice(2).every(r => typeof r.ifNoneMatch === 'string'))
  assert.equal(second.length, 150)
  assert.ok(second.every((row, index) => row === first[index]), 'rows must be the cached join, not a re-derivation')
})

test('a conditional registry scan picks up a changed page among 304s', async () => {
  const backend = makeEtaggingBackend(Array.from({ length: 150 }, (_, index) => ({ id: `s${index}`, profile: 'work', last_active: 150 - index })))
  const sources = [{ connectionId: 'gw', kind: 'local' as const, backends: [{ descriptor: { baseUrl: 'http://gw' }, profileLabel: 'work' as const }] }]
  const cache = createConditionalReadCache()
  const conditional = { cache, keyOf: (descriptor: unknown) => String((descriptor as { baseUrl: string }).baseUrl) }
  const params = new URLSearchParams({ limit: '150', offset: '0' })

  const first = await fetchRegistrySessionRows(sources, params, backend.getJson, undefined, conditional)
  backend.state.requests.length = 0

  backend.mutate(5, { last_active: 999999 }) // row 5 sits in the first page

  const second = await fetchRegistrySessionRows(sources, params, backend.getJson, undefined, conditional)

  // The changed page answers 200 with the new body (fresh objects for its
  // whole window, like a fresh parse); the untouched page 304s and its rows
  // keep their identity from the stored page.
  assert.deepEqual(backend.state.requests.map(r => r.outcome), ['200', '304'])
  assert.equal(second.length, 150)
  assert.equal((second[5] as { last_active: number }).last_active, 999999)
  assert.ok(second.every((row, index) => (index >= 100 ? row === first[index] : row !== first[index])), '304-page rows keep identity, 200-page rows are fresh')
})

test('mergeRegistrySessionRows adds nothing on an all-304 rescan', async () => {
  const backend = makeEtaggingBackend(Array.from({ length: 30 }, (_, index) => ({ id: `s${index}`, profile: 'work', last_active: 30 - index })))
  const sources = [{ connectionId: 'gw', kind: 'local' as const, backends: [{ descriptor: { baseUrl: 'http://gw' }, profileLabel: 'work' as const }] }]
  const cache = createConditionalReadCache()
  const conditional = { cache, keyOf: (descriptor: unknown) => String((descriptor as { baseUrl: string }).baseUrl) }
  const params = new URLSearchParams({ limit: '30', offset: '0' })

  const merged: unknown[] = []
  const profileTotals: Record<string, number> = {}

  const firstContribution = await mergeRegistrySessionRows(sources, params, backend.getJson, merged, profileTotals, {}, conditional)
  assert.equal(firstContribution, 30)
  assert.equal(merged.length, 30)

  const secondContribution = await mergeRegistrySessionRows(sources, params, backend.getJson, merged, profileTotals, {}, conditional)

  assert.equal(secondContribution, 0)
  assert.equal(merged.length, 30)
  assert.deepEqual(backend.state.requests.slice(1).map(r => r.outcome), ['304'])
})

test('a conditional remote-profile list revalidates and serves the cached derived list on all-304', async () => {
  const backend = makeEtaggingBackend(Array.from({ length: 150 }, (_, index) => ({ id: `s${index}`, last_active: 150 - index })))
  const cache = createConditionalReadCache()
  const params = new URLSearchParams({ limit: '150', offset: '0' })

  const first = await fetchRemoteProfileSessions('remote-a', params, async (_profile, _path, ifNoneMatch) => {
    // Route through the etagging backend with the profile stripped from the
    // path (the read is always /api/sessions?profile=...).
    const { pathname, searchParams } = new URL(_path, 'http://x')

    return backend.getJson({}, `${pathname}?${searchParams}`, ifNoneMatch)
  }, { conditional: { cache } })

  assert.equal(first.sessions.length, 150)
  assert.deepEqual(backend.state.requests.map(r => r.outcome), ['200', '200'])

  const second = await fetchRemoteProfileSessions('remote-a', params, async (_profile, _path, ifNoneMatch) => {
    const { pathname, searchParams } = new URL(_path, 'http://x')

    return backend.getJson({}, `${pathname}?${searchParams}`, ifNoneMatch)
  }, { conditional: { cache } })

  // All pages 304'd: the previously derived list comes back as-is — same
  // object, zero re-derivation, zero 200 bodies.
  assert.deepEqual(backend.state.requests.slice(2).map(r => r.outcome), ['304', '304'])
  assert.equal(second, first)

  backend.mutate(10, { last_active: 424242 })

  const third = await fetchRemoteProfileSessions('remote-a', params, async (_profile, _path, ifNoneMatch) => {
    const { pathname, searchParams } = new URL(_path, 'http://x')

    return backend.getJson({}, `${pathname}?${searchParams}`, ifNoneMatch)
  }, { conditional: { cache } })

  // The changed page 200s; the derived list is recomputed with the new row.
  assert.deepEqual(backend.state.requests.slice(4).map(r => r.outcome), ['200', '304'])
  assert.notEqual(third, first)
  assert.equal((third.sessions[10] as { last_active: number }).last_active, 424242)
})

test('an unconditional splice read is untouched by the revalidation machinery', async () => {
  const rows = Array.from({ length: 30 }, (_, index) => ({ id: `s${index}`, profile: 'work', last_active: 30 - index }))
  const sources = [{ connectionId: 'gw', kind: 'local' as const, backends: [{ descriptor: { baseUrl: 'http://gw' }, profileLabel: 'work' as const }] }]
  const params = new URLSearchParams({ limit: '30', offset: '0' })
  const validatorArgs: Array<string | undefined> = []

  // A legacy getJson: plain parsed body, no envelope, no validator arg read.
  const plainGetJson = async (_descriptor: unknown, path: string, ifNoneMatch?: string) => {
    validatorArgs.push(ifNoneMatch)
    const query = new URL(path, 'http://x').searchParams

    return { sessions: rows.slice(Number(query.get('offset') || '0'), Number(query.get('offset') || '0') + Number(query.get('limit'))), total: 30 }
  }

  const first = await fetchRegistrySessionRows(sources, params, plainGetJson)

  // No conditional source: If-None-Match is never sent and the plain body
  // flows through exactly as before.
  assert.equal(first.length, 30)
  assert.ok(validatorArgs.every(arg => arg === undefined))
})
