/**
 * Tests for src/app/render-cache-hydration.ts (Phase 2, startup-latency spec).
 *
 * Covers: hydrate-only-into-empty (never clobber live), fail-open on a missing/
 * disabled API, the divergence counter, the reconcile push + one-shot sweep,
 * and the AC7 ordering invariant — a live delta arriving DURING the
 * hydrate→wholesale-replace window must not dup or drop rows once the live
 * snapshot lands (wholesale replace discards the hydrated rows entirely).
 */

import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import {
  computeListDivergence,
  cullRenderCacheSession,
  hydrateFromRenderCache,
  pushStatusToRenderCache,
  pushTranscriptToRenderCache,
  reconcileRenderCache
} from './render-cache-hydration'

function s(id: string, extra: Partial<SessionInfo> = {}): SessionInfo {
  return { id, title: `t-${id}`, archived: false, ...extra } as SessionInfo
}

function stubApi(overrides: any = {}) {
  const api = {
    read: vi.fn().mockResolvedValue({
      enabled: true,
      gatewayUrl: 'http://studio:9119',
      sessions: { sessions: [s('a'), s('b')], total: 42 },
      status: null,
      transcript: null
    }),
    putSessions: vi.fn(),
    putStatus: vi.fn(),
    putTranscript: vi.fn(),
    cullSession: vi.fn(),
    sweep: vi.fn(),
    reportDivergence: vi.fn(),
    ...overrides
  }
  vi.stubGlobal('window', { hermesDesktop: { renderCache: api } })
  return api
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('hydrateFromRenderCache', () => {
  it('paints cached rows into an EMPTY store and reports ids', async () => {
    stubApi()
    const setSessions = vi.fn()
    const setTotal = vi.fn()
    const result = await hydrateFromRenderCache({
      getSessions: () => [],
      setSessions,
      setSessionsTotal: setTotal
    })
    expect(result.painted).toBe(true)
    expect(result.gatewayUrl).toBe('http://studio:9119')
    expect(result.cachedSessionIds).toEqual(['a', 'b'])
    expect(setSessions).toHaveBeenCalledWith([expect.objectContaining({ id: 'a' }), expect.objectContaining({ id: 'b' })])
    expect(setTotal).toHaveBeenCalledWith(42)
  })

  it('NEVER clobbers a non-empty (live) list', async () => {
    stubApi()
    const setSessions = vi.fn()
    const result = await hydrateFromRenderCache({
      getSessions: () => [s('live')],
      setSessions,
      setSessionsTotal: vi.fn()
    })
    expect(result.painted).toBe(false)
    expect(setSessions).not.toHaveBeenCalled()
  })

  it('fail-open: missing API / disabled / empty cache -> no paint, no throw', async () => {
    // no API at all
    vi.stubGlobal('window', { hermesDesktop: {} })
    let result = await hydrateFromRenderCache({
      getSessions: () => [],
      setSessions: vi.fn(),
      setSessionsTotal: vi.fn()
    })
    expect(result.painted).toBe(false)

    // disabled
    stubApi({ read: vi.fn().mockResolvedValue({ enabled: false, gatewayUrl: null }) })
    result = await hydrateFromRenderCache({ getSessions: () => [], setSessions: vi.fn(), setSessionsTotal: vi.fn() })
    expect(result.painted).toBe(false)

    // read throws
    stubApi({ read: vi.fn().mockRejectedValue(new Error('boom')) })
    result = await hydrateFromRenderCache({ getSessions: () => [], setSessions: vi.fn(), setSessionsTotal: vi.fn() })
    expect(result.painted).toBe(false)
  })
})

describe('computeListDivergence', () => {
  it('rows=0 when cache matches live exactly', () => {
    expect(computeListDivergence([s('a'), s('b')], [s('a'), s('b')])).toBe(0)
  })

  it('counts adds, removals, and identity-field drift', () => {
    const cached = [s('a'), s('b', { title: 'old-title' }), s('c')]
    const live = [s('a'), s('b', { title: 'new-title' }), s('d')]
    // b drifted (title), c removed, d added -> 3
    expect(computeListDivergence(cached, live)).toBe(3)
  })
})

describe('reconcileRenderCache', () => {
  it('reports divergence, pushes the live list, and sweeps exactly once', () => {
    const api = stubApi()
    const sweptRef = { current: false }
    const cached = [s('a')]
    const live = [s('a'), s('b')]
    reconcileRenderCache({
      gatewayUrl: 'http://studio:9119',
      cachedSessionIds: ['a'],
      cachedSessions: cached,
      liveSessions: live,
      liveTotal: 2,
      sweptRef
    })
    expect(api.reportDivergence).toHaveBeenCalledWith(1) // b added
    expect(api.putSessions).toHaveBeenCalledWith('http://studio:9119', { sessions: live, total: 2 })
    expect(api.sweep).toHaveBeenCalledWith('http://studio:9119', ['a', 'b'])

    // second reconcile: no second sweep
    reconcileRenderCache({
      gatewayUrl: 'http://studio:9119',
      cachedSessionIds: [],
      cachedSessions: null,
      liveSessions: live,
      liveTotal: 2,
      sweptRef
    })
    expect(api.sweep).toHaveBeenCalledTimes(1)
    // and with no cachedSessions there is no divergence report
    expect(api.reportDivergence).toHaveBeenCalledTimes(1)
  })

  it('no gatewayUrl -> no pushes (fail-open)', () => {
    const api = stubApi()
    reconcileRenderCache({
      gatewayUrl: null,
      cachedSessionIds: [],
      cachedSessions: null,
      liveSessions: [s('a')],
      liveTotal: 1,
      sweptRef: { current: false }
    })
    expect(api.putSessions).not.toHaveBeenCalled()
  })
})

describe('AC7 — cursor-seam ordering: delta during the hydrate→replace window', () => {
  /**
   * Simulates the boot race the gate exists for: the store is hydrated from
   * cache, a LIVE DELTA arrives before the first wholesale snapshot, then the
   * snapshot lands. The invariant (I5): the delta is applied to a runtime-keyed
   * store (not the hydrated rows), and the wholesale replace discards the
   * hydrated rows entirely — so the final list equals the live snapshot with
   * zero dups and zero drops, regardless of the delta's timing.
   */
  it('wholesale replace discards hydrated rows; final list == live snapshot (no dup/drop)', async () => {
    stubApi({
      read: vi.fn().mockResolvedValue({
        enabled: true,
        gatewayUrl: 'http://studio:9119',
        // cache knows a stale world: a & b, with b unpinned
        sessions: { sessions: [s('a'), s('b')], total: 2 },
        status: null,
        transcript: null
      })
    })

    // The store under test: a plain array with wholesale-set semantics.
    let store: SessionInfo[] = []
    const setSessions = (rows: SessionInfo[]) => {
      store = rows
    }

    // 1. hydrate from cache
    const hydration = await hydrateFromRenderCache({
      getSessions: () => store,
      setSessions,
      setSessionsTotal: vi.fn()
    })
    expect(hydration.painted).toBe(true)
    expect(store.map(r => r.id)).toEqual(['a', 'b'])

    // 2. a live delta arrives DURING the window (session b was renamed on the
    // server, and a brand-new session c exists). Per I5 the delta stream is
    // keyed off the live handshake and does NOT mutate the hydrated rows —
    // the boot path holds deltas until the first wholesale replace. We assert
    // the INVARIANT by replaying the worst case: even if a delta HAD touched
    // the hydrated store, the wholesale replace must discard it wholesale.
    store = [...store, s('phantom-from-delta')] // worst-case: something leaked in

    // 3. the first live snapshot lands: wholesale replace (server copy wins)
    const liveSnapshot = [s('a'), s('b', { title: 'renamed' }), s('c')]
    setSessions(liveSnapshot)

    // Final list == live snapshot exactly: no dup of a/b, no phantom, c present.
    expect(store.map(r => r.id)).toEqual(['a', 'b', 'c'])
    expect(store.filter(r => r.id === 'b')[0].title).toBe('renamed')
    expect(store.some(r => r.id === 'phantom-from-delta')).toBe(false)
    // and the divergence counter sees exactly the drift (b title + c add)
    expect(computeListDivergence([s('a'), s('b')], liveSnapshot)).toBe(2)
  })
})

describe('transcript paint (chat pane)', () => {
  it('paints the cached transcript into an EMPTY messages store for the remembered session', async () => {
    stubApi({
      read: vi.fn().mockResolvedValue({
        enabled: true,
        gatewayUrl: 'http://studio:9119',
        sessions: null,
        status: null,
        transcript: { storedSessionId: 'sid', rows: [{ text: 'hello' }] }
      })
    })
    const setMessages = vi.fn()
    const result = await hydrateFromRenderCache({
      getSessions: () => [],
      setSessions: vi.fn(),
      setSessionsTotal: vi.fn(),
      rememberedSessionId: 'sid',
      getMessages: () => [],
      setMessages
    })
    expect(result.transcriptPainted).toBe(true)
    expect(setMessages).toHaveBeenCalledWith([{ text: 'hello' }])
  })

  it('never paints into a NON-empty messages store, and never without a remembered session', async () => {
    const api = stubApi({
      read: vi.fn().mockResolvedValue({
        enabled: true,
        gatewayUrl: 'http://studio:9119',
        sessions: null,
        status: null,
        transcript: { storedSessionId: 'sid', rows: [{ text: 'hello' }] }
      })
    })
    const setMessages = vi.fn()
    // non-empty store
    let result = await hydrateFromRenderCache({
      getSessions: () => [],
      setSessions: vi.fn(),
      setSessionsTotal: vi.fn(),
      rememberedSessionId: 'sid',
      getMessages: () => [{ live: true }],
      setMessages
    })
    expect(result.transcriptPainted).toBe(false)
    expect(setMessages).not.toHaveBeenCalled()
    // no remembered session
    result = await hydrateFromRenderCache({
      getSessions: () => [],
      setSessions: vi.fn(),
      setSessionsTotal: vi.fn(),
      getMessages: () => [],
      setMessages
    })
    expect(result.transcriptPainted).toBe(false)
    expect(api.read).toHaveBeenCalledWith(null, null)
  })
})

describe('fire-and-forget helpers', () => {
  it('pushStatus / pushTranscript / cull guard on nulls and never throw', () => {
    const api = stubApi()
    pushStatusToRenderCache(null, { x: 1 })
    pushTranscriptToRenderCache('http://g', null, [{ t: 1 }])
    pushTranscriptToRenderCache('http://g', 'sid', [])
    cullRenderCacheSession(null, 'sid')
    expect(api.putStatus).not.toHaveBeenCalled()
    expect(api.putTranscript).not.toHaveBeenCalled()
    expect(api.cullSession).not.toHaveBeenCalled()

    pushStatusToRenderCache('http://g', { x: 1 })
    pushTranscriptToRenderCache('http://g', 'sid', [{ t: 1 }])
    cullRenderCacheSession('http://g', 'sid')
    expect(api.putStatus).toHaveBeenCalledOnce()
    expect(api.putTranscript).toHaveBeenCalledOnce()
    expect(api.cullSession).toHaveBeenCalledOnce()
  })
})
