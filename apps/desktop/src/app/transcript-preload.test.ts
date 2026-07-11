/**
 * Tests for src/app/transcript-preload.ts.
 */

import { describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { MAX_PRELOAD, preloadTranscripts, selectPreloadSessions } from './transcript-preload'

function s(id: string, extra: Partial<SessionInfo> = {}): SessionInfo {
  return { id, title: `t-${id}`, pinned: false, archived: false, ...extra } as SessionInfo
}

const noSleep = () => Promise.resolve()

describe('selectPreloadSessions', () => {
  it('pinned first, then visible order, capped', () => {
    const rows = [s('v1'), s('p1', { pinned: true }), s('v2'), s('p2', { pinned: true })]
    expect(selectPreloadSessions(rows, 3).map(r => r.id)).toEqual(['p1', 'p2', 'v1'])
  })

  it('excludes archived even when pinned', () => {
    const rows = [s('a', { pinned: true, archived: true }), s('b')]
    expect(selectPreloadSessions(rows).map(r => r.id)).toEqual(['b'])
  })

  it('default cap is sane', () => {
    const rows = Array.from({ length: 50 }, (_, i) => s(`x${i}`))
    expect(selectPreloadSessions(rows).length).toBe(MAX_PRELOAD)
  })
})

describe('preloadTranscripts', () => {
  it('fetches sequentially, pushes rows to the cache, returns the count', async () => {
    const fetched: string[] = []
    const pushed: string[] = []
    const n = await preloadTranscripts({
      gatewayUrl: 'http://g',
      sessions: [s('p1', { pinned: true }), s('v1')],
      fetchMessages: (async (id: string) => {
        fetched.push(id)
        return { messages: [{ role: 'user', content: 'hi' }] }
      }) as never,
      push: ((_url: string, id: string) => pushed.push(id)) as never,
      sleep: noSleep
    })
    expect(n).toBe(2)
    expect(fetched).toEqual(['p1', 'v1'])
    expect(pushed).toEqual(['p1', 'v1'])
  })

  it('skips fresh-cached sessions and empty transcripts', async () => {
    const pushed: string[] = []
    const n = await preloadTranscripts({
      gatewayUrl: 'http://g',
      sessions: [s('cached'), s('empty'), s('real')],
      freshCached: new Set(['cached']),
      fetchMessages: (async (id: string) => ({ messages: id === 'empty' ? [] : [{ role: 'user', content: 'hi' }] })) as never,
      push: ((_u: string, id: string) => pushed.push(id)) as never,
      sleep: noSleep
    })
    expect(n).toBe(1)
    expect(pushed).toEqual(['real'])
  })

  it('a fetch error skips that session and continues (fail-open)', async () => {
    const pushed: string[] = []
    const n = await preloadTranscripts({
      gatewayUrl: 'http://g',
      sessions: [s('bad'), s('good')],
      fetchMessages: (async (id: string) => {
        if (id === 'bad') throw new Error('boom')
        return { messages: [{ role: 'user', content: 'hi' }] }
      }) as never,
      push: ((_u: string, id: string) => pushed.push(id)) as never,
      sleep: noSleep
    })
    expect(n).toBe(1)
    expect(pushed).toEqual(['good'])
  })

  it('stops when shouldStop flips true; no gatewayUrl -> no-op', async () => {
    let calls = 0
    const n = await preloadTranscripts({
      gatewayUrl: 'http://g',
      sessions: [s('a'), s('b'), s('c')],
      fetchMessages: (async () => {
        calls += 1
        return { messages: [{ role: 'user', content: 'hi' }] }
      }) as never,
      push: (() => undefined) as never,
      sleep: noSleep,
      shouldStop: () => calls >= 1
    })
    expect(n).toBe(1)

    expect(
      await preloadTranscripts({ gatewayUrl: null, sessions: [s('a')], sleep: noSleep })
    ).toBe(0)
  })

  it('passes the session profile through to the fetch', async () => {
    const seen: Array<[string, string | null | undefined]> = []
    await preloadTranscripts({
      gatewayUrl: 'http://g',
      sessions: [s('x', { profile: 'daedalus' } as never)],
      fetchMessages: (async (id: string, profile?: string | null) => {
        seen.push([id, profile])
        return { messages: [{ role: 'user', content: 'hi' }] }
      }) as never,
      push: (() => undefined) as never,
      sleep: noSleep
    })
    expect(seen).toEqual([['x', 'daedalus']])
  })

  it('stores ChatMessage-shaped rows (parts[]), not raw SessionMessage rows', async () => {
    const pushedRows: unknown[][] = []
    await preloadTranscripts({
      gatewayUrl: 'http://g',
      sessions: [s('a')],
      fetchMessages: (async () => ({
        messages: [
          { role: 'user', content: 'hi' },
          { role: 'assistant', content: 'yo' }
        ]
      })) as never,
      push: ((_u: string, _id: string, rows: unknown[]) => pushedRows.push(rows)) as never,
      sleep: noSleep
    })
    expect(pushedRows).toHaveLength(1)
    for (const row of pushedRows[0] as Array<{ parts?: unknown }>) {
      expect(Array.isArray(row.parts)).toBe(true)
    }
  })
})
