import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { $attentionSessionIds, mergeSessionPage, sessionPinId, setSessionAttention } from './session'

const session = (over: Partial<SessionInfo>): SessionInfo => ({
  archived: false,
  cwd: null,
  ended_at: null,
  id: 'live',
  input_tokens: 0,
  is_active: false,
  last_active: 0,
  message_count: 0,
  model: null,
  output_tokens: 0,
  preview: null,
  source: null,
  started_at: 0,
  title: null,
  tool_call_count: 0,
  ...over
})

describe('setSessionAttention', () => {
  it('adds and removes a session id without duplicating it', () => {
    $attentionSessionIds.set([])

    setSessionAttention('s1', true)
    setSessionAttention('s1', true)
    expect($attentionSessionIds.get()).toEqual(['s1'])

    setSessionAttention('s2', true)
    expect($attentionSessionIds.get()).toEqual(['s1', 's2'])

    setSessionAttention('s1', false)
    expect($attentionSessionIds.get()).toEqual(['s2'])

    $attentionSessionIds.set([])
  })

  it('ignores empty ids and no-op clears', () => {
    $attentionSessionIds.set([])

    setSessionAttention(null, true)
    setSessionAttention(undefined, true)
    setSessionAttention('', true)
    setSessionAttention('missing', false)
    expect($attentionSessionIds.get()).toEqual([])
  })
})

describe('sessionPinId', () => {
  it('uses the live id when there is no compression lineage', () => {
    expect(sessionPinId(session({ id: 'abc' }))).toBe('abc')
  })

  it('uses the lineage root so a pin survives compression', () => {
    // After auto-compression the entry surfaces under a fresh tip id but keeps
    // the original root — pinning on the root keeps the pin stable.
    expect(sessionPinId(session({ id: 'tip', _lineage_root_id: 'root' }))).toBe('root')
  })
})

describe('mergeSessionPage', () => {
  it('returns the server page untouched when there is nothing to keep', () => {
    const previous = [session({ id: 'a' }), session({ id: 'b' })]
    const incoming = [session({ id: 'a' })]

    expect(mergeSessionPage(previous, incoming, [])).toBe(incoming)
  })

  it('keeps a still-working session the server omitted', () => {
    // Repro of the disappearing-sessions bug: A finished and is returned by the
    // server, but B and C are mid-first-response (message_count 0 in the DB) so
    // listSessions(min_messages=1) skips them. They must survive the refresh.
    const previous = [session({ id: 'c' }), session({ id: 'b' }), session({ id: 'a' })]
    const incoming = [session({ id: 'a', message_count: 2 })]

    const merged = mergeSessionPage(previous, incoming, ['b', 'c'])

    expect(merged.map(s => s.id)).toEqual(['c', 'b', 'a'])
    // The finished session comes from the fresh server payload, not the stale
    // optimistic copy.
    expect(merged.find(s => s.id === 'a')?.message_count).toBe(2)
  })

  it('does not duplicate a working session the server already returned', () => {
    const previous = [session({ id: 'b' }), session({ id: 'a' })]
    const incoming = [session({ id: 'b', message_count: 4 }), session({ id: 'a' })]

    const merged = mergeSessionPage(previous, incoming, ['b'])

    expect(merged.map(s => s.id)).toEqual(['b', 'a'])
    expect(merged.find(s => s.id === 'b')?.message_count).toBe(4)
  })

  it('never resurrects a session the server dropped that is not in the keep set', () => {
    // A deleted/archived session is removed from `previous` optimistically and
    // is not in the keep set, so it must stay gone after a refresh.
    const previous = [session({ id: 'b' }), session({ id: 'gone' })]
    const incoming = [session({ id: 'b' })]

    expect(mergeSessionPage(previous, incoming, ['b']).map(s => s.id)).toEqual(['b'])
  })

  it('keeps a pinned session that has aged off the recent page', () => {
    // Repro of "loses pins until you refresh": a pinned chat falls off the
    // most-recent page, so the server stops returning it. A hard replace would
    // evict it and the Pinned section would go empty. The keep set (which
    // carries pinned ids) must hold it in memory.
    const previous = [session({ id: 'recent' }), session({ id: 'pinned' })]
    const incoming = [session({ id: 'recent' })]

    const merged = mergeSessionPage(previous, incoming, ['pinned'])

    expect(merged.map(s => s.id)).toEqual(['pinned', 'recent'])
  })

  it('keeps a pinned session matched by its lineage root after compression', () => {
    // The pin is stored on the lineage-root id, but the loaded row surfaces
    // under its live compression tip. Matching on _lineage_root_id keeps it.
    const previous = [session({ id: 'tip', _lineage_root_id: 'root' })]
    const incoming = [session({ id: 'other' })]

    const merged = mergeSessionPage(previous, incoming, ['root'])

    expect(merged.map(s => s.id)).toEqual(['tip', 'other'])
  })
})
