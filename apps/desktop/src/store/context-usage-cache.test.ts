import { afterEach, describe, expect, it } from 'vitest'

import {
  clearContextUsageSnapshots,
  dropContextUsageSnapshot,
  dropContextUsageSnapshotEverywhere,
  loadContextUsageSnapshot,
  saveContextUsageSnapshot,
  type ContextUsageVersion
} from './context-usage-cache'

const VERSION: ContextUsageVersion = { input_tokens: 100, message_count: 4, model: 'm', output_tokens: 50 }
const USAGE = { context_max: 200_000, context_percent: 34, context_used: 68_000, model: 'm' }

afterEach(() => {
  window.localStorage.clear()
})

describe('context-usage-cache', () => {
  it('round-trips a snapshot and marks it restored', () => {
    saveContextUsageSnapshot('s1', USAGE, { profile: 'coder' }, VERSION)

    const loaded = loadContextUsageSnapshot('s1', { profile: 'coder' }, VERSION)

    expect(loaded).toMatchObject({
      context_estimated: true,
      context_max: 200_000,
      context_percent: 34,
      context_source: 'restored',
      context_used: 68_000
    })
  })

  it('isolates scopes — a twin id on another backend never paints', () => {
    saveContextUsageSnapshot('s1', USAGE, { connectionId: 'a', profile: 'coder' }, VERSION)

    expect(loadContextUsageSnapshot('s1', { connectionId: 'b', profile: 'coder' }, VERSION)).toBeNull()
    expect(loadContextUsageSnapshot('s1', { connectionId: 'a', profile: 'coder' }, VERSION)).not.toBeNull()
  })

  it('evicts on version drift — a moved-on transcript invalidates the paint', () => {
    saveContextUsageSnapshot('s1', USAGE, { profile: 'coder' }, VERSION)

    expect(loadContextUsageSnapshot('s1', { profile: 'coder' }, { ...VERSION, message_count: 5 })).toBeNull()
    // Evicted, so even the old version no longer paints.
    expect(loadContextUsageSnapshot('s1', { profile: 'coder' }, VERSION)).toBeNull()
  })

  it('evicts corrupt entries instead of painting them', () => {
    window.localStorage.setItem(
      'hermes.context-usage.v1:["", "coder", "s1"]',
      '{"usage":{"context_max":"lots"}}'
    )

    expect(loadContextUsageSnapshot('s1', { profile: 'coder' }, VERSION)).toBeNull()
  })

  it('refuses to cache empties — zero needs no cache', () => {
    saveContextUsageSnapshot('s1', { ...USAGE, context_percent: 0, context_used: 0 }, { profile: 'coder' }, VERSION)

    expect(loadContextUsageSnapshot('s1', { profile: 'coder' }, VERSION)).toBeNull()
  })

  it('drops one session without touching its twins', () => {
    saveContextUsageSnapshot('s1', USAGE, { connectionId: 'a', profile: 'coder' }, VERSION)
    saveContextUsageSnapshot('s1', USAGE, { connectionId: 'b', profile: 'coder' }, VERSION)

    dropContextUsageSnapshot('s1', { connectionId: 'a', profile: 'coder' })

    expect(loadContextUsageSnapshot('s1', { connectionId: 'a', profile: 'coder' }, VERSION)).toBeNull()
    expect(loadContextUsageSnapshot('s1', { connectionId: 'b', profile: 'coder' }, VERSION)).not.toBeNull()
  })

  it('sweeps every scope on delete, keeping other sessions', () => {
    saveContextUsageSnapshot('s1', USAGE, { connectionId: 'a', profile: 'coder' }, VERSION)
    saveContextUsageSnapshot('s1', USAGE, { profile: 'coder' }, VERSION)
    saveContextUsageSnapshot('s2', USAGE, { connectionId: 'a', profile: 'coder' }, VERSION)

    dropContextUsageSnapshotEverywhere('s1')

    expect(loadContextUsageSnapshot('s1', { connectionId: 'a', profile: 'coder' }, VERSION)).toBeNull()
    expect(loadContextUsageSnapshot('s1', { profile: 'coder' }, VERSION)).toBeNull()
    expect(loadContextUsageSnapshot('s2', { connectionId: 'a', profile: 'coder' }, VERSION)).not.toBeNull()
  })

  it('caps entries LRU-style', () => {
    for (let index = 0; index < 60; index += 1) {
      saveContextUsageSnapshot(`s${index}`, USAGE, { profile: 'coder' }, VERSION)
    }

    expect(loadContextUsageSnapshot('s0', { profile: 'coder' }, VERSION)).toBeNull()
    expect(loadContextUsageSnapshot('s59', { profile: 'coder' }, VERSION)).not.toBeNull()
  })

  it('clears everything on request', () => {
    saveContextUsageSnapshot('s1', USAGE, { profile: 'coder' }, VERSION)
    clearContextUsageSnapshots()

    expect(loadContextUsageSnapshot('s1', { profile: 'coder' }, VERSION)).toBeNull()
  })
})
