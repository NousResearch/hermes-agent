import { describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'

import { resolveResumedBusy, resumeRequestBaseline } from './resume-busy'

describe('resolveResumedBusy', () => {
  const idle = { awaitingResponse: false, busy: false, sawAssistantPayload: false }
  const streaming = { awaitingResponse: false, busy: true, sawAssistantPayload: true }
  const submitted = { awaitingResponse: true, busy: true, sawAssistantPayload: false }

  it('keeps a turn that went busy after the RPC was issued when the snapshot reports idle (#70449)', () => {
    // Started during the RPC: no cached state at request time.
    expect(resolveResumedBusy(false, streaming, undefined)).toBe(true)
    // Streamed during the RPC: the cache was rewritten after the request.
    expect(resolveResumedBusy(false, streaming, { ...streaming })).toBe(true)
    expect(resolveResumedBusy(false, streaming, idle)).toBe(true)
  })

  it('keeps a prompt the backend has not started yet, even when it predates the RPC', () => {
    expect(resolveResumedBusy(false, submitted, submitted)).toBe(true)
  })

  it('clears a busy claim older than the snapshot once the backend reports the turn over', () => {
    // A mid-turn snapshot whose terminal events were lost with its socket.
    expect(resolveResumedBusy(false, streaming, streaming)).toBe(false)
  })

  it('cannot clear anything when the snapshot does not report running (older backends)', () => {
    expect(resolveResumedBusy(undefined, streaming, streaming)).toBe(true)
    expect(resolveResumedBusy(null, streaming, streaming)).toBe(true)
  })

  it('clears busy when both the snapshot and the live cache agree the turn ended', () => {
    expect(resolveResumedBusy(false, idle, idle)).toBe(false)
    expect(resolveResumedBusy(undefined, idle, undefined)).toBe(false)
    expect(resolveResumedBusy(false, undefined, undefined)).toBe(false)
  })

  it('adopts a running turn reported by the snapshot even without live state', () => {
    expect(resolveResumedBusy(true, undefined, undefined)).toBe(true)
    expect(resolveResumedBusy(true, streaming, streaming)).toBe(true)
  })
})

describe('resumeRequestBaseline', () => {
  const streaming = { ...createClientSessionState('stored-A'), busy: true, sawAssistantPayload: true }

  it('settles a claim cached before the request went out, not one written after it', () => {
    const cache = { current: new Map([['rt-A', streaming]]) }
    const baseline = resumeRequestBaseline(cache, 'stored-A')

    expect(baseline.capture(() => 'sent')).toBe('sent')
    expect(baseline.resolve('rt-A', false)).toBe(false)

    cache.current.set('rt-A', { ...streaming })

    expect(baseline.resolve('rt-A', false)).toBe(true)
  })

  it('keeps every busy claim when it joined a resume issued by another caller', () => {
    const baseline = resumeRequestBaseline({ current: new Map([['rt-A', streaming]]) }, 'stored-A')

    expect(baseline.resolve('rt-A', false)).toBe(true)
  })
})
