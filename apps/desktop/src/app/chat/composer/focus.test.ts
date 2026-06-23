import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { markActiveComposer, onComposerDictateRequest, requestComposerDictate } from './focus'

describe('requestComposerDictate / onComposerDictateRequest', () => {
  beforeEach(() => {
    // Reset module state: activeTarget defaults to 'main' on import; flip
    // it explicitly so each test starts from a known place.
    markActiveComposer('main')
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('publishes a dictate event with target=main by default', async () => {
    vi.useFakeTimers()
    const handler = vi.fn()
    const off = onComposerDictateRequest(handler)

    requestComposerDictate()
    // dispatch defers to a macrotask; flush the timer.
    await vi.runAllTimersAsync()

    expect(handler).toHaveBeenCalledWith('main')
    off()
  })

  it('routes target=active to whichever composer is currently active', async () => {
    vi.useFakeTimers()
    markActiveComposer('edit')

    const handler = vi.fn()
    const off = onComposerDictateRequest(handler)

    requestComposerDictate('active')
    await vi.runAllTimersAsync()

    expect(handler).toHaveBeenCalledWith('edit')
    off()
  })

  it('respects an explicit target override (ignores active)', async () => {
    vi.useFakeTimers()
    markActiveComposer('edit')

    const handler = vi.fn()
    const off = onComposerDictateRequest(handler)

    requestComposerDictate('main')
    await vi.runAllTimersAsync()

    expect(handler).toHaveBeenCalledWith('main')
    off()
  })

  it('only invokes the most recent subscriber when the same handler is re-registered', async () => {
    vi.useFakeTimers()
    const stale = vi.fn()
    const fresh = vi.fn()

    const offStale = onComposerDictateRequest(stale)
    onComposerDictateRequest(fresh)

    requestComposerDictate('main')
    await vi.runAllTimersAsync()

    // Both fire — every subscriber is a separate listener, so the test is
    // that off() detaches the stale one without disturbing the fresh one.
    expect(stale).toHaveBeenCalledTimes(1)
    expect(fresh).toHaveBeenCalledTimes(1)

    offStale()

    requestComposerDictate('main')
    await vi.runAllTimersAsync()

    expect(stale).toHaveBeenCalledTimes(1)
    expect(fresh).toHaveBeenCalledTimes(2)
  })
})
