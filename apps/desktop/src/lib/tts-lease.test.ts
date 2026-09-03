import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const setTtsLease = vi.fn(async (_lease: string, _active: boolean) => ({ ok: true }))

vi.mock('@/hermes', () => ({
  setTtsLease: (lease: string, active: boolean) => setTtsLease(lease, active)
}))

import { CONVERSATION_LEASE, READ_ALOUD_LEASE, resetTtsLeasesForTests, syncTtsLease } from './tts-lease'

describe('syncTtsLease', () => {
  beforeEach(() => {
    resetTtsLeasesForTests()
    setTtsLease.mockReset()
    setTtsLease.mockImplementation(async () => ({ ok: true }))
  })

  afterEach(() => {
    resetTtsLeasesForTests()
  })

  it('acquires on the first on and releases on off', async () => {
    await syncTtsLease(READ_ALOUD_LEASE, true)
    await syncTtsLease(READ_ALOUD_LEASE, false)

    expect(setTtsLease.mock.calls).toEqual([
      [READ_ALOUD_LEASE, true],
      [READ_ALOUD_LEASE, false]
    ])
  })

  it('skips an initial off — never releases a lease it did not hold', async () => {
    await syncTtsLease(CONVERSATION_LEASE, false)

    expect(setTtsLease).not.toHaveBeenCalled()
  })

  it('dedupes a repeat of the last sent state', async () => {
    await syncTtsLease(READ_ALOUD_LEASE, true)
    await syncTtsLease(READ_ALOUD_LEASE, true)
    await syncTtsLease(READ_ALOUD_LEASE, true)

    expect(setTtsLease).toHaveBeenCalledTimes(1)
  })

  it('queues an off behind an in-flight on so the wire never sees them reordered', async () => {
    let finishAcquire: () => void = () => undefined
    setTtsLease.mockImplementationOnce(
      () =>
        new Promise(resolve => {
          finishAcquire = () => resolve({ ok: true })
        })
    )

    const on = syncTtsLease(CONVERSATION_LEASE, true)
    // Let the acquire actually go out (it runs on a microtask).
    await Promise.resolve()
    expect(setTtsLease.mock.calls).toEqual([[CONVERSATION_LEASE, true]])

    const off = syncTtsLease(CONVERSATION_LEASE, false)
    await Promise.resolve()
    // Still only the acquire — the release waits for it to finish.
    expect(setTtsLease).toHaveBeenCalledTimes(1)

    finishAcquire()
    await Promise.all([on, off])

    expect(setTtsLease.mock.calls).toEqual([
      [CONVERSATION_LEASE, true],
      [CONVERSATION_LEASE, false]
    ])
  })

  it('coalesces a flip that reverses before its call went out — latest intent wins', async () => {
    const on = syncTtsLease(CONVERSATION_LEASE, true)
    const off = syncTtsLease(CONVERSATION_LEASE, false)
    await Promise.all([on, off])

    // The acquire never had a chance to go out; only the terminal state is sent
    // (a release of a never-held lease is a backend no-op).
    expect(setTtsLease.mock.calls).toEqual([[CONVERSATION_LEASE, false]])
  })

  it('forgets the sent state on failure so the next flip retries', async () => {
    setTtsLease.mockImplementationOnce(async () => {
      throw new Error('backend not ready')
    })

    await expect(syncTtsLease(READ_ALOUD_LEASE, true)).resolves.toBeUndefined()
    await syncTtsLease(READ_ALOUD_LEASE, true)

    expect(setTtsLease).toHaveBeenCalledTimes(2)
  })

  it('conversation lease is per renderer, read-aloud lease is shared', () => {
    expect(CONVERSATION_LEASE).toMatch(/^desktop:conversation:[a-z0-9]+$/)
    expect(READ_ALOUD_LEASE).toBe('desktop:read-aloud')
  })
})
