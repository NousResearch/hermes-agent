import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const setSttLease = vi.fn(async (_lease: string, _active: boolean) => ({ ok: true }))

vi.mock('@/hermes', () => ({
  setSttLease: (lease: string, active: boolean) => setSttLease(lease, active)
}))

import { resetSttLeasesForTests, syncSttLease, VOICE_INPUT_LEASE } from './stt-lease'

describe('syncSttLease', () => {
  beforeEach(() => {
    resetSttLeasesForTests()
    setSttLease.mockReset()
    setSttLease.mockImplementation(async () => ({ ok: true }))
  })

  afterEach(() => {
    resetSttLeasesForTests()
  })

  it('acquires on the first on and releases on off', async () => {
    await syncSttLease(VOICE_INPUT_LEASE, true)
    await syncSttLease(VOICE_INPUT_LEASE, false)

    expect(setSttLease.mock.calls).toEqual([
      [VOICE_INPUT_LEASE, true],
      [VOICE_INPUT_LEASE, false]
    ])
  })

  it('skips an initial off — never releases a lease it did not hold', async () => {
    await syncSttLease(VOICE_INPUT_LEASE, false)

    expect(setSttLease).not.toHaveBeenCalled()
  })

  it('dedupes a repeat of the last sent state', async () => {
    await syncSttLease(VOICE_INPUT_LEASE, true)
    await syncSttLease(VOICE_INPUT_LEASE, true)
    await syncSttLease(VOICE_INPUT_LEASE, true)

    expect(setSttLease).toHaveBeenCalledTimes(1)
  })

  it('queues an off behind an in-flight on so the wire never sees them reordered', async () => {
    let finishAcquire: () => void = () => undefined
    setSttLease.mockImplementationOnce(
      () =>
        new Promise(resolve => {
          finishAcquire = () => resolve({ ok: true })
        })
    )

    const on = syncSttLease(VOICE_INPUT_LEASE, true)
    // Let the acquire actually go out (it runs on a microtask).
    await Promise.resolve()
    expect(setSttLease.mock.calls).toEqual([[VOICE_INPUT_LEASE, true]])

    const off = syncSttLease(VOICE_INPUT_LEASE, false)
    await Promise.resolve()
    // Still only the acquire — the release waits for it to finish.
    expect(setSttLease).toHaveBeenCalledTimes(1)

    finishAcquire()
    await Promise.all([on, off])

    expect(setSttLease.mock.calls).toEqual([
      [VOICE_INPUT_LEASE, true],
      [VOICE_INPUT_LEASE, false]
    ])
  })

  it('coalesces a flip that reverses before its call went out — latest intent wins', async () => {
    const on = syncSttLease(VOICE_INPUT_LEASE, true)
    const off = syncSttLease(VOICE_INPUT_LEASE, false)
    await Promise.all([on, off])

    // The acquire never had a chance to go out; only the terminal state is sent
    // (a release of a never-held lease is a backend no-op).
    expect(setSttLease.mock.calls).toEqual([[VOICE_INPUT_LEASE, false]])
  })

  it('forgets the sent state on failure so the next flip retries', async () => {
    setSttLease.mockImplementationOnce(async () => {
      throw new Error('backend not ready')
    })

    await expect(syncSttLease(VOICE_INPUT_LEASE, true)).resolves.toBeUndefined()
    await syncSttLease(VOICE_INPUT_LEASE, true)

    expect(setSttLease).toHaveBeenCalledTimes(2)
  })

  it('never rejects — recording must not depend on warm-up', async () => {
    setSttLease.mockRejectedValue(new Error('backend gone'))

    await expect(syncSttLease(VOICE_INPUT_LEASE, true)).resolves.toBeUndefined()
    await expect(syncSttLease(VOICE_INPUT_LEASE, false)).resolves.toBeUndefined()
  })

  it('voice-input lease is per renderer', () => {
    expect(VOICE_INPUT_LEASE).toMatch(/^desktop:voice-input:[a-z0-9]+$/)
  })
})
