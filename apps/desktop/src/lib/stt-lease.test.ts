import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ResolvedOwner } from '@/hermes'

const setSttLease = vi.fn(async (_lease: string, _active: boolean, _owner: ResolvedOwner) => ({ ok: true }))

vi.mock('@/hermes', () => ({
  setSttLease: (lease: string, active: boolean, owner: ResolvedOwner) => setSttLease(lease, active, owner)
}))

import { resetSttLeasesForTests, syncSttLease, VOICE_INPUT_LEASE } from './stt-lease'

const PRIMARY: ResolvedOwner = { connectionId: null, profile: null }
const ALPHA: ResolvedOwner = { connectionId: 'gateway-a', profile: 'worker_alpha' }
const BETA: ResolvedOwner = { connectionId: 'gateway-b', profile: 'worker_beta' }

/** Make the next setSttLease call hang until the returned function is called. */
function holdNextCall(): () => void {
  let finish: () => void = () => undefined
  setSttLease.mockImplementationOnce(
    () =>
      new Promise(resolve => {
        finish = () => resolve({ ok: true })
      })
  )

  return () => finish()
}

describe('syncSttLease', () => {
  beforeEach(() => {
    resetSttLeasesForTests()
    setSttLease.mockReset()
    setSttLease.mockImplementation(async () => ({ ok: true }))
  })

  afterEach(() => resetSttLeasesForTests())

  it('acquires on the first on and releases on off', async () => {
    await syncSttLease(VOICE_INPUT_LEASE, true, PRIMARY)
    await syncSttLease(VOICE_INPUT_LEASE, false, PRIMARY)

    expect(setSttLease.mock.calls).toEqual([
      [VOICE_INPUT_LEASE, true, PRIMARY],
      [VOICE_INPUT_LEASE, false, PRIMARY]
    ])
  })

  it('skips an initial off — never releases a lease it did not hold', async () => {
    await syncSttLease(VOICE_INPUT_LEASE, false, PRIMARY)

    expect(setSttLease).not.toHaveBeenCalled()
  })

  it('sends every settled-apart acquire — the backend, not a client clock, knows residency', async () => {
    // The idle-unload watcher can evict the model between two listening
    // starts at any moment; each start re-warms (a cache hit when resident).
    await syncSttLease(VOICE_INPUT_LEASE, true, PRIMARY)
    await syncSttLease(VOICE_INPUT_LEASE, true, PRIMARY)

    expect(setSttLease).toHaveBeenCalledTimes(2)
  })

  it('coalesces concurrent identical acquires onto the one in flight', async () => {
    const finish = holdNextCall()
    const first = syncSttLease(VOICE_INPUT_LEASE, true, PRIMARY)
    await Promise.resolve()
    const second = syncSttLease(VOICE_INPUT_LEASE, true, PRIMARY)

    expect(second).toBe(first)
    finish()
    await Promise.all([first, second])
    expect(setSttLease).toHaveBeenCalledTimes(1)
  })

  it('queues an off behind an in-flight on so the wire never sees them reordered', async () => {
    const finish = holdNextCall()
    const on = syncSttLease(VOICE_INPUT_LEASE, true, PRIMARY)
    await Promise.resolve()
    const off = syncSttLease(VOICE_INPUT_LEASE, false, PRIMARY)
    await Promise.resolve()
    expect(setSttLease).toHaveBeenCalledTimes(1)

    finish()
    await Promise.all([on, off])
    expect(setSttLease.mock.calls).toEqual([
      [VOICE_INPUT_LEASE, true, PRIMARY],
      [VOICE_INPUT_LEASE, false, PRIMARY]
    ])
  })

  it('drops a flip that reverses before its call went out — latest intent wins', async () => {
    const on = syncSttLease(VOICE_INPUT_LEASE, true, PRIMARY)
    const off = syncSttLease(VOICE_INPUT_LEASE, false, PRIMARY)
    await Promise.all([on, off])

    expect(setSttLease.mock.calls).toEqual([[VOICE_INPUT_LEASE, false, PRIMARY]])
  })

  // #128668 review: an acquire queued behind a release must not coalesce onto
  // an earlier acquire — it would resolve before the latest acquire settled
  // and leave the backend released.
  it('on → off → on across turns ends acquired, and the last on waits for its own acquire', async () => {
    await syncSttLease(VOICE_INPUT_LEASE, true, PRIMARY)
    setSttLease.mockClear()

    const pending: Array<() => void> = []
    setSttLease.mockImplementation(() => new Promise(resolve => pending.push(() => resolve({ ok: true }))))

    const on1 = syncSttLease(VOICE_INPUT_LEASE, true, PRIMARY)
    await Promise.resolve()
    const off = syncSttLease(VOICE_INPUT_LEASE, false, PRIMARY)
    const on2 = syncSttLease(VOICE_INPUT_LEASE, true, PRIMARY)
    expect(on2).not.toBe(on1)

    let on2Settled = false
    void on2.then(() => (on2Settled = true))

    pending[0]()
    await on1
    await new Promise(resolve => setTimeout(resolve, 0))
    // on2's own acquire is on the wire and unsettled — its caller still waits.
    expect(on2Settled).toBe(false)
    expect(pending).toHaveLength(2)

    pending[1]()
    await Promise.all([off, on2])

    const wire = setSttLease.mock.calls.map(([, active]) => active)
    expect(wire.at(-1)).toBe(true)
    expect(on2Settled).toBe(true)
  })

  it('an acquire queued behind an on-the-wire release is sent after it', async () => {
    await syncSttLease(VOICE_INPUT_LEASE, true, PRIMARY)
    setSttLease.mockClear()

    const finishRelease = holdNextCall()
    const off = syncSttLease(VOICE_INPUT_LEASE, false, PRIMARY)
    await Promise.resolve()
    const on = syncSttLease(VOICE_INPUT_LEASE, true, PRIMARY)

    finishRelease()
    await Promise.all([off, on])
    expect(setSttLease.mock.calls).toEqual([
      [VOICE_INPUT_LEASE, false, PRIMARY],
      [VOICE_INPUT_LEASE, true, PRIMARY]
    ])
  })

  it('forgets the sent state on failure so the next flip retries', async () => {
    setSttLease.mockImplementationOnce(async () => {
      throw new Error('backend not ready')
    })

    await expect(syncSttLease(VOICE_INPUT_LEASE, true, PRIMARY)).resolves.toBeUndefined()
    await syncSttLease(VOICE_INPUT_LEASE, true, PRIMARY)

    expect(setSttLease).toHaveBeenCalledTimes(2)
  })

  it('never rejects — recording must not depend on warm-up', async () => {
    setSttLease.mockRejectedValue(new Error('backend gone'))

    await expect(syncSttLease(VOICE_INPUT_LEASE, true, PRIMARY)).resolves.toBeUndefined()
    await expect(syncSttLease(VOICE_INPUT_LEASE, false, PRIMARY)).resolves.toBeUndefined()
  })

  it('voice-input lease is per renderer', () => {
    expect(VOICE_INPUT_LEASE).toMatch(/^desktop:voice-input:[a-z0-9]+$/)
  })

  it('keeps each owner on its own queue — A never satisfies or delays B', async () => {
    const finishA = holdNextCall()
    const a = syncSttLease(VOICE_INPUT_LEASE, true, ALPHA)
    await Promise.resolve()
    await syncSttLease(VOICE_INPUT_LEASE, true, BETA)
    finishA()
    await a
    await syncSttLease(VOICE_INPUT_LEASE, false, ALPHA)

    expect(setSttLease.mock.calls).toEqual([
      [VOICE_INPUT_LEASE, true, ALPHA],
      [VOICE_INPUT_LEASE, true, BETA],
      [VOICE_INPUT_LEASE, false, ALPHA]
    ])
  })
})
