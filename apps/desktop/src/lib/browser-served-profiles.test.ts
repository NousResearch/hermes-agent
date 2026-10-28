import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesApiRequest } from '@/global'

import { createServedProfiles } from './browser-served-profiles'

type Status = { gateway_shared_with?: null | string[] }

// Each `/api/status` read waits until the test answers it, so reads can overlap.
function controlledStatus() {
  const answers: ((status: Status) => void)[] = []

  const api = vi.fn(
    (_request: HermesApiRequest) => new Promise<Status>(resolve => answers.push(resolve))
  ) as unknown as (<T>(request: HermesApiRequest) => Promise<T>) & ReturnType<typeof vi.fn>

  return { answer: (index: number, shared: string[]) => answers[index]({ gateway_shared_with: shared }), api }
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('overlapping served set reads', () => {
  it('re-reads for a newer installed list instead of settling it with the older read', async () => {
    const { answer, api } = controlledStatus()
    const served = createServedProfiles(api)

    const older = served.refresh(['default', 'solo'])
    // Another inventory read finds a profile created while the first status read is in flight.
    let newerSettled = false
    const newer = served.refresh(['default', 'solo', 'fresh']).then(() => (newerSettled = true))

    answer(0, ['default'])
    await older
    await vi.waitFor(() => expect(api).toHaveBeenCalledTimes(2))
    expect(newerSettled).toBe(false)

    answer(1, ['default', 'fresh'])
    await newer
    expect(served.serves('fresh')).toBe(true)
    expect(served.serves('solo')).toBe(false)
  })

  it('re-reads after an invalidation during a read in flight', async () => {
    const { answer, api } = controlledStatus()
    const served = createServedProfiles(api)

    const before = served.refresh(['default', 'research'])
    // The gateway reconnected (and may serve another set) while that read was out.
    served.invalidate()
    const after = served.refresh(['default', 'research'])

    answer(0, ['default', 'research'])
    await before
    await vi.waitFor(() => expect(api).toHaveBeenCalledTimes(2))

    answer(1, ['default'])
    await after
    expect(served.serves('research')).toBe(false)
  })

  it('shares one read among callers that want the same list', async () => {
    const { answer, api } = controlledStatus()
    const served = createServedProfiles(api)

    const first = served.refresh(['default', 'solo'])
    const second = served.refresh(['solo', 'default'])
    answer(0, ['default'])
    await Promise.all([first, second])

    expect(api).toHaveBeenCalledTimes(1)
    expect(served.serves('solo')).toBe(false)
  })
})

// What the gateway's Webapp answers a caller it does not count as signed in, as a signed-in tab
// whose access token expired since its last gated request reads it.
const SIGNED_OUT_STATUS = {
  auth_flows: ['cookie'],
  auth_providers: ['password'],
  auth_required: true,
  components: {},
  gateway_running: true,
  gateway_state: 'running',
  overall: 'ok',
  release_date: '2026.10.28',
  version: '0.0.0'
}

describe('a signed-out status read', () => {
  it('keeps the last served set and asks again on the next inventory read', async () => {
    const api = vi
      .fn()
      .mockResolvedValueOnce({ active_sessions: 0, gateway_shared_with: ['default'], install_id: 'i' })
      .mockResolvedValueOnce(SIGNED_OUT_STATUS)
      .mockResolvedValueOnce({ active_sessions: 0, gateway_shared_with: ['default', 'solo'], install_id: 'i' })

    const served = createServedProfiles(api as unknown as <T>(request: HermesApiRequest) => Promise<T>)

    await served.refresh(['default', 'solo'])
    expect(served.serves('solo')).toBe(false)

    // The list changed, so the set is re-read, and that read comes back signed out.
    await served.refresh(['default', 'solo', 'fresh'])
    expect(api).toHaveBeenCalledTimes(2)
    expect(served.serves('solo')).toBe(false)
    expect(served.serves('fresh')).toBe(false)

    // The signed-out read settled nothing, so the same list is read again inside the TTL.
    await served.refresh(['default', 'solo', 'fresh'])
    expect(api).toHaveBeenCalledTimes(3)
    expect(served.serves('solo')).toBe(true)
  })

  it('still reads a full body without the set as an older backend', async () => {
    const api = vi
      .fn()
      .mockResolvedValueOnce({ active_sessions: 0, gateway_shared_with: ['default'], install_id: 'i' })
      .mockResolvedValueOnce({ active_sessions: 0, auth_required: true, install_id: 'i' })

    const served = createServedProfiles(api as unknown as <T>(request: HermesApiRequest) => Promise<T>)

    await served.refresh(['default', 'solo'])
    await served.refresh(['default', 'solo', 'fresh'])
    expect(served.serves('solo')).toBe(true)
  })
})
