import { describe, expect, it } from 'vitest'

import { isTimeoutError } from '@/lib/with-timeout'

import { awaitBackendPastUpdateWait, isDesktopQuittingError } from './use-gateway-boot'

const tick = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

describe('Desktop opened while an update runs', () => {
  it('waits out main’s update park, then fails on the first budget after it ends', async () => {
    let resolveBackend!: (connection: string) => void
    const backend = new Promise<string>(resolve => (resolveBackend = resolve))

    const parkedThenReady = awaitBackendPastUpdateWait({
      isCancelled: () => false,
      isParkedForUpdate: () => true,
      message: 'Timed out connecting to Hermes backend',
      pending: backend,
      timeoutMs: 5
    })

    await tick(40)
    resolveBackend('connection')
    await expect(parkedThenReady).resolves.toBe('connection')

    let parked = true

    const parkEndsWithoutBackend = awaitBackendPastUpdateWait({
      isCancelled: () => false,
      isParkedForUpdate: () => parked,
      message: 'Timed out connecting to Hermes backend',
      pending: new Promise<string>(() => undefined),
      timeoutMs: 5
    })

    await tick(20)
    parked = false
    expect(isTimeoutError(await parkEndsWithoutBackend.catch(err => err))).toBe(true)
  })

  it('stops extending once the park outlasts main’s own update-wait cap', async () => {
    const wedged = awaitBackendPastUpdateWait({
      isCancelled: () => false,
      isParkedForUpdate: () => true,
      maxParkMs: 30,
      message: 'Timed out connecting to Hermes backend',
      pending: new Promise<string>(() => undefined),
      timeoutMs: 5
    })

    expect(isTimeoutError(await wedged.catch(err => err))).toBe(true)
  }, 1000)

  it('treats the quit-teardown rejection as a closing window, not a boot failure', () => {
    expect(
      isDesktopQuittingError(
        new Error("Error invoking remote method 'hermes:connection': Error: Hermes Desktop is quitting.")
      )
    ).toBe(true)
    expect(isDesktopQuittingError(new Error('Timed out connecting to Hermes backend'))).toBe(false)
  })
})
