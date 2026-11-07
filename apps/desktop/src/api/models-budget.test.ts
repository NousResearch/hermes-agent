import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const transport = vi.hoisted(() => ({
  responseDelay: 0,
  reply: { model: 'test-model', effective_context_length: 262000 }
}))
vi.mock('./client', () => ({
  STARTUP_REQUEST_TIMEOUT_MS: 60000,
  capabilityScoped: (profile?: string) => ({ profile }),
  profileScoped: (profile?: string) => ({ profile }),
  hermesApi: (request: { timeoutMs: number; profile?: string }) =>
    new Promise((resolve, reject) => {
      const deadline = setTimeout(() => reject(new Error('client deadline')), request.timeoutMs)
      if (Number.isFinite(transport.responseDelay)) {
        setTimeout(() => {
          clearTimeout(deadline)
          resolve({ ...transport.reply, profile: request.profile })
        }, transport.responseDelay)
      }
    })
}))

import { getGlobalModelInfo } from './models'

describe('model information response budget', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    transport.responseDelay = 0
  })
  afterEach(() => {
    vi.clearAllTimers()
    vi.useRealTimers()
  })

  it('accepts a response just after the backend finishes its five-second probe', async () => {
    transport.responseDelay = 5500
    const response = getGlobalModelInfo('test-profile')
    const result = response.then(
      value => ({ ok: true, value }),
      error => ({ ok: false, error: error.message })
    )
    await vi.advanceTimersByTimeAsync(5500)
    expect(await result).toMatchObject({
      ok: true,
      value: { effective_context_length: 262000, profile: 'test-profile' }
    })
  })

  it('still returns a fast response immediately', async () => {
    transport.responseDelay = 6
    const response = getGlobalModelInfo()
    await vi.advanceTimersByTimeAsync(6)
    expect(await response).toMatchObject({ effective_context_length: 262000 })
  })

  it('keeps a finite deadline for a backend that never responds', async () => {
    transport.responseDelay = Infinity
    const response = getGlobalModelInfo()
    const result = response.then(
      () => ({ ok: true }),
      error => ({ ok: false, error: error.message })
    )
    await vi.advanceTimersByTimeAsync(10000)
    expect(await result).toEqual({ ok: false, error: 'client deadline' })
  })
})
