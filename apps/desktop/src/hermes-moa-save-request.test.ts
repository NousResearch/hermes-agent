import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createMoaModelsSaveRequest, setApiRequestProfile } from './hermes'
import type { MoaConfigResponse } from './types/hermes'

function deferred<T>() {
  let reject!: (reason?: unknown) => void
  let resolve!: (value: T) => void

  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })

  return { promise, reject, resolve }
}

const moaConfig = (model: string): MoaConfigResponse => ({
  active_preset: 'default',
  aggregator: { model, provider: 'nous' },
  aggregator_temperature: null,
  default_preset: 'default',
  enabled: true,
  max_tokens: 4096,
  presets: {
    default: {
      aggregator: { model, provider: 'nous' },
      aggregator_temperature: null,
      enabled: true,
      max_tokens: 4096,
      reference_models: [{ model: `reference-${model}`, provider: 'nous' }],
      reference_temperature: null
    }
  },
  reference_models: [{ model: `reference-${model}`, provider: 'nous' }],
  reference_temperature: null
})

const saved = (body: MoaConfigResponse): MoaConfigResponse & { ok: boolean } => ({ ...body, ok: true })

describe('createMoaModelsSaveRequest', () => {
  let api: ReturnType<typeof vi.fn>

  beforeEach(() => {
    api = vi.fn()
    setApiRequestProfile(null)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
  })

  afterEach(() => {
    setApiRequestProfile(null)
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('captures profile and body before invocation, then enters the captured-profile FIFO and continues after rejection', async () => {
    const firstTransport = deferred<MoaConfigResponse & { ok: boolean }>()
    const bodyA = moaConfig('model-a')
    const bodyB = moaConfig('model-b')
    api.mockReturnValueOnce(firstTransport.promise).mockResolvedValueOnce(saved(bodyB))
    setApiRequestProfile('profile-a')

    const requestA = createMoaModelsSaveRequest(bodyA)
    const requestB = createMoaModelsSaveRequest(bodyB)

    expect(api).not.toHaveBeenCalled()

    setApiRequestProfile('profile-b')
    const writeA = requestA()
    const writeB = requestB()

    expect(api).toHaveBeenCalledTimes(1)
    expect(api).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({ body: bodyA, method: 'PUT', path: '/api/model/moa', profile: 'profile-a' })
    )

    const rejectedA = expect(writeA).rejects.toThrow('write A failed')
    firstTransport.reject(new Error('write A failed'))
    await rejectedA
    await expect(writeB).resolves.toEqual(saved(bodyB))

    expect(api).toHaveBeenCalledTimes(2)
    expect(api).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ body: bodyB, method: 'PUT', path: '/api/model/moa', profile: 'profile-a' })
    )
  })
})
