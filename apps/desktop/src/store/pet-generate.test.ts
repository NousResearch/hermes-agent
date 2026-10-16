import { afterEach, describe, expect, it, vi } from 'vitest'

import type { GatewayRequest } from './pet-gallery'
import { $petGenPreview, $petGenStatus, adoptHatched, cleanupPetGenOnClose, resetPetGen } from './pet-generate'

const HATCHED = { enabled: true, slug: 'ember', displayName: 'Ember' }

function gatewayWithSelect(select: () => Promise<unknown>) {
  const request = vi.fn((method: string, _params?: Record<string, unknown>) =>
    method === 'pet.select' ? select() : Promise.resolve({})
  )

  return {
    request: request as unknown as GatewayRequest,
    removed: () => request.mock.calls.filter(([method]) => method === 'pet.remove').map(([, params]) => params)
  }
}

describe('closing the generate overlay around an adoption', () => {
  afterEach(() => resetPetGen())

  it('leaves a pet that is being adopted to the adoption', async () => {
    let finishSelect!: () => void

    const gateway = gatewayWithSelect(
      () => new Promise(resolve => (finishSelect = () => resolve({ ok: true, slug: 'ember', displayName: 'Ember' })))
    )

    $petGenPreview.set(HATCHED)
    $petGenStatus.set('preview')

    const adoption = adoptHatched(gateway.request)
    cleanupPetGenOnClose(gateway.request)
    finishSelect()

    expect(gateway.removed()).toEqual([])
    await expect(adoption).resolves.toMatchObject({ ok: true, slug: 'ember' })
  })

  it('discards the pet once a failed adoption hands it back as an unadopted preview', async () => {
    const gateway = gatewayWithSelect(() => Promise.reject(new Error('adopt failed')))

    $petGenPreview.set(HATCHED)
    $petGenStatus.set('preview')

    await expect(adoptHatched(gateway.request)).resolves.toMatchObject({ ok: false })
    cleanupPetGenOnClose(gateway.request)

    expect(gateway.removed()).toEqual([{ slug: 'ember' }])
  })
})
