import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $petBusy, $petGallery, $petGalleryError, type GatewayRequest, importPet, resetPetGallery } from './pet-gallery'

const messages = { failed: 'Import failed', tooLarge: 'Too large' }

describe('pet package import', () => {
  beforeEach(() => resetPetGallery())

  it('uploads bytes and preserves cached remote gallery pets', async () => {
    const calls: Array<{ method: string; params?: Record<string, unknown> }> = []

    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'pet.import') {
        return { ok: true, slug: 'fox', displayName: 'Fox' }
      }

      throw new Error(`unexpected method: ${method}`)
    }) as GatewayRequest

    $petGallery.set({
      enabled: false,
      active: '',
      pets: [
        {
          slug: 'boba',
          displayName: 'Boba',
          installed: false,
          spritesheetUrl: 'https://petdex.dev/boba.webp'
        }
      ]
    })

    const file = new File([new Uint8Array([0, 1, 2, 255])], 'fox.zip', {
      type: 'application/zip'
    })

    await expect(importPet(request, file, messages)).resolves.toBe(true)

    const upload = calls.find(call => call.method === 'pet.import')
    expect(upload?.params).toMatchObject({ filename: 'fox.zip', dataBase64: 'AAEC/w==' })
    expect(calls.filter(call => call.method === 'pet.gallery')).toHaveLength(0)
    expect($petGallery.get()?.pets.find(pet => pet.slug === 'boba')).toMatchObject({
      displayName: 'Boba',
      spritesheetUrl: 'https://petdex.dev/boba.webp'
    })
    expect($petGallery.get()?.pets.find(pet => pet.slug === 'fox')).toMatchObject({
      createdBy: 'import',
      installed: true,
      managedLocal: true
    })
    expect($petBusy.get()).toBeNull()
  })

  it('rejects an oversized file before reading or calling the gateway', async () => {
    const arrayBuffer = vi.fn()
    const request = vi.fn() as unknown as GatewayRequest
    const file = { name: 'huge.zip', size: 32 * 1024 * 1024 + 1, arrayBuffer } as unknown as File

    await expect(importPet(request, file, messages)).resolves.toBe(false)

    expect(arrayBuffer).not.toHaveBeenCalled()
    expect(request).not.toHaveBeenCalled()
    expect($petGalleryError.get()).toBe(messages.tooLarge)
    expect($petBusy.get()).toBeNull()
  })
})
