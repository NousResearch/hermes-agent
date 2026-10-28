import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $companionInfo, setCompanionInfo } from './companion'
import { $companionGallery, adoptCompanion, type GatewayRequest, loadCompanionGallery, resetCompanionGallery } from './companion-gallery'

function localGallery() {
  return {
    enabled: true,
    active: 'boba',
    companions: [{ slug: 'boba', displayName: 'Boba', installed: true }]
  }
}

describe('companion gallery companion.info sync', () => {
  beforeEach(() => {
    resetCompanionGallery()
    setCompanionInfo({ enabled: false })
  })

  afterEach(() => {
    resetCompanionGallery()
    setCompanionInfo({ enabled: false })
    vi.restoreAllMocks()
  })

  it('uses companion.info.meta and keeps the cached spritesheet when the revision is current', async () => {
    setCompanionInfo({
      enabled: true,
      slug: 'boba',
      displayName: 'Old Boba',
      scale: 0.33,
      spritesheetBase64: 'large-sprite-payload',
      spritesheetRevision: '100:2048',
      frameW: 192,
      frameH: 208
    })

    const requestMock = vi.fn(async (method: string) => {
      if (method === 'companion.gallery') {
        return localGallery()
      }

      if (method === 'companion.info.meta') {
        return {
          enabled: true,
          slug: 'boba',
          displayName: 'Boba',
          scale: 0.5,
          spritesheetRevision: '100:2048'
        }
      }

      if (method === 'companion.info') {
        throw new Error('full companion.info should not be called for an unchanged sprite')
      }

      throw new Error(`unexpected method: ${method}`)
    })

    const request = requestMock as unknown as GatewayRequest

    await loadCompanionGallery(request)

    const methods = requestMock.mock.calls.map(([method]) => method)
    expect(methods).toContain('companion.info.meta')
    expect(methods).not.toContain('companion.info')
    expect($companionInfo.get()).toMatchObject({
      enabled: true,
      slug: 'boba',
      displayName: 'Boba',
      scale: 0.5,
      spritesheetBase64: 'large-sprite-payload',
      spritesheetRevision: '100:2048',
      frameW: 192,
      frameH: 208
    })
  })

  it('fetches full companion.info when metadata reports a new spritesheet revision', async () => {
    setCompanionInfo({
      enabled: true,
      slug: 'boba',
      displayName: 'Boba',
      scale: 0.33,
      spritesheetBase64: 'old-sprite-payload',
      spritesheetRevision: '100:2048'
    })

    const requestMock = vi.fn(async (method: string) => {
      if (method === 'companion.gallery') {
        return localGallery()
      }

      if (method === 'companion.info.meta') {
        return {
          enabled: true,
          slug: 'boba',
          displayName: 'Boba',
          scale: 0.33,
          spritesheetRevision: '101:4096'
        }
      }

      if (method === 'companion.info') {
        return {
          enabled: true,
          slug: 'boba',
          displayName: 'Boba',
          scale: 0.33,
          spritesheetBase64: 'new-sprite-payload',
          spritesheetRevision: '101:4096'
        }
      }

      throw new Error(`unexpected method: ${method}`)
    })

    const request = requestMock as unknown as GatewayRequest

    await loadCompanionGallery(request)

    const methods = requestMock.mock.calls.map(([method]) => method)
    expect(methods).toContain('companion.info.meta')
    expect(methods).toContain('companion.info')
    expect($companionInfo.get().spritesheetBase64).toBe('new-sprite-payload')
    expect($companionInfo.get().spritesheetRevision).toBe('101:4096')
  })

  it('falls back to full companion.info when an older gateway lacks metadata', async () => {
    const requestMock = vi.fn(async (method: string) => {
      if (method === 'companion.gallery') {
        return localGallery()
      }

      if (method === 'companion.info.meta') {
        throw new Error('JSON-RPC -32601: Method not found')
      }

      if (method === 'companion.info') {
        return {
          enabled: true,
          slug: 'boba',
          displayName: 'Boba from legacy gateway',
          scale: 0.4,
          spritesheetBase64: 'legacy-full-payload',
          spritesheetRevision: '99:1024'
        }
      }

      throw new Error(`unexpected method: ${method}`)
    })

    const request = requestMock as unknown as GatewayRequest

    await loadCompanionGallery(request)

    const methods = requestMock.mock.calls.map(([method]) => method)
    expect(methods).toContain('companion.info.meta')
    expect(methods).toContain('companion.info')
    expect($companionInfo.get()).toMatchObject({
      enabled: true,
      slug: 'boba',
      displayName: 'Boba from legacy gateway',
      spritesheetBase64: 'legacy-full-payload',
      spritesheetRevision: '99:1024'
    })
  })

  it('keeps mutation sync on metadata when the selected companion sprite is unchanged', async () => {
    $companionGallery.set(localGallery())
    setCompanionInfo({
      enabled: true,
      slug: 'boba',
      displayName: 'Boba',
      scale: 0.33,
      spritesheetBase64: 'large-sprite-payload',
      spritesheetRevision: '100:2048'
    })

    const requestMock = vi.fn(async (method: string) => {
      if (method === 'companion.select') {
        return { ok: true, slug: 'boba', displayName: 'Boba' }
      }

      if (method === 'companion.info.meta') {
        return {
          enabled: true,
          slug: 'boba',
          displayName: 'Boba',
          scale: 0.33,
          spritesheetRevision: '100:2048'
        }
      }

      if (method === 'companion.info') {
        throw new Error('full companion.info should not be called after an unchanged select')
      }

      throw new Error(`unexpected method: ${method}`)
    })

    const request = requestMock as unknown as GatewayRequest

    await expect(adoptCompanion(request, 'boba', 'Could not adopt companion.')).resolves.toBe(true)

    const methods = requestMock.mock.calls.map(([method]) => method)
    expect(methods).toEqual(['companion.select', 'companion.info.meta'])
    expect($companionInfo.get().spritesheetBase64).toBe('large-sprite-payload')
  })
})
