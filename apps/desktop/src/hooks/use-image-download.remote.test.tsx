import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { imageFilename, useImageDownload } from './use-image-download'

const source = `hermes-media://remote/${encodeURIComponent('/srv/art/図 full.png')}?connectionId=studio&profile=artist`
const originalDesktop = window.hermesDesktop

function installBridge(value: unknown) {
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value })
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  installBridge(originalDesktop)
  $connection.set(null)
})

describe('streamed image save action', () => {
  it('saves through the authenticated file bridge instead of the public URL downloader', async () => {
    const saveGatewayFile = vi.fn().mockResolvedValue({ saved: true })
    const saveImageFromUrl = vi.fn()
    installBridge({ saveGatewayFile, saveImageFromUrl })
    $connection.set({ mode: 'local', connectionId: 'local' } as never)

    const { result } = renderHook(() => useImageDownload(source))
    await act(async () => {
      await result.current.download()
    })

    expect(saveGatewayFile).toHaveBeenCalledWith({
      connectionId: 'studio',
      path: '/srv/art/図 full.png',
      profile: 'artist',
      suggestedName: '図 full.png'
    })
    expect(saveImageFromUrl).not.toHaveBeenCalled()
    expect(result.current.saving).toBe(false)
    expect(imageFilename(source)).toBe('図 full.png')
  })

  it('settles a canceled save without invoking another downloader', async () => {
    const saveGatewayFile = vi.fn().mockResolvedValue({ canceled: true, saved: false })
    const saveImageFromUrl = vi.fn()
    installBridge({ saveGatewayFile, saveImageFromUrl })

    const { result } = renderHook(() => useImageDownload(source))
    await act(async () => {
      await result.current.download()
    })

    expect(saveGatewayFile).toHaveBeenCalledOnce()
    expect(saveImageFromUrl).not.toHaveBeenCalled()
    expect(result.current.saving).toBe(false)
  })
})
