import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useMediaImage } from './use-media-image'

// useMediaImage's gateway-proxy retry (#74564): a direct inline https image
// that fails to load (client blocked from the CDN) is fetched once through
// the gateway's authenticated /api/media/proxy and swapped in as a data URL.

const CDN_URL = 'https://v3.fal.media/media/abc123'

describe('useMediaImage gateway proxy fallback (#74564)', () => {
  const api = vi.fn(async ({ path }: { path: string }) => {
    if (path.startsWith('/api/media/proxy?')) {
      return { dataUrl: 'data:image/png;base64,cHJveGllZA==' }
    }

    throw new Error(`unexpected path ${path}`)
  })

  beforeEach(() => {
    api.mockClear()
    vi.stubGlobal('window', { hermesDesktop: { api } })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('retries a failed inline https load once through the gateway proxy', async () => {
    const { result } = renderHook(() => useMediaImage(CDN_URL, 4 / 3))

    // Inline https src paints directly; no resolution effect runs.
    expect(result.current.src).toBe(CDN_URL)
    expect(api).not.toHaveBeenCalled()

    act(() => {
      result.current.onError()
    })

    expect(result.current.failed).toBe(true)
    expect(api).toHaveBeenCalledTimes(1)
    expect(api).toHaveBeenCalledWith({ path: `/api/media/proxy?url=${encodeURIComponent(CDN_URL)}` })

    await vi.waitFor(() => expect(result.current.failed).toBe(false))
    expect(result.current.src).toBe('data:image/png;base64,cHJveGllZA==')
  })

  it('does not retry a failed proxied data URL or a non-https source', async () => {
    const { result } = renderHook(() => useMediaImage(CDN_URL, 4 / 3))

    act(() => {
      result.current.onError()
    })

    await vi.waitFor(() => expect(result.current.src).toBe('data:image/png;base64,cHJveGllZA=='))

    // The proxied data URL failing must not trigger another proxy request.
    act(() => {
      result.current.onError()
    })

    expect(api).toHaveBeenCalledTimes(1)

    // A data: source never reaches the proxy at all.
    const dataUrl = renderHook(() => useMediaImage('data:image/png;base64,aGk=', 4 / 3))

    act(() => {
      dataUrl.result.current.onError()
    })

    expect(api).toHaveBeenCalledTimes(1)
  })

  it('keeps the failed state when the gateway proxy cannot fetch the image', async () => {
    api.mockRejectedValueOnce(new Error('403 Image host not allowed'))

    const { result } = renderHook(() => useMediaImage(CDN_URL, 4 / 3))

    act(() => {
      result.current.onError()
    })

    // One attempt, then it stays failed — no retry loop.
    await vi.waitFor(() => expect(api).toHaveBeenCalledTimes(1))
    expect(result.current.failed).toBe(true)
    expect(result.current.src).toBe(CDN_URL)
  })
})
