// @vitest-environment jsdom
import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import * as media from '@/lib/media'
import { $connection } from '@/store/session'

import { useOpenMediaFile } from './use-open-media-file'

describe('useOpenMediaFile', () => {
  afterEach(() => {
    $connection.set(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('labels and downloads gateway-local files in remote mode', async () => {
    const downloadGatewayMediaFile = vi.spyOn(media, 'downloadGatewayMediaFile').mockResolvedValue()

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { openExternal: vi.fn() }
    })
    $connection.set({ mode: 'remote', baseUrl: 'https://gateway.example', token: 'token' } as never)
    const { result } = renderHook(() => useOpenMediaFile('/tmp/generated-image.png'))

    expect(result.current.downloadsRemoteFile).toBe(true)
    act(() => result.current.open())

    await waitFor(() => expect(downloadGatewayMediaFile).toHaveBeenCalledWith('/tmp/generated-image.png'))
  })

  it('reports rejected native open requests', async () => {
    const openExternal = vi.fn(async () => {
      throw new Error('open failed')
    })

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { openExternal }
    })
    $connection.set({ mode: 'local' } as never)
    const { result } = renderHook(() => useOpenMediaFile('/tmp/generated-image.png'))

    act(() => result.current.open())

    await waitFor(() => expect(result.current.openFailed).toBe(true))
  })

  it('reports a missing native open capability', async () => {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {}
    })
    $connection.set({ mode: 'local' } as never)
    const { result } = renderHook(() => useOpenMediaFile('/tmp/generated-image.png'))

    act(() => result.current.open())

    await waitFor(() => expect(result.current.openFailed).toBe(true))
  })

  it('reports unsupported inline-data sources without invoking native open', async () => {
    const openExternal = vi.fn()
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { openExternal }
    })
    const { result } = renderHook(() => useOpenMediaFile('data:image/png;base64,ZHVtbXk='))

    act(() => result.current.open())

    await waitFor(() => expect(result.current.openFailed).toBe(true))
    expect(openExternal).not.toHaveBeenCalled()
  })

  it('trims HTTP sources before opening them', () => {
    const openExternal = vi.fn(async () => true)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { openExternal }
    })
    const { result } = renderHook(() => useOpenMediaFile('  https://images.example/generated.png  '))

    act(() => result.current.open())

    expect(openExternal).toHaveBeenCalledWith('https://images.example/generated.png')
  })

  it('clears a prior failure when the source changes', async () => {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {}
    })

    const { rerender, result } = renderHook(({ path }) => useOpenMediaFile(path), {
      initialProps: { path: '/tmp/first.png' }
    })

    act(() => result.current.open())
    await waitFor(() => expect(result.current.openFailed).toBe(true))
    rerender({ path: '/tmp/second.png' })

    await waitFor(() => expect(result.current.openFailed).toBe(false))
  })
})
