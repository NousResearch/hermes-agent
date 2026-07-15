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

  it.each(['~/generated-image.png', 'outputs/generated-image.png'])(
    'downloads supported gateway path %s before local path validation',
    async path => {
      const downloadGatewayMediaFile = vi.spyOn(media, 'downloadGatewayMediaFile').mockResolvedValue()

      Object.defineProperty(window, 'hermesDesktop', {
        configurable: true,
        value: { openExternal: vi.fn() }
      })
      $connection.set({ mode: 'remote', baseUrl: 'https://gateway.example', token: 'token' } as never)
      const { result } = renderHook(() => useOpenMediaFile(path))

      expect(result.current.downloadsRemoteFile).toBe(true)
      act(() => result.current.open())

      await waitFor(() => expect(downloadGatewayMediaFile).toHaveBeenCalledWith(path))
    }
  )

  it.each(['data:image/png;base64,ZHVtbXk=', 'javascript:alert(1)', 'blob:https://gateway.example/id'])(
    'does not treat unsupported URL source %s as a gateway filesystem path',
    async source => {
      const downloadGatewayMediaFile = vi.spyOn(media, 'downloadGatewayMediaFile').mockResolvedValue()
      const openExternal = vi.fn()

      Object.defineProperty(window, 'hermesDesktop', {
        configurable: true,
        value: { openExternal }
      })
      $connection.set({ mode: 'remote', baseUrl: 'https://gateway.example', token: 'token' } as never)
      const { result } = renderHook(() => useOpenMediaFile(source))

      expect(result.current.downloadsRemoteFile).toBe(false)
      act(() => result.current.open())

      await waitFor(() => expect(result.current.openFailed).toBe(true))
      expect(downloadGatewayMediaFile).not.toHaveBeenCalled()
      expect(openExternal).not.toHaveBeenCalled()
    }
  )

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

  it.each([
    ['C:\\Users\\alice\\image.png', 'file:///C:/Users/alice/image.png'],
    ['C:/Users/alice/image.png', 'file:///C:/Users/alice/image.png'],
    ['\\\\server\\share\\image.png', 'file://server/share/image.png']
  ])('opens Windows path %s as a local file URL', (path, expectedUrl) => {
    const openExternal = vi.fn(async () => true)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { openExternal }
    })
    const { result } = renderHook(() => useOpenMediaFile(path))

    act(() => result.current.open())

    expect(openExternal).toHaveBeenCalledWith(expectedUrl)
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

  it('ignores a rejected open request after the source changes', async () => {
    let rejectOpen: (error: Error) => void = () => undefined

    const openExternal = vi.fn(
      () =>
        new Promise<boolean>((_resolve, reject) => {
          rejectOpen = reject
        })
    )

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { openExternal }
    })

    const { rerender, result } = renderHook(({ path }) => useOpenMediaFile(path), {
      initialProps: { path: '/tmp/first.png' }
    })

    act(() => result.current.open())
    rerender({ path: '/tmp/second.png' })
    await act(async () => rejectOpen(new Error('stale failure')))

    expect(result.current.openFailed).toBe(false)
  })

  it('ignores an old rejection across an A-to-B-to-A source transition', async () => {
    let rejectFirst: (error: Error) => void = () => undefined

    const openExternal = vi.fn(
      () =>
        new Promise<boolean>((_resolve, reject) => {
          rejectFirst = reject
        })
    )

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { openExternal }
    })

    const { rerender, result } = renderHook(({ path }) => useOpenMediaFile(path), {
      initialProps: { path: '/tmp/a.png' }
    })

    act(() => result.current.open())
    rerender({ path: '/tmp/b.png' })
    rerender({ path: '/tmp/a.png' })
    await act(async () => rejectFirst(new Error('old A failed')))

    expect(result.current.openFailed).toBe(false)
  })

  it('ignores an older same-source retry after a newer request succeeds', async () => {
    let rejectFirst: (error: Error) => void = () => undefined

    const openExternal = vi
      .fn()
      .mockImplementationOnce(
        () =>
          new Promise<boolean>((_resolve, reject) => {
            rejectFirst = reject
          })
      )
      .mockResolvedValueOnce(true)

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { openExternal }
    })
    const { result } = renderHook(() => useOpenMediaFile('/tmp/retry.png'))

    act(() => {
      result.current.open()
      result.current.open()
    })
    await act(async () => rejectFirst(new Error('first retry failed')))

    expect(result.current.openFailed).toBe(false)
  })
})
