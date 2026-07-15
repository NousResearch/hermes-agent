// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { GeneratedImage } from './generated-image-result'

const useImageDownload = vi.hoisted(() => vi.fn(() => ({ download: vi.fn(), saving: false })))

vi.mock('@/hooks/use-image-download', () => ({ useImageDownload }))
vi.mock('@/components/chat/image-generation-placeholder', () => ({ DiffusionCanvas: () => null }))

describe('GeneratedImage actions', () => {
  afterEach(() => {
    $connection.set(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  it('retains the original filename and exposes the local full file', async () => {
    const openExternal = vi.fn(async () => true)
    const readFileDataUrl = vi.fn(async () => 'data:image/png;base64,ZHVtbXk=')

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { openExternal, readFileDataUrl }
    })
    $connection.set({ mode: 'local' } as never)

    render(<GeneratedImage result={{ image: '/tmp/generated-image.png', success: true }} />)

    await waitFor(() =>
      expect(useImageDownload).toHaveBeenLastCalledWith('data:image/png;base64,ZHVtbXk=', 'generated-image.png')
    )

    fireEvent.click(screen.getByRole('button', { name: 'Open full file' }))
    expect(openExternal).toHaveBeenCalledWith('file:///tmp/generated-image.png')
  })

  it('opens remote HTTP results externally instead of treating them as gateway file paths', () => {
    const openExternal = vi.fn(async () => true)

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { openExternal }
    })
    $connection.set({ mode: 'remote', baseUrl: 'https://gateway.example', token: 'token' } as never)

    render(<GeneratedImage result={{ image: 'https://images.example/generated.png', success: true }} />)

    fireEvent.click(screen.getByRole('button', { name: 'Open full file' }))
    expect(openExternal).toHaveBeenCalledWith('https://images.example/generated.png')
  })

  it('labels gateway-local paths as downloads in remote mode', () => {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { openExternal: vi.fn() }
    })
    $connection.set({ mode: 'remote', baseUrl: 'https://gateway.example', token: 'token' } as never)
    vi.stubGlobal(
      'fetch',
      vi.fn(() => new Promise(() => undefined))
    )

    render(<GeneratedImage result={{ image: '/tmp/generated.png', success: true }} />)

    expect(screen.getByRole('button', { name: 'Download full file' })).toBeTruthy()
  })

  it('does not offer a full-file action when the result only contains inline image data', () => {
    const image = 'data:image/png;base64,ZHVtbXk='
    render(<GeneratedImage result={{ image, success: true }} />)

    expect(screen.queryByRole('button', { name: 'Open full file' })).toBeNull()
    expect(useImageDownload).toHaveBeenLastCalledWith(image, undefined)
  })

  it('trims inline image sources before classifying their action', () => {
    const image = '  data:image/png;base64,ZHVtbXk=  '
    render(<GeneratedImage result={{ image, success: true }} />)

    expect(screen.queryByRole('button', { name: /full file/i })).toBeNull()
    expect(useImageDownload).toHaveBeenLastCalledWith(image.trim(), undefined)
  })
})
