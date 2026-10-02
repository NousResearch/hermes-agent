// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'
import { $toolDisclosureStates } from '@/store/tool-view'

vi.mock('@assistant-ui/react', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useAuiState: (select: (state: unknown) => unknown) =>
    select({ message: { id: 'msg-1', status: { type: 'complete' } }, thread: { isRunning: false } })
}))

const { ToolFallback } = await import('./fallback')

const IMAGE_PATH = '/tmp/analyzed image.png'
const DATA_URL = 'data:image/png;base64,YW5hbHl6ZWQ='

// The native-vision fast path hands the desktop a text-only receipt: the image
// is only known from the call's `image_url` argument.
function renderVisionRow() {
  const props = {
    args: { image_url: IMAGE_PATH, question: 'Inspect this image' },
    result: 'Image attached natively for the main model (12.3 KB). Answer using built-in vision.',
    toolCallId: 'call-vision',
    toolName: 'vision_analyze'
  } as unknown as ComponentProps<typeof ToolFallback>

  render(<ToolFallback {...props} />)
}

const api = vi.fn(async ({ path }: { path: string }) => {
  if (path.startsWith('/api/fs/read-data-url?')) {
    return { dataUrl: DATA_URL }
  }

  throw new Error(`unexpected path ${path}`)
})

const readFileDataUrl = vi.fn(async () => DATA_URL)

let originalDesktop: typeof window.hermesDesktop

beforeEach(() => {
  api.mockClear()
  readFileDataUrl.mockClear()
  originalDesktop = window.hermesDesktop
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { api, readFileDataUrl }
  })
})

afterEach(() => {
  cleanup()
  $connection.set(null)
  $toolDisclosureStates.set({})
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: originalDesktop
  })
})

describe('vision_analyze activity image', () => {
  it('exposes the analyzed local image behind the disclosure and resolves it through the local file bridge', async () => {
    $connection.set({ mode: 'local' } as never)
    renderVisionRow()

    const toggle = screen.getByRole('button', { expanded: false })

    fireEvent.click(toggle)

    const img = await screen.findByRole('img')

    await waitFor(() => expect(img.getAttribute('src')).toBe(DATA_URL))
    expect(readFileDataUrl).toHaveBeenCalledWith(IMAGE_PATH)
    expect(api).not.toHaveBeenCalled()

    // "cannot be expanded": the inline preview opens the shared lightbox.
    fireEvent.click(img)
    expect(await screen.findByRole('dialog')).toBeTruthy()
  })

  it('reads a remote-gateway image through the profile-scoped fs API, never the local reader', async () => {
    $connection.set({ mode: 'remote', profile: 'wsl-work' } as never)
    renderVisionRow()

    fireEvent.click(screen.getByRole('button', { expanded: false }))

    const img = await screen.findByRole('img')

    await waitFor(() => expect(img.getAttribute('src')).toBe(DATA_URL))
    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: `/api/fs/read-data-url?path=${encodeURIComponent(IMAGE_PATH)}`,
        profile: 'wsl-work'
      })
    )
    expect(readFileDataUrl).not.toHaveBeenCalled()
  })

  it('says so when the image cannot be read instead of silently dropping the preview', async () => {
    $connection.set({ mode: 'local' } as never)
    readFileDataUrl.mockRejectedValueOnce(new Error('ENOENT'))
    renderVisionRow()

    fireEvent.click(screen.getByRole('button', { expanded: false }))

    expect(await screen.findByText(/Couldn't load/)).toBeTruthy()
    expect(screen.queryByRole('img')).toBeNull()
  })
})
