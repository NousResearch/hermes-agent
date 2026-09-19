// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $activeSessionId, $connection, $selectedStoredSessionId, $sessions, setSessionOwnerHint } from '@/store/session'
import { $toolDisclosureStates } from '@/store/tool-view'

import { stubThreadEnvironment } from '../test-utils'

stubThreadEnvironment()

vi.mock('@assistant-ui/react', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useAuiState: (select: (state: unknown) => unknown) =>
    select({ message: { id: 'msg-1', status: { type: 'complete' } }, thread: { isRunning: false } })
}))

const { ToolFallback } = await import('./fallback')

const IMAGE_PATH = '/tmp/analyzed image.png'
const DATA_URL = 'data:image/png;base64,YW5hbHl6ZWQ='
const STORED_SESSION = 'vision-session'

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
  // The reader is owner-scoped by design: it resolves the file against the
  // session that produced the call instead of guessing a host.
  $activeSessionId.set('runtime-vision-session')
  $selectedStoredSessionId.set(STORED_SESSION)
  setSessionOwnerHint(STORED_SESSION, { connectionId: 'local', profile: 'default' })
})

afterEach(() => {
  cleanup()
  $connection.set(null)
  $toolDisclosureStates.set({})
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $sessions.set([])
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: originalDesktop
  })
})

describe('vision_analyze activity image', () => {
  it('exposes the analyzed local image behind the disclosure and resolves it through the local file bridge', async () => {
    $connection.set({ mode: 'local' } as never)
    renderVisionRow()

    fireEvent.click(screen.getByRole('button', { name: /open image/i }))

    const img = await screen.findByRole('img')

    await waitFor(() => expect(img.getAttribute('src')).toBe(DATA_URL))
    // A local session's file is still read through its own owner route, never
    // the unscoped local reader: the origin decides, not the connection mode.
    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        connectionId: 'local',
        profile: 'default',
        path: `/api/fs/read-data-url?path=${encodeURIComponent(IMAGE_PATH)}&session_id=${STORED_SESSION}`
      })
    )
    expect(readFileDataUrl).not.toHaveBeenCalled()
  })

  it('says so when the image cannot be read instead of silently dropping the preview', async () => {
    $connection.set({ mode: 'local' } as never)
    api.mockRejectedValueOnce(new Error('ENOENT'))
    renderVisionRow()

    fireEvent.click(screen.getByRole('button', { name: /open image/i }))

    expect(await screen.findByText('Preview unavailable')).toBeTruthy()
    expect(screen.getByRole('button', { name: /retry/i })).toBeTruthy()
    expect(screen.queryByRole('img')).toBeNull()
  })
})
