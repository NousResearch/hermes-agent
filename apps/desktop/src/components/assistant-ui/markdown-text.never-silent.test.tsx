import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { closePreviewForSource } from '@/store/preview'
import { $connection } from '@/store/session'

import { MarkdownImage, MarkdownTextContent } from './markdown-text'

const MEDIA_PATH = '/home/user/project/renders/hero-shot.png'

function primeRegistry(path: string, size: number) {
  // Imported lazily so the test controls registry contents per case.
  return import('@/lib/media-store').then(({ recordMediaDeliverable }) => {
    recordMediaDeliverable({ kind: 'image', mime: 'image/png', path, size })
  })
}

describe('never-silent media cards (M4)', () => {
  const api = vi.fn<(args: { path: string }) => Promise<unknown>>(async ({ path }: { path: string }) => {
    if (path.startsWith('/api/fs/read-data-url?')) {
      const error: Error & { statusCode?: number } = new Error('403: outside media roots')

      error.statusCode = 403
      throw error
    }

    throw new Error(`unexpected path ${path}`)
  })

  let originalDesktop: typeof window.hermesDesktop

  beforeEach(() => {
    api.mockClear()
    originalDesktop = window.hermesDesktop
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
    $connection.set({ mode: 'remote', profile: 'remote-work' } as never)
  })

  afterEach(() => {
    cleanup()

    // The click-through test opens a preview tab in the shared store; close
    // it so later tests see the default "Open preview" state.
    closePreviewForSource('/tmp/data.bin')

    return import('@/lib/media-store').then(({ resetMediaDeliverables }) => {
      resetMediaDeliverables()
      $connection.set(null)
      Object.defineProperty(window, 'hermesDesktop', {
        configurable: true,
        value: originalDesktop
      })
    })
  })

  it('renders a fallback card with name, kind, size, and reason when the gateway denies the read', async () => {
    await primeRegistry(MEDIA_PATH, 1234)

    render(<MarkdownTextContent isRunning={false} text={`![Hero shot](${MEDIA_PATH})`} />)

    const card = await screen.findByText(content => content.startsWith('hero-shot.png'))

    expect(card).toBeTruthy()
    expect(await screen.findByText('1.2 KB')).toBeTruthy()
    expect(screen.getByText(/media.roots/)).toBeTruthy()
    expect(screen.getByRole('button', { name: /save as/i })).toBeTruthy()
  })

  it('renders the href-carried size on the structured file card when no event row is in memory', async () => {
    render(<MarkdownTextContent isRunning={false} text={`[File: data.bin](#media:%2Ftmp%2Fdata.bin?~=16)`} />)

    // Post-#97812 file-kind refs render PreviewAttachment: a structured card
    // with name + Download + Open preview — never a bare anchor and never a
    // dead "Loading" link. The href-size codec feeding the card is covered
    // unit-level in media.meta.test.ts.
    expect(await screen.findByText('data.bin')).toBeTruthy()
    expect(screen.getByRole('button', { name: /download/i })).toBeTruthy()
    expect(screen.getByRole('button', { name: /open preview/i })).toBeTruthy()
    expect(screen.queryByText(/^Loading /)).toBeNull()

    // Click-through is wired: Open preview toggles to the active "Hide"
    // state (the local preview route accepts the target without throwing).
    screen.getByRole('button', { name: /open preview/i }).click()
    expect(await screen.findByRole('button', { name: /hide/i })).toBeTruthy()
  })

  it('renders a structured card for a denied non-media file link instead of a bare link', async () => {
    render(<MarkdownTextContent isRunning={false} text={`[File: data.bin](#media:%2Ftmp%2Fdata.bin)`} />)

    expect(await screen.findByText('data.bin')).toBeTruthy()
    expect(screen.getByRole('button', { name: /download/i })).toBeTruthy()
    expect(screen.getByRole('button', { name: /open preview/i })).toBeTruthy()
    expect(screen.queryByText(/^Loading /)).toBeNull()
  })

  it('still renders an inline image when the file resolves', async () => {
    api.mockImplementation(async ({ path }: { path: string }) => {
      if (path.startsWith('/api/fs/read-data-url?')) {
        return { dataUrl: 'data:image/png;base64,aGVybw==' }
      }

      throw new Error(`unexpected path ${path}`)
    })

    render(<MarkdownImage alt="Hero shot" src={MEDIA_PATH} />)

    const image = await screen.findByRole('img', { name: 'Hero shot' })

    expect(image.getAttribute('src')).toBe('data:image/png;base64,aGVybw==')
  })

  it('renders a Save-as fallback card when a legacy image has no metadata at all', async () => {
    // Test isolation: the success case above replaces the shared mock's
    // implementation (mockClear only resets calls), so restore the denial.
    api.mockImplementation(async ({ path }: { path: string }) => {
      if (path.startsWith('/api/fs/read-data-url?')) {
        const error: Error & { statusCode?: number } = new Error('403: outside media roots')

        error.statusCode = 403
        throw error
      }

      throw new Error(`unexpected path ${path}`)
    })

    render(<MarkdownTextContent isRunning={false} text={`![Remote preview](${MEDIA_PATH})`} />)

    // The 403 bridge denies the fetch. Even with no event row and no size
    // query, the card knows the ref path from the markdown itself — Save-as
    // through the authenticated download bridge is offered, never silence.
    expect(await screen.findByRole('button', { name: /save as/i })).toBeTruthy()
    expect(screen.getByText('hero-shot.png')).toBeTruthy()
    expect(screen.getByText(/media.roots/i)).toBeTruthy()
    // Unknown size must be ABSENT, not rendered as a junk label: the size
    // element only appears when formatMediaSize returns a string.
    expect(screen.queryByText(/^B$|NaN|undefined/i)).toBeNull()
  })
})
