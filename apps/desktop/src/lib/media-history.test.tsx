import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { toChatMessages } from '@/lib/chat-messages'
import { mediaCardMeta, resetMediaDeliverables, seedMediaDeliverablesFromHistory } from '@/lib/media-store'
import { $connection } from '@/store/session'
import type { SessionMessage } from '@/types/hermes'

import { MarkdownTextContent } from '../components/assistant-ui/markdown-text'

// ── M5 / D5 — History media projection (desktop consumer) ───────────────────
//
// A reopened transcript has no media.deliverable events in memory, so before
// this contract the only card metadata was the capture-time href size. The
// server now derives refs from stored history (`include_media=true`) and the
// hydration path seeds them into the same registry live events write. These
// tests run the REAL chain: stored rows → toChatMessages → markdown render.

const KEPT = '/tmp/hermes-media/kept.png'
const LOST = '/tmp/hermes-media/lost.png'

const REOPENED_PAGE: SessionMessage[] = [
  {
    content: 'Here is the chart.\n\nMEDIA:/tmp/hermes-media/kept.png',
    media: [{ available: true, kind: 'image', mime: 'image/png', path: KEPT, size: 1234 }],
    role: 'assistant',
    timestamp: 1700000000
  },
  {
    content: 'The older render:\n\nMEDIA:/tmp/hermes-media/lost.png',
    media: [{ available: false, kind: 'image', mime: 'image/png', name: 'lost.png', path: LOST }],
    role: 'assistant',
    timestamp: 1700000001
  },
  {
    content: 'plain text turn, no refs',
    media: [],
    role: 'assistant',
    timestamp: 1700000002
  }
]

function textPartsOf(message: ReturnType<typeof toChatMessages>[number]): string {
  return (message?.parts ?? [])
    .filter(part => part.type === 'text')
    .map(part => (part.type === 'text' ? part.text : ''))
    .join('\n\n')
}

describe('history media projection (M5)', () => {
  const api = vi.fn<(args: { path: string }) => Promise<unknown>>(async ({ path }: { path: string }) => {
    const error: Error & { statusCode?: number } = new Error('404: no such file')

    error.statusCode = 404
    throw error
  })

  let originalDesktop: typeof window.hermesDesktop

  beforeEach(() => {
    api.mockClear()
    resetMediaDeliverables()
    originalDesktop = window.hermesDesktop
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
    $connection.set({ mode: 'remote', profile: 'remote-work' } as never)
  })

  afterEach(() => {
    cleanup()
    resetMediaDeliverables()
    $connection.set(null)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: originalDesktop
    })
  })

  it('seeds projected rows into the deliverable registry', () => {
    expect(seedMediaDeliverablesFromHistory(REOPENED_PAGE[0].media)).toBe(1)
    expect(seedMediaDeliverablesFromHistory(REOPENED_PAGE[1].media)).toBe(1)
    expect(seedMediaDeliverablesFromHistory(REOPENED_PAGE[2].media)).toBe(0)
    expect(seedMediaDeliverablesFromHistory(undefined)).toBe(0)
    expect(seedMediaDeliverablesFromHistory('garbage')).toBe(0)

    // Existing file: full D1-shaped row. Missing file: metadata without size.
    expect(mediaCardMeta(KEPT)).toMatchObject({ kind: 'image', mime: 'image/png', size: 1234 })
    expect(mediaCardMeta(LOST)).toMatchObject({ kind: 'image', mime: 'image/png' })
    expect(mediaCardMeta(LOST)?.size).toBeUndefined()
  })

  it('drops garbage rows without throwing', () => {
    expect(
      seedMediaDeliverablesFromHistory([{ path: '' }, { path: 42 }, null, { path: LOST, kind: 'image' }])
    ).toBe(1)
    expect(mediaCardMeta(LOST)?.kind).toBe('image')
  })

  it('toChatMessages primes the registry from the projected page', () => {
    toChatMessages(REOPENED_PAGE)

    expect(mediaCardMeta(KEPT)?.size).toBe(1234)
    expect(mediaCardMeta(LOST)?.kind).toBe('image')
    expect(mediaCardMeta(LOST)?.size).toBeUndefined()
  })

  it('renders the kept ref with projected metadata on a reopened session', async () => {
    const chats = toChatMessages(REOPENED_PAGE)

    // Hydration-time label carries the projected size and the href carries it
    // in the `?~=` codec — proof the seeded row reached mediaCardMeta during
    // render (an unprimed registry renders no size anywhere).
    expect(textPartsOf(chats[0])).toMatch(/1\.2 KB/)
    expect(textPartsOf(chats[0])).toContain('?~=1234')

    render(<MarkdownTextContent isRunning={false} text={textPartsOf(chats[0])} />)

    // Resolve fails (404 mock) → the never-silent fallback card, now WITH the
    // projected size a bare reopened transcript could not know.
    expect(await screen.findByText(/couldn't display this image/i)).toBeTruthy()
    expect(screen.getByText('1.2 KB')).toBeTruthy()
    expect(screen.getByRole('button', { name: /save as/i })).toBeTruthy()
  })

  it('renders the missing ref as a named fallback card without a fabricated size', async () => {
    const chats = toChatMessages(REOPENED_PAGE)

    render(<MarkdownTextContent isRunning={false} text={textPartsOf(chats[1])} />)

    expect(await screen.findByText(/no longer exists/i)).toBeTruthy()
    expect(screen.getByRole('button', { name: /save as/i })).toBeTruthy()
    // No size chip: an unavailable row must never invent one.
    expect(screen.queryByText(/^\d+(\.\d+)? (B|KB)$/)).toBeNull()
  })
})
