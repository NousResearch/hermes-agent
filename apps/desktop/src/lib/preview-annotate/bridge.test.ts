import { beforeEach, describe, expect, it, vi } from 'vitest'

import { type AnnotateFlushEnvelope, installAnnotateFlushReceiver, postPopoutAnnotateFlush } from './bridge'
import type { AnnotatePin } from './stack'

const png =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='

function pin(partial: Partial<AnnotatePin> = {}): AnnotatePin {
  return {
    id: 'annotate-1',
    imageDataUrl: png,
    kind: 'element',
    note: 'This button overflows on mobile.',
    number: 1,
    pageTitle: 'Pricing',
    pageUrl: 'http://127.0.0.1:4173/',
    rect: { height: 40, width: 120, x: 8, y: 8 },
    ...partial
  }
}

function setBridge(bridge: unknown) {
  ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = bridge as never
}

const tick = () => new Promise(resolve => setTimeout(resolve, 0))

describe('postPopoutAnnotateFlush', () => {
  beforeEach(() => {
    setBridge(undefined)
  })

  it('reports undelivered when the shell predates the flush bridge', async () => {
    await expect(postPopoutAnnotateFlush([pin()], 'http://127.0.0.1:4173/')).resolves.toEqual({
      delivered: false
    })
  })

  it('packages the pin stack into the posted envelope', async () => {
    const postAnnotateFlush = vi.fn(async (envelope: AnnotateFlushEnvelope) => ({ ok: true }))
    setBridge({ postAnnotateFlush })

    const result = await postPopoutAnnotateFlush([pin()], 'http://127.0.0.1:4173/')

    expect(result).toEqual({ delivered: true })
    expect(postAnnotateFlush).toHaveBeenCalledTimes(1)
    const envelope = postAnnotateFlush.mock.calls[0][0]
    expect(typeof envelope.id).toBe('string')
    expect(envelope.pageUrl).toBe('http://127.0.0.1:4173/')
    expect(envelope.items).toHaveLength(1)
    expect(envelope.items[0].number).toBe(1)
    expect(envelope.items[0].imageDataUrl).toBe(png)
  })

  it('rejects when the main process refuses the envelope so the caller keeps its pins', async () => {
    setBridge({ postAnnotateFlush: vi.fn(async () => ({ error: 'no-main-window', ok: false })) })

    await expect(postPopoutAnnotateFlush([pin()])).resolves.toEqual({ delivered: false })
  })
})

describe('installAnnotateFlushReceiver', () => {
  beforeEach(() => {
    setBridge(undefined)
  })

  it('is a no-op without the shell bridge', () => {
    expect(() => installAnnotateFlushReceiver()()).not.toThrow()
  })

  it('attaches crops and inserts the prompt into the local composer bus', async () => {
    const slot: { fire: ((envelope: unknown) => void) | null } = { fire: null }
    setBridge({
      onAnnotateFlushed: vi.fn((callback: (envelope: unknown) => void) => {
        slot.fire = callback

        return () => {
          slot.fire = null
        }
      })
    })

    const seen: { detail: unknown; name: string }[] = []

    const record = (name: string) => (event: Event) => {
      seen.push({ detail: (event as CustomEvent).detail, name })
    }

    window.addEventListener('hermes:composer-attach-images', record('hermes:composer-attach-images'))
    window.addEventListener('hermes:composer-insert', record('hermes:composer-insert'))

    const unsubscribe = installAnnotateFlushReceiver()
    slot.fire?.({
      id: 'flush-1',
      items: [
        {
          imageDataUrl: png,
          note: 'This button overflows on mobile.',
          number: 1,
          prompt: 'Comment 1 prompt'
        }
      ],
      pageUrl: 'http://127.0.0.1:4173/'
    })
    await tick()
    await tick()

    const attach = seen.find(event => event.name === 'hermes:composer-attach-images')
    expect(attach).toBeDefined()
    const blobs = (attach!.detail as { blobs: Blob[] }).blobs
    expect(blobs).toHaveLength(1)
    expect(blobs[0]).toBeInstanceOf(Blob)

    const insert = seen.find(event => event.name === 'hermes:composer-insert')
    expect(insert).toBeDefined()
    expect((insert!.detail as { text: string }).text).toContain('http://127.0.0.1:4173/')

    unsubscribe()
  })

  it('ignores malformed envelopes instead of touching the composer', async () => {
    const slot: { fire: ((envelope: unknown) => void) | null } = { fire: null }
    setBridge({
      onAnnotateFlushed: vi.fn((callback: (envelope: unknown) => void) => {
        slot.fire = callback

        return () => undefined
      })
    })

    const inserts: unknown[] = []
    window.addEventListener('hermes:composer-insert', event => {
      inserts.push((event as CustomEvent).detail)
    })

    installAnnotateFlushReceiver()
    slot.fire?.({ id: 'flush-bad' })
    slot.fire?.({ id: 'flush-empty', items: [] })
    await tick()
    await tick()

    expect(inserts).toHaveLength(0)
  })
})
