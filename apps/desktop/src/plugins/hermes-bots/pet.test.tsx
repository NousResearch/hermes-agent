/**
 * Pet tiles: frame 0 of a petdex spritesheet, cropped server-side by the
 * gateway's `pet.thumb` RPC and cached per slug.
 *
 * Why the tile goes through the gateway rather than fetching the CDN sheet
 * itself: a locally hatched pet has no manifest entry, so `pet.gallery` reports
 * an EMPTY spritesheetUrl — a client-side fetch could never render or select
 * it ("Could not load that pet"). `pet.thumb` reads the installed sheet off
 * disk, so the slug alone is enough.
 *
 * The regression the cache created: a FAILED load was left parked in the
 * cache as a resolved-null promise, so one blip poisoned that pet for the rest
 * of the session. A failure must be evicted; a success must not be re-requested.
 */

import { fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { hostMock, UnboundedCache, useQueryMock } = vi.hoisted(() => ({
  hostMock: { notify: vi.fn(), request: vi.fn() },
  // Stand-in for the SDK's LruCache. Its ceiling has its own unit test and no
  // fixture here approaches it, so the double just drops the bound.
  UnboundedCache: class extends Map {
    constructor(_max: number) {
      super()
    }
  },
  useQueryMock: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', () => ({
  Button: (props: React.ComponentProps<'button'>) => <button {...props} />,
  cn: (...parts: unknown[]) => parts.filter(Boolean).join(' '),
  GlyphSpinner: () => <span />,
  host: hostMock,
  Input: (props: React.ComponentProps<'input'>) => <input {...props} />,
  LruCache: UnboundedCache,
  RowButton: (props: React.ComponentProps<'button'>) => <button {...props} />,
  useQuery: useQueryMock
}))

vi.mock('./i18n', () => ({
  useBots: () => ({
    avatar: { petLoadFailed: 'Could not load that pet.', pickPet: 'Pick a pet', removeBackToShape: 'Remove' }
  })
}))

vi.mock('./shared', () => ({ ID: 'hermes-bots' }))

const SHEET = 'https://pets.example/a.webp'
const ICON = 'data:image/png;base64,ok'

/** Every `pet.thumb` call, so the cache behaviour is observable. */
const thumbs: Array<{ slug: string; url: string }> = []

function stubThumb(handler: () => Promise<{ dataUri?: string; ok: boolean }>) {
  hostMock.request.mockImplementation(async (method: string, params: { slug: string; url: string }) => {
    if (method !== 'pet.thumb') {
      throw new Error(`unexpected RPC ${method}`)
    }

    thumbs.push(params)

    return handler()
  })
}

async function loadPetTab() {
  vi.resetModules()

  return (await import('./pet')).PetTab
}

beforeEach(() => {
  vi.clearAllMocks()
  thumbs.length = 0
  useQueryMock.mockReturnValue({ data: { pets: [{ displayName: 'Axolotl', slug: 'axolotl', spritesheetUrl: SHEET }] } })
  stubThumb(async () => ({ ok: true, dataUri: ICON }))
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('the pet gallery', () => {
  it('reserves paint space around boundary tiles inside the bounded scroller', async () => {
    const PetTab = await loadPetTab()
    const view = render(<PetTab image={null} onImage={vi.fn()} />)
    await waitFor(() => expect(view.container.querySelector('img')).toBeTruthy())
    const tile = view.getByText('Axolotl').closest('button')!
    const scroller = tile.parentElement!.parentElement!

    // The selection ring paints one pixel outside each tile. The scrollport
    // must leave room for it even at the first/last row and outer columns.
    const style = getComputedStyle(scroller)

    for (const side of ['paddingTop', 'paddingRight', 'paddingBottom', 'paddingLeft'] as const) {
      expect(parseFloat(style[side]) || 0).toBeGreaterThanOrEqual(1)
    }

    expect(parseFloat(style.maxHeight)).toBeGreaterThan(0)
    view.unmount()
  })

  it('keeps selection while scrolling for more and resets the search window', async () => {
    useQueryMock.mockReturnValue({
      data: {
        pets: Array.from({ length: 60 }, (_, i) => ({
          displayName: `Pet ${i}`,
          slug: `pet-${i}`,
          spritesheetUrl: SHEET
        }))
      }
    })
    const PetTab = await loadPetTab()
    const onImage = vi.fn()
    const view = render(<PetTab image={null} onImage={onImage} />)
    const first = view.getByText('Pet 0').closest('button')!
    fireEvent.click(first)
    await waitFor(() => expect(onImage).toHaveBeenCalledWith(ICON))
    const scroller = first.parentElement!.parentElement!
    Object.defineProperties(scroller, {
      clientHeight: { value: 220 },
      scrollHeight: { value: 600 },
      scrollTop: { value: 400 }
    })
    fireEvent.scroll(scroller)
    expect(view.getByText('Pet 47')).toBeTruthy()
    expect(view.queryByText('Pet 48')).toBeNull()
    expect(first.className).toContain('ring-1')
    fireEvent.change(view.getByRole('textbox'), { target: { value: 'Pet 59' } })
    expect(view.getByText('Pet 59')).toBeTruthy()
    fireEvent.change(view.getByRole('textbox'), { target: { value: '' } })
    expect(view.queryByText('Pet 24')).toBeNull()
    expect(onImage).toHaveBeenCalledTimes(1)
    view.unmount()
  })
})

describe('the pet thumb cache', () => {
  it('selects a locally hatched pet that has no spritesheet URL', async () => {
    // Generator-hatched pets are absent from the petdex manifest, so the
    // gallery reports spritesheetUrl: "". The gateway crops the installed
    // sheet off disk — the slug is the identity, not the URL.
    useQueryMock.mockReturnValue({
      data: { pets: [{ displayName: 'Mine', installed: true, slug: 'mine', spritesheetUrl: '' }] }
    })

    const PetTab = await loadPetTab()
    const onImage = vi.fn()
    const view = render(<PetTab image={null} onImage={onImage} />)

    fireEvent.click(view.getByText('Mine').closest('button')!)
    await waitFor(() => expect(onImage).toHaveBeenCalledWith(ICON))
    expect(thumbs.length).toBeGreaterThan(0)
    expect(thumbs.every(call => call.slug === 'mine')).toBe(true)
    expect(hostMock.notify).not.toHaveBeenCalled()
  })

  it('never leaves a failed load parked in the cache', async () => {
    stubThumb(async () => {
      throw new Error('gateway')
    })

    const PetTab = await loadPetTab()
    const first = render(<PetTab image={null} onImage={vi.fn()} />)

    await waitFor(() => expect(thumbs).toHaveLength(1))
    first.unmount()

    const second = render(<PetTab image={null} onImage={vi.fn()} />)

    // Reopening retries rather than serving the poisoned null.
    await waitFor(() => expect(thumbs).toHaveLength(2))
    expect(second.container.querySelector('img')).toBeNull()
  })

  it('requests a successful thumb once and reuses it across mounts', async () => {
    const PetTab = await loadPetTab()
    const first = render(<PetTab image={null} onImage={vi.fn()} />)

    await waitFor(() => expect(first.container.querySelector('img')?.getAttribute('src')).toBe(ICON))
    expect(thumbs).toHaveLength(1)

    first.unmount()

    const second = render(<PetTab image={null} onImage={vi.fn()} />)

    await waitFor(() => expect(second.container.querySelector('img')?.getAttribute('src')).toBe(ICON))
    expect(thumbs).toHaveLength(1)
  })
})
