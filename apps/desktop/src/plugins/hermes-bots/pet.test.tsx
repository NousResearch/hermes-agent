/**
 * Pet tiles: frame 0 of a petdex spritesheet, extracted once and cached.
 *
 * A "spritesheet" is the FULL animation sheet (1536×1872 webp, ~2MB, an 8×9
 * grid) — using it directly as an <img> downloads megabytes per tile and shows
 * the sheet squashed. So each sheet is fetched once, cropped to frame 0,
 * downscaled, and the resulting data URL is cached per URL.
 *
 * The regression the cache created: a FAILED fetch was left parked in the
 * cache as a resolved-null promise, so one network blip poisoned that pet for
 * the rest of the session — the tile never recovered, not even on reopen. A
 * failure must be evicted; a success must not be refetched.
 */

import { render, waitFor } from '@testing-library/react'
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

/** Record every fetch so the cache behaviour is observable. */
const fetches: Array<{ init?: RequestInit; url: string }> = []

function stubFetch(handler: () => Promise<unknown>) {
  vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
    fetches.push({ init, url })

    return handler()
  })
}

async function loadPetTab() {
  vi.resetModules()

  return (await import('./pet')).PetTab
}

beforeEach(() => {
  vi.clearAllMocks()
  fetches.length = 0
  useQueryMock.mockReturnValue({ data: { pets: [{ displayName: 'Axolotl', slug: 'axolotl', spritesheetUrl: SHEET }] } })
  vi.stubGlobal('createImageBitmap', async () => ({ close: () => undefined }))
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    drawImage: () => undefined
  } as unknown as CanvasRenderingContext2D)
  vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue('data:image/png;base64,ok')
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('the sprite-frame cache', () => {
  it('never leaves a failed fetch parked in the cache', async () => {
    stubFetch(async () => {
      throw new Error('network')
    })

    const PetTab = await loadPetTab()
    const first = render(<PetTab image={null} onImage={vi.fn()} />)

    await waitFor(() => expect(fetches).toHaveLength(1))
    // The fetch is abortable: a hung sheet must not hold a slot forever.
    expect(fetches[0].init?.signal).toBeTruthy()

    first.unmount()

    const second = render(<PetTab image={null} onImage={vi.fn()} />)

    // Reopening retries rather than serving the poisoned null.
    await waitFor(() => expect(fetches).toHaveLength(2))
    expect(second.container.querySelector('img')).toBeNull()
  })

  it('fetches a successful sheet once and reuses the extracted frame', async () => {
    stubFetch(async () => ({ blob: async () => new Blob() }))

    const PetTab = await loadPetTab()
    const first = render(<PetTab image={null} onImage={vi.fn()} />)

    await waitFor(() =>
      expect(first.container.querySelector('img')?.getAttribute('src')).toBe('data:image/png;base64,ok')
    )
    expect(fetches).toHaveLength(1)

    first.unmount()

    const second = render(<PetTab image={null} onImage={vi.fn()} />)

    await waitFor(() =>
      expect(second.container.querySelector('img')?.getAttribute('src')).toBe('data:image/png;base64,ok')
    )
    expect(fetches).toHaveLength(1)
  })

  it('shares one fetch between tiles that point at the same sheet', async () => {
    useQueryMock.mockReturnValue({
      data: {
        pets: [
          { displayName: 'Axolotl', slug: 'axolotl', spritesheetUrl: SHEET },
          { displayName: 'Axolotl (shiny)', slug: 'axolotl-shiny', spritesheetUrl: SHEET }
        ]
      }
    })
    stubFetch(async () => ({ blob: async () => new Blob() }))

    const PetTab = await loadPetTab()
    const { container } = render(<PetTab image={null} onImage={vi.fn()} />)

    await waitFor(() => expect(container.querySelectorAll('img')).toHaveLength(2))
    expect(fetches).toHaveLength(1)
  })

  it('does not fetch for a pet with no spritesheet', async () => {
    useQueryMock.mockReturnValue({ data: { pets: [{ displayName: 'Ghost', slug: 'ghost', spritesheetUrl: null }] } })
    stubFetch(async () => ({ blob: async () => new Blob() }))

    const PetTab = await loadPetTab()
    const { findByText } = render(<PetTab image={null} onImage={vi.fn()} />)

    await findByText('Ghost')
    expect(fetches).toHaveLength(0)
  })
})
