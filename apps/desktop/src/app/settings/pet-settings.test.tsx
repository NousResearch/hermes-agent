// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { $petGallery, $petGalleryStatus, type GalleryPet, resetPetGallery } from '@/store/pet-gallery'
import { $gatewayState } from '@/store/session'

import { PET_ROW_ESTIMATE_PX, PetSettings } from './pet-settings'

// The virtualizer is mocked with a window that mounts EVERY row, the way the
// real one does as the user scrolls through: the catalog must stay reachable
// past any cap (#123177). `options` is captured so the count/estimate contract
// can be asserted, mirroring virtual-session-list.test.tsx.
type VirtualizerOptions = {
  count: number
  estimateSize: (index: number) => number
  getItemKey: (index: number) => string
}

let virtualizerOptions: VirtualizerOptions

vi.mock('@tanstack/react-virtual', () => ({
  useVirtualizer: (options: VirtualizerOptions) => {
    virtualizerOptions = options

    return {
      getVirtualItems: () =>
        Array.from({ length: options.count }, (_, index) => ({
          index,
          key: options.getItemKey(index),
          start: index * options.estimateSize(index)
        })),
      getTotalSize: () => options.count * options.estimateSize(0)
    }
  }
}))

// jsdom has no IntersectionObserver; the thumbs are not what's under test.
vi.mock('@/components/pet/pet-thumb', () => ({ PetThumb: () => null }))

const { requestGateway } = vi.hoisted(() => ({ requestGateway: vi.fn() }))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

vi.mock('@/i18n', () => ({ useI18n: () => ({ t: en }) }))

// Column count comes from the Tailwind breakpoints; pin it per test so the
// row math is deterministic in jsdom (no real matchMedia).
const mediaQueries = new Map<string, boolean>()

vi.mock('@/hooks/use-media-query', () => ({ useMediaQuery: (query: string) => mediaQueries.get(query) ?? false }))

// The issue's repro: a catalog larger than the old 60-entry render cap.
const pets: GalleryPet[] = Array.from({ length: 61 }, (_, i) => ({
  displayName: `Pet ${String(i).padStart(3, '0')}`,
  installed: false,
  slug: `pet-${String(i).padStart(3, '0')}`
}))

const copy = en.settings.appearance.pet

beforeEach(() => {
  mediaQueries.clear()
  $gatewayState.set('open')
  // A ready gallery makes the mount-time `loadPetGallery` a cached no-op, so
  // `requestGateway` is never called and the store state is the test's own.
  $petGallery.set({ active: '', enabled: true, pets })
  $petGalleryStatus.set('ready')
})

afterEach(() => {
  cleanup()
  resetPetGallery()
  $gatewayState.set('idle')
})

describe('PetSettings virtualized grid', () => {
  it('keeps every catalog entry reachable instead of capping the render at 60 (#123177)', () => {
    render(<PetSettings />)

    // All 61 tiles mount through the virtual rows — rankedGalleryPets already
    // ranks the full list, so a missed tile means a mount/window bug, not rank.
    for (const pet of pets) {
      expect(screen.getByText(pet.displayName)).toBeTruthy()
    }

    // The status line reports the whole catalog, with no "Showing 60 of 61".
    expect(screen.getByText(copy.count(61))).toBeTruthy()
    expect(screen.queryByText(/Showing \d+ of \d+/)).toBeNull()
  })

  it('drives the virtualizer with a row count that follows the column count and a fixed row estimate', () => {
    const { rerender } = render(<PetSettings />)

    // One column (narrow): a row per pet, each estimated at the fixed tile height.
    expect(virtualizerOptions.count).toBe(61)
    expect(virtualizerOptions.estimateSize(0)).toBe(PET_ROW_ESTIMATE_PX)
    expect(virtualizerOptions.getItemKey(0)).toBe('pet-000')

    // Two columns (Tailwind sm): the same catalog in half the rows.
    mediaQueries.set('(min-width: 640px)', true)
    rerender(<PetSettings />)
    expect(virtualizerOptions.count).toBe(31)

    // Three columns (Tailwind xl): ceil(61 / 3).
    mediaQueries.set('(min-width: 1280px)', true)
    rerender(<PetSettings />)
    expect(virtualizerOptions.count).toBe(21)
  })
})
