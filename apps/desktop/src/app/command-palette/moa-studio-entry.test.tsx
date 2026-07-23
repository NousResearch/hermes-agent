import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesModule from '@/hermes'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesModule>()),
  getHermesConfigRecord: vi.fn(async () => ({})),
  listAllProfileSessions: vi.fn(async () => ({ sessions: [] }))
}))
vi.mock('@/themes/context', () => ({
  useTheme: () => ({
    availableThemes: [],
    resolvedMode: 'light',
    setMode: vi.fn(),
    setTheme: vi.fn(),
    themeName: 'nous'
  })
}))
vi.mock('./contrib', () => ({ usePaletteContributions: () => [] }))
vi.mock('./marketplace-theme-page', () => ({ MarketplaceThemePage: () => null }))
vi.mock('./pet-palette-page', () => ({ PetInlineToggle: () => null, PetPalettePage: () => null }))

import { $commandPaletteOpen } from '@/store/command-palette'

import { CommandPalette } from './index'

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  globalThis.ResizeObserver = class ResizeObserver {
    disconnect() {}
    observe() {}
    unobserve() {}
  }
})

function LocationProbe() {
  const location = useLocation()

  return <output data-testid="location">{`${location.pathname}${location.search}`}</output>
}

beforeEach(() => {
  $commandPaletteOpen.set(true)
})

afterEach(() => {
  cleanup()
  $commandPaletteOpen.set(false)
  vi.clearAllMocks()
})

describe('Command palette MoA Studio entry', () => {
  it('opens the dedicated settings deep link', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

    render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={['/']}>
          <CommandPalette />
          <LocationProbe />
        </MemoryRouter>
      </QueryClientProvider>
    )

    fireEvent.click(await screen.findByText('MoA Studio'))

    expect(screen.getByTestId('location').textContent).toBe('/settings?tab=moa')
    expect($commandPaletteOpen.get()).toBe(false)
  })
})
