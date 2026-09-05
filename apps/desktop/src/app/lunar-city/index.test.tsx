import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $worldEnabled, $worldOnboardingDismissed, WORLD_ONBOARDING_DISMISSED_STORAGE_KEY } from '@/store/lunar-city'

import { LunarCity } from './index'

const renderWorld = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <QueryClientProvider client={queryClient}>
      <I18nProvider configClient={null}>
        <MemoryRouter>
          <LunarCity />
        </MemoryRouter>
      </I18nProvider>
    </QueryClientProvider>
  )
}

beforeEach(() => {
  localStorage.clear()
  $worldEnabled.set(true)
  $worldOnboardingDismissed.set(false)
})

afterEach(cleanup)

describe('LunarCity', () => {
  it('shows first-open onboarding with Hermes defaults', () => {
    renderWorld()

    expect(screen.getByRole('dialog', { name: 'Lunar City world setup' })).toBeTruthy()
    expect(screen.getByText('Hermes built-in pets and assets')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Use Hermes defaults' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Use Hermes defaults' }))

    expect(screen.queryByRole('dialog', { name: 'Lunar City world setup' })).toBeNull()
    expect(localStorage.getItem(WORLD_ONBOARDING_DISMISSED_STORAGE_KEY)).toBe('true')
  })

  it('accepts local files as review-only asset metadata', () => {
    renderWorld()
    const dropZone = screen.getByTestId('world-asset-drop-zone')
    const file = new File(['placeholder'], 'moon.glb', { type: 'model/gltf-binary' })

    fireEvent.drop(dropZone, { dataTransfer: { files: [file] } })

    expect(screen.getByText('moon.glb')).toBeTruthy()
    expect(screen.getAllByText(/Review only/).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/no importer is connected/i).length).toBeGreaterThan(0)
  })

  it('makes the world unavailable when Enable World is disabled', () => {
    $worldEnabled.set(false)

    renderWorld()

    expect(screen.getByRole('status', { name: 'World disabled' })).toBeTruthy()
    expect(screen.queryByRole('dialog', { name: 'Lunar City world setup' })).toBeNull()
    expect(screen.queryByTestId('world-asset-drop-zone')).toBeNull()
  })
})
