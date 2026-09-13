import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider, type Locale, TRANSLATIONS } from '@/i18n'
import { $worldEnabled, $worldOnboardingDismissed, WORLD_ONBOARDING_DISMISSED_STORAGE_KEY } from '@/store/lunar-city'

import { LunarCity } from './index'

const renderWorld = (initialLocale: Locale = 'en') => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <QueryClientProvider client={queryClient}>
      <I18nProvider configClient={null} initialLocale={initialLocale}>
        <MemoryRouter>
          <LunarCity />
        </MemoryRouter>
      </I18nProvider>
    </QueryClientProvider>
  )
}

const localizedLocales = (Object.keys(TRANSLATIONS).filter(locale => locale !== 'en') as Locale[])

beforeEach(() => {
  localStorage.clear()
  $worldEnabled.set(true)
  $worldOnboardingDismissed.set(false)
})

afterEach(cleanup)

describe('LunarCity', () => {
  it('renders its surface copy through the active locale', () => {
    renderWorld('ja')
    fireEvent.click(screen.getByRole('button', { name: TRANSLATIONS.ja.lunarCity.useDefaults }))

    expect(screen.getByRole('heading', { name: TRANSLATIONS.ja.lunarCity.heading })).toBeTruthy()
    expect(TRANSLATIONS.ja.lunarCity.heading).not.toBe(TRANSLATIONS.en.lunarCity.heading)
  })

  it.each(localizedLocales)('renders the full Lunar City surface in %s', locale => {
    const copy = TRANSLATIONS[locale].lunarCity
    localStorage.clear()
    $worldEnabled.set(true)
    $worldOnboardingDismissed.set(false)
    renderWorld(locale as Locale)

    expect(screen.getByRole('dialog', { name: copy.worldSetup })).toBeTruthy()
    expect(screen.getByRole('button', { name: copy.skip })).toBeTruthy()
    expect(screen.getByRole('button', { name: copy.useDefaults })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: copy.useDefaults }))

    expect(screen.getByRole('heading', { name: copy.heading })).toBeTruthy()
    expect(screen.getByRole('region', { name: copy.scene.ariaLabel })).toBeTruthy()
    expect(screen.getByRole('region', { name: copy.dispatcher.companion })).toBeTruthy()
    expect(screen.getByRole('button', { name: copy.dispatcher.newTask })).toBeTruthy()
    expect(copy.heading).not.toBe(TRANSLATIONS.en.lunarCity.heading)
    cleanup()
  })

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
