import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import AppRoot, { appCompositionMode, jarvisViewForLocation } from './index'

const windowMode = vi.hoisted(() => ({
  auxiliary: false
}))

vi.mock('@/store/windows', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  isAuxiliaryWindow: () => windowMode.auxiliary
}))

vi.mock('./contrib', async () => {
  const router = await vi.importActual('react-router')
  const useLocation = (router as { useLocation: () => { pathname: string; search: string } }).useLocation

  return {
    ContribController: ({ layoutMode }: { layoutMode?: string }) => {
      const location = useLocation()

      return (
        <main
          data-layout-mode={layoutMode ?? 'viewport'}
          data-path={`${location.pathname}${location.search}`}
          data-testid="contrib-runtime"
        />
      )
    }
  }
})

function renderRoot(initialEntry = '/') {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <I18nProvider configClient={null} initialLocale="pl">
        <AppRoot />
      </I18nProvider>
    </MemoryRouter>
  )
}

afterEach(() => {
  cleanup()
  windowMode.auxiliary = false
})

describe('desktop app root Jarvis integration', () => {
  it('reaches the Jarvis shell through the default app root while keeping the contrib runtime mounted', () => {
    renderRoot('/')

    expect(screen.getByRole('navigation', { name: 'Główna nawigacja' })).toBeTruthy()
    expect(screen.getAllByRole('navigation', { name: 'Główna nawigacja' })).toHaveLength(1)
    expect(screen.getAllByRole('main')).toHaveLength(1)
    expect(screen.getByTestId('contrib-runtime').getAttribute('data-path')).toBe('/')
    expect(screen.getByTestId('contrib-runtime').getAttribute('data-layout-mode')).toBe('embedded')
  })

  it.each([
    ['Zadania', '/cron'],
    ['Pamięć', '/settings?tab=config:memory'],
    ['Narzędzia', '/skills?tab=toolsets'],
    ['Ustawienia', '/settings'],
    ['Profil', '/profiles'],
    ['Jarvis', '/']
  ])('delegates %s to an existing production route', (label, route) => {
    renderRoot('/settings')

    fireEvent.click(screen.getByRole('button', { name: label }))

    expect(screen.getByTestId('contrib-runtime').getAttribute('data-path')).toBe(route)
  })

  it('derives Jarvis navigation state from existing runtime routes', () => {
    expect(jarvisViewForLocation('/cron', '')).toBe('tasks')
    expect(jarvisViewForLocation('/settings', '?tab=config:memory')).toBe('memory')
    expect(jarvisViewForLocation('/skills', '?tab=toolsets')).toBe('tools')
    expect(jarvisViewForLocation('/settings', '')).toBe('settings')
    expect(jarvisViewForLocation('/profiles', '')).toBe('profile')
    expect(jarvisViewForLocation('/some-session', '')).toBe('jarvis')
  })

  it('keeps HUD windows on the existing chrome-free contrib root', () => {
    windowMode.auxiliary = true

    renderRoot('/')

    expect(screen.queryByRole('navigation', { name: 'Główna nawigacja' })).toBeNull()
    expect(screen.queryByTestId('jarvis-dashboard')).toBeNull()
    expect(screen.getByTestId('contrib-runtime').getAttribute('data-layout-mode')).toBe('viewport')
  })

  it('keeps Browser Popout windows on the existing popout root', () => {
    windowMode.auxiliary = true

    renderRoot('/')

    expect(screen.queryByRole('navigation', { name: 'Główna nawigacja' })).toBeNull()
    expect(screen.getByTestId('contrib-runtime').getAttribute('data-layout-mode')).toBe('viewport')
  })

  it('keeps secondary single-chat windows on the existing chrome-free controller path', () => {
    windowMode.auxiliary = true

    renderRoot('/session-123')

    expect(screen.queryByRole('navigation', { name: 'Główna nawigacja' })).toBeNull()
    expect(screen.getByTestId('contrib-runtime').getAttribute('data-path')).toBe('/session-123')
    expect(screen.getByTestId('contrib-runtime').getAttribute('data-layout-mode')).toBe('viewport')
  })

  it('keeps watch session windows on the same secondary-window controller path', () => {
    windowMode.auxiliary = true

    renderRoot('/session-123')

    expect(screen.queryByRole('navigation', { name: 'Główna nawigacja' })).toBeNull()
    expect(screen.getByTestId('contrib-runtime').getAttribute('data-path')).toBe('/session-123')
    expect(screen.getByTestId('contrib-runtime').getAttribute('data-layout-mode')).toBe('viewport')
  })

  it('classifies product shell composition without coupling tests to the host window', () => {
    expect(appCompositionMode({ auxiliary: false })).toBe('product-shell')
    expect(appCompositionMode({ auxiliary: true })).toBe('special-window')
  })
})
