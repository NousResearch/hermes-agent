import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { pl } from '@/i18n/pl'
import { $composerPrefillRequest } from '@/store/composer'

import { ConnectionsView } from './index'

function Where() {
  const location = useLocation()

  return <output data-testid="where">{`${location.pathname}${location.search}`}</output>
}

function renderPage() {
  render(
    <MemoryRouter initialEntries={['/connections']}>
      <I18nProvider configClient={null} initialLocale="pl">
        <Routes>
          <Route element={<ConnectionsView />} path="/connections" />
          <Route element={null} path="*" />
        </Routes>
        <Where />
      </I18nProvider>
    </MemoryRouter>
  )
}

afterEach(() => {
  cleanup()
  $composerPrefillRequest.set(null)
  window.localStorage.clear()
})

describe('ConnectionsView', () => {
  it('starts a guided setup in a fresh conversation, with the request waiting in the composer', () => {
    renderPage()

    const google = screen.getByText(pl.jarvisConnections.entries.google.name).closest('article')!

    fireEvent.click(within(google as HTMLElement).getByRole('button', { name: pl.jarvisConnections.setupWithJarvis }))

    expect(screen.getByTestId('where').textContent).toBe('/')
    expect($composerPrefillRequest.get()?.text).toBe(pl.jarvisConnections.entries.google.prompt)
  })

  it('opens the real settings page for connections configured in settings', () => {
    renderPage()

    const home = screen.getByText(pl.jarvisConnections.entries.smartHome.name).closest('article')!

    fireEvent.click(within(home as HTMLElement).getByRole('button', { name: pl.jarvisConnections.openSettings }))

    expect(screen.getByTestId('where').textContent).toBe('/messaging?platform=homeassistant')
  })

  it('keeps communication channels out of the integrations catalog', () => {
    renderPage()

    expect(screen.queryByText(pl.jarvisConnections.entries.messaging.name)).toBeNull()
    expect(screen.queryByText(pl.jarvisConnections.entries.email.name)).toBeNull()
    expect(screen.queryByText(pl.jarvisConnections.entries.phone.name)).toBeNull()
    expect(screen.getByText(pl.jarvisConnections.entries.google.name)).toBeTruthy()
    expect(screen.getByText(pl.jarvisConnections.entries.github.name)).toBeTruthy()
  })

  it('explains the Agent Czesiek API with a copyable example that uses the real address', () => {
    renderPage()

    fireEvent.click(screen.getByRole('button', { name: pl.jarvisConnections.api.tab }))

    expect(screen.getByText(/127\.0\.0\.1:8642\/v1\/chat\/completions/)).toBeTruthy()
  })
})
