import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { pl } from '@/i18n/pl'
import { $composerPrefillRequest } from '@/store/composer'

import { ConnectionsView } from './index'

function Where() {
  const location = useLocation()

  return <output data-testid="where">{`${location.pathname}${location.search}`}</output>
}

const api = vi.fn(async ({ path }: { path: string }) =>
  path.startsWith('/api/connections/status')
    ? { github: 'connected', google: 'missing', notion: 'missing' }
    : { can_send_mail: false, client_secret: false, connected: false, services: [], token: false }
)

function renderPage() {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { api, openExternal: vi.fn() }

  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter initialEntries={['/connections']}>
        <I18nProvider configClient={null} initialLocale="pl">
          <Routes>
            <Route element={<ConnectionsView />} path="/connections" />
            <Route element={null} path="*" />
          </Routes>
          <Where />
        </I18nProvider>
      </MemoryRouter>
    </QueryClientProvider>
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

    const notion = screen.getByText(pl.jarvisConnections.entries.notion.name).closest('article')!

    fireEvent.click(within(notion as HTMLElement).getByRole('button', { name: pl.jarvisConnections.setupWithJarvis }))

    expect(screen.getByTestId('where').textContent).toBe('/')
    expect($composerPrefillRequest.get()?.text).toBe(pl.jarvisConnections.entries.notion.prompt)
  })

  it('connects Google through the wizard, not through a prompt to the agent, and shows what is already connected', async () => {
    renderPage()

    const google = screen.getByText(pl.jarvisConnections.entries.google.name).closest('article')!

    expect(within(google as HTMLElement).queryByRole('button', { name: pl.jarvisConnections.setupWithJarvis })).toBeNull()
    fireEvent.click(within(google as HTMLElement).getByRole('button', { name: 'Połącz Google' }))

    // With no client file yet the wizard starts at the first step.
    await waitFor(() => expect(screen.getByText('1 · Twoje dane logowania Google')).toBeTruthy())
    expect($composerPrefillRequest.get()).toBeNull()

    // GitHub reports connected: its card says so.
    await waitFor(() => {
      const github = screen.getByText(pl.jarvisConnections.entries.github.name).closest('article') as HTMLElement

      expect(within(github).getByText('Połączono')).toBeTruthy()
    })
  })

  it('opens the real settings page for connections configured in settings', () => {
    renderPage()

    const home = screen.getByText(pl.jarvisConnections.entries.smartHome.name).closest('article')!

    fireEvent.click(within(home as HTMLElement).getByRole('button', { name: pl.jarvisConnections.openSettings }))

    expect(screen.getByTestId('where').textContent).toBe('/messaging?platform=homeassistant')
  })

  it('opens communicator setup from integrations without a separate rail entry', () => {
    renderPage()

    const messaging = screen.getByText(pl.jarvisConnections.entries.messaging.name).closest('article')!

    expect(screen.queryByText(pl.jarvisConnections.entries.email.name)).toBeNull()
    expect(screen.queryByText(pl.jarvisConnections.entries.phone.name)).toBeNull()
    expect(screen.getByText(pl.jarvisConnections.entries.google.name)).toBeTruthy()
    expect(screen.getByText(pl.jarvisConnections.entries.github.name)).toBeTruthy()
    fireEvent.click(within(messaging as HTMLElement).getByRole('button', { name: pl.jarvisConnections.openSettings }))
    expect(screen.getByTestId('where').textContent).toBe('/messaging?platform=telegram')
  })

  it('explains the Agent Czesiek API with a copyable example that uses the real address', () => {
    renderPage()

    fireEvent.click(screen.getByRole('button', { name: pl.jarvisConnections.api.tab }))

    expect(screen.getByText(/127\.0\.0\.1:8642\/v1\/chat\/completions/)).toBeTruthy()
  })
})
