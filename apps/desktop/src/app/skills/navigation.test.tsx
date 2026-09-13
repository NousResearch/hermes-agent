// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { type InitialEntry, MemoryRouter, Route, Routes, useLocation, useNavigate } from 'react-router'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { $selectedStoredSessionId } from '@/store/session'

import { NEW_CHAT_ROUTE, sessionRoute } from '../routes'
import type * as SettingsModule from '../settings'
import { useOverlayRouting } from '../shell/hooks/use-overlay-routing'

import type * as SkillsPageModule from './page'

import type * as SkillsModule from './index'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getSkills: async () => [],
  getToolsets: async () => [],
  getProfiles: async () => ({ profiles: [] }),
  getOfficialSkills: async () => ({ skills: [] }),
  getUsageAnalytics: async () => ({ tools: [] })
}))

let SettingsView: typeof SettingsModule.SettingsView
let SkillsView: typeof SkillsModule.SkillsView
let SkillsPage: typeof SkillsPageModule.SkillsPage

beforeAll(async () => {
  ;[{ SettingsView }, { SkillsView }, { SkillsPage }] = await Promise.all([
    import('../settings'),
    import('./index'),
    import('./page')
  ])
}, 60_000)

afterEach(() => {
  cleanup()
  $selectedStoredSessionId.set(null)
})

function Chat() {
  const location = useLocation()

  return <h1>Chat {location.pathname}</h1>
}

function CurrentRoute() {
  const location = useLocation()

  return (
    <output aria-label="Current route">
      {location.pathname}
      {location.search}
      {location.hash}
    </output>
  )
}

function Navigation({ surface }: { surface: 'page' | 'tile' | 'dialog' }) {
  const navigate = useNavigate()
  const { closeOverlayToPreviousRoute, openCapabilitiesFromOverlay, settingsOpen } = useOverlayRouting()

  return (
    <>
      <CurrentRoute />
      <button onClick={() => navigate('/settings?tab=keybinds')}>Open Settings</button>
      <button onClick={() => navigate(-1)}>Back</button>
      {settingsOpen && (
        <SettingsView onClose={closeOverlayToPreviousRoute} onOpenCapabilities={openCapabilitiesFromOverlay} />
      )}
      <Routes>
        <Route
          element={surface === 'page' ? <SkillsPage /> : <SkillsView embedded={surface === 'dialog'} />}
          path="skills"
        />
        <Route element={null} path="settings" />
        <Route element={<Chat />} path=":sessionId" />
        <Route element={<Chat />} index />
      </Routes>
    </>
  )
}

async function openRoute(entry: InitialEntry, surface: 'page' | 'tile' | 'dialog' = 'page') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  await act(async () => {
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={[entry]}>
          <Navigation surface={surface} />
        </MemoryRouter>
      </QueryClientProvider>
    )
  })

  return client
}

describe('Capabilities navigation', () => {
  it('returns to the Settings underlying route without leaving an overlay in history', async () => {
    const selectedSession = 'my saved chat'

    for (const path of ['/artifacts?tab=images#recent:2', `${sessionRoute(selectedSession)}?focus=reply#message:2`]) {
      $selectedStoredSessionId.set(selectedSession)
      const client = await openRoute(path)

      fireEvent.click(screen.getByRole('button', { name: 'Open Settings' }))
      fireEvent.click(screen.getByRole('button', { name: /Plugins/ }))
      expect(screen.getByLabelText('Current route').textContent).toBe('/skills?tab=plugins')
      fireEvent.click(screen.getByRole('button', { name: 'MCP' }))
      fireEvent.click(screen.getByRole('button', { name: 'Close' }))

      expect(screen.getByLabelText('Current route').textContent).toBe(path)
      fireEvent.click(screen.getByRole('button', { name: 'Back' }))
      expect(screen.getByLabelText('Current route').textContent).toBe(path)
      expect(screen.queryByRole('button', { name: 'Close settings' })).toBeNull()
      expect($selectedStoredSessionId.get()).toBe(selectedSession)
      cleanup()
      client.clear()
    }
  })

  it('can leave a direct link without adding page navigation to tiles or dialogs', async () => {
    for (const returnTo of [
      undefined,
      'https://example.test',
      '//example.test',
      '/\\example.test',
      '/settings?tab=keys',
      '/skills?tab=mcp',
      '/%'
    ]) {
      const selectedSession = returnTo === undefined ? null : 'previous chat'
      $selectedStoredSessionId.set(selectedSession)
      const client = await openRoute({ pathname: '/skills', search: '?tab=plugins', state: { returnTo } })

      fireEvent.click(screen.getByRole('button', { name: 'Close' }))

      const destination = selectedSession ? sessionRoute(selectedSession) : NEW_CHAT_ROUTE
      expect(screen.getByRole('heading', { name: `Chat ${destination}` })).toBeTruthy()
      expect($selectedStoredSessionId.get()).toBe(selectedSession)
      cleanup()
      client.clear()
    }

    for (const surface of ['tile', 'dialog'] as const) {
      cleanup()
      const embeddedClient = await openRoute('/skills?tab=plugins', surface)

      expect(screen.queryByRole('button', { name: 'Close' })).toBeNull()
      expect(screen.getByLabelText('Current route').textContent).toBe('/skills?tab=plugins')
      embeddedClient.clear()
    }
  })
})
