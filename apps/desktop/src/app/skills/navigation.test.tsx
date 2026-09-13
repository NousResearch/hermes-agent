// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { $selectedStoredSessionId } from '@/store/session'

import { NEW_CHAT_ROUTE, sessionRoute } from '../routes'
import type * as SettingsModule from '../settings'

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
    </output>
  )
}

async function openRoute(path: string, surface: 'page' | 'tile' | 'dialog' = 'page') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  await act(async () => {
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={[path]}>
          <CurrentRoute />
          <Routes>
            <Route element={<SettingsView onClose={vi.fn()} />} path="settings" />
            <Route
              element={surface === 'page' ? <SkillsPage /> : <SkillsView embedded={surface === 'dialog'} />}
              path="skills"
            />
            <Route element={<Chat />} path=":sessionId" />
            <Route element={<Chat />} index />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>
    )
  })

  return client
}

describe('Capabilities navigation', () => {
  it('makes Plugins discoverable in Settings and returns to the selected chat after switching tabs', async () => {
    const sessionId = 'my saved chat'
    $selectedStoredSessionId.set(sessionId)
    const client = await openRoute('/settings?tab=keybinds')

    fireEvent.click(screen.getByRole('button', { name: /Plugins/ }))
    expect(screen.getByLabelText('Current route').textContent).toBe('/skills?tab=plugins')
    fireEvent.click(screen.getByRole('button', { name: 'MCP' }))
    fireEvent.click(screen.getByRole('button', { name: 'Close' }))

    expect(screen.getByRole('heading', { name: `Chat ${sessionRoute(sessionId)}` })).toBeTruthy()
    expect($selectedStoredSessionId.get()).toBe(sessionId)
    client.clear()
  })

  it('can leave a direct link without adding page navigation to tiles or dialogs', async () => {
    const client = await openRoute('/skills?tab=plugins')

    fireEvent.click(screen.getByRole('button', { name: 'Close' }))

    expect(screen.getByRole('heading', { name: `Chat ${NEW_CHAT_ROUTE}` })).toBeTruthy()
    expect($selectedStoredSessionId.get()).toBeNull()
    client.clear()

    for (const surface of ['tile', 'dialog'] as const) {
      cleanup()
      const embeddedClient = await openRoute('/skills?tab=plugins', surface)

      expect(screen.queryByRole('button', { name: 'Close' })).toBeNull()
      expect(screen.getByLabelText('Current route').textContent).toBe('/skills?tab=plugins')
      embeddedClient.clear()
    }
  })
})
