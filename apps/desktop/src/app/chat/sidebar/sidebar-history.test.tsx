import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { CommandPalette } from '@/app/command-palette'
import { SidebarProvider } from '@/components/ui/sidebar'
import { $commandPaletteOpen, closeCommandPalette } from '@/store/command-palette'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { ChatSidebar } from './index'

const noop = () => {}

const noopAsync = async () => {}
const themes: never[] = []
const catalog = { subpageEntries: [], settingEntries: [], configEntries: [], pluginEntries: [], credentialEntries: [] }
vi.mock('@/themes/context', () => ({
  useTheme: () => ({
    availableThemes: themes,
    clearThemePreview: noop,
    previewTheme: noop,
    mode: 'light',
    resolvedMode: 'light',
    setMode: noop,
    setTheme: noop,
    themeName: 'nous'
  })
}))
vi.mock('@/app/settings/use-settings-search', () => ({ useSettingsSearchCatalog: () => catalog }))
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<object>()),
  getHermesConfigRecord: async () => ({}),
  listAllProfileSessions: async () => ({ sessions: [{ id: 'older', title: 'Older conversation' }] })
}))

stubResizeObserver()
stubMenuDomApis()
afterEach(() => {
  cleanup()
  closeCommandPalette()
})

it('exposes a focusable History button even with no loaded sessions and opens the existing searchable modal', async () => {
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter>
        <SidebarProvider>
          <ChatSidebar
            currentView="chat"
            onArchiveSession={noop}
            onBranchSession={noop}
            onDeleteSession={noop}
            onLoadMoreSessions={noop}
            onManageCronJob={noop}
            onNavigate={noop}
            onNewSessionInWorkspace={noop}
            onNewSessionSplit={noop}
            onResumeSession={noop}
            onRetrySessions={noopAsync}
            onTriggerCronJob={noopAsync}
          />
        </SidebarProvider>
        <CommandPalette />
      </MemoryRouter>
    </QueryClientProvider>
  )
  const button = screen.getByRole('button', { name: 'Session history…' })
  expect(button.tabIndex).toBe(0)
  button.focus()
  expect(window.document.activeElement).toBe(button)
  // Native buttons map Enter/Space to click in Chromium; jsdom requires click.
  fireEvent.click(button)
  await screen.findByRole('option', { name: /Older conversation/ })
  expect(screen.getByRole('dialog', { name: 'Session history' })).toBeTruthy()
  const search = screen.getByRole('combobox', { name: 'Search sessions…' })
  expect(window.document.activeElement).toBe(search)
  fireEvent.keyDown(search, { key: 'Escape', code: 'Escape' })
  expect($commandPaletteOpen.get()).toBe(false)
})
