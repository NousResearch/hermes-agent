import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { group } from '@/components/pane-shell/tree/model'
import { $layoutTree } from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { $commandPaletteOpen, closeCommandPalette } from '@/store/command-palette'
import { $activeConnectionId } from '@/store/connections'
import {
  $activeSessionId,
  $connection,
  $selectedStoredSessionId,
  $sessionResumeRequest,
  $sessions
} from '@/store/session'
import { $sessionTiles } from '@/store/session-states'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { useKeybinds } from '../hooks/use-keybinds'

import { CommandPalette } from './index'

const deps = {
  archiveSelectedSession: vi.fn(),
  openNewSessionTab: vi.fn(),
  requestGateway: vi.fn(),
  startFreshSession: vi.fn(),
  toggleCommandCenter: vi.fn(),
  toggleSelectedPin: vi.fn()
}

function Keyboard() {
  useKeybinds(deps)
  const location = useLocation()

  return <output data-testid="route">{location.pathname}</output>
}

const mocks = vi.hoisted(() => ({
  list: vi.fn(),
  open: vi.fn(),
  clearThemePreview: vi.fn(),
  previewTheme: vi.fn(),
  setMode: vi.fn(),
  setTheme: vi.fn(),
  themes: [],
  catalog: { subpageEntries: [], settingEntries: [], configEntries: [], pluginEntries: [], credentialEntries: [] }
}))

vi.mock('@/themes/context', () => ({
  useTheme: () => ({
    availableThemes: mocks.themes,
    clearThemePreview: mocks.clearThemePreview,
    previewTheme: mocks.previewTheme,
    mode: 'light',
    resolvedMode: 'light',
    setMode: mocks.setMode,
    setTheme: mocks.setTheme,
    themeName: 'nous'
  })
}))
vi.mock('../settings/use-settings-search', () => ({ useSettingsSearchCatalog: () => mocks.catalog }))
vi.mock('../open-session', async importOriginal => {
  const original = await importOriginal<{ openSessionFromPicker: (...args: unknown[]) => void }>()
  mocks.open.mockImplementation(original.openSessionFromPicker)

  return { ...original, openSessionFromPicker: mocks.open }
})
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<object>()),
  getHermesConfigRecord: async () => ({}),
  listAllProfileSessions: async (_limit: number, _preview: number, archived: string) => ({
    sessions:
      mocks.list.getMockImplementation() && $activeConnectionId.get() !== 'other-gateway'
        ? await mocks.list(archived)
        : $activeConnectionId.get() === 'other-gateway'
          ? await mocks.list()
          : archived === 'only'
            ? [{ id: 'archived-alpha', title: 'Archived previous gateway conversation' }]
            : [
                { id: 'alpha', title: 'Alpha conversation', preview: 'first topic' },
                { id: 'beta', title: 'Beta conversation', preview: 'second topic' }
              ]
  })
}))

stubResizeObserver()
stubMenuDomApis()

afterEach(() => {
  cleanup()
  closeCommandPalette()
  mocks.open.mockClear()
  $connection.set(null)
  mocks.list.mockReset()
  $selectedStoredSessionId.set(null)
  $activeSessionId.set(null)
  $sessions.set([])
  $sessionTiles.set([])
  $layoutTree.set(null)
})

async function openHistory() {
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter>
        <Keyboard />
        <CommandPalette />
      </MemoryRouter>
    </QueryClientProvider>
  )
  fireEvent.keyDown(window, { key: 'h', code: 'KeyH', ctrlKey: true })
  await screen.findByRole('option', { name: /Alpha conversation/ })

  return screen.getByRole('combobox')
}

describe('session history in the existing palette', () => {
  it('keeps older pinned conversations in addition to 200 recent conversations', async () => {
    mocks.list.mockImplementation(async (archived: string) =>
      archived === 'only'
        ? []
        : [
            ...Array.from({ length: 200 }, (_, index) => ({ id: `recent-${index}`, title: `Recent ${index}` })),
            { id: 'old-pin', title: 'Older pinned conversation', pinned: true }
          ]
    )
    render(
      <QueryClientProvider client={new QueryClient()}>
        <MemoryRouter>
          <Keyboard />
          <CommandPalette />
        </MemoryRouter>
      </QueryClientProvider>
    )
    fireEvent.keyDown(window, { key: 'h', code: 'KeyH', ctrlKey: true })
    await screen.findByRole('option', { name: /Older pinned conversation/ })
    expect(screen.getAllByRole('option')).toHaveLength(201)
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'Older pinned' } })
    fireEvent.keyDown(screen.getByRole('combobox'), { key: 'Enter', code: 'Enter' })
    expect($sessionResumeRequest.get()?.sessionId).toBe('old-pin')
  })

  it('Enter preserves occupied main by opening a real session tile', async () => {
    $selectedStoredSessionId.set('current')
    $layoutTree.set(group(['workspace']))
    const input = await openHistory()
    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' })
    expect($selectedStoredSessionId.get()).toBe('current')
    expect($sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['alpha'])
    expect(screen.getByTestId('route').textContent).toBe('/')
    expect($sessionResumeRequest.get()?.sessionId).toBe('alpha')
  })

  it('Enter reuses blank main through the real open-session route', async () => {
    const input = await openHistory()
    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' })
    expect(screen.getByTestId('route').textContent).toBe('/alpha')
    expect($sessionTiles.get()).toHaveLength(0)
  })

  it('Enter focuses an already-open real session tile without duplicating it', async () => {
    $selectedStoredSessionId.set('current')
    registry.register({
      area: 'panes',
      id: 'session-tile:alpha',
      title: 'Alpha',
      data: { placement: 'main' },
      render: () => null
    })
    $sessionTiles.set([{ storedSessionId: 'alpha', dir: 'center', workspaceMode: 'sessions' }])
    $layoutTree.set(group(['workspace', 'session-tile:alpha'], { active: 'workspace' }))
    const input = await openHistory()
    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' })
    expect($sessionTiles.get()).toHaveLength(1)
    const tree = $layoutTree.get()
    expect(tree?.type === 'group' && tree.active).toBe('session-tile:alpha')
    expect(screen.getByTestId('route').textContent).toBe('/')
  })
  it.each(['pending', 'failed'])('drops previous gateway rows while the new gateway request is %s', async status => {
    $connection.set({
      connectionId: 'first-gateway',
      baseUrl: '',
      isFullscreen: false,
      nativeOverlayWidth: 0,
      token: '',
      wsUrl: '',
      logs: [],
      windowButtonPosition: null
    })
    await openHistory()
    mocks.list.mockImplementation(() =>
      status === 'pending' ? new Promise(() => {}) : Promise.reject(new Error('offline'))
    )
    await act(async () => {
      $connection.set({ ...$connection.get()!, connectionId: 'other-gateway' })
    })
    expect(screen.queryByRole('option', { name: /Alpha conversation/ })).toBeNull()
    expect(screen.queryByRole('option', { name: /Beta conversation/ })).toBeNull()
    await waitFor(() => expect(mocks.list).toHaveBeenCalled())
  })
  it('drops cached archived rows on a gateway switch while refetch is pending', async () => {
    $connection.set({
      connectionId: 'first-gateway',
      baseUrl: '',
      isFullscreen: false,
      nativeOverlayWidth: 0,
      token: '',
      wsUrl: '',
      logs: [],
      windowButtonPosition: null
    })
    const input = await openHistory()
    fireEvent.click(screen.getByRole('button', { name: /Back/ }))
    fireEvent.change(input, { target: { value: 'Archived previous gateway' } })
    await screen.findByRole('option', { name: /Archived previous gateway conversation/ })
    mocks.list.mockImplementation(() => new Promise(() => {}))
    await act(async () => {
      $connection.set({ ...$connection.get()!, connectionId: 'other-gateway' })
    })
    expect(screen.queryByRole('option', { name: /Archived previous gateway conversation/ })).toBeNull()
  })

  it('Control+H clears the existing palette page search', async () => {
    const input = await openHistory()
    fireEvent.change(input, { target: { value: 'second topic' } })
    fireEvent.keyDown(input, { key: 'h', code: 'KeyH', ctrlKey: true })
    await waitFor(() => expect((input as HTMLInputElement).value).toBe(''))
    expect(screen.getAllByRole('option')).toHaveLength(2)
  })

  it('Escape closes history in one press without navigating', async () => {
    const input = await openHistory()
    fireEvent.keyDown(input, { key: 'Escape', code: 'Escape' })
    expect($commandPaletteOpen.get()).toBe(false)
    expect(mocks.open).not.toHaveBeenCalled()
  })

  it('arrows announce selection and Enter opens only the selected session', async () => {
    const input = await openHistory()
    fireEvent.keyDown(input, { key: 'ArrowDown', code: 'ArrowDown' })
    const beta = screen.getByRole('option', { name: /Beta conversation/, selected: true })
    expect(input.getAttribute('aria-activedescendant')).toBe(beta.id)
    expect(mocks.open).not.toHaveBeenCalled()
    fireEvent.keyDown(input, { key: 'ArrowUp', code: 'ArrowUp' })
    expect(screen.getByRole('option', { name: /Alpha conversation/, selected: true })).toBeTruthy()
    fireEvent.keyDown(input, { key: 'ArrowDown', code: 'ArrowDown' })
    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' })
    expect(mocks.open).toHaveBeenCalledWith('beta', expect.any(Function), 'stack')
    expect($commandPaletteOpen.get()).toBe(false)
  })

  it('search filters saved sessions by title and preview', async () => {
    const input = await openHistory()
    fireEvent.change(input, { target: { value: 'second topic' } })
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    expect(screen.getByRole('option', { name: /Beta conversation/ })).toBeTruthy()
    expect(mocks.open).not.toHaveBeenCalled()
  })

  it('opens a named session-only dialog with focused search, without navigation', async () => {
    const input = await openHistory()
    expect(screen.getByRole('dialog').getAttribute('aria-labelledby')).toBeTruthy()
    expect(screen.getByRole('dialog', { name: 'Session history' })).toBeTruthy()
    expect(screen.getByRole('combobox', { name: 'Search sessions…' })).toBe(input)
    expect(window.document.activeElement).toBe(input)
    expect(screen.getAllByRole('option')).toHaveLength(2)
    expect(mocks.open).not.toHaveBeenCalled()
  })
})
