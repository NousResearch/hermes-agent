import * as shared from '@hermes/shared'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $narrowOverlayPaneIds } from '@/components/pane-shell/narrow-overlay-state'
import { group, split } from '@/components/pane-shell/tree/model'
import {
  $dismissedPanes,
  $hiddenTreePanes,
  $layoutTree,
  $narrowViewport,
  noteActiveTreeGroup,
  setTreeGroupMinimized
} from '@/components/pane-shell/tree/store'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import { registry } from '@/contrib/registry'
import type { SessionInfo } from '@/hermes'
import { $interfaceMode } from '@/store/interface-mode'
import { $rightRailActiveTabId } from '@/store/layout'
import { $previewTabs } from '@/store/preview'
import { $reviewScopeCwd, $reviewScopeTarget, closeReview } from '@/store/review'
import { $currentCwd, $selectedStoredSessionId, setSessions } from '@/store/session'
import { $agentSidebarChoice } from '@/store/simplified-agent-mode'

import { DeveloperRail } from './developer-rail'

vi.mock('@/store/preview', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  toggleBrowserTab: vi.fn()
}))
vi.mock('../right-sidebar/terminal/reveal-focus', () => ({ toggleTerminalPane: vi.fn() }))

const disposers: (() => void)[] = []

beforeEach(() => {
  $interfaceMode.set('advanced')
  $previewTabs.set([])
  $rightRailActiveTabId.set(null)
  $agentSidebarChoice.set(true)
  $narrowViewport.set(false)
  $narrowOverlayPaneIds.set(new Set())
  $dismissedPanes.set(new Set())
  $hiddenTreePanes.set(new Set())
  noteActiveTreeGroup(null)
  disposers.push(registry.register({ area: 'panes', id: 'files', data: { placement: 'right' }, render: () => null }))
  $layoutTree.set(split('row', [group(['workspace'], { id: 'main' }), group(['files'], { id: 'tools' })]))
})

afterEach(() => {
  cleanup()
  closeReview()
  setSessions([])
  $selectedStoredSessionId.set(null)
  $currentCwd.set('')
  $previewTabs.set([])
  $rightRailActiveTabId.set(null)
  disposers.splice(0).forEach(dispose => dispose())
  $agentSidebarChoice.set(false)
  $layoutTree.set(null)
  $narrowViewport.set(false)
  $narrowOverlayPaneIds.set(new Set())
  $dismissedPanes.set(new Set())
  vi.clearAllMocks()
})

function mount(path = '/') {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <RootTooltipProvider>
        <DeveloperRail />
      </RootTooltipProvider>
    </MemoryRouter>
  )
}

it('reflects actual pane visibility and reveals a minimized pane on the first click', () => {
  mount()
  const files = screen.getByRole('button', { name: 'Files' })
  expect(files.getAttribute('aria-pressed')).toBe('true')
  act(() => setTreeGroupMinimized('tools', true))
  expect(files.getAttribute('aria-pressed')).toBe('false')
  fireEvent.click(files)
  expect(files.getAttribute('aria-pressed')).toBe('true')
  fireEvent.click(files)
  expect(files.getAttribute('aria-pressed')).toBe('false')
})

it('reviews the focused tile worktree after focus enters chrome, then returns to primary scope', () => {
  setSessions([
    { id: 'primary', cwd: '/repo-primary' },
    { id: 'secondary', cwd: '/repo-secondary' }
  ] as SessionInfo[])
  $selectedStoredSessionId.set('primary')
  $currentCwd.set('/repo-primary')
  $layoutTree.set(
    split('row', [
      group(['workspace'], { id: 'main' }),
      group(['session-tile:secondary'], { id: 'tile' }),
      group(['files'], { id: 'tools' })
    ])
  )
  noteActiveTreeGroup('tile')
  mount()
  act(() => noteActiveTreeGroup('tools'))
  fireEvent.click(screen.getByRole('button', { name: 'Review' }))
  expect($reviewScopeCwd.get()).toBe('/repo-secondary')
  expect($reviewScopeTarget.get()).toBe('tile:secondary')

  act(() => {
    closeReview()
    noteActiveTreeGroup('main')
  })
  fireEvent.click(screen.getByRole('button', { name: 'Review' }))
  expect($reviewScopeCwd.get()).toBeNull()
  expect($reviewScopeTarget.get()).toBe('main')
})

it('reads the visible overlay rather than docked state on narrow windows', () => {
  $narrowViewport.set(true)
  mount()
  const files = screen.getByRole('button', { name: 'Files' })
  expect(files.getAttribute('aria-pressed')).toBe('false')
  act(() => $narrowOverlayPaneIds.set(new Set(['files'])))
  expect(files.getAttribute('aria-pressed')).toBe('true')
  act(() => $narrowOverlayPaneIds.set(new Set()))
  expect(files.getAttribute('aria-pressed')).toBe('false')
})

it('keeps classic and full-page destinations free of the chat tool rail', () => {
  const view = mount('/settings')
  expect(screen.queryByRole('navigation')).toBeNull()
  view.unmount()
  mount()
  expect(screen.getByRole('navigation', { name: 'Right sidebar panels' })).toBeTruthy()
  act(() => $agentSidebarChoice.set(false))
  expect(screen.queryByRole('navigation')).toBeNull()
})

it('does not allocate browser tab identities while rendering an empty preview rail', () => {
  const mint = vi.spyOn(shared, 'randomUuid')

  try {
    mount()
    act(() => $narrowViewport.set(true))
    act(() => $narrowViewport.set(false))
    expect(screen.getByRole('button', { name: 'Preview', pressed: false })).toBeTruthy()
    expect(mint).not.toHaveBeenCalled()
  } finally {
    mint.mockRestore()
  }
})

it('updates Preview visibility when the selected browser changes without a layout change', () => {
  $previewTabs.set(
    ['first', 'second'].map(name => ({
      id: `url:${name}`,
      pinned: true,
      target: { kind: 'url', label: name, source: 'about:blank', url: 'about:blank' }
    }))
  )
  $layoutTree.set(
    group(['preview-tile:url:first', 'preview-tile:url:second'], {
      active: 'preview-tile:url:first',
      id: 'browsers'
    })
  )
  $rightRailActiveTabId.set('url:second')
  mount()
  expect(screen.getByRole('button', { name: 'Preview', pressed: false })).toBeTruthy()
  act(() => $rightRailActiveTabId.set('url:first'))
  expect(screen.getByRole('button', { name: 'Preview', pressed: true })).toBeTruthy()
})

it('shows developer tools only in Advanced mode with the redesign enabled', () => {
  $interfaceMode.set('simple')
  mount()
  expect(screen.queryByRole('button', { name: 'Terminal' })).toBeNull()
  act(() => $interfaceMode.set('advanced'))
  expect(screen.getByRole('button', { name: 'Terminal' })).toBeTruthy()
  act(() => $interfaceMode.set('simple'))
  expect(screen.queryByRole('button', { name: 'Terminal' })).toBeNull()
  act(() => {
    $interfaceMode.set('advanced')
    $agentSidebarChoice.set(false)
  })
  expect(screen.queryByRole('navigation')).toBeNull()
})
