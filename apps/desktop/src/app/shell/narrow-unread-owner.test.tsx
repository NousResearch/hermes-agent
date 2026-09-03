import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { SIDEBAR_COLLAPSE_MEDIA_QUERY } from '@/app/layout-constants'
import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import { group, split } from '@/components/pane-shell/tree/model'
import { NarrowOverlays } from '@/components/pane-shell/tree/renderer/narrow-overlays'
import { $hiddenStripTabs, $hiddenTreePanes, $layoutTree, $narrowViewport } from '@/components/pane-shell/tree/store'
import { $workspaceMode } from '@/components/pane-shell/workspace-scope'
import { registry } from '@/contrib/registry'
import { $panesFlipped, setFileBrowserOpen, setSidebarOpen } from '@/store/layout'
import { stubResizeObserver } from '@/test/jsdom'

import { SessionsTabTitle } from './sessions-tab-title'
import { TitlebarControls } from './titlebar-controls'

vi.mock('@/store/session-dot-state', async () => {
  const { atom } = await import('nanostores')

  return { $unreadSessionCount: atom(3) }
})

// Review B's rendered narrow repro, extended to require exactly one owner
// across reveal, close and breakpoint transitions, including flipped layouts.
const open = vi.fn()
const disposers: (() => void)[] = []

const register = (id: string, placement: string, collapsible = true) => {
  disposers.push(
    registry.register({
      area: 'panes',
      id,
      title: id,
      data: {
        placement,
        collapsible,
        width: '237px',
        ...(id === 'sessions'
          ? {
              hideOnly: true,
              revealAliases: ['chat-sidebar'],
              tabTitle: () => <SessionsTabTitle onOpenNextUnread={open} unread={3} />
            }
          : {})
      },
      render: () => <div data-testid={`${id}-body`}>{id}</div>
    })
  )
}

const reveal = (id = 'sessions') =>
  act(() => {
    window.dispatchEvent(new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id, mode: 'open' } }))
  })

const count = () => screen.getAllByRole('button', { name: /3 unread sessions/ })

const mount = () =>
  render(
    <MemoryRouter>
      <TitlebarControls onOpenSettings={vi.fn()} />
      <NarrowOverlays />
    </MemoryRouter>
  )

beforeEach(() => {
  vi.clearAllMocks()
  stubResizeObserver()
  vi.stubGlobal('matchMedia', (media: string) => ({
    matches: media === SIDEBAR_COLLAPSE_MEDIA_QUERY && $narrowViewport.get(),
    media,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn()
  }))
  $workspaceMode.set('sessions')
  $panesFlipped.set(false)
  $hiddenTreePanes.set(new Set())
  $hiddenStripTabs.set(new Set())
  setSidebarOpen(true)
  setFileBrowserOpen(true)
  $narrowViewport.set(true)
})
afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
  $narrowViewport.set(false)
  $layoutTree.set(null)
  vi.unstubAllGlobals()
})

describe('narrow unread affordance', () => {
  it.each([false, true])('hands off one count between titlebar and single-pane overlay (flipped=%s)', flipped => {
    $panesFlipped.set(flipped)
    register('sessions', 'left')
    register('workspace', 'main', false)
    $layoutTree.set(split('row', [group(['sessions']), group(['workspace'])]))
    mount()
    expect(screen.queryByTestId('sessions-body')).toBeNull()
    expect(count()).toHaveLength(1)
    expect(count()[0].getAttribute('aria-label')).toMatch(/show/i)
    fireEvent.click(count()[0])
    expect(screen.getByTestId('sessions-body')).toBeDefined()
    expect(count()).toHaveLength(1)
    fireEvent.click(count()[0])
    expect(open).toHaveBeenCalledOnce()
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(screen.queryByTestId('sessions-body')).toBeNull()
    expect(count()).toHaveLength(1)
    act(() => $narrowViewport.set(false))
    expect(screen.queryByRole('button', { name: /3 unread sessions/ })).toBeNull()
    act(() => $narrowViewport.set(true))
    expect(count()).toHaveLength(1)
  })

  it('keeps one live count in a shared overlay without activating Bots on count clicks', () => {
    register('sessions', 'left')
    register('bots', 'left')
    register('workspace', 'main', false)
    $layoutTree.set(split('row', [group(['sessions', 'bots']), group(['workspace'])]))
    mount()
    reveal('bots')
    expect(count()).toHaveLength(1)
    fireEvent.pointerDown(count()[0], { button: 0, pointerType: 'touch' })
    fireEvent.click(count()[0], { detail: 0 })
    expect(open).toHaveBeenCalledOnce()
    expect(screen.getByTestId('bots-body')).toBeDefined()
  })

  it.each(['bots', 'terminal'])('does not put a Sessions count on a narrow %s toggle', active => {
    register('sessions', 'left')
    register(active, 'left')
    register('workspace', 'main', false)
    $layoutTree.set(split('row', [group(['sessions', active], { active }), group(['workspace'])]))

    if (active === 'bots') {
      $workspaceMode.set('bots')
    }

    mount()
    expect(screen.queryByRole('button', { name: /3 unread sessions/ })).toBeNull()
  })
})
