import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { StrictMode, useEffect } from 'react'
import { MemoryRouter, useLocation, useNavigate } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as Hermes from '@/hermes'

const h = vi.hoisted(() => ({ getSession: vi.fn(), rowOpen: vi.fn(), noop: () => undefined }))
vi.mock('@/hermes', async original => ({ ...(await original<typeof Hermes>()), getSession: h.getSession }))

import { SidebarSessionsSection } from '@/app/chat/sidebar/sessions-section'
import { SIDEBAR_COLLAPSE_MEDIA_QUERY } from '@/app/layout-constants'
import { SessionsTabTitle } from '@/app/shell/sessions-tab-title'
import { TitlebarControls } from '@/app/shell/titlebar-controls'
import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import { findGroup, group, setSplitWeights, split } from '@/components/pane-shell/tree/model'
import { $narrowOverlayChrome } from '@/components/pane-shell/tree/renderer/narrow-overlay-state'
import { NarrowOverlays } from '@/components/pane-shell/tree/renderer/narrow-overlays'
import {
  $activeTreeGroup,
  $hiddenStripTabs,
  $hiddenTreePanes,
  $layoutTree,
  $narrowViewport,
  activateTreePane,
  noteActiveTreeGroup,
  trackActiveTreeGroup
} from '@/components/pane-shell/tree/store'
import { setWorkspaceScope } from '@/components/pane-shell/workspace-scope'
import { registry } from '@/contrib/registry'
import { $panesFlipped, $sidebarOpen, setFileBrowserOpen, setSidebarOpen } from '@/store/layout'
import { $notifications } from '@/store/notifications'
import { $activeGatewayProfile, $showAllProfiles } from '@/store/profile'
import {
  $connection,
  $cronSessions,
  $lastReadAtBySessionId,
  $messagingSessions,
  $selectedStoredSessionId,
  $sessionResumeRequest,
  $sessions,
  $unreadFinishedSessionIds,
  _resetSessionOwnerHintsForTests,
  getSessionOwnerHints,
  setSessionOwnerHint
} from '@/store/session'
import { $unreadSessionCount } from '@/store/session-dot-state'
import { $focusedStoredSessionId, $sessionTiles, clearAllSessionStates } from '@/store/session-states'
import { $sessionSeenCounts, $unreadFinishedMarkers } from '@/store/session-unread'
import { $openNextUnreadRequest, requestOpenNextUnread } from '@/store/session-unread-navigation'
import { $unreadWriteGuard } from '@/store/session-unread-remote'
import { stubResizeObserver } from '@/test/jsdom'
import type { SessionInfo } from '@/types/hermes'

import { useUnreadNavigation } from './use-unread-navigation'

// Composition regressions from independent A RA1/RA2 and B B5. Real router,
// pane actions, count, sidebar rows and stores; only backend I/O is stubbed.
const target: SessionInfo = {
  id: 'unread',
  connection_id: 'source-a',
  profile: 'ops',
  source: 'desktop',
  title: 'Unread conversation',
  ended_at: null,
  input_tokens: 0,
  output_tokens: 0,
  is_active: false,
  model: null,
  preview: null,
  tool_call_count: 0,
  started_at: 1,
  last_active: 100,
  message_count: 2,
  unread: true
}

const disposers: (() => void)[] = []

function Rows() {
  return (
    <SidebarSessionsSection
      activeSessionId={useStore($selectedStoredSessionId)}
      emptyState={null}
      label="Recent conversations"
      onArchiveSession={h.noop}
      onDeleteSession={h.noop}
      onResumeSession={h.rowOpen}
      onToggle={h.noop}
      onTogglePin={h.noop}
      onToggleUnread={h.noop}
      open
      pinned={false}
      sessions={useStore($sessions)}
    />
  )
}

function LiveTitle() {
  return <SessionsTabTitle onOpenNextUnread={requestOpenNextUnread} unread={useStore($unreadSessionCount)} />
}

function Host() {
  const location = useLocation()
  const navigate = useNavigate()
  const narrow = useStore($narrowViewport)
  useUnreadNavigation(navigate, `${location.key}:${location.pathname}:${location.search}:${location.hash}`)
  useEffect(() => trackActiveTreeGroup(), [])

  return (
    <>
      <output data-testid="route">{location.pathname}</output>
      {!narrow && (
        <div data-tree-group="side">
          <LiveTitle />
          <button onPointerDown={() => activateTreePane('side', 'terminal')}>Terminal tab</button>
          <button onPointerDown={() => activateTreePane('side', 'sessions')}>Sessions tab</button>
        </div>
      )}
      <TitlebarControls onOpenSettings={h.noop} />
      <NarrowOverlays />
    </>
  )
}

const mount = () =>
  render(
    <StrictMode>
      <MemoryRouter initialEntries={['/initial']}>
        <Host />
      </MemoryRouter>
    </StrictMode>
  )

const path = () => screen.getByTestId('route').textContent

const reveal = (id: string) =>
  act(() => {
    window.dispatchEvent(new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id, mode: 'open' } }))
  })

const clickCount = async () =>
  act(async () => {
    const count = screen.getByRole('button', { name: '1 unread session' })
    fireEvent.pointerDown(count, { button: 0, pointerType: 'touch' })
    fireEvent.click(count)
  })

function deferPreflight() {
  let resolve!: () => void
  h.getSession.mockReturnValue(
    new Promise(done => {
      resolve = () => done({ session: target })
    })
  )

  return () => act(async () => resolve())
}

const snapshot = () => ({
  route: path(),
  focused: $focusedStoredSessionId.get(),
  activeGroup: $activeTreeGroup.get(),
  tiles: $sessionTiles.get(),
  hints: getSessionOwnerHints('unread'),
  resume: $sessionResumeRequest.get(),
  unread: $unreadFinishedSessionIds.get(),
  markers: $unreadFinishedMarkers.get(),
  seen: $sessionSeenCounts.get(),
  readAt: $lastReadAtBySessionId.get(),
  notifications: $notifications.get()
})

beforeEach(() => {
  vi.clearAllMocks()
  h.getSession.mockReset().mockResolvedValue({ session: target })
  window.hermesDesktop = { getProfileRoutes: vi.fn(async () => []) } as unknown as typeof window.hermesDesktop
  stubResizeObserver()
  vi.stubGlobal('matchMedia', (media: string) => ({
    matches: media === SIDEBAR_COLLAPSE_MEDIA_QUERY && $narrowViewport.get(),
    media,
    addEventListener: h.noop,
    removeEventListener: h.noop,
    addListener: h.noop,
    removeListener: h.noop,
    dispatchEvent: h.noop
  }))
  clearAllSessionStates()
  _resetSessionOwnerHintsForTests()
  $sessionTiles.set([])
  $openNextUnreadRequest.set(0)
  $connection.set({
    connectionId: 'primary',
    baseUrl: 'http://unused.invalid',
    wsUrl: 'ws://unused.invalid',
    token: '',
    isFullscreen: false,
    nativeOverlayWidth: 0,
    logs: [],
    windowButtonPosition: null
  })
  $activeGatewayProfile.set('default')
  $showAllProfiles.set(true)
  setWorkspaceScope('sessions')
  $selectedStoredSessionId.set('initial')
  $sessionResumeRequest.set(null)
  $sessions.set([target])
  $messagingSessions.set([])
  $cronSessions.set([])
  $unreadFinishedSessionIds.set(['unread'])
  $unreadFinishedMarkers.set({ ops: ['unread'] })
  $sessionSeenCounts.set({ ops: { unread: 1 } })
  $lastReadAtBySessionId.set({})
  $unreadWriteGuard.set(new Map())
  $notifications.set([])
  $panesFlipped.set(false)
  $hiddenStripTabs.set(new Set())
  $hiddenTreePanes.set(new Set())
  $narrowViewport.set(false)
  setSidebarOpen(true)
  setFileBrowserOpen(true)

  for (const id of ['sessions', 'bots', 'terminal', 'workspace', 'session-tile:unread']) {
    const sidebar = ['sessions', 'bots', 'terminal'].includes(id)
    disposers.push(
      registry.register({
        area: 'panes',
        id,
        title: id,
        data: {
          placement: sidebar ? 'left' : 'main',
          collapsible: sidebar,
          width: '237px',
          ...(id === 'sessions'
            ? { hideOnly: true, revealAliases: ['chat-sidebar'], tabTitle: () => <LiveTitle /> }
            : {})
        },
        render: id === 'sessions' ? () => <Rows /> : () => <div data-testid={`${id}-body`}>{id}</div>
      })
    )
  }

  $layoutTree.set(
    split('row', [
      group(['sessions', 'bots', 'terminal'], { id: 'side' }),
      group(['workspace', 'session-tile:unread'], { id: 'main' })
    ])
  )
  noteActiveTreeGroup('side')
})
afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
  $narrowViewport.set(false)
  vi.unstubAllGlobals()
})

describe('unread navigation pane intent', () => {
  it.each([false, true])(
    'cancels a same-group Terminal selection, including a round trip (%s)',
    async returnToSessions => {
      const finish = deferPreflight()
      mount()
      await clickCount()
      expect(h.getSession).toHaveBeenCalledOnce()
      fireEvent.pointerDown(screen.getByText('Terminal tab'), { button: 0 })
      expect(findGroup($layoutTree.get()!, 'side')?.active).toBe('terminal')
      expect($activeTreeGroup.get()).toBe('side')
      expect($focusedStoredSessionId.get()).toBe('initial')

      if (returnToSessions) {
        fireEvent.pointerDown(screen.getByText('Sessions tab'), { button: 0 })
      }

      const before = snapshot()
      await finish()
      expect(snapshot()).toEqual(before)
    }
  )

  it('does not self-cancel its docked reveal or cancel on a layout resize', async () => {
    activateTreePane('side', 'terminal')
    const finish = deferPreflight()
    mount()
    await clickCount()
    expect(findGroup($layoutTree.get()!, 'side')?.active).toBe('sessions')
    act(() => {
      const tree = $layoutTree.get()!
      $layoutTree.set(setSplitWeights(tree, tree.id, [2, 3]))
    })
    await finish()
    expect(path()).toBe('/unread')
    expect($sessionResumeRequest.get()?.ownerRoute).toEqual({ connectionId: 'source-a', profile: 'ops' })
  })

  it.each(['bots', 'terminal'])(
    'reveals Sessions from %s without self-cancelling a successful preflight',
    async sibling => {
      setSidebarOpen(false)
      $narrowViewport.set(true)
      const finish = deferPreflight()
      mount()
      reveal(sibling)
      await clickCount()
      expect(screen.getByText('Unread conversation')).toBeDefined()
      expect($narrowOverlayChrome.get()?.paneId).toBe('sessions')
      expect($sidebarOpen.get()).toBe(false)
      await clickCount()
      act(() => {
        const tree = $layoutTree.get()!
        $layoutTree.set(setSplitWeights(tree, tree.id, [2, 3]))
      })
      expect(h.getSession).toHaveBeenCalledOnce()
      await finish()
      expect(path()).toBe('/unread')
    }
  )

  it.each(['tab', 'reveal-event', 'close'])('cancels later narrow pane intent (%s)', async intent => {
    $narrowViewport.set(true)
    const finish = deferPreflight()
    const view = mount()
    reveal('sessions')
    await clickCount()

    if (intent === 'tab') {
      const tab = view.container.querySelector<HTMLElement>('[data-narrow-overlay-tab="terminal"]')!
      fireEvent.pointerDown(tab, { button: 0 })
    } else if (intent === 'reveal-event') {
      reveal('terminal')
    } else {
      fireEvent.keyDown(window, { key: 'Escape' })
    }

    expect($narrowOverlayChrome.get()?.paneId ?? null).toBe(intent === 'close' ? null : 'terminal')
    const before = snapshot()
    await finish()
    expect(snapshot()).toEqual(before)
  })

  it('observes later Terminal intent even if the request started before narrow collapse', async () => {
    const finish = deferPreflight()
    mount()
    await clickCount()
    act(() => $narrowViewport.set(true))
    reveal('terminal')
    expect($narrowOverlayChrome.get()?.paneId).toBe('terminal')
    const before = snapshot()
    await finish()
    expect(snapshot()).toEqual(before)
  })
})

describe('unread navigation recovery reveal', () => {
  it.each([
    ['bots', 'ambiguous'],
    ['bots', 'offline'],
    ['terminal', 'ambiguous'],
    ['terminal', 'offline']
  ])('reveals usable Sessions rows from %s after %s refusal', async (sibling, failure) => {
    if (failure === 'ambiguous') {
      const ownerRoute = { connectionId: 'source-b', profile: 'ops' }
      $sessionTiles.set([{ storedSessionId: 'unread', ownerRoute, workspaceMode: 'sessions' }])
      setSessionOwnerHint('unread', ownerRoute)
    } else {
      h.getSession.mockRejectedValue(new Error('offline'))
    }

    $narrowViewport.set(true)
    noteActiveTreeGroup('main')
    mount()
    reveal(sibling)
    expect(screen.getByTestId(`${sibling}-body`)).toBeDefined()
    const before = snapshot()
    await clickCount()
    expect(snapshot()).toEqual(before)
    expect($sessionTiles.get()).toBe(before.tiles)
    expect(h.getSession).toHaveBeenCalledOnce()
    expect($narrowOverlayChrome.get()?.paneId).toBe('sessions')
    expect(screen.queryByTestId(`${sibling}-body`)).toBeNull()
    fireEvent.click(screen.getByText('Unread conversation'))
    expect(h.rowOpen).toHaveBeenCalledWith('unread', target)
  })
})
