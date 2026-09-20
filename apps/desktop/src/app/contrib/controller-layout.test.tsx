import { cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const windowMode = vi.hoisted(() => ({
  browser: false,
  hud: false
}))

vi.mock('@/store/windows', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  isBrowserWindow: () => windowMode.browser,
  isHudWindow: () => windowMode.hud
}))

vi.mock('@/components/pane-shell/tree/renderer', () => ({
  LayoutTreeRoot: () => <div data-testid="layout-tree-root" />
}))

vi.mock('@/components/pane-shell/tree/store', () => ({
  $layoutTree: atom(null),
  bindPaneVisibility: vi.fn(),
  bindToolPaneCollapse: vi.fn(),
  bindTreeSideVisibility: vi.fn(),
  declareDefaultTree: vi.fn(),
  dismissTreePane: vi.fn(),
  isPaneVisible: vi.fn(() => true),
  markCollapsePane: vi.fn(),
  mirrorLayoutTree: vi.fn(),
  paneRootSide: vi.fn(() => 'left'),
  registerLayoutResetHandler: vi.fn(),
  registerPaneCloser: vi.fn(),
  registerPaneOpener: vi.fn(),
  removeTreePane: vi.fn(),
  resetLayoutTree: vi.fn(),
  revealTreePane: vi.fn(),
  setStripTabHidden: vi.fn(),
  targetZoneTabStripVisible: vi.fn(() => true),
  togglePaneVisible: vi.fn(),
  toggleTargetZoneTabStrip: vi.fn(),
  watchContributedPanes: vi.fn()
}))

vi.mock('@/components/pane-shell/tree/model', () => ({
  allPaneIds: vi.fn(() => []),
  group: vi.fn((children, opts) => ({ children, opts, type: 'group' })),
  groupLeafIds: vi.fn(() => []),
  split: vi.fn((orientation, children, weights, id) => ({ children, id, orientation, type: 'split', weights }))
}))

vi.mock('@/app/chat/session-draft-title', () => ({ SessionDraftTitle: () => <div /> }))
vi.mock('@/app/chat/session-status-dot', () => ({ SessionStatusDot: () => <div /> }))
vi.mock('@/components/assistant-ui/inline-preview-directive', () => ({ InlinePreviewDirective: () => <div /> }))
vi.mock('@/contrib/plugins', () => ({ discoverBundledPlugins: vi.fn() }))
vi.mock('@/contrib/runtime-loader', () => ({ discoverRuntimePlugins: vi.fn() }))
vi.mock('@/store/layout', () => ({
  $fileBrowserOpen: atom(true),
  $panesFlipped: atom(false),
  $sidebarOpen: atom(true),
  FILE_BROWSER_DEFAULT_WIDTH: '280px',
  FILE_BROWSER_MAX_WIDTH: '520px',
  FILE_BROWSER_MIN_WIDTH: '220px',
  setFileBrowserOpen: vi.fn(),
  setSidebarOpen: vi.fn(),
  SIDEBAR_DEFAULT_WIDTH: '256px',
  SIDEBAR_MAX_WIDTH: '420px'
}))
vi.mock('@/store/statusbar-prefs', () => ({ $statusbarVisible: atom(true) }))
vi.mock('@/store/session', () => ({
  $currentCwd: atom('/repo'),
  $selectedStoredSessionId: atom(null),
  $sessions: atom([]),
  $yoloActive: atom(false),
  sessionMatchesStoredId: vi.fn(() => false)
}))
vi.mock('@/store/session-states', () => ({ $botChatScopes: atom({}) }))
vi.mock('@/components/pane-shell/workspace-scope', () => ({
  $workspaceOwnerLabels: atom({}),
  workspaceOwnerTitle: (title: string) => title
}))
vi.mock('@/store/composer-popout', () => ({ pruneComposerPopoutZones: vi.fn() }))
vi.mock('@/store/profile-share', () => ({ runExportProfileFlow: vi.fn(), runImportProfileFlow: vi.fn() }))
vi.mock('@/store/review', () => ({
  $reviewOpen: atom(false),
  $reviewScopeCwd: atom(null),
  $reviewScopeTarget: atom(null),
  closeReview: vi.fn(),
  openReview: vi.fn(),
  REVIEW_PANE_ID: 'review'
}))
vi.mock('@/store/session-pin-sync', () => ({ watchSessionPins: vi.fn() }))
vi.mock('@/store/session-unread-remote', () => ({ watchUnreadWriteGuard: vi.fn() }))
vi.mock('@/app/right-sidebar/store', () => ({ $terminalTakeover: atom(false), setTerminalTakeover: vi.fn() }))
vi.mock('@/lib/yolo-session', () => ({ setYoloEnabled: vi.fn() }))
vi.mock('../routes', () => ({ $workspaceIsPage: atom(false) }))
vi.mock('../chat/session-drag', () => ({ startSessionDrag: vi.fn() }))
vi.mock('../chat/preview-tile', () => ({ watchPreviewTiles: vi.fn() }))
vi.mock('../chat/route-tile', () => ({ watchRouteTiles: vi.fn() }))
vi.mock('../chat/session-tile', () => ({
  SessionTileCloseConfirm: () => <div data-testid="session-close-confirm" />,
  stackSessionTilesIntoMain: vi.fn(),
  startUnrestoredTileTitleBackfill: vi.fn(),
  watchSessionTiles: vi.fn(),
  WorkspaceTabMenu: ({ children }: { children: ReactNode }) => <>{children}</>
}))
vi.mock('../context-menu/app-context-menu', () => ({ AppContextMenu: () => <div data-testid="app-context-menu" /> }))
vi.mock('../hud/hud-shell', () => ({ HudShell: () => <div data-testid="hud-shell" /> }))
vi.mock('../chat/browser-popout-shell', () => ({ BrowserPopoutShell: () => <div data-testid="browser-popout-shell" /> }))
vi.mock('./panes', () => ({
  FilesPane: () => <div />,
  LogsPane: () => <div />,
  ReviewPaneContent: () => <div />
}))
vi.mock('./wiring', () => ({
  ContribWiring: ({ children }: { children: ReactNode }) => <div data-testid="contrib-wiring">{children}</div>,
  WiredPane: ({ part }: { part: string }) => <div data-testid={`wired-${part}`} />
}))

const { ContribController } = await import('./controller')

afterEach(() => {
  cleanup()
  windowMode.browser = false
  windowMode.hud = false
})

describe('ContribController layout composition', () => {
  it('uses container dimensions instead of viewport dimensions when embedded in the product shell', () => {
    const { container } = render(<ContribController layoutMode="embedded" />)

    const shell = container.querySelector('[data-contrib-shell]')

    expect(shell).toBeTruthy()
    expect(shell?.className).toContain('h-full')
    expect(shell?.className).toContain('w-full')
    expect(shell?.className).not.toContain('h-screen')
    expect(shell?.className).not.toContain('w-screen')
    expect(screen.getByTestId('layout-tree-root')).toBeTruthy()
  })

  it('keeps the old viewport root for the standalone main controller path', () => {
    const { container } = render(<ContribController />)

    const shell = container.querySelector('[data-contrib-shell]')

    expect(shell?.className).toContain('h-screen')
    expect(shell?.className).toContain('w-screen')
  })

  it('renders the HUD path without the layout tree shell', () => {
    windowMode.hud = true

    const { container } = render(<ContribController layoutMode="embedded" />)

    expect(screen.getByTestId('hud-shell')).toBeTruthy()
    expect(screen.queryByTestId('layout-tree-root')).toBeNull()
    expect(container.querySelector('[data-contrib-shell]')).toBeNull()
  })

  it('renders the Browser Popout path without the layout tree shell', () => {
    windowMode.browser = true

    const { container } = render(<ContribController layoutMode="embedded" />)

    expect(screen.getByTestId('browser-popout-shell')).toBeTruthy()
    expect(screen.queryByTestId('layout-tree-root')).toBeNull()
    expect(container.querySelector('[data-contrib-shell]')).toBeNull()
  })
})
