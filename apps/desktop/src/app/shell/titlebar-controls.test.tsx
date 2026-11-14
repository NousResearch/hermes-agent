// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { useEffect } from 'react'
import { MemoryRouter, useNavigate } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import { $narrowOverlayPaneIds } from '@/components/pane-shell/narrow-overlay-state'
import { $narrowViewport } from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { I18nProvider } from '@/i18n'
import { $interfaceMode } from '@/store/interface-mode'
import {
  $fileBrowserOpen,
  $leftSideOpen,
  $sidebarOpen,
  CHAT_SIDEBAR_PANE_ID,
  FILE_BROWSER_PANE_ID,
  setSidebarOpen
} from '@/store/layout'
import { REVIEW_PANE_ID } from '@/store/review'
import { $shellChoice } from '@/store/shell-variant'
import { setTitlebarAppActionsSide } from '@/store/titlebar-app-actions'

import { $widthClass } from '../redesign/breakpoints'
import { $workspaceIsPage, ROUTES_AREA } from '../routes'

import { $shellSheet } from './shell-sheet'
import { TITLEBAR_CHROME_CHANGED_EVENT } from './titlebar'
import { TitlebarControls, type TitlebarTool } from './titlebar-controls'

const PLUGIN_TOOL: TitlebarTool = { icon: <span />, id: 'plugin-tool', label: 'plugin tool' }

let navigateTo: (to: string) => void = () => {}

function NavigateProbe() {
  navigateTo = useNavigate()

  return null
}

function renderControls(pathname: string, props?: { leftTools?: TitlebarTool[]; tools?: TitlebarTool[] }) {
  return render(
    <MemoryRouter initialEntries={[pathname]}>
      <I18nProvider configClient={null} initialLocale="en">
        <NavigateProbe />
        <TitlebarControls leftTools={props?.leftTools} onOpenSettings={() => {}} tools={props?.tools} />
      </I18nProvider>
    </MemoryRouter>
  )
}

const windowControls = () => screen.queryByLabelText('Window controls')
const appControls = () => screen.queryByLabelText('App controls')
const pluginChrome = () => screen.queryByText('plugin-chrome')
const pluginTool = () => screen.queryByLabelText('plugin tool')

describe('TitlebarControls fixed clusters', () => {
  let dispose: () => void

  beforeEach(() => {
    dispose = registry.registerMany([
      {
        area: ROUTES_AREA,
        data: { path: '/kanban' },
        id: 'test-kanban-route',
        render: () => null
      },
      {
        area: ROUTES_AREA,
        data: { path: '/plain' },
        id: 'test-plain-route',
        render: () => null
      }
    ])
  })

  afterEach(() => {
    dispose()
    cleanup()
  })

  it('keeps the app clusters on a contributed page that mounts no titlebar chrome', () => {
    renderControls('/plain')

    expect(windowControls()).not.toBeNull()
    expect(appControls()).not.toBeNull()
  })

  it('hides the app clusters on an overlay', () => {
    $shellChoice.set(false)
    renderControls('/settings')

    expect(windowControls()).toBeNull()
    expect(appControls()).toBeNull()
    $shellChoice.set(null)
  })

  it('parks the redesign’s clusters on its overlay destinations, keeping titleBar slots mounted', () => {
    const life = { cleanups: 0, mounts: 0 }

    function PluginRight() {
      useEffect(() => {
        life.mounts += 1

        return () => {
          life.cleanups += 1
        }
      }, [])

      return <span>plugin-right</span>
    }

    $shellChoice.set(true)

    const disposeRight = registry.register({
      area: 'titleBar.right',
      id: 'test-plugin-right',
      render: () => <PluginRight />
    })

    try {
      renderControls('/')
      expect(life).toEqual({ cleanups: 0, mounts: 1 })

      for (const page of ['/settings', '/profiles', '/cron']) {
        act(() => navigateTo(page))
        // Out of sight and out of reach while the destination owns the band, but never torn down.
        expect(windowControls()?.hidden).toBe(true)
        expect(screen.getByText('plugin-right').closest('[data-titlebar-cluster]')?.hasAttribute('hidden')).toBe(true)
      }

      act(() => navigateTo('/'))
      expect(windowControls()?.hidden).toBe(false)
      expect(life).toEqual({ cleanups: 0, mounts: 1 })
    } finally {
      act(disposeRight)
      $shellChoice.set(null)
    }
  })

  it('a titleBar.tools item alone does not claim the band', () => {
    renderControls('/plain', { leftTools: [PLUGIN_TOOL] })

    expect(windowControls()).not.toBeNull()
    expect(pluginTool()).not.toBeNull()
  })

  it('keeps a titleBar.center component mounted across a chat -> page -> chat round trip', () => {
    // Plugins tear down global side effects (style tags, observers) in their
    // effect cleanup; a remount on navigation ran the OLD cleanup after the
    // NEW setup and left the plugin dead until reload (#114290).
    const life = { cleanups: 0, mounts: 0 }

    function PluginCenter() {
      useEffect(() => {
        life.mounts += 1

        return () => {
          life.cleanups += 1
        }
      }, [])

      return <span>plugin-center</span>
    }

    const disposeTitle = registry.register({
      area: 'titleBar.center',
      id: 'test-plugin-center',
      render: () => <PluginCenter />
    })

    try {
      renderControls('/')
      expect(life).toEqual({ cleanups: 0, mounts: 1 })

      act(() => navigateTo('/skills'))
      expect(screen.getByText('plugin-center')).not.toBeNull()
      expect(windowControls()).not.toBeNull()
      expect(appControls()).not.toBeNull()

      act(() => navigateTo('/'))
      expect(life).toEqual({ cleanups: 0, mounts: 1 })
    } finally {
      act(disposeTitle)
    }
  })

  describe('when the page projects titlebar chrome', () => {
    let disposeChrome: () => void

    beforeEach(() => {
      disposeChrome = registry.register({
        area: 'titleBar.left',
        id: 'test-plugin-chrome',
        render: () => <span>plugin-chrome</span>
      })
    })

    afterEach(() => {
      // The mounted controls subscribe to titleBar.* areas — dispose inside
      // act so the unmount-time registry update doesn't warn.
      act(() => disposeChrome())
    })

    it('keeps plugin titlebar contributions on a contributed full-page route', () => {
      renderControls('/kanban')

      expect(pluginChrome()).not.toBeNull()
      expect(windowControls()).toBeNull()
      expect(appControls()).toBeNull()
    })

    it('keeps contributed titlebar tools on a chrome-owning page', () => {
      renderControls('/kanban', { leftTools: [PLUGIN_TOOL] })

      expect(pluginTool()).not.toBeNull()
    })

    it('hides plugin titlebar contributions on an overlay: unmounted in classic, parked in the redesign', () => {
      $shellChoice.set(false)
      renderControls('/settings')
      expect(pluginChrome()).toBeNull()
      cleanup()

      $shellChoice.set(true)
      renderControls('/settings')
      expect(pluginChrome()?.closest('[data-titlebar-cluster]')?.hasAttribute('hidden')).toBe(true)
      $shellChoice.set(null)
    })

    it('keeps measurable titlebar clusters on a chrome-owning contributed page', () => {
      // usePanelTitlebar positions the sessions tab strip from these hooks;
      // without them the tabs keep their stale chat-view offset and slide
      // under the page's switcher (kanban's board switcher).
      const { container } = renderControls('/kanban')

      expect(container.querySelector('[data-titlebar-cluster="left"]')).not.toBeNull()
      expect(container.querySelector('[data-titlebar-cluster="right"]')).not.toBeNull()
    })

    it('announces its chrome so the sessions tab reservation re-measures', () => {
      const listener = vi.fn()
      window.addEventListener(TITLEBAR_CHROME_CHANGED_EVENT, listener)

      try {
        renderControls('/kanban')

        expect(listener).toHaveBeenCalled()
      } finally {
        window.removeEventListener(TITLEBAR_CHROME_CHANGED_EVENT, listener)
      }
    })
  })
})

describe('titlebar app-action cluster', () => {
  beforeEach(() => {
    vi.stubGlobal('hermesDesktop', { ...window.hermesDesktop, hud: { open: vi.fn() } })
  })

  afterEach(() => {
    $narrowViewport.set(false)
    $narrowOverlayPaneIds.set(new Set())
    setTitlebarAppActionsSide('right')
    cleanup()
    $shellChoice.set(null)
    vi.unstubAllGlobals()
    delete window.document.documentElement.dataset.hermesDesktopHost
  })

  // A narrow native window keeps main's titlebar: the side buttons read the
  // docked sides and flip the overlay through the shared ⌘B/⌘J toggle, whatever
  // the overlay shows, and carry no aria-expanded.
  it('names and toggles the sides from docked state in a narrow native window, as on main', () => {
    vi.stubGlobal(
      'matchMedia',
      vi.fn().mockReturnValue({ matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn() })
    )
    $narrowViewport.set(true)
    $narrowOverlayPaneIds.set(new Set([CHAT_SIDEBAR_PANE_ID, REVIEW_PANE_ID]))
    const intents: unknown[] = []
    const listener = (event: Event) => intents.push((event as CustomEvent).detail)
    window.addEventListener(PANE_TOGGLE_REVEAL_EVENT, listener)

    try {
      renderControls('/')
      const sidebar = screen.getByRole('button', { name: $leftSideOpen.get() ? 'Hide sidebar' : 'Show sidebar' })

      const rightSidebar = screen.getByRole('button', {
        name: $fileBrowserOpen.get() ? 'Hide right sidebar' : 'Show right sidebar'
      })

      expect(sidebar.hasAttribute('aria-expanded')).toBe(false)
      expect(rightSidebar.hasAttribute('aria-expanded')).toBe(false)
      fireEvent.click(sidebar)
      fireEvent.click(rightSidebar)
      expect(intents).toEqual([
        { id: CHAT_SIDEBAR_PANE_ID, mode: 'toggle' },
        { id: FILE_BROWSER_PANE_ID, mode: 'toggle' }
      ])
    } finally {
      window.removeEventListener(PANE_TOGGLE_REVEAL_EVENT, listener)
    }
  })

  it('names the narrow sidebar action from its visible overlay, not docked state', () => {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
    $narrowViewport.set(true)
    renderControls('/')
    const left = screen.getByLabelText('Window controls')
    expect(within(left).getByRole('button', { name: 'Show sidebar' }).getAttribute('aria-expanded')).toBe('false')
    act(() => $narrowOverlayPaneIds.set(new Set([CHAT_SIDEBAR_PANE_ID])))
    expect(within(left).getByRole('button', { name: 'Hide sidebar' }).getAttribute('aria-expanded')).toBe('true')
    act(() => $narrowOverlayPaneIds.set(new Set()))
    expect(within(left).getByRole('button', { name: 'Show sidebar' })).toBeTruthy()
  })

  it('explicitly closes a visible overlay rather than pinning a hover reveal', () => {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
    vi.stubGlobal(
      'matchMedia',
      vi.fn().mockReturnValue({ matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn() })
    )
    $narrowViewport.set(true)
    $narrowOverlayPaneIds.set(new Set([CHAT_SIDEBAR_PANE_ID]))
    const intents: unknown[] = []
    const listener = (event: Event) => intents.push((event as CustomEvent).detail)
    window.addEventListener(PANE_TOGGLE_REVEAL_EVENT, listener)

    try {
      renderControls('/')
      fireEvent.click(screen.getByRole('button', { name: 'Hide sidebar' }))
      expect(intents).toEqual([{ id: CHAT_SIDEBAR_PANE_ID, mode: 'close' }])
    } finally {
      window.removeEventListener(PANE_TOGGLE_REVEAL_EVENT, listener)
    }
  })

  it('hides a review-only right sidebar instead of opening files', () => {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
    $narrowViewport.set(true)
    $narrowOverlayPaneIds.set(new Set([REVIEW_PANE_ID]))
    const intents: unknown[] = []
    const listener = (event: Event) => intents.push((event as CustomEvent).detail)
    window.addEventListener(PANE_TOGGLE_REVEAL_EVENT, listener)

    try {
      renderControls('/')
      fireEvent.click(screen.getByRole('button', { name: 'Hide right sidebar' }))
      expect(intents).toEqual([{ id: 'review', mode: 'close' }])
    } finally {
      window.removeEventListener(PANE_TOGGLE_REVEAL_EVENT, listener)
    }
  })

  it('moves classic settings, layout, and HUD to the left when the appearance setting says left', () => {
    $shellChoice.set(false)
    setTitlebarAppActionsSide('left')
    renderControls('/')

    const left = screen.getByLabelText('Window controls')
    const right = screen.getByLabelText('App controls')

    expect(within(left).getByLabelText('Open settings')).toBeTruthy()
    expect(within(left).getByLabelText('Layout editor')).toBeTruthy()
    expect(within(left).getByLabelText('HUD mode')).toBeTruthy()
    expect(within(right).queryByLabelText('Open settings')).toBeNull()
  })

  it('keeps HUD hidden in the browser on either app-action side', () => {
    $shellChoice.set(false)
    vi.stubGlobal('hermesDesktop', { ...window.hermesDesktop, hud: undefined })
    renderControls('/')

    expect(screen.queryByLabelText('HUD mode')).toBeNull()
    expect(screen.getByLabelText('Open settings')).toBeTruthy()

    // Each side assembles its own cluster, so the move must keep the filter.
    act(() => setTitlebarAppActionsSide('left'))
    expect(screen.queryByLabelText('HUD mode')).toBeNull()
    expect(within(screen.getByLabelText('Window controls')).getByLabelText('Open settings')).toBeTruthy()
    expect(screen.getByLabelText('Layout editor')).toBeTruthy()
  })
})

it('keeps settings only in the global rail when simplified mode is enabled', () => {
  $shellChoice.set(false)
  renderControls('/')
  expect(screen.getByRole('button', { name: 'Open settings' })).toBeTruthy()
  act(() => $shellChoice.set(true))
  expect(screen.queryByRole('button', { name: 'Open settings' })).toBeNull()
  act(() => $shellChoice.set(false))
  expect(screen.getByRole('button', { name: 'Open settings' })).toBeTruthy()
  cleanup()
  $shellChoice.set(null)
})

it('offers the Advanced band only the way back to a hidden sidebar: the open Chats destination folds it', () => {
  $shellChoice.set(null)
  setSidebarOpen(true)
  renderControls('/')
  expect(screen.queryByRole('button', { name: 'Hide sidebar' })).toBeNull()

  act(() => setSidebarOpen(false))
  expect(screen.queryByRole('button', { name: 'Show chats' })).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Show sidebar' }))
  expect($leftSideOpen.get()).toBe(true)
  expect(screen.queryByRole('button', { name: 'Hide sidebar' })).toBeNull()
  cleanup()
})

it('brings a hidden sidebar back in Simple mode from one floating button at the stage, not the band', () => {
  const previousMode = $interfaceMode.get()
  $shellChoice.set(null)
  $interfaceMode.set('simple')
  setSidebarOpen(false)
  renderControls('/')

  expect(screen.queryByRole('button', { name: 'Show sidebar' })).toBeNull()
  const reveal = screen.getByRole('button', { name: 'Show chats' })
  expect(reveal.getAttribute('data-slot')).toBe('panel-reveal')
  fireEvent.click(reveal)
  expect($sidebarOpen.get()).toBe(true)
  expect(screen.queryByRole('button', { name: 'Show chats' })).toBeNull()

  // Below wide the band's button is the sheet's menu in every mode, and no floating button shows.
  act(() => {
    setSidebarOpen(false)
    $narrowViewport.set(true)
  })
  expect(screen.queryByRole('button', { name: 'Show chats' })).toBeNull()
  expect(screen.getByRole('button', { name: 'Show sidebar' })).toBeTruthy()

  act(() => $narrowViewport.set(false))
  cleanup()
  setSidebarOpen(true)
  $interfaceMode.set(previousMode)
})

describe('the redesigned band on a compact window', () => {
  beforeEach(() => {
    act(() => {
      $widthClass.set('compact')
      $narrowViewport.set(true)
    })
  })

  afterEach(() => {
    cleanup()
    act(() => {
      $widthClass.set('wide')
      $narrowViewport.set(false)
      $shellSheet.set(null)
      $workspaceIsPage.set(false)
    })
  })

  it('is menu, title and new session: no Back or Forward (the browser has them)', () => {
    renderControls('/')
    const left = screen.getByLabelText('Window controls')
    expect(within(left).queryByRole('button', { name: 'Back' })).toBeNull()
    expect(within(left).queryByRole('button', { name: 'Forward' })).toBeNull()
    expect(within(left).getByRole('button', { name: /^(Show|Hide) sidebar/ })).toBeTruthy()
  })

  it('opens the rail as a sheet from a workspace page, which has no chat sidebar', () => {
    act(() => $workspaceIsPage.set(true))
    renderControls('/capabilities')
    fireEvent.click(
      within(screen.getByLabelText('Window controls')).getByRole('button', { name: /^(Show|Hide) sidebar/ })
    )
    expect($shellSheet.get()).toBe('nav')
  })

  it('keeps the rail, and so no menu, on a medium workspace page', () => {
    act(() => {
      $widthClass.set('medium')
      $workspaceIsPage.set(true)
    })
    renderControls('/capabilities')
    expect(
      within(screen.getByLabelText('Window controls')).queryByRole('button', { name: /^(Show|Hide) sidebar/ })
    ).toBeNull()
  })
})

describe('Back and Forward in the redesigned band', () => {
  afterEach(() => {
    cleanup()
    delete document.documentElement.dataset.hermesDesktopHost
  })

  it('leads the native app’s band at desktop width', () => {
    renderControls('/')
    const left = screen.getByLabelText('Window controls')

    expect(within(left).getByRole('button', { name: 'Back' })).toBeTruthy()
    expect(within(left).getByRole('button', { name: 'Forward' })).toBeTruthy()
  })

  it('is left to the browser when the app is hosted in one, at any width', () => {
    document.documentElement.dataset.hermesDesktopHost = 'browser'
    renderControls('/')
    const left = screen.getByLabelText('Window controls')

    expect(within(left).queryByRole('button', { name: 'Back' })).toBeNull()
    expect(within(left).queryByRole('button', { name: 'Forward' })).toBeNull()
  })
})
