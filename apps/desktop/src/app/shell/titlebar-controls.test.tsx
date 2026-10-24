// @vitest-environment jsdom
import { act, cleanup, render, screen, within } from '@testing-library/react'
import { useEffect } from 'react'
import { MemoryRouter, useNavigate } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { I18nProvider } from '@/i18n'
import { setTitlebarAppActionsSide } from '@/store/titlebar-app-actions'

import { ROUTES_AREA } from '../routes'

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

  it('hides the app clusters on a contributed page — it is an overlay now', () => {
    // Contributed pages float over the workspace like Settings: the card owns
    // the window, so there is no band for the app clusters even when the page
    // mounts no chrome of its own.
    renderControls('/plain')

    expect(windowControls()).toBeNull()
    expect(appControls()).toBeNull()
  })

  it('hides the app clusters on an overlay', () => {
    renderControls('/settings')

    expect(windowControls()).toBeNull()
    expect(appControls()).toBeNull()
  })

  it('a titleBar.tools item does not claim the band on a contributed overlay', () => {
    renderControls('/plain', { leftTools: [PLUGIN_TOOL] })

    expect(windowControls()).toBeNull()
    expect(pluginTool()).toBeNull()
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

    it('hides plugin titlebar contributions on a contributed route — an overlay owns the window', () => {
      renderControls('/kanban')

      expect(pluginChrome()).toBeNull()
      expect(windowControls()).toBeNull()
      expect(appControls()).toBeNull()
    })

    it('hides contributed titlebar tools on a contributed route', () => {
      renderControls('/kanban', { leftTools: [PLUGIN_TOOL] })

      expect(pluginTool()).toBeNull()
    })

    it('hides plugin titlebar contributions on an overlay', () => {
      renderControls('/settings')

      expect(pluginChrome()).toBeNull()
    })

    it('renders no titlebar clusters on a contributed route — it is an overlay now', () => {
      // The old hooks existed to position the sessions tab strip around a
      // page-mounted switcher; an overlay card has no band to measure.
      const { container } = renderControls('/kanban')

      expect(container.querySelector('[data-titlebar-cluster="left"]')).toBeNull()
      expect(container.querySelector('[data-titlebar-cluster="right"]')).toBeNull()
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
  afterEach(() => {
    setTitlebarAppActionsSide('right')
    cleanup()
  })

  it('moves settings, layout, and HUD to the left when the appearance setting says left', () => {
    setTitlebarAppActionsSide('left')
    renderControls('/')

    const left = screen.getByLabelText('Window controls')
    const right = screen.getByLabelText('App controls')

    expect(within(left).getByLabelText('Open settings')).toBeTruthy()
    expect(within(left).getByLabelText('Layout editor')).toBeTruthy()
    expect(within(left).getByLabelText('HUD mode')).toBeTruthy()
    expect(within(right).queryByLabelText('Open settings')).toBeNull()
  })
})
