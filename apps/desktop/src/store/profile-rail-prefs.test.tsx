// @vitest-environment jsdom
import { act, cleanup, render, renderHook } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

// Switching profiles has exactly one door at a time: the rail at the sidebar
// foot, or the statusbar picker beside the gateway switcher when the user hides
// the rail. The Webapp (browser-hosted Desktop) always draws the rail as its
// dropdown, never the squares; native Desktop keeps the squares until the
// dropdown threshold. The host is fixed per page load, so each case loads a
// fresh module graph after declaring its host — in a hook, whose timeout
// absorbs the cold sidebar/statusbar import.

type SurfaceWindow = Window & { __HERMES_UI_SURFACE__?: string }

const noop = () => {}

const noopAsync = async () => {}

afterEach(() => {
  cleanup()
  localStorage.clear()
})

async function importHost(webapp: boolean) {
  if (webapp) {
    ;(window as SurfaceWindow).__HERMES_UI_SURFACE__ = 'webapp'
  }

  vi.resetModules()

  const [{ ChatSidebar }, { SidebarProvider }, { useStatusbarItems }, { group }, { $layoutTree }, prefs] =
    await Promise.all([
      import('@/app/chat/sidebar'),
      import('@/components/ui/sidebar'),
      import('@/app/shell/hooks/use-statusbar-items'),
      import('@/components/pane-shell/tree/model'),
      import('@/components/pane-shell/tree/store'),
      import('@/store/profile-rail-prefs')
    ])

  return { $layoutTree, ChatSidebar, group, prefs, SidebarProvider, useStatusbarItems }
}

type HostModules = Awaited<ReturnType<typeof importHost>>

function renderDoors({ $layoutTree, ChatSidebar, group, SidebarProvider, useStatusbarItems }: HostModules) {
  // The sessions pane is on screen: both doors live with it.
  $layoutTree.set(group(['sessions'], { active: 'sessions', id: 'sessions-group' }))

  const sidebar = render(
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
    </MemoryRouter>
  )

  const statusbar = renderHook(
    () =>
      useStatusbarItems({
        agentsOpen: false,
        chatOpen: true,
        commandCenterOpen: false,
        extraLeftItems: [],
        extraRightItems: [],
        freshDraftReady: false,
        gatewayState: 'ready',
        inferenceStatus: null,
        openAgents: noop,
        openCommandCenterSection: noop,
        requestGateway: async () => undefined as never,
        statusSnapshot: null,
        toggleCommandCenter: noop
      }),
    { wrapper: MemoryRouter }
  )

  const doors = () => ({
    dropdown: sidebar.container.querySelector('[data-slot="profile-dropdown"]') !== null,
    picker: statusbar.result.current.leftStatusbarItems.some(item => item.id === 'profile-switcher' && !item.hidden),
    rail: sidebar.container.querySelector('[data-slot="profile-rail"]') !== null
  })

  return doors
}

describe.each([
  { host: 'Webapp', webapp: true, dropdown: true },
  { host: 'native Desktop', webapp: false, dropdown: false }
])('$host', ({ dropdown, webapp }) => {
  let host: HostModules

  beforeAll(async () => {
    host = await importHost(webapp)
  })

  afterAll(() => {
    delete (window as SurfaceWindow).__HERMES_UI_SURFACE__
    vi.resetModules()
  })

  it('keeps exactly one profile door, the Webapp as a dropdown', () => {
    const doors = renderDoors(host)

    // The rail stays at the sidebar foot whether or not the statusbar is shown.
    expect(doors()).toEqual({ dropdown, picker: false, rail: true })

    // Hiding the rail hands the door to the statusbar picker on either host.
    act(() => host.prefs.toggleProfileRailVisible())
    expect(doors()).toEqual({ dropdown: false, picker: true, rail: false })
  })
})
