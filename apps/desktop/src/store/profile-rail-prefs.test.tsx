// @vitest-environment jsdom
import { act, cleanup, render, renderHook } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

// Switching profiles has exactly one door at a time: the rail at the sidebar
// foot, or the statusbar picker beside the gateway switcher when the user hides
// the rail. The Webapp (browser-hosted Desktop) always draws the rail as its
// dropdown, never the squares; native Desktop keeps the squares until the
// dropdown threshold. The host is fixed per page load, so each case loads a
// fresh module graph after declaring its host.

const noop = () => {}

const noopAsync = async () => {}

afterEach(() => {
  cleanup()
  delete (window as Window & { __HERMES_UI_SURFACE__?: string }).__HERMES_UI_SURFACE__
  localStorage.clear()
  vi.resetModules()
})

async function loadHost(webapp: boolean) {
  if (webapp) {
    ;(window as Window & { __HERMES_UI_SURFACE__?: string }).__HERMES_UI_SURFACE__ = 'webapp'
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

  return { doors, prefs }
}

it.each([
  { host: 'Webapp', webapp: true, dropdown: true },
  { host: 'native Desktop', webapp: false, dropdown: false }
])('$host keeps exactly one profile door, the Webapp as a dropdown', async ({ dropdown, webapp }) => {
  const { doors, prefs } = await loadHost(webapp)

  // The rail stays at the sidebar foot whether or not the statusbar is shown.
  expect(doors()).toEqual({ dropdown, picker: false, rail: true })

  // Hiding the rail hands the door to the statusbar picker on either host.
  act(() => prefs.toggleProfileRailVisible())
  expect(doors()).toEqual({ dropdown: false, picker: true, rail: false })
})
