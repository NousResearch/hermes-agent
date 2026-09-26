// @vitest-environment jsdom
import { act, cleanup, render, renderHook } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

// Switching profiles has exactly one door at a time: the colored rail at the
// sidebar foot, or the statusbar picker beside the gateway switcher. The
// Webapp (browser-hosted Desktop) always takes the picker; native Desktop
// keeps the rail as the user's preference. The host is fixed per page load,
// so each case loads a fresh module graph after declaring its host.

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
    picker: statusbar.result.current.leftStatusbarItems.some(item => item.id === 'profile-switcher' && !item.hidden),
    rail: sidebar.container.querySelector('[data-slot="profile-rail"]') !== null
  })

  return { doors, prefs }
}

it.each([
  { host: 'Webapp', webapp: true, rail: false },
  { host: 'native Desktop', webapp: false, rail: true }
])('$host shows exactly one profile door, the rail only where it is a preference', async ({ rail, webapp }) => {
  const { doors, prefs } = await loadHost(webapp)

  expect(doors()).toEqual({ picker: !rail, rail })

  // The user's rail toggle flips native Desktop onto the picker; it can never
  // put the rail into the Webapp.
  act(() => prefs.toggleProfileRailVisible())
  expect(doors()).toEqual({ picker: true, rail: false })
})
