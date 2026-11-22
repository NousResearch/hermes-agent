import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, useParams } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { $cronJobs } from '@/store/cron'
import { bindDesktopMetrics, resetDesktopMetricsForTests } from '@/store/desktop-metrics'
import { $discoveryLabUsageStatsConsent, discoveryLabSignUpOn, setDiscoveryLabSignUp } from '@/store/discovery-lab'
import { $firstTimeSplash } from '@/store/first-time-splash'
import { $guidanceLedger } from '@/store/guidance-ledger'
import { $heldPrompt, takeHeldPrompt } from '@/store/held-prompt'
import { $newVersionReady } from '@/store/new-version'
import { $trayCleared, $traySeen } from '@/store/notification-tray'
import { $renameSplash } from '@/store/rename-splash'
import { $sessions, $sessionsLoadError, $sessionsLoading } from '@/store/session'
import { $sharedMetricsConsent, $sharedMetricsDetailsOpen } from '@/store/shared-metrics'
import { $shellChoice } from '@/store/shell-variant'
import { $signInFailed, $signUpRequired, $visitorAccess, installVisitorAccess } from '@/store/visitor-access'
import { ThemeProvider } from '@/themes/context'
import type { SessionInfo } from '@/types/hermes'

import { ChatDiscovery } from '../discovery/chat-discovery'
import { discoveryCopy } from '../discovery/discovery-copy'
import { firstTimeCopy } from '../discovery/first-time-copy'
import { NotificationBell } from '../discovery/notification-tray'
import { SplashCoveredRoutes } from '../discovery/splash-layer'
import { LandingGreeting, StarterIdeas } from '../discovery/starter-ideas'
import { usageStatsCopy } from '../discovery/usage-stats-copy'

import { reloadAt } from './ui-driver'

import { DiscoveryLabView } from '.'

// jsdom cannot load a page: the lab's reload is where its part ends.
vi.mock('./ui-driver', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  reloadAt: vi.fn()
}))

const copy = firstTimeCopy('en')
const discovery = discoveryCopy('en')
const chat = { id: 'chat-1', title: 'Quarterly plan', last_active: 1_700_000_000 } as SessionInfo

function StoredChat() {
  return <ChatDiscovery primary routedSessionId={useParams().sessionId ?? null} />
}

/** The redesigned shell's pieces the lab drives: the rail's bell, a new chat's landing, a stored
 *  chat, and the lab page itself, under the splash layer, on a machine someone has used. */
function renderShell(at = '/discovery-lab') {
  render(
    <ThemeProvider>
      <MemoryRouter initialEntries={[at]}>
        <nav data-slot="global-nav">
          <NotificationBell />
        </nav>
        <main>
          <SplashCoveredRoutes
            chat={
              <>
                <Route
                  element={
                    <>
                      <textarea aria-label="Message" />
                      <LandingGreeting target="main" />
                      <StarterIdeas target="main" />
                    </>
                  }
                  index
                />
                <Route element={<StoredChat />} path=":sessionId" />
              </>
            }
            pages={<Route element={<DiscoveryLabView />} path="discovery-lab" />}
          />
        </main>
      </MemoryRouter>
    </ThemeProvider>
  )

  act(() => {
    $sessions.set([chat])
    $sessionsLoading.set(false)
  })
}

function show(title: string) {
  fireEvent.click(screen.getByRole('button', { name: `Show ${title}` }))
}

const region = (name: string) => screen.queryByRole('region', { name })

beforeEach(() => {
  window.localStorage.clear()
  window.sessionStorage.clear()
  resetDesktopMetricsForTests()
  $apiRequestScope.set({ connectionId: 'gateway', profile: 'default' })
  bindDesktopMetrics(null, 'gateway|default')
  $shellChoice.set(true)
  // Someone who already saw the rename splash, so the lab page is not covered when it opens.
  $guidanceLedger.set({ renameSplash: 'seen', firstTimeSplash: 'not_applicable' })
  $traySeen.set([])
  $trayCleared.set([])
  $renameSplash.set(null)
  $firstTimeSplash.set(null)
  $newVersionReady.set(false)
  $visitorAccess.set({ kind: 'open' })
  $signInFailed.set(false)
  $sharedMetricsConsent.set({ decided: true, enabled: false, send: false })
  $sharedMetricsDetailsOpen.set(false)
  $discoveryLabUsageStatsConsent.set(null)
  takeHeldPrompt()
  $cronJobs.set([])
  $sessions.set([])
  $sessionsLoadError.set(false)
  $sessionsLoading.set(true)
  // jsdom lays nothing out; the lab presses only controls that are on screen.
  vi.spyOn(Element.prototype, 'getClientRects').mockReturnValue([new DOMRect(0, 0, 10, 10)] as unknown as DOMRectList)
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.mocked(reloadAt).mockClear()
  window.history.replaceState(null, '', '/')
  resetDesktopMetricsForTests()
})

describe('the discovery lab', () => {
  it('lists every state with its own button, and reads out the marks', () => {
    renderShell()

    expect(screen.getByRole('heading', { name: 'Discovery lab' })).toBeTruthy()
    expect(within(screen.getByLabelText('Guidance marks')).getByText('seen')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Show Welcome to Hermes' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Show Hermes got a new look' })).toBeTruthy()
  })

  it('opens the welcome splash for a brand-new person, though this machine has chats', () => {
    renderShell()

    fireEvent.click(screen.getAllByRole('button', { name: 'Reset' })[0])

    expect(region(copy.splash.title)).not.toBeNull()
    expect($guidanceLedger.get()).toEqual({ renameSplash: 'not_applicable' })
  })

  it('takes the welcome splash to its second step', async () => {
    renderShell()

    show('Make it yours')

    await waitFor(() => expect(region(copy.splash.theme.title)).not.toBeNull())
  })

  it('opens the rename splash for a returning user', () => {
    renderShell()

    fireEvent.click(screen.getAllByRole('button', { name: 'Reset' })[1])

    expect(region(discovery.renameSplash.title)).not.toBeNull()
    expect($firstTimeSplash.get()).toBeNull()
  })

  it('lands a newcomer on the new chat with the greeting and the ideas, then the check-in when due', () => {
    renderShell()

    show('Newcomer landing')
    expect(screen.getByRole('heading', { name: copy.landing.title })).toBeTruthy()
    expect(screen.getByRole('list', { name: copy.landing.ideasLabel })).toBeTruthy()

    cleanup()
    renderShell()
    show("How's Hermes going so far?")
    expect(screen.getByRole('heading', { name: copy.checkIn.title })).toBeTruthy()
  })

  it('opens the tray with the usage-stats question put away, never asking the backend', async () => {
    renderShell()

    show('Usage stats: put away to the tray')

    const tray = await screen.findByRole('dialog', { name: discovery.tray.title })
    expect(within(tray).getByText(usageStatsCopy('en').tray.title)).toBeTruthy()
    expect($sharedMetricsConsent.get()?.decided).toBe(true)
  })

  it('opens the usage-stats details from its stand-in', () => {
    renderShell()

    show('Usage stats: details dialog')

    expect($sharedMetricsDetailsOpen.get()).toBe(true)
    expect($discoveryLabUsageStatsConsent.get()?.decided).toBe(false)
  })

  it('says a newer build over the newest stored chat', () => {
    renderShell()

    show('A newer build is waiting')

    expect(region(discovery.newVersion.title)).not.toBeNull()
  })

  it('makes sign up required with a message held, as the next load will read it', () => {
    renderShell()

    show('A message held until sign up')

    expect(discoveryLabSignUpOn()).toBe(true)
    // At once on this page too, so the held message cannot run before the load.
    expect($signUpRequired.get()).toBe(true)
    expect($heldPrompt.get()).not.toBeNull()
    expect(new URL(vi.mocked(reloadAt).mock.calls[0][0]).hash).toBe('#/')
  })

  it('brings sign up back without signing in to the recovery screen on the next load', () => {
    setDiscoveryLabSignUp(true)
    window.history.replaceState(null, '', '/?hermes_sign_in=failed#/')

    installVisitorAccess()
    renderShell('/')

    expect(region(copy.recovery.title)).not.toBeNull()
  })

  it('shows its stand-in sign-up page only while the stand-in is on', () => {
    renderShell('/discovery-lab?stand-in=sign-up')
    expect(screen.queryByRole('button', { name: 'Finish sign up' })).toBeNull()
    cleanup()

    setDiscoveryLabSignUp(true)
    renderShell('/discovery-lab?stand-in=sign-up')
    expect(screen.getByRole('button', { name: 'Finish sign up' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Come back' })).toBeTruthy()
  })

  it('cannot open the developer-view offer outside Simple, and says why', () => {
    renderShell()

    const button = screen.getByRole('button', { name: 'Show Want the developer view?' })
    expect(button.hasAttribute('disabled')).toBe(true)
    expect(screen.getByText(/Shown only in Simple/)).toBeTruthy()
  })
})
