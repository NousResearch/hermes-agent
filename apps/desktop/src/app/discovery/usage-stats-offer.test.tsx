import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { ComposerStatusStack } from '@/app/chat/composer/status-stack'
import { CreateAccountCard } from '@/app/chat/sidebar/portal/create-account-card'
import { UsageStatsCard } from '@/app/chat/sidebar/portal/usage-stats-card'
import { SharedMetricsConsentDialog } from '@/components/shared-metrics/consent-dialog'
import { en } from '@/i18n/en'
import { bindDesktopMetrics, resetDesktopMetricsForTests } from '@/store/desktop-metrics'
import { immediateMetricsLocks } from '@/store/desktop-metrics-test-locks'
import { $firstTimeSplash } from '@/store/first-time-splash'
import type * as GatewayStore from '@/store/gateway'
import { $guidanceLedger, GUIDANCE_LEDGER_KEY } from '@/store/guidance-ledger'
import { $traySeen } from '@/store/notification-tray'
import { $desktopOnboarding } from '@/store/onboarding'
import { markOnboardingDecided } from '@/store/onboarding-presence'
import { $activeGatewayProfile } from '@/store/profile'
import { $renameSplash } from '@/store/rename-splash'
import { $sharedMetricsConsent, $sharedMetricsDetailsOpen, type SharedMetricsConsent } from '@/store/shared-metrics'
import { $shellChoice } from '@/store/shell-variant'
import { $usageStatsOfferPlace } from '@/store/usage-stats-offer'
import { $visitorAccess } from '@/store/visitor-access'

import { discoveryCopy } from './discovery-copy'
import { firstTimeCopy } from './first-time-copy'
import { NotificationBell } from './notification-tray'
import { usageStatsCopy } from './usage-stats-copy'

const requestGateway = vi.fn()

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

// The consent host and the sidebar card dial the focused (connection, profile) directly.
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<typeof GatewayStore>()),
  requestGatewayForAgent: (_connection: unknown, _profile: unknown, method: string, params?: Record<string, unknown>) =>
    requestGateway(method, params)
}))

// Onboarding's due check answered "not due": the consent host may read the answer.
markOnboardingDecided()

const copy = usageStatsCopy('en')
const tray = discoveryCopy('en').tray
const initialOnboarding = $desktopOnboarding.get()
const initialProfile = $activeGatewayProfile.get()
const initialScope = $apiRequestScope.get()
const UNDECIDED: SharedMetricsConsent = { enabled: false, send: false, decided: false }

/** The backend: `shared_metrics.status` reads the stored answer, `shared_metrics.set` writes it. */
function installBackend(initial: SharedMetricsConsent = UNDECIDED) {
  let stored = initial

  requestGateway.mockImplementation(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'shared_metrics.set') {
      const enabled = params?.enabled === true
      stored = { enabled, send: enabled && params?.send === true, decided: true }
    }

    return stored
  })
}

const writes = () => requestGateway.mock.calls.filter(([method]) => method === 'shared_metrics.set')

/** The redesign's pieces the offer moves between: the rail's bell, the chat sidebar's foot, the
 *  composer's status stack, and the app-level consent host that paints the dialog. */
function renderApp() {
  return render(
    <MemoryRouter>
      <nav>
        <NotificationBell />
      </nav>
      <div className="portal-sidebar" data-slot="sidebar">
        <div data-slot="new-session-split">
          <button type="button">New chat</button>
        </div>
        <CreateAccountCard />
        <UsageStatsCard />
      </div>
      <ComposerStatusStack queue={null} sessionId="chat" />
      <SharedMetricsConsentDialog enabled profile="havoc" />
    </MemoryRouter>
  )
}

const card = () => screen.queryByRole('region', { name: copy.card.title })
const strip = () => screen.queryByText(en.sharedMetrics.stripBody)

async function openTray() {
  fireEvent.click(screen.getByRole('button', { name: tray.title }))

  return within(await screen.findByRole('dialog', { name: tray.title }))
}

beforeEach(() => {
  window.localStorage.clear()
  // The metrics hook binds the focused owner; a first-run answer applies its gate to that owner.
  immediateMetricsLocks()
  $apiRequestScope.set({ connectionId: null, profile: 'havoc' })
  resetDesktopMetricsForTests()
  bindDesktopMetrics(null, '|havoc')
  $shellChoice.set(true)
  $guidanceLedger.set({})
  $traySeen.set([])
  $activeGatewayProfile.set('havoc')
  $desktopOnboarding.set({ ...initialOnboarding, configured: true })
  installBackend()
})

afterEach(() => {
  cleanup()
  resetDesktopMetricsForTests()
  $apiRequestScope.set(initialScope)
  requestGateway.mockReset()
  $shellChoice.set(null)
  $guidanceLedger.set({})
  $activeGatewayProfile.set(initialProfile)
  $desktopOnboarding.set(initialOnboarding)
  $sharedMetricsConsent.set(null)
  $sharedMetricsDetailsOpen.set(false)
  $firstTimeSplash.set(null)
  $renameSplash.set(null)
  $visitorAccess.set({ kind: 'open' })
  vi.unstubAllGlobals()
})

describe('usage-stats offer placement', () => {
  it('asks in the chat sidebar in the redesign, never in the composer', async () => {
    renderApp()

    await waitFor(() => expect(card()).not.toBeNull())
    expect(strip()).toBeNull()
  })

  it('asks in the composer in classic, never in the sidebar', async () => {
    $shellChoice.set(false)
    renderApp()

    await waitFor(() => expect(strip()).not.toBeNull())
    expect(card()).toBeNull()
  })

  it('asks nowhere until the backend says the question is open', async () => {
    installBackend({ enabled: false, send: false, decided: true })
    renderApp()

    await waitFor(() => expect($sharedMetricsConsent.get()?.decided).toBe(true))
    expect(card()).toBeNull()
    expect(strip()).toBeNull()
  })
})

describe('the sidebar’s foot while sign up is required', () => {
  const account = firstTimeCopy('en').account

  it('offers Create an Account first, and the usage-stats question only once sign up is behind them', async () => {
    $visitorAccess.set({ kind: 'sign_up_required', signUpUrl: 'https://portal.example/sign-up' })
    const assign = vi.fn()
    vi.stubGlobal('location', { href: 'https://hermes.example/#/', assign })
    const first = renderApp()

    const offer = await screen.findByRole('button', { name: name => name.startsWith(account.title) })
    await waitFor(() => expect($sharedMetricsConsent.get()?.decided).toBe(false))
    expect(card()).toBeNull()
    fireEvent.click(offer)
    expect(assign).toHaveBeenCalledTimes(1)
    first.unmount()

    $visitorAccess.set({ kind: 'open' })
    renderApp()
    await waitFor(() => expect(card()).not.toBeNull())
    expect(screen.queryByRole('button', { name: name => name.startsWith(account.title) })).toBeNull()
  })
})

describe('the card’s ×', () => {
  // A click in the navigation closes an open splash for good, and the card stands in the navigation:
  // answering it there would end the welcome the person was reading.
  it.each([
    ['the welcome splash', $firstTimeSplash],
    ['the rename splash', $renameSplash]
  ])('waits while %s is open', async (_name, $splash) => {
    $splash.set('shown')
    renderApp()
    await waitFor(() => expect($usageStatsOfferPlace.get()).toBe('card'))
    expect(card()).toBeNull()

    act(() => $splash.set(null))
    expect(card()).not.toBeNull()
  })

  it('moves the question to the tray unanswered, and that survives a reload', async () => {
    const first = renderApp()
    await waitFor(() => expect(card()).not.toBeNull())

    fireEvent.click(screen.getByRole('button', { name: copy.card.later }))

    expect(card()).toBeNull()
    expect(writes()).toEqual([])
    expect(JSON.parse(window.localStorage.getItem(GUIDANCE_LEDGER_KEY) ?? '{}')).toEqual({
      usageStatsOffer: 'deferred'
    })

    // A reload: the same stored ledger, a fresh read of the still-open question.
    first.unmount()
    $sharedMetricsConsent.set(null)
    renderApp()
    await waitFor(() => expect($usageStatsOfferPlace.get()).toBe('tray'))
    expect(card()).toBeNull()

    const items = await openTray()
    expect(items.getByText(copy.tray.title)).toBeTruthy()

    fireEvent.click(items.getByRole('button', { name: copy.tray.action }))

    expect(await screen.findByRole('dialog', { name: copy.dialog.title })).toBeTruthy()
    expect(writes()).toEqual([])
  })

  it('clearing the tray item answers nothing and offers it no more', async () => {
    renderApp()
    await waitFor(() => expect(card()).not.toBeNull())
    fireEvent.click(screen.getByRole('button', { name: copy.card.later }))

    const items = await openTray()
    fireEvent.click(items.getByRole('button', { name: copy.tray.clear }))

    await waitFor(() => expect(screen.queryByRole('dialog', { name: tray.title })).toBeNull())
    expect($usageStatsOfferPlace.get()).toBeNull()
    expect(writes()).toEqual([])
    expect(card()).toBeNull()
  })
})

describe('answering', () => {
  it.each([
    ['share', copy.card.share, { enabled: true, send: true }],
    ['off', copy.card.decline, { enabled: false, send: false }]
  ] as const)('the card’s %s answer is saved for the focused agent and retires the offer', async (_, label, flags) => {
    renderApp()
    await waitFor(() => expect(card()).not.toBeNull())

    fireEvent.click(screen.getByRole('button', { name: label }))

    await waitFor(() => expect(card()).toBeNull())
    expect(writes()).toEqual([['shared_metrics.set', { ...flags, first_run: true, profile: 'havoc' }]])
    expect($usageStatsOfferPlace.get()).toBeNull()
  })

  it('"Share stats" in the dialog opened from the tray saves the yes and empties the tray', async () => {
    $guidanceLedger.set({ usageStatsOffer: 'deferred' })
    renderApp()
    await waitFor(() => expect($usageStatsOfferPlace.get()).toBe('tray'))

    fireEvent.click((await openTray()).getByRole('button', { name: copy.tray.action }))
    const dialog = within(await screen.findByRole('dialog', { name: copy.dialog.title }))
    fireEvent.click(dialog.getByRole('button', { name: copy.dialog.share }))

    await waitFor(() => expect(screen.queryByRole('dialog', { name: copy.dialog.title })).toBeNull())
    expect(writes()).toEqual([['shared_metrics.set', { enabled: true, send: true, first_run: true, profile: 'havoc' }]])
    expect($usageStatsOfferPlace.get()).toBeNull()
    expect((await openTray()).queryByText(copy.tray.title)).toBeNull()
  })
})

describe('the dialog', () => {
  it.each([
    ['Not now', () => fireEvent.click(screen.getByRole('button', { name: copy.dialog.notNow }))],
    ['its ×', () => fireEvent.click(screen.getByRole('button', { name: en.common.close }))],
    ['Escape', () => fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })]
  ])('%s only closes it: no answer, the card stays, focus back on Details', async (_, close) => {
    renderApp()
    await waitFor(() => expect(card()).not.toBeNull())

    const details = screen.getByRole('button', { name: copy.card.details })
    details.focus()
    fireEvent.click(details)
    expect(await screen.findByRole('dialog', { name: copy.dialog.title })).toBeTruthy()
    // The short explainer, not classic's long one; neither answer is focused for the person.
    expect(screen.queryByText(en.sharedMetrics.whatIsCollected)).toBeNull()
    const dialog = screen.getByRole('dialog')
    expect(dialog.ownerDocument.activeElement).toBe(dialog)

    close()

    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    expect(writes()).toEqual([])
    expect($guidanceLedger.get()).toEqual({})
    expect(card()).not.toBeNull()
    // Focus goes back to Details, not to the page.
    await waitFor(() => expect(details.ownerDocument.activeElement).toBe(details))
  })

  it('says why it asks again when an earlier build saved an answer unseen', async () => {
    installBackend({ ...UNDECIDED, reask: true })
    renderApp()
    await waitFor(() => expect(card()).not.toBeNull())

    fireEvent.click(screen.getByRole('button', { name: copy.card.details }))

    expect(await screen.findByText(en.sharedMetrics.stripReaskBody)).toBeTruthy()
  })
})
