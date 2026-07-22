import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  billingDevFixtures,
  loggedOutBillingState,
  loggedOutSubscriptionState,
  okBilling,
  okSubscription,
  postTrainBillingState,
  postTrainSubscriptionState,
  todayBillingState,
  todaySubscriptionState
} from './fixtures.test-util'
import { formatUsageUpdatedAgo } from './use-billing-state'

import { BillingSettings } from './index'

const apiMocks = vi.hoisted(() => ({
  charge: vi.fn(),
  chargeStatus: vi.fn(),
  fetchBillingState: vi.fn(),
  fetchSubscriptionState: vi.fn(),
  openExternal: vi.fn(),
  previewSubscriptionChange: vi.fn(),
  resumeSubscription: vi.fn(),
  scheduleSubscriptionChange: vi.fn(),
  stepUp: vi.fn(),
  updateAutoReload: vi.fn()
}))

vi.mock('./api', () => ({
  // Pass-through provider — the mocked useBillingApi ignores any override anyway.
  BillingApiProvider: ({ children }: { children: ReactNode }) => children,
  useBillingApi: () => ({
    charge: apiMocks.charge,
    chargeStatus: apiMocks.chargeStatus,
    fetchBillingState: apiMocks.fetchBillingState,
    fetchSubscriptionState: apiMocks.fetchSubscriptionState,
    previewSubscriptionChange: apiMocks.previewSubscriptionChange,
    resumeSubscription: apiMocks.resumeSubscription,
    scheduleSubscriptionChange: apiMocks.scheduleSubscriptionChange,
    stepUp: apiMocks.stepUp,
    updateAutoReload: apiMocks.updateAutoReload
  })
}))

function renderBilling(initialEntries: string[] = ['/settings?tab=billing']) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  render(
    <MemoryRouter initialEntries={initialEntries}>
      <QueryClientProvider client={client}>
        <BillingSettings />
      </QueryClientProvider>
    </MemoryRouter>
  )

  return client
}

beforeEach(() => {
  apiMocks.fetchBillingState.mockResolvedValue(okBilling(todayBillingState))
  apiMocks.fetchSubscriptionState.mockResolvedValue(okSubscription(todaySubscriptionState))
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      openExternal: apiMocks.openExternal
    }
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('BillingSettings', () => {
  it('renders the deployed-today payload with buy controls hidden and usage rows visible', async () => {
    renderBilling()

    expect(await screen.findByText('$996.47')).toBeTruthy()
    expect(screen.getByText('Ultra · $200/mo')).toBeTruthy()
    expect(screen.getByText('Visa •••• 3206')).toBeTruthy()
    expect(
      screen.getByText(
        "Remote spending is off for this account — a billing admin can turn it on from the portal's Hermes Agent page."
      )
    ).toBeTruthy()
    expect(screen.queryByRole('button', { name: '$100' })).toBeNull()
    expect(screen.getByText('Charges $10 automatically when your balance falls below $5.')).toBeTruthy()
    expect(screen.getByText('$120 of $220 left')).toBeTruthy()
    expect(screen.getByText('$876.47')).toBeTruthy()
    expect(screen.getByText('$10 of $100 used').classList.contains('tabular-nums')).toBe(true)
    expect(screen.getByText('Default ceiling')).toBeTruthy()
  })

  it('renders the post-train payload with enabled buy controls and card provenance', async () => {
    apiMocks.fetchBillingState.mockResolvedValue(okBilling(postTrainBillingState))
    apiMocks.fetchSubscriptionState.mockResolvedValue(okSubscription(postTrainSubscriptionState))

    renderBilling()

    expect(await screen.findByText('$142.50')).toBeTruthy()
    expect(screen.getByText('Visa •••• 4242 - subscription card')).toBeTruthy()
    expect(screen.getByRole('button', { name: '$25' }).hasAttribute('disabled')).toBe(false)
    expect(screen.getByRole('button', { name: '$50' }).hasAttribute('disabled')).toBe(false)
    expect(screen.getByRole('button', { name: '$100' }).hasAttribute('disabled')).toBe(false)
    expect(screen.getByRole('spinbutton', { name: 'Custom credit amount' })).toBeTruthy()
    expect(screen.getByRole('button', { name: /^Buy$/ }).hasAttribute('disabled')).toBe(false)
  })

  it('disables buy controls when no card is on file', async () => {
    const fixture = billingDevFixtures['no-card']

    apiMocks.fetchBillingState.mockResolvedValue(fixture.billing)
    apiMocks.fetchSubscriptionState.mockResolvedValue(fixture.subscription)

    renderBilling()

    expect(await screen.findByText('No card on file')).toBeTruthy()
    expect(screen.getByRole('button', { name: '$25' }).hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('button', { name: '$50' }).hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('button', { name: '$100' }).hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('spinbutton', { name: 'Custom credit amount' }).hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('button', { name: /^Buy$/ }).hasAttribute('disabled')).toBe(true)

    fireEvent.click(screen.getByRole('button', { name: /^Buy$/ }))

    expect(apiMocks.charge).not.toHaveBeenCalled()
  })

  it('saves enabled auto-refill edits and refreshes billing state', async () => {
    const client = renderBilling()
    const invalidate = vi.spyOn(client, 'invalidateQueries')

    apiMocks.updateAutoReload.mockResolvedValue({ data: { ok: true }, ok: true })

    fireEvent.click(await screen.findByRole('button', { name: 'Manage' }))
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Auto-refill threshold' }), {
      target: { value: '15' }
    })
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Auto-refill reload-to amount' }), {
      target: { value: '20' }
    })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() =>
      expect(apiMocks.updateAutoReload).toHaveBeenCalledWith({
        enabled: true,
        reload_to_usd: '20',
        threshold_usd: '15'
      })
    )
    await waitFor(() => expect(invalidate).toHaveBeenCalledWith({ queryKey: ['billing', 'state'] }))
    expect(await screen.findByText('Auto-refill updated.')).toBeTruthy()
  })

  it('rejects auto-refill amounts outside the billing bounds', async () => {
    renderBilling()

    fireEvent.click(await screen.findByRole('button', { name: 'Manage' }))
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Auto-refill threshold' }), {
      target: { value: '7.50' }
    })

    expect(screen.getByText('Threshold: minimum is $10.')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Save' }).hasAttribute('disabled')).toBe(true)

    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    expect(apiMocks.updateAutoReload).not.toHaveBeenCalled()
  })

  it('renders the enabled auto-refill row without crashing when the card is null', async () => {
    apiMocks.fetchBillingState.mockResolvedValue(
      okBilling({ ...todayBillingState, auto_reload: { ...todayBillingState.auto_reload, card: null } })
    )

    renderBilling()

    expect(await screen.findByText('Charges $10 automatically when your balance falls below $5.')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Manage' })).toBeTruthy()
  })

  it('requires inline confirmation before disabling auto-refill', async () => {
    renderBilling()

    apiMocks.updateAutoReload.mockResolvedValue({ data: { ok: true }, ok: true })

    fireEvent.click(await screen.findByRole('button', { name: 'Manage' }))
    fireEvent.click(screen.getByRole('button', { name: 'Disable' }))

    expect(screen.getByText('Turn off auto-refill?')).toBeTruthy()
    expect(apiMocks.updateAutoReload).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Turn off' }))

    // The gateway requires threshold + top_up_amount even to disable, so the current
    // amounts ride along (todayBillingState: threshold $5, reload-to $10).
    await waitFor(() =>
      expect(apiMocks.updateAutoReload).toHaveBeenCalledWith({
        enabled: false,
        reload_to_usd: '10',
        threshold_usd: '5'
      })
    )
  })

  it('opens auto-refill edit without a validation error even when the saved config is below the minimum', async () => {
    // todayBillingState: threshold $5 with min_usd $10 — invalid, but opening
    // Manage must stay silent until the user edits or attempts to save (spec §9).
    renderBilling()

    fireEvent.click(await screen.findByRole('button', { name: 'Manage' }))

    expect(screen.getByRole('spinbutton', { name: 'Auto-refill threshold' })).toBeTruthy()
    expect(screen.queryByText('Threshold: minimum is $10.')).toBeNull()
    // Save is disabled because the prefilled config is invalid — but no error yet.
    expect(screen.getByRole('button', { name: 'Save' }).hasAttribute('disabled')).toBe(true)
  })

  it('navigates to the in-app plans grid from the plan card and back', async () => {
    const fixture = billingDevFixtures['free-personal']

    apiMocks.fetchBillingState.mockResolvedValue(fixture.billing)
    apiMocks.fetchSubscriptionState.mockResolvedValue(fixture.subscription)

    renderBilling()

    fireEvent.click(await screen.findByRole('button', { name: 'View plans' }))

    expect(await screen.findByText('Plans')).toBeTruthy()
    // No subscription → the free tier is the inert current plan, the three paid
    // tiers are "Choose ↗" upgrades (no "subscribe to Free").
    expect(screen.getByText('Current plan')).toBeTruthy()
    expect(screen.getAllByRole('button', { name: /Choose/ }).length).toBe(3)
    expect(screen.queryByRole('button', { name: 'Downgrade' })).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Back to billing' }))

    expect(await screen.findByRole('button', { name: 'View plans' })).toBeTruthy()
  })

  it('renders the current marker and an actionable downgrade when deep-linked to the plans grid', async () => {
    const fixture = billingDevFixtures['subscriber-personal']

    apiMocks.fetchBillingState.mockResolvedValue(fixture.billing)
    apiMocks.fetchSubscriptionState.mockResolvedValue(fixture.subscription)

    renderBilling(['/settings?tab=billing&bview=plans'])

    expect(await screen.findByText('Current plan')).toBeTruthy()
    // Free sits below Plus → an in-app (enabled) "Downgrade" button, not disabled.
    expect(screen.getByRole('button', { name: 'Downgrade' }).hasAttribute('disabled')).toBe(false)
    // Super + Ultra are upgrades.
    expect(screen.getAllByRole('button', { name: /Choose/ }).length).toBe(2)
  })

  it('falls back to overview (no live Choose grid) when a team deep-links bview=plans', async () => {
    // Default beforeEach uses todaySubscriptionState (context: 'team') — no in-app
    // plans capability, so the URL must not surface a grid of Choose buttons.
    renderBilling(['/settings?tab=billing&bview=plans'])

    expect(await screen.findByText('Payment')).toBeTruthy()
    expect(screen.queryByText('Plans')).toBeNull()
    expect(screen.queryByRole('button', { name: /Choose/ })).toBeNull()
  })

  it('falls back to overview when a non-changer personal account deep-links bview=plans', async () => {
    apiMocks.fetchSubscriptionState.mockResolvedValue(
      okSubscription({ ...todaySubscriptionState, can_change_plan: false, context: 'personal' })
    )

    renderBilling(['/settings?tab=billing&bview=plans'])

    expect(await screen.findByText('Payment')).toBeTruthy()
    expect(screen.queryByText('Plans')).toBeNull()
    expect(screen.queryByRole('button', { name: /Choose/ })).toBeNull()
  })

  it('runs an in-app downgrade: preview → confirm → schedule with the tier id → refetch → overview', async () => {
    const fixture = billingDevFixtures['subscriber-personal']

    apiMocks.fetchBillingState.mockResolvedValue(fixture.billing)
    apiMocks.fetchSubscriptionState.mockResolvedValue(fixture.subscription)
    apiMocks.previewSubscriptionChange.mockResolvedValue({
      data: {
        effect: 'scheduled',
        effective_at: '2026-08-15T00:00:00Z',
        monthly_credits_delta: '-88',
        ok: true,
        target_tier_name: 'Free'
      },
      ok: true
    })
    apiMocks.scheduleSubscriptionChange.mockResolvedValue({ data: { ok: true }, ok: true })

    const client = renderBilling(['/settings?tab=billing&bview=plans'])
    const invalidate = vi.spyOn(client, 'invalidateQueries')

    fireEvent.click(await screen.findByRole('button', { name: 'Downgrade' }))

    await waitFor(() =>
      expect(apiMocks.previewSubscriptionChange).toHaveBeenCalledWith('cltier000free0000personal')
    )
    expect(await screen.findByText(/No charge now/)).toBeTruthy()
    // Credits delta renders as signed dollars, not the raw wire string "-88".
    expect(screen.getByText(/Monthly credits change: −\$88\/mo\./)).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Confirm downgrade' }))

    await waitFor(() =>
      expect(apiMocks.scheduleSubscriptionChange).toHaveBeenCalledWith('cltier000free0000personal')
    )
    await waitFor(() => expect(invalidate).toHaveBeenCalledWith({ queryKey: ['billing', 'subscription'] }))
    // Scheduled → back on the overview.
    expect(await screen.findByText('Payment')).toBeTruthy()
    expect(screen.queryByText('Plans')).toBeNull()
  })

  it('surfaces the step-up affordance when scheduling a downgrade needs approval', async () => {
    const fixture = billingDevFixtures['subscriber-personal']

    apiMocks.fetchBillingState.mockResolvedValue(fixture.billing)
    apiMocks.fetchSubscriptionState.mockResolvedValue(fixture.subscription)
    apiMocks.previewSubscriptionChange.mockResolvedValue({
      data: { effect: 'scheduled', effective_at: '2026-08-15T00:00:00Z', ok: true, target_tier_name: 'Free' },
      ok: true
    })
    apiMocks.scheduleSubscriptionChange.mockResolvedValue({
      ok: false,
      refusal: { kind: 'insufficient_scope', message: 'billing:manage required' }
    })

    renderBilling(['/settings?tab=billing&bview=plans'])

    fireEvent.click(await screen.findByRole('button', { name: 'Downgrade' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Confirm downgrade' }))

    expect(await screen.findByText('Remote Spending needs approval:')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Verify to continue' })).toBeTruthy()
    // The failed schedule offers a retry in place.
    expect(screen.getByRole('button', { name: 'Try again' })).toBeTruthy()
    expect(apiMocks.scheduleSubscriptionChange).toHaveBeenCalledTimes(1)
  })

  it('undoes a scheduled downgrade from the plan card via resume', async () => {
    const fixture = billingDevFixtures['pending-downgrade']

    apiMocks.fetchBillingState.mockResolvedValue(fixture.billing)
    apiMocks.fetchSubscriptionState.mockResolvedValue(fixture.subscription)
    apiMocks.resumeSubscription.mockResolvedValue({ data: { ok: true }, ok: true })

    const client = renderBilling()
    const invalidate = vi.spyOn(client, 'invalidateQueries')

    expect(await screen.findByText('Changes to Free on Aug 15.')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Undo' }))

    await waitFor(() => expect(apiMocks.resumeSubscription).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(invalidate).toHaveBeenCalledWith({ queryKey: ['billing', 'subscription'] }))
  })

  it('undoes a scheduled cancellation from the plan card via resume', async () => {
    const fixture = billingDevFixtures['pending-cancellation']

    apiMocks.fetchBillingState.mockResolvedValue(fixture.billing)
    apiMocks.fetchSubscriptionState.mockResolvedValue(fixture.subscription)
    apiMocks.resumeSubscription.mockResolvedValue({ data: { ok: true }, ok: true })

    renderBilling()

    expect(await screen.findByText('Cancels on Aug 15.')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Undo' }))

    await waitFor(() => expect(apiMocks.resumeSubscription).toHaveBeenCalledTimes(1))
  })

  it('locks out the other downgrade tiles and Back while a schedule is in flight', async () => {
    // Current = Ultra so Free/Plus/Super are all downgrades (three tiles).
    apiMocks.fetchBillingState.mockResolvedValue(billingDevFixtures['subscriber-personal'].billing)
    apiMocks.fetchSubscriptionState.mockResolvedValue(
      okSubscription({
        ...todaySubscriptionState,
        can_change_plan: true,
        context: 'personal',
        current: { ...todaySubscriptionState.current, tier_id: 't_ultra', tier_name: 'Ultra' },
        tiers: [
          { dollars_per_month_display: '$0', is_current: false, is_enabled: true, monthly_credits: '0.1', name: 'Free', tier_id: 't_free', tier_order: 0 },
          { dollars_per_month_display: '$20', is_current: false, is_enabled: true, monthly_credits: '22', name: 'Plus', tier_id: 't_plus', tier_order: 1 },
          { dollars_per_month_display: '$100', is_current: false, is_enabled: true, monthly_credits: '110', name: 'Super', tier_id: 't_super', tier_order: 2 },
          { dollars_per_month_display: '$200', is_current: true, is_enabled: true, monthly_credits: '220', name: 'Ultra', tier_id: 't_ultra', tier_order: 3 }
        ]
      })
    )
    apiMocks.previewSubscriptionChange.mockResolvedValue({
      data: { effect: 'scheduled', effective_at: '2026-08-15T00:00:00Z', ok: true, target_tier_name: 'Free' },
      ok: true
    })

    let settleSchedule: (value: unknown) => void = () => {}
    apiMocks.scheduleSubscriptionChange.mockReturnValue(
      new Promise(resolve => {
        settleSchedule = resolve
      })
    )

    renderBilling(['/settings?tab=billing&bview=plans'])

    const downgrades = await screen.findAllByRole('button', { name: 'Downgrade' })
    expect(downgrades.length).toBe(3)

    fireEvent.click(downgrades[0])
    fireEvent.click(await screen.findByRole('button', { name: 'Confirm downgrade' }))

    // Scheduling in flight → the two remaining tiles + Back are disabled.
    await waitFor(() => {
      const remaining = screen.getAllByRole('button', { name: 'Downgrade' })
      expect(remaining.length).toBe(2)
      expect(remaining.every(btn => btn.hasAttribute('disabled'))).toBe(true)
    })
    expect(screen.getByRole('button', { name: 'Back to billing' }).hasAttribute('disabled')).toBe(true)

    settleSchedule({ data: { ok: true }, ok: true })
  })

  it('disables Undo while the resume is in flight', async () => {
    const fixture = billingDevFixtures['pending-downgrade']

    apiMocks.fetchBillingState.mockResolvedValue(fixture.billing)
    apiMocks.fetchSubscriptionState.mockResolvedValue(fixture.subscription)

    let settleResume: (value: unknown) => void = () => {}
    apiMocks.resumeSubscription.mockReturnValue(
      new Promise(resolve => {
        settleResume = resolve
      })
    )

    renderBilling()

    fireEvent.click(await screen.findByRole('button', { name: 'Undo' }))

    await waitFor(() => expect(screen.getByRole('button', { name: 'Undoing…' }).hasAttribute('disabled')).toBe(true))

    settleResume({ data: { ok: true }, ok: true })
  })

  it('moves focus into the confirm panel (role=status) when a downgrade opens', async () => {
    const fixture = billingDevFixtures['subscriber-personal']

    apiMocks.fetchBillingState.mockResolvedValue(fixture.billing)
    apiMocks.fetchSubscriptionState.mockResolvedValue(fixture.subscription)
    apiMocks.previewSubscriptionChange.mockResolvedValue({
      data: { effect: 'scheduled', effective_at: '2026-08-15T00:00:00Z', ok: true, target_tier_name: 'Free' },
      ok: true
    })

    renderBilling(['/settings?tab=billing&bview=plans'])

    fireEvent.click(await screen.findByRole('button', { name: 'Downgrade' }))

    const panel = await screen.findByRole('status')

    expect(panel.getAttribute('aria-live')).toBe('polite')
    expect(panel).toBe(panel.ownerDocument.activeElement)
  })

  it('keeps the auto-refill edit form mounted so the row height is reserved before editing', async () => {
    renderBilling()

    await screen.findByRole('button', { name: 'Manage' })

    // Not editing: the inputs are already in the DOM (height reserved) but aria-hidden,
    // so the accessible query finds nothing while the hidden-inclusive query does.
    expect(screen.queryByRole('spinbutton', { name: 'Auto-refill threshold' })).toBeNull()
    expect(screen.getByRole('spinbutton', { name: 'Auto-refill threshold', hidden: true })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Manage' }))

    // Editing reveals the same reserved input.
    expect(screen.getByRole('spinbutton', { name: 'Auto-refill threshold' })).toBeTruthy()
  })

  it('renders auto-refill mutation refusals and step-up affordance', async () => {
    renderBilling()

    apiMocks.updateAutoReload.mockResolvedValue({
      ok: false,
      refusal: {
        kind: 'insufficient_scope',
        message: 'billing:manage required'
      }
    })

    fireEvent.click(await screen.findByRole('button', { name: 'Manage' }))
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Auto-refill threshold' }), {
      target: { value: '15' }
    })
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Auto-refill reload-to amount' }), {
      target: { value: '20' }
    })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    expect(await screen.findByText('Remote Spending needs approval:')).toBeTruthy()
    expect(screen.getByText('This needs Remote Spending allowed. Start a top-up to allow it, then retry.')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Verify to continue' })).toBeTruthy()
  })

  it('keeps disabled auto-refill portal-only with no enable control', async () => {
    apiMocks.fetchBillingState.mockResolvedValue(okBilling(postTrainBillingState))
    apiMocks.fetchSubscriptionState.mockResolvedValue(okSubscription(postTrainSubscriptionState))

    renderBilling()

    expect((await screen.findAllByText('Off')).length).toBeGreaterThan(0)
    expect(screen.getByText('Turn on auto-refill from the portal')).toBeTruthy()
    expect(screen.queryByRole('button', { name: /enable/i })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Manage' })).toBeNull()
  })

  it('disables buy controls while polling and renders the settled outcome', async () => {
    let settleStatus: (value: unknown) => void = () => {}

    const statusPromise = new Promise(resolve => {
      settleStatus = resolve
    })

    apiMocks.fetchBillingState.mockResolvedValue(okBilling(postTrainBillingState))
    apiMocks.fetchSubscriptionState.mockResolvedValue(okSubscription(postTrainSubscriptionState))
    apiMocks.charge.mockResolvedValue({
      data: {
        charge_id: 'ch_123',
        ok: true
      },
      idempotencyKey: 'key-1',
      ok: true
    })
    apiMocks.chargeStatus.mockReturnValue(statusPromise)

    renderBilling()

    fireEvent.click(await screen.findByRole('button', { name: /^Buy$/ }))

    expect(await screen.findByText('Processing… checking settlement')).toBeTruthy()
    expect(screen.getByRole('button', { name: '$25' }).hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('button', { name: '$50' }).hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('spinbutton', { name: 'Custom credit amount' }).hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('button', { name: /^Buy$/ }).hasAttribute('disabled')).toBe(true)

    settleStatus({
      data: {
        amount_usd: '25',
        ok: true,
        status: 'settled'
      },
      ok: true
    })

    await waitFor(() => expect(screen.getByText('$25 added. Balance is refreshing.')).toBeTruthy())
  })

  it('renders logged-out as a connect card without normal account rows', async () => {
    apiMocks.fetchBillingState.mockResolvedValue(okBilling(loggedOutBillingState))
    apiMocks.fetchSubscriptionState.mockResolvedValue(okSubscription(loggedOutSubscriptionState))

    renderBilling()

    expect(await screen.findByText('Connect your Nous account')).toBeTruthy()
    expect(screen.getByText('Run /portal in the TUI or open the Nous portal to connect your account.')).toBeTruthy()
    expect(screen.queryByText('Payment method')).toBeNull()
    expect(screen.queryByText('Usage')).toBeNull()
  })

  it('renders danger value text for overdrawn subscription credits', async () => {
    const fixture = billingDevFixtures['empty-overdrawn']

    apiMocks.fetchBillingState.mockResolvedValue(fixture.billing)
    apiMocks.fetchSubscriptionState.mockResolvedValue(fixture.subscription)

    renderBilling()

    expect((await screen.findByText('$0 of $220 left · $0.79 over')).classList.contains('text-destructive')).toBe(true)
    const subscriptionTrack = screen.getByRole('progressbar', { name: 'Subscription credits remaining' })

    expect(subscriptionTrack.classList.contains('dither')).toBe(true)
    expect(subscriptionTrack.classList.contains('text-destructive/60')).toBe(true)
    expect(subscriptionTrack.classList.contains('bg-destructive/10')).toBe(true)
  })

  it('renders an empty neutral usage track when a row has no bar data', async () => {
    const fixture = billingDevFixtures['no-subscription']

    apiMocks.fetchBillingState.mockResolvedValue(
      okBilling({
        ...todayBillingState,
        monthly_cap: {
          ...todayBillingState.monthly_cap,
          spent_display: '$0',
          spent_this_month_usd: '0'
        }
      })
    )
    apiMocks.fetchSubscriptionState.mockResolvedValue(fixture.subscription)

    renderBilling()

    await screen.findByText('Subscription credits')
    const subscriptionTrack = screen.getByRole('progressbar', { name: 'Subscription credits usage' })

    expect(subscriptionTrack.getAttribute('aria-valuenow')).toBe('0')
    expect(subscriptionTrack.classList.contains('text-destructive')).toBe(false)
    expect(subscriptionTrack.classList.contains('dither')).toBe(true)

    const monthlyCapTrack = screen.getByRole('progressbar', { name: 'Monthly spend cap used' })

    expect(monthlyCapTrack.getAttribute('aria-valuenow')).toBe('0')
    expect(monthlyCapTrack.classList.contains('dither')).toBe(true)
    expect(monthlyCapTrack.classList.contains('bg-(--ui-bg-elevated)')).toBe(true)
  })

  it('refreshes both billing queries from the usage refresh button', async () => {
    renderBilling()

    await screen.findByText('$120 of $220 left')
    expect(apiMocks.fetchBillingState).toHaveBeenCalledTimes(1)
    expect(apiMocks.fetchSubscriptionState).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }))

    await waitFor(() => expect(apiMocks.fetchBillingState).toHaveBeenCalledTimes(2))
    expect(apiMocks.fetchSubscriptionState).toHaveBeenCalledTimes(2)
  })

  it('disables the usage refresh button while either query is fetching', async () => {
    let settleBilling: (value: unknown) => void = () => {}

    let settleSubscription: (value: unknown) => void = () => {}

    apiMocks.fetchBillingState.mockResolvedValueOnce(okBilling(todayBillingState)).mockReturnValueOnce(
      new Promise(resolve => {
        settleBilling = resolve
      })
    )
    apiMocks.fetchSubscriptionState.mockResolvedValueOnce(okSubscription(todaySubscriptionState)).mockReturnValueOnce(
      new Promise(resolve => {
        settleSubscription = resolve
      })
    )

    renderBilling()

    const refresh = await screen.findByRole('button', { name: 'Refresh' })

    fireEvent.click(refresh)

    await waitFor(() => expect(refresh.hasAttribute('disabled')).toBe(true))

    settleBilling(okBilling(todayBillingState))
    settleSubscription(okSubscription(todaySubscriptionState))

    await waitFor(() => expect(refresh.hasAttribute('disabled')).toBe(false))
  })
})

describe('formatUsageUpdatedAgo', () => {
  it('formats sub-second and current timestamps as just now', () => {
    expect(formatUsageUpdatedAgo(1_000, 1_000)).toBe('just now')
    expect(formatUsageUpdatedAgo(1_500, 1_000)).toBe('just now')
  })

  it('formats seconds below a minute', () => {
    expect(formatUsageUpdatedAgo(1_000, 60_000)).toBe('59s ago')
  })

  it('rounds elapsed time to whole minutes from 61 seconds', () => {
    expect(formatUsageUpdatedAgo(1_000, 62_000)).toBe('1m ago')
  })

  it('formats one hour and later as hours', () => {
    expect(formatUsageUpdatedAgo(1_000, 3_601_000)).toBe('1h ago')
  })
})
