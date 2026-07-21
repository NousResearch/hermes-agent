import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
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
  stepUp: vi.fn(),
  updateAutoReload: vi.fn()
}))

vi.mock('./api', () => ({
  useBillingApi: () => ({
    charge: apiMocks.charge,
    chargeStatus: apiMocks.chargeStatus,
    fetchBillingState: apiMocks.fetchBillingState,
    fetchSubscriptionState: apiMocks.fetchSubscriptionState,
    stepUp: apiMocks.stepUp,
    updateAutoReload: apiMocks.updateAutoReload
  })
}))

function renderBilling() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  render(
    <QueryClientProvider client={client}>
      <BillingSettings />
    </QueryClientProvider>
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
    expect(screen.getByText('Refill $10 when balance falls below $5')).toBeTruthy()
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

  it('requires inline confirmation before disabling auto-refill', async () => {
    renderBilling()

    apiMocks.updateAutoReload.mockResolvedValue({ data: { ok: true }, ok: true })

    fireEvent.click(await screen.findByRole('button', { name: 'Manage' }))
    fireEvent.click(screen.getByRole('button', { name: 'Disable' }))

    expect(screen.getByText('Turn off auto-refill?')).toBeTruthy()
    expect(apiMocks.updateAutoReload).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Turn off' }))

    await waitFor(() => expect(apiMocks.updateAutoReload).toHaveBeenCalledWith({ enabled: false }))
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
