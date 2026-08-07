import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { ar } from '@/i18n/ar'
import { en } from '@/i18n/en'
import { zh } from '@/i18n/zh'

import {
  loggedOutBillingState,
  loggedOutSubscriptionState,
  okBilling,
  okSubscription,
  postTrainBillingState,
  postTrainSubscriptionState
} from './fixtures.test-util'

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

// `initialLocale` undefined → the provider's own default (en). `configClient` is
// null so the provider never touches the desktop config bridge.
function renderBilling(initialLocale?: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  render(
    <MemoryRouter initialEntries={['/settings?tab=billing']}>
      <QueryClientProvider client={client}>
        <I18nProvider configClient={null} initialLocale={initialLocale}>
          <BillingSettings />
        </I18nProvider>
      </QueryClientProvider>
    </MemoryRouter>
  )
}

function mockLoggedOut() {
  apiMocks.fetchBillingState.mockResolvedValue(okBilling(loggedOutBillingState))
  apiMocks.fetchSubscriptionState.mockResolvedValue(okSubscription(loggedOutSubscriptionState))
}

beforeEach(() => {
  apiMocks.fetchBillingState.mockResolvedValue(okBilling(postTrainBillingState))
  apiMocks.fetchSubscriptionState.mockResolvedValue(okSubscription(postTrainSubscriptionState))
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { openExternal: apiMocks.openExternal }
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('billing page localization', () => {
  it('(a) English keeps the literal copy the page shipped with', async () => {
    mockLoggedOut()
    renderBilling()

    // The header renders over the loading skeleton too, so wait on the banner
    // (payload-gated) before asserting the rest synchronously.
    expect(await screen.findByText('Connect your Nous account')).toBeTruthy()
    expect(screen.getByText('Billing')).toBeTruthy()
    expect(screen.getByText('Run /portal in the TUI or open the Nous portal to connect your account.')).toBeTruthy()
    expect(screen.getByText('Open portal ↗')).toBeTruthy()
    expect(screen.getByText('Balance')).toBeTruthy()
    expect(screen.getByText('Plan')).toBeTruthy()
    expect(screen.getByText('Auto-refill')).toBeTruthy()
  })

  it('(b) Arabic renders the catalog copy', async () => {
    mockLoggedOut()
    renderBilling('ar')

    expect(await screen.findByText('اربط حساب Nous')).toBeTruthy()
    expect(screen.getByText('الفوترة')).toBeTruthy()
    expect(screen.getByText('شغّل /portal في واجهة الطرفية أو افتح بوابة Nous لربط حسابك.')).toBeTruthy()
    expect(screen.getByText('افتح البوابة ↗')).toBeTruthy()
    expect(screen.getByText('الرصيد')).toBeTruthy()
    expect(screen.getByText('الخطة')).toBeTruthy()
    expect(screen.getByText('التعبئة التلقائية')).toBeTruthy()
    expect(screen.queryByText('Connect your Nous account')).toBeNull()
  })

  it('(c) a locale without the billing section falls back to the English literals', async () => {
    mockLoggedOut()
    renderBilling('zh')

    expect(zh.settings.billing).toBeUndefined()
    expect(await screen.findByText('Connect your Nous account')).toBeTruthy()
    expect(screen.getByText('Billing')).toBeTruthy()
    expect(screen.getByText('Balance')).toBeTruthy()
  })

  it('(d) section headings and buy controls follow the locale too', async () => {
    renderBilling()

    expect(await screen.findByText('Payment & credits')).toBeTruthy()
    expect(screen.getByText('Usage')).toBeTruthy()
    expect(screen.getByRole('button', { name: /^Buy$/ })).toBeTruthy()
    expect(screen.getByRole('spinbutton', { name: 'Custom credit amount' })).toBeTruthy()

    cleanup()
    renderBilling('ar')

    expect(await screen.findByText('الدفع والرصيد')).toBeTruthy()
    expect(screen.getByText('الاستخدام')).toBeTruthy()
    expect(screen.getByRole('button', { name: /^شراء$/ })).toBeTruthy()
    expect(screen.getByRole('spinbutton', { name: 'مبلغ رصيد مخصص' })).toBeTruthy()
  })

  it('(e) the inline "title: message" join keeps its exact English separator', async () => {
    apiMocks.charge.mockResolvedValue({ data: { charge_id: 'ch_1', ok: true }, idempotencyKey: 'key-1', ok: true })
    apiMocks.chargeStatus.mockResolvedValue({ data: { ok: true, reason: 'card_declined', status: 'failed' }, ok: true })

    renderBilling()
    fireEvent.click(await screen.findByRole('button', { name: /^Buy$/ }))

    // Title and message are server/error copy (out of catalog scope) — only the
    // ': ' between them moved, so the joined text must be unchanged.
    expect(
      await screen.findByText('Charge failed: Your card was declined. Try another card on the portal.')
    ).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()

    cleanup()
    renderBilling('ar')
    fireEvent.click(await screen.findByRole('button', { name: /^شراء$/ }))

    expect(
      await screen.findByText('Charge failed: Your card was declined. Try another card on the portal.')
    ).toBeTruthy()
    expect(screen.getByRole('button', { name: 'إعادة المحاولة' })).toBeTruthy()
  })

  it('(f) the catalog section stays optional and locale-additive', () => {
    // en (and zh) never gained the section — the component's `?? 'literal'`
    // fallbacks are what English renders, so the base catalogs stay untouched.
    expect(en.settings.billing).toBeUndefined()
    expect(zh.settings.billing).toBeUndefined()

    expect(ar.settings.billing?.title).toBe('الفوترة')
    expect(ar.settings.billing?.creditsAdded('$25')).toBe('أُضيف $25. يجري تحديث الرصيد.')
    expect(ar.settings.billing?.usageBarLabel('Top-up credits')).toBe('استخدام Top-up credits')
    // Colon + space rides the catalog so RTL can retune it without a code edit.
    expect(ar.settings.billing?.labelSeparator).toBe(': ')
  })
})
