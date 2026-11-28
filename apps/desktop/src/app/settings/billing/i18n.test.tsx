import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { ar } from '@/i18n/ar'
import { en } from '@/i18n/en'

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
  updateAutoReload: vi.fn(),
  updateKeepGoingOnCredits: vi.fn()
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
    updateAutoReload: apiMocks.updateAutoReload,
    updateKeepGoingOnCredits: apiMocks.updateKeepGoingOnCredits
  })
}))

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

beforeEach(() => {
  apiMocks.fetchBillingState.mockResolvedValue(okBilling(loggedOutBillingState))
  apiMocks.fetchSubscriptionState.mockResolvedValue(okSubscription(loggedOutSubscriptionState))
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
  it('keeps the English logged-out shell on the current copy', async () => {
    renderBilling()

    expect(await screen.findByText('Connect your Nous account')).toBeTruthy()
    expect(screen.getByText('Sign in with your Nous account to see your balance, plan and usage here.')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Sign in' })).toBeTruthy()
    expect(screen.getByText('Balance')).toBeTruthy()
    expect(screen.getByText('Plan')).toBeTruthy()
    expect(screen.getByText('Auto-refill')).toBeTruthy()
    expect(screen.getAllByText('Billing').length).toBeGreaterThan(0)
  })

  it('renders the Arabic logged-out shell from the catalog', async () => {
    renderBilling('ar')

    expect(await screen.findByText('اربط حساب Nous')).toBeTruthy()
    expect(screen.getByText('سجّل الدخول بحساب Nous لترى رصيدك وخطتك واستخدامك هنا.')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'تسجيل الدخول' })).toBeTruthy()
    expect(screen.getByText('الرصيد')).toBeTruthy()
    expect(screen.getByText('الخطة')).toBeTruthy()
    expect(screen.getByText('التعبئة التلقائية')).toBeTruthy()
    expect(screen.getAllByText('الفوترة').length).toBeGreaterThan(0)
    expect(screen.queryByText('Connect your Nous account')).toBeNull()
  })

  it('localizes the logged-in section headings and buy controls', async () => {
    apiMocks.fetchBillingState.mockResolvedValue(okBilling(postTrainBillingState))
    apiMocks.fetchSubscriptionState.mockResolvedValue(okSubscription(postTrainSubscriptionState))

    renderBilling()
    expect(await screen.findByRole('button', { name: /^Buy$/ })).toBeTruthy()
    expect(screen.getByText('Payment & credits')).toBeTruthy()
    expect(screen.getByText('Usage')).toBeTruthy()
    expect(screen.getByRole('spinbutton', { name: 'Custom credit amount' })).toBeTruthy()

    cleanup()
    renderBilling('ar')
    expect(await screen.findByRole('button', { name: /^شراء$/ })).toBeTruthy()
    expect(screen.getByText('الدفع والرصيد')).toBeTruthy()
    expect(screen.getByText('الاستخدام')).toBeTruthy()
    expect(screen.getByRole('spinbutton', { name: 'مبلغ رصيد مخصص' })).toBeTruthy()
  })

  it('pins the Arabic shell helpers without changing English', () => {
    expect(en.settings.billing.title).toBe('Billing')
    expect(en.settings.billing.buyCredits.added('$25')).toBe('$25 added. Balance is refreshing.')
    expect(en.settings.billing.usageLabel('Top-up credits')).toBe('Top-up credits usage')
    expect(en.settings.nav.billing).toBe('Billing')

    expect(ar.settings.billing.title).toBe('الفوترة')
    expect(ar.settings.nav.billing).toBe('الفوترة')
    expect(ar.settings.billing.buyCredits.added('$25')).toBe('أُضيف $25. يجري تحديث الرصيد.')
    expect(ar.settings.billing.usageLabel('Top-up credits')).toBe('استخدام Top-up credits')
    expect(ar.settings.billing.state.notice.loggedOut.action).toBe('تسجيل الدخول')
  })
})
