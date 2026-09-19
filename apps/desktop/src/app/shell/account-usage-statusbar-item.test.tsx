import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import type { AccountUsageOwnerScope, GatewayRequester } from '@/hooks/use-account-usage'
import { I18nProvider } from '@/i18n'
import type { AccountUsageResponse, AccountUsageSnapshot, UsageStats } from '@/types/hermes'

import {
  accountUsageChipLabel,
  accountUsageMinRemaining,
  accountUsageRemaining,
  formatCompactQuotaRatio,
  formatCreditsBalance,
  tightestAccountUsageWindow,
  useAccountUsageStatusbarItem
} from './account-usage-statusbar-item'
import { StatusbarControls } from './statusbar-controls'

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', TestResizeObserver)
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.setPointerCapture ??= () => undefined
  Element.prototype.releasePointerCapture ??= () => undefined
  HTMLElement.prototype.scrollIntoView ??= () => undefined
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

const EMPTY_USAGE: UsageStats = { calls: 0, input: 0, output: 0, total: 0 }

const windowsSnapshot: AccountUsageSnapshot = {
  available: true,
  details: ['Credits balance: $12.50'],
  fetched_at: '2026-07-16T01:02:03+00:00',
  plan: 'Plus',
  provider: 'openai-codex',
  source: 'usage_api',
  title: 'Account limits',
  unavailable_reason: null,
  windows: [
    { label: 'Session', reset_at: '2026-07-16T03:02:03+00:00', used_percent: 17 },
    { label: 'Weekly', reset_at: '2026-07-20T03:02:03+00:00', used_percent: 59 }
  ]
}

function Harness({
  connectionScope = 'local:',
  gatewayState = 'open',
  owner = { connectionId: 'local', profile: 'default' },
  profile = 'default',
  provider = 'openai-codex',
  requestGateway,
  sessionId = 'runtime-1',
  usage = EMPTY_USAGE
}: {
  connectionScope?: string
  gatewayState?: string
  owner?: AccountUsageOwnerScope
  profile?: string
  provider?: string
  requestGateway: GatewayRequester
  sessionId?: null | string
  usage?: UsageStats
}) {
  const item = useAccountUsageStatusbarItem({
    connectionScope,
    gatewayState,
    owner,
    profile,
    provider,
    requestGateway,
    sessionId,
    usage
  })

  return (
    <MemoryRouter>
      <StatusbarControls items={[item]} />
    </MemoryRouter>
  )
}

function renderHarness(props: ComponentProps<typeof Harness>) {
  const client = new QueryClient({
    defaultOptions: { queries: { gcTime: Number.POSITIVE_INFINITY, retry: false } }
  })

  return render(<Harness {...props} />, {
    wrapper: ({ children }) => (
      <I18nProvider configClient={null} initialLocale="en">
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      </I18nProvider>
    )
  })
}

function requester(fn: () => Promise<unknown>): GatewayRequester {
  return vi.fn(fn) as unknown as GatewayRequester
}

describe('Account usage statusbar item', () => {
  it('derives remaining from each window and the tightest window across a snapshot', () => {
    expect(accountUsageRemaining({ label: 'Session', used_percent: 21 })).toBe(79)
    expect(accountUsageRemaining({ label: 'Session', used_percent: 140 })).toBe(0)
    expect(accountUsageRemaining({ label: 'Session' })).toBeNull()
    expect(accountUsageMinRemaining(windowsSnapshot)).toBe(41)
    expect(tightestAccountUsageWindow(windowsSnapshot)?.label).toBe('Weekly')
  })

  it('formats credits plus quota as Name: $amount (remaining%)', async () => {
    const requestGateway = requester(async () => ({
      account_usage: {
        ...windowsSnapshot,
        credits_balance: 31.44,
        provider: 'openrouter',
        windows: [{ label: 'Daily', used_percent: 0 }]
      },
      status: 'ok'
    }))

    renderHarness({ provider: 'openrouter', requestGateway })

    expect(await screen.findByRole('button', { name: /OpenRouter: \$31\.44 \(100%\)/ })).toBeTruthy()
  })

  it('formats credits plus rounded key quota as Name: $amount ($remaining/$limit, remaining%)', async () => {
    const requestGateway = requester(async () => ({
      account_usage: {
        ...windowsSnapshot,
        credits_balance: 46.07,
        provider: 'openrouter',
        windows: [
          {
            label: 'API key quota',
            label_key: 'api_key_quota',
            limit: 20,
            limit_remaining: 19.29,
            used_percent: ((20 - 19.29) / 20) * 100
          }
        ]
      },
      status: 'ok'
    }))

    renderHarness({ provider: 'openrouter', requestGateway })

    expect(await screen.findByRole('button', { name: /OpenRouter: \$46\.07 \(\$19\/20, 96%\)/ })).toBeTruthy()
  })

  it('formats windows-only remaining as Name: N% left', async () => {
    const requestGateway = requester(async () => ({
      account_usage: {
        ...windowsSnapshot,
        windows: [{ label: 'Session', used_percent: 17 }]
      },
      status: 'ok'
    }))

    renderHarness({ requestGateway })

    expect(await screen.findByRole('button', { name: /Codex: 83% left/i })).toBeTruthy()
  })

  it('formats credits-only as Name: $amount', async () => {
    const requestGateway = requester(async () => ({
      account_usage: {
        ...windowsSnapshot,
        credits_balance: 31.44,
        provider: 'openrouter',
        windows: [{ detail: 'No percent window', label: 'Balance' }]
      },
      status: 'ok'
    }))

    renderHarness({ provider: 'openrouter', requestGateway })

    expect(await screen.findByRole('button', { name: /OpenRouter: \$31\.44$/ })).toBeTruthy()
  })

  it('uses the minimum remaining across multiple windows for the compact label', async () => {
    const requestGateway = requester(async () => ({ account_usage: windowsSnapshot, status: 'ok' }))

    renderHarness({ requestGateway })

    expect(await screen.findByRole('button', { name: /Codex: 41% left/i })).toBeTruthy()
  })

  it('stays hidden while the first snapshot is loading', async () => {
    const requestGateway = requester(() => new Promise<AccountUsageResponse>(() => undefined))

    renderHarness({ requestGateway })
    await act(async () => undefined)

    expect(screen.queryByRole('button', { name: /Codex|OpenRouter|Account usage/i })).toBeNull()
    expect(requestGateway).toHaveBeenCalled()
  })

  it('stays hidden when a session has no resolved owner', async () => {
    const requestGateway = requester(async () => ({ account_usage: windowsSnapshot, status: 'ok' }))

    renderHarness({ owner: null, requestGateway })
    await act(async () => undefined)

    expect(requestGateway).not.toHaveBeenCalled()
    expect(screen.queryByRole('button', { name: /Codex|Account usage/i })).toBeNull()
  })

  it('stays hidden when the provider is unsupported', async () => {
    const requestGateway = requester(async () => ({ account_usage: null, status: 'unsupported' }))

    renderHarness({ provider: 'openai', requestGateway })
    await act(async () => undefined)
    await waitFor(() => expect(requestGateway).toHaveBeenCalled())

    expect(screen.queryByRole('button', { name: /Account usage|Openai/i })).toBeNull()
  })

  it('marks remaining at or below 20 as amber and at or below 5 as destructive', async () => {
    const amberGateway = requester(async () => ({
      account_usage: { ...windowsSnapshot, windows: [{ label: 'Session', used_percent: 80 }] },
      status: 'ok'
    }))
    const { unmount } = renderHarness({ requestGateway: amberGateway })
    const amber = await screen.findByRole('button', { name: /Codex: 20% left/i })
    expect(amber.className).toContain('text-amber-600')
    expect(amber.className).not.toContain('text-destructive')
    unmount()

    const redGateway = requester(async () => ({
      account_usage: { ...windowsSnapshot, windows: [{ label: 'Session', used_percent: 95 }] },
      status: 'ok'
    }))
    renderHarness({ requestGateway: redGateway })
    const red = await screen.findByRole('button', { name: /Codex: 5% left/i })
    expect(red.className).toContain('text-destructive')
  })

  it('keeps the last good snapshot visible and amber when a refresh fails', async () => {
    const requestGateway = vi
      .fn<() => Promise<AccountUsageResponse>>()
      .mockResolvedValueOnce({
        account_usage: { ...windowsSnapshot, windows: [{ label: 'Session', used_percent: 17 }] },
        status: 'ok'
      })
      .mockRejectedValueOnce(new Error('auth expired'))

    renderHarness({ requestGateway: requestGateway as never })
    const chip = await screen.findByRole('button', { name: /Codex: 83% left/i })
    fireEvent.pointerDown(chip, { button: 0 })
    fireEvent.click(await screen.findByRole('button', { name: 'Refresh' }))

    await waitFor(() => {
      expect(screen.getByText(/Showing the last successful result/i)).toBeTruthy()
      expect(screen.getByText('Codex: 83% left')).toBeTruthy()
    })
    expect(screen.getByText('Codex: 83% left').closest('button')?.className).toContain('text-amber-600')
  })

  it('renders windows, details, and a This session section in the popover', async () => {
    renderHarness({
      requestGateway: requester(async () => ({ account_usage: windowsSnapshot, status: 'ok' })),
      usage: { calls: 12, input: 1200, output: 800, total: 2000 }
    })

    fireEvent.pointerDown(await screen.findByRole('button', { name: /Codex: 41% left/i }), { button: 0 })

    expect(await screen.findByText('83% remaining')).toBeTruthy()
    expect(screen.getByText('41% remaining')).toBeTruthy()
    expect(screen.getByText('Credits balance: $12.50')).toBeTruthy()
    expect(screen.getByText('This session')).toBeTruthy()
    expect(screen.getByText(/1\.2k in · 800 out · 2k total/)).toBeTruthy()
    expect(screen.queryByText(/12 calls/)).toBeNull()
    expect(screen.getByText(/Updated/)).toBeTruthy()
    expect(screen.getByRole('link', { name: 'Open usage settings' }).getAttribute('href')).toBe(
      'https://chatgpt.com/codex/settings/usage'
    )
  })

  it('never paints a new provider name with the previous snapshot numbers', async () => {
    const first = requester(async () => ({
      account_usage: { ...windowsSnapshot, credits_balance: 31.44 },
      status: 'ok'
    }))
    const second = requester(() => new Promise<AccountUsageResponse>(() => undefined))

    const { rerender } = renderHarness({ provider: 'openai-codex', requestGateway: first })
    expect(await screen.findByRole('button', { name: /Codex: \$31\.44 \(41%\)/ })).toBeTruthy()

    rerender(<Harness provider="openrouter" requestGateway={second} />)
    await act(async () => undefined)

    expect(screen.queryByRole('button', { name: /OpenRouter/i })).toBeNull()
    expect(screen.queryByText(/OpenRouter: \$31\.44/)).toBeNull()
    expect(screen.queryByText(/OpenRouter: 41%/)).toBeNull()
    expect(screen.queryByRole('button', { name: /Codex/i })).toBeNull()
  })

  it('hides the chip when switching to another profile of the same provider', async () => {
    const first = requester(async () => ({
      account_usage: { ...windowsSnapshot, credits_balance: 50, provider: 'openrouter' },
      status: 'ok'
    }))
    const second = requester(() => new Promise<AccountUsageResponse>(() => undefined))

    const { rerender } = renderHarness({
      profile: 'default',
      provider: 'openrouter',
      requestGateway: first
    })
    expect(await screen.findByRole('button', { name: /OpenRouter: \$50\.00 \(41%\)/ })).toBeTruthy()

    rerender(<Harness profile="work" provider="openrouter" requestGateway={second} />)
    await act(async () => undefined)

    expect(screen.queryByRole('button', { name: /OpenRouter/i })).toBeNull()
    expect(screen.queryByText(/\$50\.00/)).toBeNull()
  })

  it('omits the This session section when only stale calls remain', async () => {
    renderHarness({
      requestGateway: requester(async () => ({ account_usage: windowsSnapshot, status: 'ok' })),
      usage: { calls: 25, input: 0, output: 0, total: 0 }
    })

    fireEvent.pointerDown(await screen.findByRole('button', { name: /Codex: 41% left/i }), { button: 0 })

    expect(await screen.findByText('83% remaining')).toBeTruthy()
    expect(screen.queryByText('This session')).toBeNull()
    expect(screen.queryByText(/25 calls/)).toBeNull()
  })

  it('omits the This session section when session usage is all zeros', async () => {
    renderHarness({
      requestGateway: requester(async () => ({ account_usage: windowsSnapshot, status: 'ok' })),
      usage: EMPTY_USAGE
    })

    fireEvent.pointerDown(await screen.findByRole('button', { name: /Codex: 41% left/i }), { button: 0 })

    expect(await screen.findByText('83% remaining')).toBeTruthy()
    expect(screen.queryByText('This session')).toBeNull()
  })

  it('does not render a billing link for an unmapped provider', async () => {
    renderHarness({
      provider: 'custom-llm',
      requestGateway: requester(async () => ({
        account_usage: { ...windowsSnapshot, provider: 'custom-llm' },
        status: 'ok'
      }))
    })

    fireEvent.pointerDown(await screen.findByRole('button', { name: /Custom Llm: 41% left/i }), { button: 0 })

    expect(await screen.findByText('83% remaining')).toBeTruthy()
    expect(screen.queryByRole('link', { name: 'Open usage settings' })).toBeNull()
  })

  it('stays hidden when the backend reports method-unavailable', async () => {
    const requestGateway = requester(async () => {
      throw Object.assign(new Error('Method not found'), { code: -32601 })
    })

    renderHarness({ requestGateway })
    await act(async () => undefined)
    await waitFor(() => expect(requestGateway).toHaveBeenCalled())

    expect(screen.queryByRole('button', { name: /Codex|Account usage/i })).toBeNull()
  })

  it('renders localized structured rows instead of raw details', async () => {
    const snapshot: AccountUsageSnapshot = {
      available: true,
      credits_balance: 31.44,
      details: [
        'Credits balance: $31.44',
        'API key usage: $14.25 total · $4.09 this week · $0.44 this month'
      ],
      details_structured: true,
      fetched_at: '2026-07-16T01:02:03+00:00',
      plan: 'Plus',
      provider: 'openrouter',
      rows: [
        { args: { currency: 'USD', value: 31.44 }, key: 'credits_balance' },
        { args: { monthly: 0.44, total: 14.25, weekly: 4.09 }, key: 'api_key_usage' },
        { args: { foo: 1 }, key: 'future_metric' }
      ],
      source: 'usage_api',
      title: 'Account limits',
      unavailable_reason: null,
      windows: [
        {
          label: 'API key quota',
          label_key: 'api_key_quota',
          limit: 14.25,
          limit_remaining: 4.09,
          reset_interval: 'weekly',
          used_percent: 71
        }
      ]
    }

    renderHarness({
      provider: 'openrouter',
      requestGateway: requester(async () => ({ account_usage: snapshot, status: 'ok' }))
    })

    fireEvent.pointerDown(await screen.findByRole('button', { name: /OpenRouter: \$31\.44 \(\$4\/14, 29%\)/ }), { button: 0 })

    const creditsLabel = await screen.findByText('Credits balance')
    expect(creditsLabel.tagName).toBe('P')
    expect(creditsLabel.nextElementSibling?.textContent).toBe('$31.44')
    expect(creditsLabel.textContent).toBe('Credits balance')

    const usageLabel = screen.getByText('API key usage')
    const usageValue = usageLabel.nextElementSibling
    expect(usageValue?.textContent).toBe('$14.25 total · $4.09 this week · $0.44 this month')
    expect(usageValue?.className.split(/\s+/)).toContain('break-words')
    expect(usageValue?.className.split(/\s+/)).not.toContain('truncate')
    expect(screen.getByText('API key quota')).toBeTruthy()
    expect(screen.getByText('$4.09 of $14.25 remaining · resets weekly')).toBeTruthy()
    expect(screen.queryByText('Credits balance: $31.44')).toBeNull()
    expect(screen.queryByText(/API key usage: \$14\.25/)).toBeNull()
  })

  it('falls back to truncated raw details when details_structured is absent', async () => {
    renderHarness({
      requestGateway: requester(async () => ({
        account_usage: {
          ...windowsSnapshot,
          rows: [{ args: { currency: 'USD', value: 12.5 }, key: 'credits_balance' }]
        },
        status: 'ok'
      }))
    })

    fireEvent.pointerDown(await screen.findByRole('button', { name: /Codex: 41% left/i }), { button: 0 })

    const rawDetail = await screen.findByText('Credits balance: $12.50')
    expect(rawDetail.tagName).toBe('LI')
    expect(rawDetail.className.split(/\s+/)).toContain('truncate')
    expect(screen.queryByText('Credits balance')).toBeNull()
    expect(screen.queryByText('API key usage')).toBeNull()
  })

  it('uses a localized window label for known label_key and the raw label otherwise', async () => {
    renderHarness({
      requestGateway: requester(async () => ({
        account_usage: {
          ...windowsSnapshot,
          details: [],
          windows: [
            { label: 'Session', label_key: 'current_session', reset_at: '2026-07-16T03:02:03+00:00', used_percent: 10 },
            { label: 'Daily burst', label_key: 'daily_burst', reset_at: '2026-07-20T03:02:03+00:00', used_percent: 20 }
          ]
        },
        status: 'ok'
      }))
    })

    fireEvent.pointerDown(await screen.findByRole('button', { name: /Codex: 80% left/i }), { button: 0 })

    expect(await screen.findByText('Current session')).toBeTruthy()
    expect(screen.getByText('Daily burst')).toBeTruthy()
    expect(screen.queryByText('Session')).toBeNull()
  })

  it('omits the resets phrase for an unknown reset_interval without throwing', async () => {
    renderHarness({
      provider: 'openrouter',
      requestGateway: requester(async () => ({
        account_usage: {
          ...windowsSnapshot,
          credits_balance: 31.44,
          details: [],
          provider: 'openrouter',
          windows: [
            {
              label: 'API key quota',
              label_key: 'api_key_quota',
              limit: 100,
              limit_remaining: 70,
              reset_interval: 'hourly',
              used_percent: 30
            }
          ]
        },
        status: 'ok'
      }))
    })

    fireEvent.pointerDown(await screen.findByRole('button', { name: /OpenRouter: \$31\.44 \(\$70\/100, 70%\)/ }), { button: 0 })

    expect(await screen.findByText('$70.00 of $100.00 remaining')).toBeTruthy()
    expect(screen.queryByText(/resets /)).toBeNull()
  })

  it('renders credits_unlimited, banked_resets, and extra_usage rows', async () => {
    renderHarness({
      requestGateway: requester(async () => ({
        account_usage: {
          ...windowsSnapshot,
          details: [
            'Credits balance: unlimited',
            'You have 2 resets banked - use /usage reset to activate',
            'Extra usage: 1.50 / 10.00 USD'
          ],
          details_structured: true,
          rows: [
            { key: 'credits_unlimited' },
            { args: { count: 2 }, key: 'banked_resets' },
            { args: { currency: 'USD', limit: 10, used: 1.5 }, key: 'extra_usage' }
          ],
          windows: [{ label: 'Session', reset_at: '2026-07-16T03:02:03+00:00', used_percent: 10 }]
        },
        status: 'ok'
      }))
    })

    fireEvent.pointerDown(await screen.findByRole('button', { name: /Codex: 90% left/i }), { button: 0 })

    const creditsLabel = await screen.findByText('Credits balance')
    expect(creditsLabel.nextElementSibling?.textContent).toBe('Unlimited')
    expect(screen.getByText('You have 2 resets banked - use /usage reset to activate')).toBeTruthy()
    const extraLabel = screen.getByText('Extra usage')
    expect(extraLabel.nextElementSibling?.textContent).toBe('$1.50 / $10.00')
    expect(screen.queryByText('Credits balance: unlimited')).toBeNull()
  })
})

describe('formatCreditsBalance', () => {
  it('formats USD by default and falls back for unknown currency codes', () => {
    expect(formatCreditsBalance(31.44, 'en')).toBe('$31.44')
    expect(formatCreditsBalance(12.5, 'en', 'USD')).toBe('$12.50')
    expect(formatCreditsBalance(12.5, 'en', 'USDX')).toBe('12.50 USDX')
  })
})

describe('formatCompactQuotaRatio', () => {
  it('rounds remaining and limit to nearest integers', () => {
    expect(formatCompactQuotaRatio(19.29, 20)).toBe('$19/20')
    expect(formatCompactQuotaRatio(19.5, 20)).toBe('$20/20')
    expect(formatCompactQuotaRatio(0.4, 20)).toBe('$0/20')
  })

  it('omits a ratio when the rounded limit is not a positive dollar amount', () => {
    expect(formatCompactQuotaRatio(5, 0.4)).toBeNull()
    expect(formatCompactQuotaRatio(1, 0)).toBeNull()
    expect(formatCompactQuotaRatio(Number.NaN, 20)).toBeNull()
    expect(formatCompactQuotaRatio(19, undefined)).toBeNull()
  })

  it('clamps a rounded remaining that would exceed the rounded limit', () => {
    expect(formatCompactQuotaRatio(19.6, 19.4)).toBe('$19/19')
  })
})

describe('accountUsageChipLabel', () => {
  const copy = {
    accountUsage: 'Account usage',
    accountUsageLeft: (remaining: number) => `${remaining}% left`
  }

  it('puts rounded key quota ahead of remaining percent when both exist', () => {
    expect(
      accountUsageChipLabel({
        copy,
        credits: '$46.07',
        providerName: 'OpenRouter',
        quotaRatio: '$19/20',
        remaining: 96
      })
    ).toBe('OpenRouter: $46.07 ($19/20, 96%)')
  })

  it('keeps credits-only remaining percent when the tightest window has no dollar limit', () => {
    expect(
      accountUsageChipLabel({
        copy,
        credits: '$31.44',
        providerName: 'OpenRouter',
        quotaRatio: null,
        remaining: 100
      })
    ).toBe('OpenRouter: $31.44 (100%)')
  })

  it('keeps the localized remaining-only phrasing for windows without credits or a dollar limit', () => {
    expect(
      accountUsageChipLabel({
        copy,
        credits: null,
        providerName: 'Codex',
        quotaRatio: null,
        remaining: 83
      })
    ).toBe('Codex: 83% left')
  })
})
