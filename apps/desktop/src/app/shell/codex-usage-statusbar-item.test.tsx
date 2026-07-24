import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ComponentProps } from 'react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import type { GatewayRequester } from '@/hooks/use-codex-account-usage'
import type { AccountUsageResponse, AccountUsageSnapshot } from '@/types/hermes'

import {
  accountUsageRemaining,
  primaryAccountUsageWindow,
  useCodexUsageStatusbarItem
} from './codex-usage-statusbar-item'
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

const snapshot: AccountUsageSnapshot = {
  available: true,
  details: ['Credits balance: $12.50'],
  fetched_at: '2026-07-16T01:02:03+00:00',
  plan: 'Plus',
  provider: 'openai-codex',
  source: 'usage_api',
  title: 'Account limits',
  unavailable_reason: null,
  windows: [
    { label: 'Session', reset_at: '2026-07-16T03:02:03+00:00', used_percent: 21 },
    { label: 'Weekly', reset_at: '2026-07-20T03:02:03+00:00', used_percent: 59 }
  ]
}

function Harness({
  connectionScope = 'local:',
  gatewayState = 'open',
  profile = 'default',
  provider = 'openai-codex',
  requestGateway,
  sessionId = 'runtime-1'
}: {
  connectionScope?: string
  gatewayState?: string
  profile?: string
  provider?: string
  requestGateway: GatewayRequester
  sessionId?: null | string
}) {
  const item = useCodexUsageStatusbarItem({
    connectionScope,
    gatewayState,
    profile,
    provider,
    requestGateway,
    sessionId
  })

  return (
    <I18nProvider configClient={null} initialLocale="en">
      <MemoryRouter>
        <StatusbarControls items={[item]} />
      </MemoryRouter>
    </I18nProvider>
  )
}

function renderHarness(props: ComponentProps<typeof Harness>) {
  const client = new QueryClient({
    defaultOptions: { queries: { gcTime: Number.POSITIVE_INFINITY, retry: false } }
  })

  return render(<Harness {...props} />, {
    wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
  })
}

describe('Codex usage statusbar item', () => {
  it('derives the compact remaining allowance from the first valid window', () => {
    expect(accountUsageRemaining({ label: 'Session', used_percent: 21 })).toBe(79)
    expect(accountUsageRemaining({ label: 'Session', used_percent: 140 })).toBe(0)
    expect(accountUsageRemaining({ label: 'Session' })).toBeNull()
    expect(primaryAccountUsageWindow(snapshot)?.label).toBe('Session')
  })

  it('stays hidden unless an active Codex session returns valid limits', async () => {
    const requestGateway = vi.fn(async () => ({ account_usage: snapshot }) as never)
    const { rerender } = renderHarness({ provider: 'anthropic', requestGateway })

    await act(async () => undefined)
    expect(requestGateway).not.toHaveBeenCalled()
    expect(screen.queryByText(/Codex/)).toBeNull()

    rerender(<Harness requestGateway={requestGateway} />)

    expect(await screen.findByRole('button', { name: /Codex 79% left/i })).toBeTruthy()
    expect(requestGateway).toHaveBeenCalledWith(
      'session.account_usage',
      { session_id: 'runtime-1' },
      45_000,
      expect.any(AbortSignal)
    )
  })

  it('shows both windows, credits, refresh time, and the web fallback', async () => {
    const requestGateway = vi.fn(async () => ({ account_usage: snapshot }) as never)
    renderHarness({ requestGateway })

    fireEvent.pointerDown(await screen.findByRole('button', { name: /Codex 79% left/i }), { button: 0 })

    expect(await screen.findByText('79% remaining')).toBeTruthy()
    expect(screen.getByText('41% remaining')).toBeTruthy()
    expect(screen.getByText('Credits balance: $12.50')).toBeTruthy()
    expect(screen.getByText(/Updated/)).toBeTruthy()
    expect(screen.getByRole('link', { name: 'Open Codex Usage' }).getAttribute('href')).toBe(
      'https://chatgpt.com/codex/settings/usage'
    )
  })

  it('keeps the last good snapshot visible when a refresh fails', async () => {
    const requestGateway = vi
      .fn<() => Promise<AccountUsageResponse>>()
      .mockResolvedValueOnce({ account_usage: snapshot })
      .mockRejectedValueOnce(new Error('auth expired'))

    renderHarness({ requestGateway: requestGateway as never })
    fireEvent.pointerDown(await screen.findByRole('button', { name: /Codex 79% left/i }), { button: 0 })
    fireEvent.click(await screen.findByRole('button', { name: 'Refresh Codex usage' }))

    await waitFor(() => {
      expect(screen.getByText(/Showing the last successful result/i)).toBeTruthy()
      expect(screen.getByText('Codex 79% left')).toBeTruthy()
    })
  })
})
