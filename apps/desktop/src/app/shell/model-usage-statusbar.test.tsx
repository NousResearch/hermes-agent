import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { StatusbarControls } from '@/app/shell/statusbar-controls'
import { I18nProvider } from '@/i18n'
import { $statusbarHiddenIds, STATUSBAR_HIDDEN_BY_DEFAULT } from '@/store/statusbar-prefs'
import type { UsageStats } from '@/types/hermes'

import { useModelUsageStatusbarItem } from './model-usage-statusbar'

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
  $statusbarHiddenIds.set([...STATUSBAR_HIDDEN_BY_DEFAULT])
})

// The item ships hidden (STATUSBAR_HIDDEN_BY_DEFAULT), so the harness opts it
// in before each render — the tests assert bar content, not default visibility.
beforeEach(() => {
  $statusbarHiddenIds.set([...STATUSBAR_HIDDEN_BY_DEFAULT].filter(id => id !== 'model-usage'))
})

function Harness({
  activeSessionId,
  currentModel,
  currentProvider,
  currentUsage,
  requestGateway
}: {
  activeSessionId: string | null
  currentModel: string
  currentProvider: string
  currentUsage: UsageStats
  requestGateway: (method: string, params?: Record<string, unknown>) => Promise<unknown>
}) {
  const item = useModelUsageStatusbarItem({
    activeSessionId,
    currentModel,
    currentProvider,
    currentUsage,
    requestGateway: <T,>(method: string, params?: Record<string, unknown>) =>
      requestGateway(method, params) as Promise<T>
  })

  return (
    <I18nProvider configClient={null} initialLocale="zh">
      <MemoryRouter>
        <StatusbarControls items={[item]} />
      </MemoryRouter>
    </I18nProvider>
  )
}

const EMPTY_USAGE: UsageStats = { calls: 0, input: 0, output: 0, total: 0 }

/** The model-usage button in the bar, or null when the item is hidden. */
function withinBar() {
  return screen.queryByRole('button', { name: /tokens/i })
}

describe('model usage statusbar item', () => {
  it('hides the item until there is accounted usage', () => {
    const requestGateway = vi.fn(async () => ({ routes: [], totals: EMPTY_USAGE }))

    render(
      <Harness
        activeSessionId={null}
        currentModel="xai/grok-4.5"
        currentProvider="xai-oauth"
        currentUsage={EMPTY_USAGE}
        requestGateway={requestGateway}
      />
    )

    // With no usage, the item hides entirely (hidden: totalTokens <= 0) and
    // the bar renders no model-usage button at all.
    expect(withinBar()).toBeNull()
    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('shows the session-wide cumulative total in the bar and expands every model route', async () => {
    const requestGateway = vi.fn(async () => ({
      routes: [
        {
          model: 'deepseek-v4-pro',
          provider: 'deepseek',
          billing_mode: 'api_key',
          calls: 2,
          input: 40_000,
          output: 8_000,
          cache_read: 3_000,
          cache_write: 0,
          reasoning: 1_000,
          total: 51_000,
          estimated_cost_usd: 0.12,
          actual_cost_usd: 0,
          cost_status: 'estimated',
          cost_source: 'pricing',
          last_seen: 20
        },
        {
          model: 'claude-opus-4.8',
          provider: 'openrouter',
          billing_mode: 'api_key',
          calls: 3,
          input: 50_000,
          output: 4_000,
          cache_read: 0,
          cache_write: 0,
          reasoning: 2_000,
          total: 54_000,
          estimated_cost_usd: 0.34,
          actual_cost_usd: 0,
          cost_status: 'estimated',
          cost_source: 'pricing',
          last_seen: 30
        }
      ],
      totals: {
        calls: 5,
        input: 90_000,
        output: 12_000,
        cache_read: 3_000,
        cache_write: 0,
        reasoning: 3_000,
        total: 105_000,
        estimated_cost_usd: 0.46,
        actual_cost_usd: 0
      }
    }))

    render(
      <Harness
        activeSessionId="runtime-1"
        currentModel="claude-opus-4.8"
        currentProvider="openrouter"
        currentUsage={{ calls: 5, input: 90_000, output: 12_000, total: 105_000 }}
        requestGateway={requestGateway}
      />
    )

    await waitFor(() => {
      expect(requestGateway).toHaveBeenCalledWith('session.model_usage', { session_id: 'runtime-1' })
      expect(screen.getByRole('button', { name: /Tokens.*105k/i })).toBeTruthy()
    })

    fireEvent.pointerDown(screen.getByRole('button', { name: /Tokens/i }), { button: 0 })

    expect(await screen.findByText('deepseek-v4-pro')).toBeTruthy()
    // The bar shows the cumulative total now, so the model name lives only in
    // the expanded panel row.
    expect(screen.getAllByText('claude-opus-4.8')).toHaveLength(1)
    expect(screen.getByText(/105k tokens/i)).toBeTruthy()
  })

  it('prefers backend totals over live-session totals', async () => {
    const requestGateway = vi.fn(async () => ({
      routes: [],
      totals: {
        calls: 2,
        input: 10,
        output: 5,
        cache_read: 1,
        cache_write: 0,
        reasoning: 0,
        total: 16,
        estimated_cost_usd: 0,
        actual_cost_usd: 0
      }
    }))

    render(
      <Harness
        activeSessionId="sid-1"
        currentModel="model/a"
        currentProvider="provider"
        currentUsage={{ calls: 1, input: 2, output: 1, total: 3 }}
        requestGateway={requestGateway}
      />
    )

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Tokens.*16/i })).toBeTruthy()
    })
  })
})
