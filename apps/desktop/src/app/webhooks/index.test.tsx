// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { TRANSLATIONS } from '@/i18n/catalog'
import type { WebhookRoute, WebhooksResponse } from '@/types/hermes'

const getWebhooks = vi.fn()

// Partial mock: keep the real module (WebhooksView pulls in @/store/profile,
// whose import-time subscription calls setApiRequestProfile) and stub only the
// webhook reads we assert on.
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getWebhooks: () => getWebhooks()
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/store/system-actions', () => ({
  runGatewayRestart: vi.fn()
}))

vi.mock('../hooks/use-refresh-hotkey', () => ({
  useRefreshHotkey: () => undefined
}))

function subscription(patch: Partial<WebhookRoute> = {}): WebhookRoute {
  return {
    created_at: null,
    deliver: 'log',
    deliver_only: false,
    description: '',
    enabled: true,
    events: [],
    name: 'github-push',
    prompt: '',
    secret_set: true,
    skills: [],
    url: 'http://127.0.0.1:8642/webhooks/github-push',
    ...patch
  }
}

function payload(subs: WebhookRoute[]): WebhooksResponse {
  return {
    base_url: 'http://127.0.0.1:8642',
    enabled: true,
    subscriptions: subs
  }
}

async function renderWebhooks(data: WebhooksResponse) {
  getWebhooks.mockResolvedValue(data)
  const { WebhooksView } = await import('./index')
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })

  let result: ReturnType<typeof render>
  await act(async () => {
    result = render(
      <QueryClientProvider client={client}>
        <WebhooksView onClose={() => undefined} />
      </QueryClientProvider>
    )
  })

  return result!
}

async function detailHeader(name: string): Promise<HTMLElement> {
  // Master/detail falls back to the first visible subscription, so the detail
  // heading is present as soon as the list loads.
  const heading = await screen.findByRole('heading', { name })
  const header = heading.parentElement
  expect(header).not.toBeNull()
  return header!
}

describe('WebhooksView subscription status pill', () => {
  beforeEach(() => {
    Element.prototype.hasPointerCapture ??= () => false
    Element.prototype.releasePointerCapture ??= () => undefined
    Element.prototype.setPointerCapture ??= () => undefined
    HTMLElement.prototype.scrollIntoView ??= () => undefined
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('shows an Enabled status pill for an enabled subscription detail', async () => {
    await renderWebhooks(payload([subscription({ enabled: true, name: 'github-push' })]))

    expect(within(await detailHeader('github-push')).getByText('Enabled')).toBeTruthy()
  })

  it('shows a Disabled status pill for a disabled subscription detail', async () => {
    await renderWebhooks(payload([subscription({ enabled: false, name: 'nightly-hook' })]))

    expect(within(await detailHeader('nightly-hook')).getByText('Disabled')).toBeTruthy()
  })

  it('keeps the deliver-only pill beside the status pill when both apply', async () => {
    await renderWebhooks(payload([subscription({ deliver_only: true, enabled: true, name: 'payload-only' })]))

    const header = await detailHeader('payload-only')
    expect(within(header).getByText('Enabled')).toBeTruthy()
    expect(within(header).getByText('deliver only')).toBeTruthy()
  })
})

describe('webhook status labels across locale catalogs', () => {
  it('keeps distinct non-empty status labels in every desktop locale', () => {
    const expected: Record<string, { disabled: string; enabled: string }> = {
      ar: { disabled: 'معطّل', enabled: 'مفعّل' },
      en: { disabled: 'Disabled', enabled: 'Enabled' },
      ja: { disabled: '無効', enabled: '有効' },
      zh: { disabled: '已禁用', enabled: '已启用' },
      'zh-hant': { disabled: '已停用', enabled: '已啟用' }
    }

    for (const [locale, t] of Object.entries(TRANSLATIONS)) {
      const labels = expected[locale]
      expect(labels, `missing expectation for ${locale}`).toBeTruthy()
      expect(t.webhooks.statusEnabled).toBe(labels.enabled)
      expect(t.webhooks.statusDisabled).toBe(labels.disabled)
      // Keep the old dead-key trap closed: messaging platform states must not
      // grow an `enabled` entry that webhooks no longer consume.
      expect(Object.hasOwn(t.messaging.states, 'enabled')).toBe(false)
    }
  })
})
