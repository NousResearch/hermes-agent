import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { HERMES_CONFIG_KEY } from '@/app/hooks/use-config-record'
import { $visibleModels } from '@/store/model-visibility'
import { $collapsedProviders } from '@/store/provider-collapse'

import { ModelVisibilityDialog } from './model-visibility-dialog'

const getGlobalModelOptions = vi.fn()
const getHermesConfigRecord = vi.fn()
const saveHermesConfig = vi.fn()

vi.mock('@/hermes', () => ({
  getGlobalModelOptions: (...args: unknown[]) => getGlobalModelOptions(...args),
  getHermesConfigRecord: () => getHermesConfigRecord(),
  saveHermesConfig: (config: unknown) => saveHermesConfig(config),
  setApiRequestProfile: vi.fn()
}))

const notifyError = vi.fn()

vi.mock('@/store/notifications', () => ({
  notifyError: (...args: unknown[]) => notifyError(...args)
}))

const PROVIDERS = [
  { models: ['deepseek-v4-pro', 'deepseek-chat'], name: 'DeepSeek', slug: 'deepseek' },
  { models: ['gemini-3.1-pro'], name: 'Google', slug: 'google' }
]

beforeEach(() => {
  $visibleModels.set(null)
  $collapsedProviders.set([])
  getGlobalModelOptions.mockResolvedValue({ providers: PROVIDERS })
  getHermesConfigRecord.mockResolvedValue({ agent: { reasoning_effort: 'high' } })
  saveHermesConfig.mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function renderDialog(profile?: string, client = new QueryClient({ defaultOptions: { queries: { retry: false } } })) {
  const result = render(
    <QueryClientProvider client={client}>
      <ModelVisibilityDialog onOpenChange={vi.fn()} onOpenProviders={vi.fn()} open profile={profile} />
    </QueryClientProvider>
  )

  return { ...result, client }
}

describe('ModelVisibilityDialog provider switch', () => {
  it('writes the provider into model_catalog.excluded_providers so every picker drops it', async () => {
    const content = renderDialog()

    const toggle = await content.findByRole('switch', { name: 'Enable DeepSeek' })
    fireEvent.click(toggle)

    // Config write, not a desktop-local preference — the backend builds the
    // catalog for the TUI and `hermes model` from this same key.
    await waitFor(() => {
      expect(saveHermesConfig).toHaveBeenCalledWith({
        agent: { reasoning_effort: 'high' },
        model_catalog: { excluded_providers: ['deepseek'] }
      })
    })

    // Optimistic: the model rows fold away without waiting for a refetch.
    expect(content.queryByText('Deepseek V4 Pro')).toBeNull()
    // Only that provider is off; the others keep their rows.
    expect(content.queryByText('Gemini 3.1 pro')).not.toBeNull()
  })

  it('keeps a row for an excluded provider the catalog no longer returns, so it can be switched back on', async () => {
    // The backend drops excluded providers from the payload — the only trace of
    // `copilot` is the config list.
    getHermesConfigRecord.mockResolvedValue({ model_catalog: { excluded_providers: ['copilot'] } })
    const content = renderDialog()

    const toggle = await content.findByRole('switch', { name: 'Enable copilot' })
    expect(toggle.getAttribute('data-state')).toBe('unchecked')

    fireEvent.click(toggle)

    // An explicit empty list, not a dropped key: PUT /api/config deep-merges,
    // so omitting it would leave `copilot` excluded on disk.
    await waitFor(() => {
      expect(saveHermesConfig).toHaveBeenCalledWith({ model_catalog: { excluded_providers: [] } })
    })
  })

  it('rolls the switch back and reports the failure when the config write fails', async () => {
    saveHermesConfig.mockRejectedValue(new Error('read-only managed install'))
    const content = renderDialog()

    const toggle = await content.findByRole('switch', { name: 'Enable DeepSeek' })
    fireEvent.click(toggle)

    await waitFor(() => {
      expect(notifyError).toHaveBeenCalled()
    })

    expect(content.getByRole('switch', { name: 'Enable DeepSeek' }).getAttribute('data-state')).toBe('checked')
    expect(content.queryByText('Deepseek V4 Pro')).not.toBeNull()
  })

  it('hides the select-all checkbox of a disabled provider (nothing to curate)', async () => {
    getHermesConfigRecord.mockResolvedValue({ model_catalog: { excluded_providers: ['deepseek'] } })
    const content = renderDialog()

    await content.findByText('DeepSeek')

    // One checkbox left: the enabled provider's select-all.
    expect(content.queryAllByRole('checkbox')).toHaveLength(1)
  })
})

// The config record is per-profile (`getHermesConfigRecord` targets the active
// API profile), and this dialog is mounted app-wide across profile switches.
// One shared cache key would paint — and then merge back — another profile's
// blocklist.
describe('ModelVisibilityDialog profile scoping', () => {
  // The app's real client caches for 60s (lib/query-client), so a remount
  // inside that window paints the cache with no refetch — which is exactly when
  // a shared key leaks the previous profile's data. A staleTime-0 client would
  // refetch on mount and hide the bug.
  const cachingClient = () => new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: 60_000 } } })

  it('does not show one profile’s exclusions after switching to another', async () => {
    const client = cachingClient()

    getHermesConfigRecord.mockResolvedValue({ model_catalog: { excluded_providers: ['copilot'] } })
    const alpha = renderDialog('alpha', client)
    await alpha.findByRole('switch', { name: 'Enable copilot' })
    alpha.unmount()

    // Switching profiles re-points the config endpoint at the other backend.
    getHermesConfigRecord.mockResolvedValue({ model_catalog: { excluded_providers: [] } })
    const beta = renderDialog('beta', client)
    await beta.findByText('DeepSeek')

    expect(beta.queryByRole('switch', { name: 'Enable copilot' })).toBeNull()
  })

  it('builds the write from a freshly read config, never a stale cached record', async () => {
    const client = cachingClient()

    getHermesConfigRecord.mockResolvedValue({ agent: { max_turns: 10 } })
    const first = renderDialog('alpha', client)
    await first.findByText('DeepSeek')
    first.unmount()

    // The backend's config moved on while the dialog was closed (another
    // settings page, the CLI, a profile switch). The cached record is stale.
    getHermesConfigRecord.mockResolvedValue({ agent: { max_turns: 99 } })
    const content = renderDialog('alpha', client)
    const toggle = await content.findByRole('switch', { name: 'Enable DeepSeek' })
    fireEvent.click(toggle)

    await waitFor(() => {
      expect(saveHermesConfig).toHaveBeenCalledWith({
        agent: { max_turns: 99 },
        model_catalog: { excluded_providers: ['deepseek'] }
      })
    })
  })
})

// Settings pages hold the config record under their own shared key and save it
// back whole. Leaving that cache untouched after this dialog writes means the
// next save from Settings would PUT a record whose `model_catalog` block predates
// the switch — silently re-enabling the provider.
describe('ModelVisibilityDialog cross-surface freshness', () => {
  it('invalidates the shared config-record cache after writing', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: 60_000 } } })

    client.setQueryData(HERMES_CONFIG_KEY, { agent: { max_turns: 10 } })

    const content = renderDialog('alpha', client)
    fireEvent.click(await content.findByRole('switch', { name: 'Enable DeepSeek' }))

    await waitFor(() => {
      expect(saveHermesConfig).toHaveBeenCalled()
    })

    await waitFor(() => {
      expect(client.getQueryState(HERMES_CONFIG_KEY)?.isInvalidated).toBe(true)
    })
  })
})
