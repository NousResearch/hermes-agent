import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { createRef } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as ConfigApi from '@/api/config'

import type { ConfigSettings as ConfigSettingsType } from './config-settings'

const getHermesConfigRecord = vi.fn()
const getHermesConfigSchema = vi.fn()
const saveHermesConfig = vi.fn()
const getElevenLabsVoices = vi.fn()

vi.mock('@/hermes', async () => ({
  ...(await vi.importActual<typeof ConfigApi>('@/api/config')),
  profileScopeKey: (scope?: unknown) => (typeof scope === 'string' && scope.trim()) || 'default',
  getHermesConfigRecord: (profile?: string) => getHermesConfigRecord(profile),
  getHermesConfigSchema: () => getHermesConfigSchema(),
  saveHermesConfig: (config: unknown, profile?: string) => saveHermesConfig(config, profile),
  getElevenLabsVoices: () => getElevenLabsVoices(),
  setApiRequestProfile: () => {}
}))

vi.mock('../hooks/use-on-profile-switch', () => ({
  useOnProfileSwitch: () => {}
}))

vi.mock('@/store/settings-scope', () => ({
  $settingsRequestProfile: atom<string | undefined>('default'),
  $settingsScopeEditsNonDefault: atom(false),
  $settingsScopeOverride: atom<null | string>(null),
  $settingsScopeProfile: atom<string>('default')
}))

vi.mock('@/store/projects', () => ({
  repoDiscoveryPolicyFromConfig: () => ({ enabled: true, roots: [], exclude_paths: [] }),
  repoDiscoveryPolicySignature: (policy: unknown) => JSON.stringify(policy),
  scanAndRecordRepos: vi.fn().mockResolvedValue(undefined)
}))

let ConfigSettings: typeof ConfigSettingsType

beforeAll(async () => {
  ;({ ConfigSettings } = await import('./config-settings'))
}, 60_000)

beforeEach(() => {
  getElevenLabsVoices.mockResolvedValue({ available: false })
  getHermesConfigSchema.mockResolvedValue({ fields: {} })
  saveHermesConfig.mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function renderConfigSettings(activeSectionId = 'safety') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const importInputRef = createRef<HTMLInputElement>()

  render(
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <ConfigSettings activeSectionId={activeSectionId} importInputRef={importInputRef} />
      </QueryClientProvider>
    </MemoryRouter>
  )
}

describe('ConfigSettings managed-scope fields (#135859)', () => {
  it('renders a pinned leaf read-only with the managed source in the hint', async () => {
    getHermesConfigRecord.mockResolvedValue({ checkpoints: { enabled: false } })
    getHermesConfigSchema.mockResolvedValue({
      fields: { 'checkpoints.enabled': { type: 'boolean' } },
      managed_keys: ['checkpoints.enabled'],
      managed_source: '/etc/hermes'
    })

    renderConfigSettings()

    const toggle = await screen.findByRole('switch')
    // The whole field sits inside a disabled fieldset instead of a toggle the
    // save would silently drop.
    expect(toggle.closest('fieldset')?.disabled).toBe(true)
    expect(screen.getByText(/Managed by your administrator \(\/etc\/hermes\) — read-only/)).toBeTruthy()
  })

  it('keeps unpinned fields editable when a managed scope is active', async () => {
    getHermesConfigRecord.mockResolvedValue({ checkpoints: { enabled: false } })
    getHermesConfigSchema.mockResolvedValue({
      fields: { 'checkpoints.enabled': { type: 'boolean' } },
      managed_keys: ['model.default'],
      managed_source: '/etc/hermes'
    })

    renderConfigSettings()

    const toggle = await screen.findByRole('switch')
    expect(toggle.closest('fieldset')?.disabled).toBeUndefined()
  })
})
