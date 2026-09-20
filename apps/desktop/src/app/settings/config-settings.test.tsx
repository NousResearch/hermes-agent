import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { createRef } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ConfigSettings as ConfigSettingsType } from './config-settings'

const getHermesConfigRecord = vi.fn()
const getHermesConfigSchema = vi.fn()
const saveHermesConfigRecord = vi.fn()
const getElevenLabsVoices = vi.fn()

vi.mock('@/hermes', () => ({
  getHermesConfigRecord: () => getHermesConfigRecord(),
  getHermesConfigSchema: () => getHermesConfigSchema(),
  saveHermesConfigRecord: (config: unknown, owner: unknown) => saveHermesConfigRecord(config, owner),
  getElevenLabsVoices: () => getElevenLabsVoices(),
  setApiRequestProfile: () => {}
}))

// The real stores pull in the gateway/profile stack, which needs a live
// backend connection. This page only reads the "applies to" scope, the
// connection descriptor and the repo-discovery signature, none of which this
// test touches. The scope chip it renders also reads the selected profile and
// the loud-note selector, so those are stubbed to the single-profile default shape.
vi.mock('@/store/session', () => ({ $connection: atom(null) }))
vi.mock('@/store/settings-scope', () => ({
  $settingsRequestProfile: atom<string | undefined>(undefined),
  $settingsScopeEditsNonDefault: atom(false),
  $settingsScopeOverride: atom<null | string>(null),
  $settingsScopeProfile: atom<string>('default')
}))
vi.mock('@/store/projects', () => ({
  repoDiscoveryPolicyFromConfig: () => ({ enabled: true, roots: [], exclude_paths: [] }),
  repoDiscoveryPolicySignature: (policy: unknown) => JSON.stringify(policy),
  scanAndRecordRepos: vi.fn().mockResolvedValue(undefined)
}))
// The provider list has its own tests; here it only reports which owner it was handed.
vi.mock('./memory/provider-settings', () => ({
  MemoryProviderSettings: ({ profile }: { profile: { connectionId: string | null; profile: string } }) => (
    <output data-testid="providers">{`${profile.connectionId} / ${profile.profile}`}</output>
  )
}))

// The module graph behind ConfigSettings is large (1.5s cold here, >10s on a
// saturated CI runner); load it once under the hook timeout so the 15s test
// budget is spent on the tests.
let ConfigSettings: typeof ConfigSettingsType

beforeAll(async () => {
  ;({ ConfigSettings } = await import('./config-settings'))
}, 60_000)

beforeEach(() => {
  getElevenLabsVoices.mockResolvedValue({ available: false })
  getHermesConfigSchema.mockResolvedValue({ fields: { 'compression.codex_gpt55_autoraise': { type: 'boolean' } } })
  saveHermesConfigRecord.mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function renderConfigSettings(activeSectionId = 'memory') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  render(
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <ConfigSettings activeSectionId={activeSectionId} importInputRef={createRef<HTMLInputElement>()} />
      </QueryClientProvider>
    </MemoryRouter>
  )
}

describe('ConfigSettings', () => {
  it('renders the memory provider list for the owner while the record is still loading', () => {
    getHermesConfigRecord.mockReturnValue(new Promise(() => {}))
    renderConfigSettings()
    expect(screen.getByTestId('providers').textContent).toBe('null / default')
    expect(screen.queryByRole('switch')).toBeNull()
  })

  it('saves the Codex compression auto-raise edit to the settings owner, then the revert against the saved baseline', async () => {
    getHermesConfigRecord.mockResolvedValue({ compression: { codex_gpt55_autoraise: true } })
    vi.useFakeTimers({ shouldAdvanceTime: true })

    try {
      renderConfigSettings()
      expect(await screen.findByText('Codex Compression Auto-Raise')).toBeTruthy()
      screen.getByRole('switch').click()
      await vi.advanceTimersByTimeAsync(700)
      await vi.waitFor(() =>
        expect(saveHermesConfigRecord).toHaveBeenCalledWith(
          { compression: { codex_gpt55_autoraise: false } },
          { connectionId: null, profile: 'default' }
        )
      )
      // Reverting must send a real patch; diffing against the page-load baseline would send nothing.
      screen.getByRole('switch').click()
      await vi.advanceTimersByTimeAsync(700)
      await vi.waitFor(() => expect(saveHermesConfigRecord).toHaveBeenCalledTimes(2))
      expect(saveHermesConfigRecord.mock.calls[1][0]).toEqual({ compression: { codex_gpt55_autoraise: true } })
    } finally {
      vi.useRealTimers()
    }
  })
})
