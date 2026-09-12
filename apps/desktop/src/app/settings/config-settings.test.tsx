import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { createRef } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const getHermesConfigRecord = vi.fn()
const getHermesConfigSchema = vi.fn()
const saveHermesConfig = vi.fn()
const getElevenLabsVoices = vi.fn()
const scanAndRecordRepos = vi.hoisted(() => vi.fn())

vi.mock('@/hermes', () => ({
  getHermesConfigRecord: () => getHermesConfigRecord(),
  getHermesConfigSchema: () => getHermesConfigSchema(),
  saveHermesConfig: (config: unknown, profile?: string) => saveHermesConfig(config, profile),
  getElevenLabsVoices: () => getElevenLabsVoices(),
  getProfiles: async () => ({ profiles: [] }),
  profileScopeKey: (profile: string) => profile,
  setApiRequestProfile: () => {}
}))

vi.mock('../hooks/use-on-profile-switch', () => ({
  useOnProfileSwitch: () => {}
}))

// The real stores pull in the gateway/profile stack, which needs a live
// backend connection. This page only reads the "applies to" scope override
// and the repo-discovery signature, neither of which this test touches.
vi.mock('@/store/settings-scope', () => ({
  $settingsRequestProfile: atom('default'),
  $settingsScopeOverride: atom<null | string>(null)
}))

vi.mock('@/store/projects', () => ({
  repoDiscoveryPolicyFromConfig: (config: unknown) => config,
  repoDiscoveryPolicySignature: (policy: unknown) => JSON.stringify(policy),
  scanAndRecordRepos
}))

beforeEach(async () => {
  const { $settingsRequestProfile, $settingsScopeOverride } = await import('@/store/settings-scope')
  $settingsRequestProfile.set('default')
  $settingsScopeOverride.set(null)
  getElevenLabsVoices.mockResolvedValue({ available: false })
  getHermesConfigSchema.mockResolvedValue({ fields: {} })
  saveHermesConfig.mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function renderConfigSettings() {
  const { ConfigSettings } = await import('./config-settings')
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const importInputRef = createRef<HTMLInputElement>()

  render(
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <ConfigSettings activeSectionId="safety" importInputRef={importInputRef} />
      </QueryClientProvider>
    </MemoryRouter>
  )

  return { importInputRef }
}

describe('ConfigSettings autosave', () => {
  it('sends a later revert instead of diffing it away against the stale page-load baseline', async () => {
    getHermesConfigRecord.mockResolvedValue({ checkpoints: { enabled: false }, other: 'untouched' })

    vi.useFakeTimers({ shouldAdvanceTime: true })

    try {
      await renderConfigSettings()

      const toggle = await screen.findByRole('switch')

      // Edit: flip checkpoints.enabled on, let the debounced autosave fire.
      toggle.click()
      await vi.advanceTimersByTimeAsync(700)

      await vi.waitFor(() => expect(saveHermesConfig).toHaveBeenCalledTimes(1))
      expect(saveHermesConfig.mock.calls[0][0]).toEqual({ checkpoints: { enabled: true } })

      // Revert: flip it back to its original value and let autosave fire again.
      toggle.click()
      await vi.advanceTimersByTimeAsync(700)

      await vi.waitFor(() => expect(saveHermesConfig).toHaveBeenCalledTimes(2))
      // Must still explicitly send the reverted value — diffing against the
      // never-advanced page-load baseline would produce an empty patch here
      // (the field is back to its original value) and leave disk stuck at
      // `enabled: true` from the first save.
      expect(saveHermesConfig.mock.calls[1][0]).toEqual({ checkpoints: { enabled: false } })
      expect(scanAndRecordRepos).toHaveBeenCalledTimes(2)
    } finally {
      vi.useRealTimers()
    }
  })

  it('does not rescan active-profile repositories while editing another profile', async () => {
    const { $settingsRequestProfile, $settingsScopeOverride } = await import('@/store/settings-scope')
    $settingsRequestProfile.set('profile-b')
    $settingsScopeOverride.set('profile-b')
    getHermesConfigRecord.mockResolvedValue({ checkpoints: { enabled: false } })
    vi.useFakeTimers({ shouldAdvanceTime: true })

    try {
      await renderConfigSettings()
      const toggle = await screen.findByRole('switch')
      toggle.click()
      await vi.advanceTimersByTimeAsync(700)

      await vi.waitFor(() => expect(saveHermesConfig).toHaveBeenCalledOnce())
      expect(scanAndRecordRepos).not.toHaveBeenCalled()
    } finally {
      vi.useRealTimers()
    }
  })
})
