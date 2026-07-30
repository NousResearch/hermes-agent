import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { StrictMode } from 'react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'
import { $activeGatewayProfile } from '@/store/profile'
import type { HermesConfigRecord } from '@/types/hermes'

import { HERMES_CONFIG_KEY } from '../hooks/use-config-record'

const getHermesConfigRecord = vi.fn()
const getHermesConfigSchema = vi.fn()
const saveHermesConfig = vi.fn()
const getElevenLabsVoices = vi.fn()

vi.mock('@/hermes', () => ({
  getHermesConfigRecord: () => getHermesConfigRecord(),
  getHermesConfigSchema: () => getHermesConfigSchema(),
  saveHermesConfig: (config: unknown) => saveHermesConfig(config),
  getElevenLabsVoices: () => getElevenLabsVoices(),
  getProfiles: async () => ({ profiles: [] }),
  setApiRequestProfile: () => {},
  STARTUP_REQUEST_TIMEOUT_MS: 1000
}))

vi.mock('@/store/projects', () => ({
  repoDiscoveryPolicyFromConfig: (config: HermesConfigRecord) => config?.desktop ?? {},
  repoDiscoveryPolicySignature: (policy: unknown) => JSON.stringify(policy ?? null),
  scanAndRecordRepos: vi.fn()
}))

// Heavy neighbours that aren't under test.
vi.mock('./model-settings', () => ({
  ModelSettings: () => null,
  ModelSettingsSkeleton: () => null
}))
vi.mock('./memory/connect', () => ({ MemoryConnect: () => null }))
vi.mock('./memory/provider-config-panel', () => ({ ProviderConfigPanel: () => null }))
vi.mock('./quick-entry-settings', () => ({ QuickEntrySettings: () => null }))

const workspaceConfig = (cwd: string): HermesConfigRecord => ({ terminal: { cwd } })

const SCHEMA = {
  fields: {
    'terminal.cwd': { type: 'string', description: 'Default project folder.' }
  }
}

beforeEach(() => {
  getHermesConfigRecord.mockImplementation(async () => workspaceConfig('.'))
  getHermesConfigSchema.mockResolvedValue(structuredClone(SCHEMA))
  saveHermesConfig.mockResolvedValue({ ok: true })
  getElevenLabsVoices.mockResolvedValue({ available: false, voices: [] })
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  $activeGatewayProfile.set('default')
  vi.clearAllMocks()
})

async function renderWorkspaceSettings() {
  const { ConfigSettings } = await import('./config-settings')

  return render(
    // StrictMode is load-bearing: the app runs under it, and the regression
    // this file guards (profile-switch hook double-fire wiping the draft)
    // only reproduces with Strict Mode's second effect pass.
    <StrictMode>
      <MemoryRouter>
        <QueryClientProvider client={queryClient}>
          <ConfigSettings activeSectionId="workspace" importInputRef={{ current: null }} />
        </QueryClientProvider>
      </MemoryRouter>
    </StrictMode>
  )
}

describe('ConfigSettings draft seeding', () => {
  it('keeps the seeded draft under StrictMode when the record is already cached', async () => {
    // Warm cache = the live repro: another surface fetched the config before
    // Settings opened, so the seed happens on mount and the (old) first-flag
    // profile-switch hook wiped it on Strict Mode's second effect pass —
    // permanent skeleton, because the refetch structural-shares the same
    // reference and the seed effect never re-ran.
    queryClient.setQueryData(HERMES_CONFIG_KEY, workspaceConfig('.'))

    await renderWorkspaceSettings()

    await screen.findByText('Working Directory')

    // Let the staleTime-0 background refetch (deep-equal payload) settle.
    await waitFor(() => expect(getHermesConfigRecord).toHaveBeenCalled())
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 25))
    })

    expect(screen.queryByText('Working Directory')).not.toBeNull()
    expect(screen.getByDisplayValue('.')).toBeTruthy()
  })

  it('re-seeds the other profile record after a profile switch, without autosaving', async () => {
    await renderWorkspaceSettings()
    await screen.findByText('Working Directory')
    expect(screen.getByDisplayValue('.')).toBeTruthy()

    getHermesConfigRecord.mockImplementation(async () => workspaceConfig('/profile-b'))

    act(() => {
      $activeGatewayProfile.set('coder')
    })

    // Draft cleared + record hard-reset → refetch lands profile B's config and
    // the empty draft re-seeds from it.
    await screen.findByDisplayValue('/profile-b')

    // The switch itself must never write config (that would cross-contaminate).
    expect(saveHermesConfig).not.toHaveBeenCalled()
  })

  it('re-seeds after a profile switch even when both profiles have deep-equal configs', async () => {
    await renderWorkspaceSettings()
    await screen.findByText('Working Directory')

    const callsBeforeSwitch = getHermesConfigRecord.mock.calls.length

    // Profile B's record has identical content. Without a hard cache reset,
    // React Query structural sharing keeps the SAME object reference across
    // the refetch, a reference-keyed seed effect never re-runs, and the page
    // is a skeleton forever. The state-derived seed + resetQueries must
    // recover regardless of reference identity.
    act(() => {
      $activeGatewayProfile.set('coder')
    })

    await waitFor(() => expect(getHermesConfigRecord.mock.calls.length).toBeGreaterThan(callsBeforeSwitch))
    await waitFor(() => expect(screen.getByDisplayValue('.')).toBeTruthy())
  })
})
