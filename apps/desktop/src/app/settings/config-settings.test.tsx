import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { createRef } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as ConfigApi from '@/api/config'
import type * as Notifications from '@/store/notifications'
import { notifyError } from '@/store/notifications'
import { $settingsRequestProfile } from '@/store/settings-scope'
import { $agentSidebarChoice } from '@/store/simplified-agent-mode'

import { type ConfigSettingsDraft, ConfigSettingsDraftContext } from './config-draft'
import type { ConfigSettings as ConfigSettingsType } from './config-settings'
import { SettingsOwnerContext } from './profile-scope'

// The vi.mock factory below replaces the computed (read-only) atom with a
// writable one; narrow the import back so tests can drive it.
const scopeProfileMock = $settingsRequestProfile as unknown as { set: (value: string) => void }

const getHermesConfigRecord = vi.fn()
const getHermesConfigSchema = vi.fn()
const saveHermesConfig = vi.fn()
const getElevenLabsVoices = vi.fn()

// Keep the real read-origin helpers (WeakMap peek/bind) live: the shared
// config hook reaches them through the barrel, and a bare mock would throw.
vi.mock('@/hermes', async () => ({
  ...(await vi.importActual<typeof ConfigApi>('@/api/config')),
  // use-config-record folds the scope into its cache key via the barrel; the
  // real one is a pure string fold, mirrored here for the string scopes this
  // suite passes.
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

vi.mock('@/store/notifications', async importOriginal => ({
  ...(await importOriginal<typeof Notifications>()),
  notifyError: vi.fn()
}))

// The real stores pull in the gateway/profile stack, which needs a live
// backend connection. This page only reads the "applies to" scope override
// and the repo-discovery signature, neither of which this test touches. The
// scope chip it renders also reads the selected profile and the loud-note
// selector, so those are stubbed to the single-profile default shape.
vi.mock('@/store/settings-scope', () => ({
  // The real store derives this from the displayed $settingsScopeProfile
  // (never undefined for a real profile — settings-scope.test.ts pins that);
  // here it is a plain atom so the page's threading of it can be driven.
  $settingsRequestProfile: atom<string | undefined>('default'),
  $settingsScopeEditsNonDefault: atom(false),
  $settingsScopeOverride: atom<null | string>(null),
  $settingsScopeProfile: atom<string>('default')
}))

vi.mock('@/store/projects', () => ({
  $worktreeRefreshToken: atom(0),
  repoDiscoveryPolicyFromConfig: () => ({ enabled: true, roots: [], exclude_paths: [] }),
  repoDiscoveryPolicySignature: (policy: unknown) => JSON.stringify(policy),
  scanAndRecordRepos: vi.fn().mockResolvedValue(undefined)
}))

// The module graph behind ConfigSettings is large (1.5s cold here, >10s on a
// saturated CI runner); load it once under the hook timeout so the 15s test
// budget is spent on the autosave behaviour, not on transform + import.
let ConfigSettings: typeof ConfigSettingsType

beforeAll(async () => {
  ;({ ConfigSettings } = await import('./config-settings'))
}, 60_000)

beforeEach(() => {
  scopeProfileMock.set('default')
  getElevenLabsVoices.mockResolvedValue({ available: false })
  getHermesConfigSchema.mockResolvedValue({ fields: {} })
  saveHermesConfig.mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  $agentSidebarChoice.set(null)
  vi.clearAllMocks()
})

function renderConfigSettings(activeSectionId = 'safety') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const importInputRef = createRef<HTMLInputElement>()

  render(
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <ConfigSettings
          activeSectionId={activeSectionId}
          importInputRef={importInputRef}
          subpage={activeSectionId === 'safety' ? 'checkpoints' : undefined}
        />
      </QueryClientProvider>
    </MemoryRouter>
  )

  return { importInputRef }
}

describe.each([false, true])('ConfigSettings autosave, simplified mode %s', simplified => {
  beforeEach(() => $agentSidebarChoice.set(simplified))

  it.each(['record', 'schema'])('recovers from a failed %s read without writing settings', async failedRead => {
    scopeProfileMock.set('research')
    getHermesConfigRecord.mockResolvedValue({ checkpoints: { enabled: false } })
    const failedRequest = failedRead === 'record' ? getHermesConfigRecord : getHermesConfigSchema
    failedRequest.mockRejectedValueOnce(new Error('Gateway unavailable'))

    renderConfigSettings()
    fireEvent.click(await screen.findByRole('button', { name: 'Retry' }))

    expect((await screen.findByRole('switch')).getAttribute('aria-checked')).toBe('false')
    expect(failedRequest).toHaveBeenCalledTimes(2)
    expect(getHermesConfigRecord.mock.calls.every(([profile]) => profile === 'research')).toBe(true)
    expect(saveHermesConfig).not.toHaveBeenCalled()
  })

  it('keeps the draft after a failed save and retries the edit on its original profile', async () => {
    scopeProfileMock.set('research')
    getHermesConfigRecord.mockResolvedValue({ checkpoints: { enabled: false } })
    const failure = new Error('Gateway unavailable')
    saveHermesConfig.mockRejectedValueOnce(failure)
    vi.useFakeTimers({ shouldAdvanceTime: true })

    try {
      renderConfigSettings()
      const toggle = await screen.findByRole('switch')
      fireEvent.click(toggle)
      await vi.advanceTimersByTimeAsync(700)
      await vi.waitFor(() => expect(notifyError).toHaveBeenCalledWith(failure, expect.any(String)))
      expect(toggle.getAttribute('aria-checked')).toBe('true')

      fireEvent.click(toggle)
      fireEvent.click(toggle)
      await vi.advanceTimersByTimeAsync(700)
      await vi.waitFor(() => expect(saveHermesConfig).toHaveBeenCalledTimes(2))
      expect(saveHermesConfig.mock.calls).toEqual([
        [{ checkpoints: { enabled: true } }, 'research'],
        [{ checkpoints: { enabled: true } }, 'research']
      ])
      expect(toggle.getAttribute('aria-checked')).toBe('true')
    } finally {
      vi.useRealTimers()
    }
  })

  it('sends a later revert instead of diffing it away against the stale page-load baseline', async () => {
    getHermesConfigRecord.mockResolvedValue({ checkpoints: { enabled: false }, other: 'untouched' })

    vi.useFakeTimers({ shouldAdvanceTime: true })

    try {
      renderConfigSettings()

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
    } finally {
      vi.useRealTimers()
    }
  })

  it('threads the "Applies to" request scope into both the config read and the autosave write', async () => {
    // #118432: the request scope is the concrete profile the page displays
    // (see settings-scope.test.ts). The page must carry it into the read AND
    // the write — a read scoped to B with a write that falls back to the
    // ambient (launch) profile is exactly the silent cross-profile write.
    scopeProfileMock.set('nash')
    getHermesConfigRecord.mockResolvedValue({ checkpoints: { enabled: false } })

    vi.useFakeTimers({ shouldAdvanceTime: true })

    try {
      renderConfigSettings()

      await vi.waitFor(() => expect(getHermesConfigRecord).toHaveBeenCalledWith('nash'))

      ;(await screen.findByRole('switch')).click()
      await vi.advanceTimersByTimeAsync(700)

      await vi.waitFor(() => expect(saveHermesConfig).toHaveBeenCalledWith({ checkpoints: { enabled: true } }, 'nash'))
    } finally {
      vi.useRealTimers()
    }
  })
})

it('keeps a pending agent edit pinned when the global settings scope changes', async () => {
  const owner = { connectionId: 'agent-gateway', profile: 'research' }
  getHermesConfigRecord.mockResolvedValue({ checkpoints: { enabled: false } })
  vi.useFakeTimers({ shouldAdvanceTime: true })

  try {
    render(
      <MemoryRouter>
        <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
          <SettingsOwnerContext.Provider value={owner}>
            <ConfigSettings
              activeSectionId="safety"
              importInputRef={createRef<HTMLInputElement>()}
              subpage="checkpoints"
            />
          </SettingsOwnerContext.Provider>
        </QueryClientProvider>
      </MemoryRouter>
    )
    fireEvent.click(await screen.findByRole('switch'))
    scopeProfileMock.set('other-agent')
    await vi.advanceTimersByTimeAsync(700)
    await vi.waitFor(() => expect(saveHermesConfig).toHaveBeenCalledTimes(1))
    expect(getHermesConfigRecord).toHaveBeenCalledWith(owner)
    expect(saveHermesConfig).toHaveBeenCalledWith({ checkpoints: { enabled: true } }, owner)
  } finally {
    vi.useRealTimers()
  }
})

it('does not publish a completed save after its owner editor unmounts', async () => {
  const owner = { connectionId: 'agent-gateway', profile: 'research' }
  const saved = vi.fn()
  const pending = Promise.withResolvers<{ ok: boolean }>()
  getHermesConfigRecord.mockResolvedValue({ checkpoints: { enabled: false } })
  saveHermesConfig.mockReturnValueOnce(pending.promise)
  vi.useFakeTimers({ shouldAdvanceTime: true })

  try {
    const view = render(
      <MemoryRouter>
        <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
          <SettingsOwnerContext.Provider value={owner}>
            <ConfigSettings
              activeSectionId="safety"
              importInputRef={createRef<HTMLInputElement>()}
              onConfigSaved={saved}
              subpage="checkpoints"
            />
          </SettingsOwnerContext.Provider>
        </QueryClientProvider>
      </MemoryRouter>
    )

    fireEvent.click(await screen.findByRole('switch'))
    await vi.advanceTimersByTimeAsync(700)
    expect(saveHermesConfig).toHaveBeenCalledWith({ checkpoints: { enabled: true } }, owner)
    view.unmount()
    pending.resolve({ ok: true })
    await vi.advanceTimersByTimeAsync(1)
    expect(saved).not.toHaveBeenCalled()
  } finally {
    vi.useRealTimers()
  }
})

it.each([false, true])(
  'retains an agent draft and serializes saves across presentation remounts (in flight: %s)',
  async inFlight => {
    const owner = { connectionId: 'agent-gateway', profile: 'research' }
    const draft = createRef<ConfigSettingsDraft | null>()
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const pending = Promise.withResolvers<{ ok: boolean }>()
    getHermesConfigRecord.mockResolvedValue({ checkpoints: { enabled: false }, untouched: 'server' })

    if (inFlight) {
      saveHermesConfig.mockReturnValueOnce(pending.promise)
    }

    const editor = (presentation: string) => (
      <MemoryRouter>
        <QueryClientProvider client={client}>
          <SettingsOwnerContext.Provider value={owner}>
            <ConfigSettingsDraftContext.Provider value={draft}>
              <ConfigSettings
                activeSectionId="safety"
                importInputRef={createRef<HTMLInputElement>()}
                key={presentation}
                subpage="checkpoints"
              />
            </ConfigSettingsDraftContext.Provider>
          </SettingsOwnerContext.Provider>
        </QueryClientProvider>
      </MemoryRouter>
    )
    vi.useFakeTimers({ shouldAdvanceTime: true })

    try {
      const view = render(editor('redesigned'))
      fireEvent.click(await screen.findByRole('switch'))

      if (inFlight) {
        await vi.advanceTimersByTimeAsync(700)
        expect(saveHermesConfig).toHaveBeenCalledTimes(1)
      }

      scopeProfileMock.set('other-agent')
      view.rerender(editor('classic'))
      const toggle = await screen.findByRole('switch')
      expect(toggle.getAttribute('aria-checked')).toBe('true')

      if (inFlight) {
        fireEvent.click(toggle)
        await vi.advanceTimersByTimeAsync(700)
        expect(saveHermesConfig).toHaveBeenCalledTimes(1)
        pending.resolve({ ok: true })
        await vi.waitFor(() => expect(saveHermesConfig).toHaveBeenCalledTimes(2))
        expect(saveHermesConfig.mock.calls[1]).toEqual([{ checkpoints: { enabled: false } }, owner])
      } else {
        await vi.advanceTimersByTimeAsync(700)
        await vi.waitFor(() => expect(saveHermesConfig).toHaveBeenCalledTimes(1))
      }

      expect(saveHermesConfig.mock.calls[0]).toEqual([{ checkpoints: { enabled: true } }, owner])
    } finally {
      vi.useRealTimers()
    }
  }
)

it('keeps compact Limits destinations distinct when moving between runtime, terminal and output', async () => {
  $agentSidebarChoice.set(true)
  getHermesConfigRecord.mockResolvedValue({
    agent: { max_turns: 42 },
    terminal: { timeout: 17 },
    tool_output: { max_bytes: 1234 }
  })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const input = createRef<HTMLInputElement>()
  const editor = (subpage: string) => (
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <SettingsOwnerContext.Provider value={{ connectionId: 'local', profile: 'research' }}>
          <ConfigSettings activeSectionId="advanced" agentLayout importInputRef={input} subpage={subpage} />
        </SettingsOwnerContext.Provider>
      </QueryClientProvider>
    </MemoryRouter>
  )
  const view = render(editor('runtime'))
  expect(await screen.findByDisplayValue('42')).toBeTruthy()
  expect(screen.queryByDisplayValue('17')).toBeNull()
  expect(screen.queryByDisplayValue('1234')).toBeNull()
  view.rerender(editor('terminal'))
  expect(await screen.findByDisplayValue('17')).toBeTruthy()
  expect(screen.queryByDisplayValue('42')).toBeNull()
  expect(screen.queryByDisplayValue('1234')).toBeNull()
  view.rerender(editor('output'))
  expect(await screen.findByDisplayValue('1234')).toBeTruthy()
  expect(screen.queryByDisplayValue('42')).toBeNull()
  expect(screen.queryByDisplayValue('17')).toBeNull()
})
