// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { BrowserRealProfilePanel, describeResolved } from './browser-real-profile-panel'

// Radix Select calls scrollIntoView / pointer-capture APIs jsdom lacks.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

const mocks = vi.hoisted(() => ({
  cache: vi.fn(),
  candidates: vi.fn(),
  loadedConfig: {} as Record<string, unknown>,
  notify: vi.fn(),
  notifyError: vi.fn(),
  save: vi.fn()
}))

vi.mock('@/hermes', () => ({
  getBrowserRealProfile: (profile?: unknown) => mocks.candidates(profile),
  profileScopeKey: (profile?: unknown) => String(profile ?? 'active'),
  saveHermesConfigRecord: (config: Record<string, unknown>, profile?: unknown) => mocks.save(config, profile)
}))

const pickerCopy = {
  browserLabel: 'Browser',
  browserDescription: 'Which installed browser the agent borrows logins from.',
  profileLabel: 'Browser profile',
  profileDescription: 'Which profile inside that browser.',
  systemDefault: 'System default',
  systemDefaultNamed: (browser: string) => `System default (${browser})`,
  lastUsed: 'Last used',
  lastUsedNamed: (profile: string) => `Last used (${profile})`,
  notInstalled: 'Not installed',
  noProfile: 'Never launched',
  loading: 'Looking for browsers…',
  failedLoad: 'Could not list browsers on this machine',
  browsingAs: (browser: string, profile: string) => `Browsing as ${browser} · ${profile}`,
  unsupportedPlatform: (platform: string) => `Not available on ${platform}.`,
  savedTitle: 'Browsing identity updated',
  savedMessage: (target: string) => `New sessions will browse as ${target}.`
}

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      settings: {
        toolsets: {
          browserRealProfile: {
            label: 'Use My Real Browser Profile',
            description: 'Copies your browser profile into a managed snapshot.',
            enabledTitle: 'Real-profile browsing on',
            enabledMessage: 'New sessions use the snapshot.',
            disabledTitle: 'Real-profile browsing off',
            disabledMessage: 'Snapshot will be deleted.',
            failedSave: 'Could not save the real-profile setting',
            picker: pickerCopy
          }
        }
      }
    }
  })
}))

vi.mock('@/store/notifications', () => ({
  notify: (...args: unknown[]) => mocks.notify(...args),
  notifyError: (...args: unknown[]) => mocks.notifyError(...args)
}))

vi.mock('../hooks/use-config-record', () => ({
  hermesConfigCacheWriter: () => (config: Record<string, unknown>) => mocks.cache(config),
  useHermesConfigRecord: () => ({ data: mocks.loadedConfig })
}))

/** Two browsers, one uninstalled — the shape the picker has to survive. */
function candidatesFixture(overrides: Record<string, unknown> = {}) {
  return {
    supported: true,
    platform: 'Linux',
    detected_default: 'brave',
    detected_unsupported_channel: false,
    resolved_browser: 'brave',
    resolved_profile: 'Profile 1',
    pinned_browser: null,
    pinned_profile: null,
    error: null,
    browsers: [
      {
        key: 'chrome',
        label: 'Google Chrome',
        installed: false,
        has_profile: false,
        is_system_default: false,
        data_dir: '',
        profiles: []
      },
      {
        key: 'brave',
        label: 'Brave',
        installed: true,
        has_profile: true,
        is_system_default: true,
        data_dir: '/home/u/.config/BraveSoftware/Brave-Browser',
        profiles: [
          { directory: 'Profile 1', name: 'Me', last_used: true },
          { directory: 'Profile 2', name: 'Personal', last_used: false }
        ]
      }
    ],
    ...overrides
  }
}

function renderPanel(ui: ReactNode) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>)
}

const toggleOn = { browser: { use_real_profile: true } }

describe('BrowserRealProfilePanel', () => {
  beforeEach(() => {
    mocks.loadedConfig = { browser: { allow_private_urls: false }, model: { provider: 'nous' } }
    mocks.save.mockResolvedValue({ ok: true })
    mocks.candidates.mockResolvedValue(candidatesFixture())
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders off for a config without the key and turns it on', async () => {
    renderPanel(<BrowserRealProfilePanel />)
    const toggle = screen.getByRole('switch', { name: 'Use My Real Browser Profile' })

    expect(toggle).toHaveProperty('ariaChecked', 'false')

    await act(async () => {
      fireEvent.click(toggle)
    })

    // Saves the WHOLE merged record with only use_real_profile added — sibling
    // browser keys survive.
    expect(mocks.save).toHaveBeenCalledWith(
      {
        browser: { allow_private_urls: false, use_real_profile: true },
        model: { provider: 'nous' }
      },
      undefined
    )
    expect(mocks.cache).toHaveBeenCalledWith(mocks.save.mock.calls[0][0])
    expect(mocks.notify).toHaveBeenCalled()
  })

  it('turns an enabled toggle off', async () => {
    mocks.loadedConfig = toggleOn
    renderPanel(<BrowserRealProfilePanel />)
    const toggle = screen.getByRole('switch', { name: 'Use My Real Browser Profile' })

    expect(toggle).toHaveProperty('ariaChecked', 'true')

    await act(async () => {
      fireEvent.click(toggle)
    })

    expect(mocks.save).toHaveBeenCalledWith({ browser: { use_real_profile: false } }, undefined)
  })

  it('rolls the optimistic cache write back when the save fails', async () => {
    mocks.save.mockRejectedValue(new Error('boom'))
    renderPanel(<BrowserRealProfilePanel />)

    await act(async () => {
      fireEvent.click(screen.getByRole('switch', { name: 'Use My Real Browser Profile' }))
    })

    // Last cache write restores the original record.
    expect(mocks.cache).toHaveBeenLastCalledWith(mocks.loadedConfig)
    expect(mocks.notifyError).toHaveBeenCalled()
  })

  it('does not probe for browsers until consent is on', () => {
    renderPanel(<BrowserRealProfilePanel />)

    // Enumerating browser profiles is a filesystem walk on the gateway host;
    // it must not run for users who never opted in.
    expect(mocks.candidates).not.toHaveBeenCalled()
    expect(screen.queryByText(pickerCopy.browserLabel)).toBeNull()
  })

  it('shows the browser and profile pickers once enabled', async () => {
    mocks.loadedConfig = toggleOn
    renderPanel(<BrowserRealProfilePanel />)

    expect(await screen.findByText(pickerCopy.browserLabel)).toBeTruthy()
    expect(screen.getByText(pickerCopy.profileLabel)).toBeTruthy()
    // Names the identity a launch would actually use, not "your default browser".
    expect(screen.getByText(pickerCopy.browsingAs('Brave', 'Me'))).toBeTruthy()
  })

  it('writes real_profile_pin when a profile is chosen', async () => {
    mocks.loadedConfig = toggleOn
    renderPanel(<BrowserRealProfilePanel />)

    await screen.findByText(pickerCopy.profileLabel)

    fireEvent.click(screen.getAllByRole('combobox')[1])
    await act(async () => {
      fireEvent.click(await screen.findByRole('option', { name: /Personal/ }))
    })

    expect(mocks.save).toHaveBeenCalledWith(
      { browser: { use_real_profile: true, real_profile_pin: 'Profile 2' } },
      undefined
    )
  })

  it('clears a stale profile pin when the browser changes', async () => {
    // A pin names a directory inside ONE browser's user-data dir; carrying
    // "Profile 2" from Brave to Chrome would fail closed on the next launch.
    mocks.loadedConfig = { browser: { use_real_profile: true, real_profile_pin: 'Profile 2' } }
    mocks.candidates.mockResolvedValue(
      candidatesFixture({
        browsers: [
          {
            key: 'chrome',
            label: 'Google Chrome',
            installed: true,
            has_profile: true,
            is_system_default: false,
            data_dir: '/home/u/.config/google-chrome',
            profiles: [{ directory: 'Default', name: 'Work', last_used: true }]
          }
        ]
      })
    )
    renderPanel(<BrowserRealProfilePanel />)

    await screen.findByText(pickerCopy.browserLabel)

    fireEvent.click(screen.getAllByRole('combobox')[0])
    await act(async () => {
      fireEvent.click(await screen.findByRole('option', { name: /Google Chrome/ }))
    })

    expect(mocks.save).toHaveBeenCalledWith(
      { browser: { use_real_profile: true, real_profile_browser: 'chrome', real_profile_pin: '' } },
      undefined
    )
  })

  it('writes an empty string (not the sentinel) when reverting to the system default', async () => {
    mocks.loadedConfig = { browser: { use_real_profile: true, real_profile_browser: 'brave' } }
    renderPanel(<BrowserRealProfilePanel />)

    await screen.findByText(pickerCopy.browserLabel)

    fireEvent.click(screen.getAllByRole('combobox')[0])
    await act(async () => {
      fireEvent.click(await screen.findByRole('option', { name: /System default/ }))
    })

    // The config contract is "" = follow the OS default; the UI sentinel must
    // never reach config.yaml, where the resolver would reject it.
    expect(mocks.save).toHaveBeenCalledWith(
      { browser: { use_real_profile: true, real_profile_browser: '', real_profile_pin: '' } },
      undefined
    )
  })

  it('surfaces the backend fail-closed error instead of silently ignoring it', async () => {
    mocks.loadedConfig = { browser: { use_real_profile: true, real_profile_pin: 'Profile 99' } }
    mocks.candidates.mockResolvedValue(
      candidatesFixture({
        resolved_profile: null,
        error: "browser.real_profile_pin is set to 'Profile 99' but that profile does not exist."
      })
    )
    renderPanel(<BrowserRealProfilePanel />)

    // Two surfaces on purpose: the trigger keeps showing the pin the config
    // still holds, and the error explains why it isn't being honored.
    const shown = await screen.findAllByText(/Profile 99/)

    expect(shown.length).toBeGreaterThan(1)
    expect(shown.some(node => node.textContent?.includes('does not exist'))).toBe(true)
  })

  it('reports platforms where real-profile browsing cannot work', async () => {
    mocks.loadedConfig = toggleOn
    mocks.candidates.mockResolvedValue(candidatesFixture({ supported: false, platform: 'FreeBSD' }))
    renderPanel(<BrowserRealProfilePanel />)

    expect(await screen.findByText(pickerCopy.unsupportedPlatform('FreeBSD'))).toBeTruthy()
    // No pickers on an unsupported platform — nothing to pick.
    expect(screen.queryByText(pickerCopy.browserLabel)).toBeNull()
  })

  it('keeps the toggle usable when browser discovery fails', async () => {
    mocks.loadedConfig = toggleOn
    mocks.candidates.mockRejectedValue(new Error('probe failed'))
    renderPanel(<BrowserRealProfilePanel />)

    expect(await screen.findByText(pickerCopy.failedLoad)).toBeTruthy()
    expect(screen.getByRole('switch', { name: 'Use My Real Browser Profile' })).toBeTruthy()
  })

  it('scopes the discovery fetch to the panel profile', async () => {
    // Per-profile identities are the point: the Capabilities scope selector
    // configuring another profile must read THAT profile's resolution.
    mocks.loadedConfig = toggleOn
    renderPanel(<BrowserRealProfilePanel profile="omar" />)

    await waitFor(() => expect(mocks.candidates).toHaveBeenCalledWith('omar'))
  })
})

describe('describeResolved', () => {
  it('prefers display names over directory names', () => {
    expect(describeResolved(candidatesFixture() as never, pickerCopy)).toBe('Browsing as Brave · Me')
  })

  it('is null until both halves of the identity are known', () => {
    expect(describeResolved(undefined, pickerCopy)).toBeNull()
    expect(describeResolved(candidatesFixture({ resolved_profile: null }) as never, pickerCopy)).toBeNull()
    expect(describeResolved(candidatesFixture({ resolved_browser: null }) as never, pickerCopy)).toBeNull()
  })

  it('falls back to raw keys when the browser reports no label', () => {
    const data = candidatesFixture({ resolved_browser: 'edge', resolved_profile: 'Profile 7' })

    expect(describeResolved(data as never, pickerCopy)).toBe('Browsing as edge · Profile 7')
  })
})
