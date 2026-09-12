// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $realProfilePromptClaim,
  $realProfilePromptDismissed,
  $realProfilePromptMuted
} from '@/store/real-profile-consent'

import { RealProfileConsentDialog } from './real-profile-consent-dialog'

const mocks = vi.hoisted(() => ({
  cache: vi.fn(),
  candidates: vi.fn(),
  loadedConfig: {} as Record<string, unknown> | undefined,
  notify: vi.fn(),
  notifyError: vi.fn(),
  save: vi.fn()
}))

vi.mock('@/hermes', () => ({
  getBrowserRealProfile: () => mocks.candidates(),
  profileScopeKey: (profile?: unknown) => String(profile ?? 'active'),
  saveHermesConfigRecord: (config: Record<string, unknown>, profile?: unknown) => mocks.save(config, profile)
}))

const promptCopy = {
  title: 'Stay signed in to your sites',
  body: 'Let Hermes browse with a snapshot of your default browser profile.',
  bulletSnapshot: 'Cookies and logins are copied into a managed snapshot.',
  bulletLiveProfile: 'Your live browser profile is never opened directly.',
  bulletLocal: 'Nothing leaves this computer.',
  bulletTarget: (target: string) => `Uses ${target} — change it in Settings any time.`,
  dontShowAgain: "Don't show again",
  notNow: 'Not now',
  enable: 'Use my profile'
}

const pickerCopy = {
  browsingAs: (browser: string, profile: string) => `Browsing as ${browser} · ${profile}`
}

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { close: 'Close' },
      settings: {
        toolsets: {
          browserRealProfile: {
            enabledTitle: 'Real-profile browsing on',
            enabledMessage: 'New sessions use the snapshot.',
            failedSave: 'Could not save the real-profile setting',
            picker: pickerCopy,
            prompt: promptCopy
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

vi.mock('../../hooks/use-config-record', () => ({
  hermesConfigCacheWriter: () => (config: Record<string, unknown>) => mocks.cache(config),
  useHermesConfigRecord: () => ({ data: mocks.loadedConfig })
}))

describe('RealProfileConsentDialog', () => {
  /** The dialog now names the identity it is about to copy, so it fetches the
   *  resolved browser/profile — which needs a query provider in the tree. */
  function renderDialog(ui: ReactNode) {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })

    return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>)
  }

  beforeEach(() => {
    mocks.loadedConfig = { browser: { allow_private_urls: false }, model: { provider: 'nous' } }
    mocks.save.mockResolvedValue({ ok: true })
    mocks.candidates.mockResolvedValue({
      supported: true,
      platform: 'Linux',
      detected_default: 'brave',
      detected_unsupported_channel: false,
      resolved_browser: 'brave',
      resolved_profile: 'Profile 2',
      pinned_browser: null,
      pinned_profile: null,
      error: null,
      browsers: [
        {
          key: 'brave',
          label: 'Brave',
          installed: true,
          has_profile: true,
          is_system_default: true,
          data_dir: '/home/u/.config/BraveSoftware/Brave-Browser',
          profiles: [{ directory: 'Profile 2', name: 'Personal', last_used: true }]
        }
      ]
    })
    $realProfilePromptDismissed.set(false)
    $realProfilePromptMuted.set(false)
    $realProfilePromptClaim.set(null)
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('names the browser and profile it will copy, so consent is informed', async () => {
    renderDialog(<RealProfileConsentDialog tabId="tab-1" />)

    // Resolved server-side from the same keys the picker writes, so the dialog
    // can never advertise a different identity than the launch will use.
    expect(await screen.findByText(promptCopy.bulletTarget('Browsing as Brave · Personal'))).toBeTruthy()
  })

  it('still renders when the identity cannot be resolved', async () => {
    mocks.candidates.mockRejectedValue(new Error('no browsers'))
    renderDialog(<RealProfileConsentDialog tabId="tab-1" />)

    // Discovery is decoration on this dialog — a failure must not block consent.
    expect(screen.getByText(promptCopy.title)).toBeTruthy()
    expect(screen.getByRole('button', { name: promptCopy.enable })).toBeTruthy()
  })

  it('shows when the feature is off and accepting writes browser.use_real_profile', async () => {
    renderDialog(<RealProfileConsentDialog tabId="tab-1" />)

    expect(screen.getByText(promptCopy.title)).toBeTruthy()

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: promptCopy.enable }))
    })

    // Saves the WHOLE merged record with only use_real_profile added — the
    // same shape the Capabilities toggle writes, through the same cache, so
    // the existing toggle flips on without a refetch.
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

  it('does not show when real-profile browsing is already on', () => {
    mocks.loadedConfig = { browser: { use_real_profile: true } }
    renderDialog(<RealProfileConsentDialog tabId="tab-1" />)

    expect(screen.queryByText(promptCopy.title)).toBeNull()
  })

  it('does not show before the config record loads', () => {
    mocks.loadedConfig = undefined
    renderDialog(<RealProfileConsentDialog tabId="tab-1" />)

    expect(screen.queryByText(promptCopy.title)).toBeNull()
  })

  it('"Not now" mutes for the app run without persisting the opt-out', () => {
    renderDialog(<RealProfileConsentDialog tabId="tab-1" />)

    fireEvent.click(screen.getByRole('button', { name: promptCopy.notNow }))

    expect(screen.queryByText(promptCopy.title)).toBeNull()
    expect($realProfilePromptMuted.get()).toBe(true)
    expect($realProfilePromptDismissed.get()).toBe(false)
    expect(mocks.save).not.toHaveBeenCalled()
  })

  it('"Don\'t show again" persists the opt-out', () => {
    renderDialog(<RealProfileConsentDialog tabId="tab-1" />)

    fireEvent.click(screen.getByRole('button', { name: promptCopy.dontShowAgain }))

    expect(screen.queryByText(promptCopy.title)).toBeNull()
    expect($realProfilePromptDismissed.get()).toBe(true)
    expect(mocks.save).not.toHaveBeenCalled()
  })

  it('only the claiming pane renders the dialog when several Browser panes mount', () => {
    renderDialog(
      <>
        <RealProfileConsentDialog tabId="tab-1" />
        <RealProfileConsentDialog tabId="tab-2" />
      </>
    )

    expect(screen.getAllByText(promptCopy.title)).toHaveLength(1)
    expect($realProfilePromptClaim.get()).toBe('tab-1')
  })

  it('rolls the optimistic cache write back when the save fails', async () => {
    mocks.save.mockRejectedValue(new Error('boom'))
    renderDialog(<RealProfileConsentDialog tabId="tab-1" />)

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: promptCopy.enable }))
    })

    expect(mocks.cache).toHaveBeenLastCalledWith(mocks.loadedConfig)
    expect(mocks.notifyError).toHaveBeenCalled()
  })
})
