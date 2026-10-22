// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import {
  $realProfilePromptClaim,
  $realProfilePromptDismissed,
  $realProfilePromptMuted
} from '@/store/real-profile-consent'
import { $connection } from '@/store/session'

import { RealProfileConsentDialog, shouldOfferRealProfilePrompt } from './real-profile-consent-dialog'

const mocks = vi.hoisted(() => ({
  cache: vi.fn(),
  loadedConfig: {} as Record<string, unknown> | undefined,
  notify: vi.fn(),
  notifyError: vi.fn(),
  save: vi.fn()
}))

vi.mock('@/hermes', () => ({
  saveHermesConfigRecord: (config: Record<string, unknown>, profile?: unknown) => mocks.save(config, profile)
}))

const promptCopy = {
  title: 'Stay signed in to your sites',
  body: 'Let Hermes browse with a snapshot of your default browser profile.',
  bulletSnapshot: 'Cookies and logins are copied into a managed snapshot.',
  bulletLiveProfile: 'Your live browser profile is never opened directly.',
  bulletLocal: 'Nothing leaves this computer.',
  dontShowAgain: "Don't show again",
  notNow: 'Not now',
  enable: 'Use my profile'
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

const localConnection = { mode: 'local' } as HermesConnection
const remoteConnection = { mode: 'remote', remoteKind: 'ssh' } as HermesConnection

const openGate = {
  claim: 'tab-1',
  configLoaded: true,
  connection: localConnection,
  dismissed: false,
  enabled: false,
  muted: false,
  tabId: 'tab-1'
}

describe('shouldOfferRealProfilePrompt', () => {
  it('offers the prompt for a local backend with the feature off', () => {
    expect(shouldOfferRealProfilePrompt(openGate)).toBe(true)
  })

  it('never offers it for a remote backend (#119398)', () => {
    for (const remoteKind of ['ssh', 'url', 'cloud'] as const) {
      expect(
        shouldOfferRealProfilePrompt({ ...openGate, connection: { mode: 'remote', remoteKind } as HermesConnection })
      ).toBe(false)
    }
  })

  it('fails closed while the connection is unresolved', () => {
    expect(shouldOfferRealProfilePrompt({ ...openGate, connection: null })).toBe(false)
    expect(shouldOfferRealProfilePrompt({ ...openGate, connection: {} as HermesConnection })).toBe(false)
  })

  it('keeps the existing gates', () => {
    expect(shouldOfferRealProfilePrompt({ ...openGate, configLoaded: false })).toBe(false)
    expect(shouldOfferRealProfilePrompt({ ...openGate, enabled: true })).toBe(false)
    expect(shouldOfferRealProfilePrompt({ ...openGate, dismissed: true })).toBe(false)
    expect(shouldOfferRealProfilePrompt({ ...openGate, muted: true })).toBe(false)
    expect(shouldOfferRealProfilePrompt({ ...openGate, claim: 'tab-2' })).toBe(false)
  })
})

describe('RealProfileConsentDialog', () => {
  beforeEach(() => {
    mocks.loadedConfig = { browser: { allow_private_urls: false }, model: { provider: 'nous' } }
    mocks.save.mockResolvedValue({ ok: true })
    $realProfilePromptDismissed.set(false)
    $realProfilePromptMuted.set(false)
    $realProfilePromptClaim.set(null)
    $connection.set(localConnection)
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('shows when the feature is off and accepting writes browser.use_real_profile', async () => {
    render(<RealProfileConsentDialog tabId="tab-1" />)

    expect(screen.getByText(promptCopy.title)).toBeTruthy()

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: promptCopy.enable }))
    })

    // Saves ONLY the toggled key (PUT deep-merges) — the same shape the
    // Capabilities toggle writes — while the shared cache gets the merged
    // record so the existing toggle flips on without a refetch.
    expect(mocks.save).toHaveBeenCalledWith({ browser: { use_real_profile: true } }, undefined)
    expect(mocks.cache).toHaveBeenCalledWith({
      browser: { allow_private_urls: false, use_real_profile: true },
      model: { provider: 'nous' }
    })
    expect(mocks.notify).toHaveBeenCalled()
  })

  it('does not show when real-profile browsing is already on', () => {
    mocks.loadedConfig = { browser: { use_real_profile: true } }
    render(<RealProfileConsentDialog tabId="tab-1" />)

    expect(screen.queryByText(promptCopy.title)).toBeNull()
  })

  it('does not show, or write, for a remote backend (#119398)', () => {
    $connection.set(remoteConnection)
    render(<RealProfileConsentDialog tabId="tab-1" />)

    expect(screen.queryByText(promptCopy.title)).toBeNull()
    expect(mocks.save).not.toHaveBeenCalled()
  })

  it('does not show before the config record loads', () => {
    mocks.loadedConfig = undefined
    render(<RealProfileConsentDialog tabId="tab-1" />)

    expect(screen.queryByText(promptCopy.title)).toBeNull()
  })

  it('"Not now" mutes for the app run without persisting the opt-out', () => {
    render(<RealProfileConsentDialog tabId="tab-1" />)

    fireEvent.click(screen.getByRole('button', { name: promptCopy.notNow }))

    expect(screen.queryByText(promptCopy.title)).toBeNull()
    expect($realProfilePromptMuted.get()).toBe(true)
    expect($realProfilePromptDismissed.get()).toBe(false)
    expect(mocks.save).not.toHaveBeenCalled()
  })

  it('"Don\'t show again" persists the opt-out', () => {
    render(<RealProfileConsentDialog tabId="tab-1" />)

    fireEvent.click(screen.getByRole('button', { name: promptCopy.dontShowAgain }))

    expect(screen.queryByText(promptCopy.title)).toBeNull()
    expect($realProfilePromptDismissed.get()).toBe(true)
    expect(mocks.save).not.toHaveBeenCalled()
  })

  it('only the claiming pane renders the dialog when several Browser panes mount', () => {
    render(
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
    render(<RealProfileConsentDialog tabId="tab-1" />)

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: promptCopy.enable }))
    })

    expect(mocks.cache).toHaveBeenLastCalledWith(mocks.loadedConfig)
    expect(mocks.notifyError).toHaveBeenCalled()
  })
})
