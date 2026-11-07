// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { BrowserRealProfilePanel } from './browser-real-profile-panel'

const mocks = vi.hoisted(() => ({
  loadedConfig: {} as Record<string, unknown>,
  notify: vi.fn(),
  order: [] as string[],
  save: vi.fn(),
  sync: vi.fn()
}))

vi.mock('@/hermes', () => ({
  saveHermesConfigRecord: async (config: Record<string, unknown>) => {
    mocks.order.push(`save:${JSON.stringify(config)}`)

    return mocks.save(config)
  }
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      settings: {
        toolsets: {
          browserRealProfile: {
            label: 'Use My Real Browser Profile',
            description: 'd',
            enabledTitle: 'on',
            enabledMessage: 'on',
            disabledTitle: 'off',
            disabledMessage: 'off',
            failedSave: 'failed',
            reimportSignins: 'Re-import sign-ins',
            signinsImportedTitle: 'In-app browser sign-ins',
            signinsImported: (count: number) => `imported ${count}`,
            signinsRemote: 'remote',
            signinsFailed: 'sign-in import failed'
          }
        }
      }
    }
  })
}))

vi.mock('@/store/notifications', () => ({
  notify: (...args: unknown[]) => mocks.notify(...args),
  notifyError: vi.fn()
}))

vi.mock('../hooks/use-config-record', () => ({
  hermesConfigCacheWriter: () => () => undefined,
  useHermesConfigRecord: () => ({ data: mocks.loadedConfig, writeScope: { profile: 'work' } })
}))

describe('BrowserRealProfilePanel in-app browser sign-ins', () => {
  beforeEach(() => {
    mocks.order.length = 0
    mocks.save.mockResolvedValue({ ok: true })
    mocks.sync.mockImplementation(async (request: { enabled: boolean }) => {
      mocks.order.push(`sync:${request.enabled}`)

      return { status: 'imported', imported: 2, skipped: 0, failed: 0, cleared: 0 }
    })
    window.hermesDesktop = { syncPreviewSignins: mocks.sync } as unknown as typeof window.hermesDesktop
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('syncs the in-app browser only after the consent save lands, for the scope it was saved to', async () => {
    mocks.loadedConfig = { browser: {} }
    render(<BrowserRealProfilePanel />)

    await act(async () => {
      fireEvent.click(screen.getByRole('switch', { name: 'Use My Real Browser Profile' }))
    })

    // The backend re-checks the key before exporting, so a sync racing ahead of the save would read "off".
    expect(mocks.order).toEqual(['save:{"browser":{"use_real_profile":true}}', 'sync:true'])
    expect(mocks.sync).toHaveBeenCalledWith({ connectionId: null, enabled: true, profile: 'work' })
  })

  it('Re-import sign-ins asks main again and reports counts only', async () => {
    mocks.loadedConfig = { browser: { use_real_profile: true } }
    render(<BrowserRealProfilePanel />)

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Re-import sign-ins' }))
    })

    expect(mocks.sync).toHaveBeenCalledWith({ connectionId: null, enabled: true, profile: 'work' })
    expect(mocks.notify).toHaveBeenCalledWith(expect.objectContaining({ message: 'imported 2' }))
  })

  it('offers no Re-import while the toggle is off', () => {
    mocks.loadedConfig = { browser: { use_real_profile: false } }
    render(<BrowserRealProfilePanel />)

    expect(screen.queryByRole('button', { name: 'Re-import sign-ins' })).toBeNull()
  })
})
