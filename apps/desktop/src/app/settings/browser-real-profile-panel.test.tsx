// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { BrowserRealProfilePanel } from './browser-real-profile-panel'

const mocks = vi.hoisted(() => ({
  cache: vi.fn(),
  loadedConfig: {} as Record<string, unknown>,
  notify: vi.fn(),
  notifyError: vi.fn(),
  save: vi.fn(),
  schemaOptions: undefined as string[] | undefined
}))

vi.mock('@tanstack/react-query', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useQuery: () => ({
    data: mocks.schemaOptions
      ? { fields: { 'browser.real_profile_browser': { type: 'select', options: mocks.schemaOptions } } }
      : undefined
  })
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getHermesConfigSchema: vi.fn(),
  saveHermesConfigRecord: (config: Record<string, unknown>, profile?: unknown) => mocks.save(config, profile)
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      settings: {
        config: { systemDefault: 'System default' },
        toolsets: {
          browserRealProfile: {
            label: 'Use My Real Browser Profile',
            description: 'Copies your default browser profile into a managed snapshot.',
            enabledTitle: 'Real-profile browsing on',
            enabledMessage: 'New sessions use the snapshot.',
            disabledTitle: 'Real-profile browsing off',
            disabledMessage: 'Snapshot will be deleted.',
            failedSave: 'Could not save the real-profile setting',
            browserLabel: 'Browser to copy from',
            browserDescription: 'Which Chromium browser is snapshotted.',
            browserSavedTitle: 'Real-profile browser saved',
            browserSavedMessage: 'New sessions snapshot this browser.'
          }
        }
      }
    }
  })
}))

// Radix Select needs pointer/layout APIs jsdom lacks; a native <select> keeps
// the value/onValueChange contract the panel relies on.
vi.mock('@/components/ui/select', () => ({
  Select: ({
    children,
    onValueChange,
    value
  }: {
    children: ReactNode
    onValueChange: (v: string) => void
    value: string
  }) => (
    <select aria-label="Browser to copy from" onChange={e => onValueChange(e.target.value)} value={value}>
      {children}
    </select>
  ),
  SelectContent: ({ children }: { children: ReactNode }) => <>{children}</>,
  SelectItem: ({ children, value }: { children: ReactNode; value: string }) => (
    <option value={value}>{children}</option>
  ),
  SelectTrigger: () => null,
  SelectValue: () => null
}))

vi.mock('@/store/notifications', () => ({
  notify: (...args: unknown[]) => mocks.notify(...args),
  notifyError: (...args: unknown[]) => mocks.notifyError(...args)
}))

vi.mock('../hooks/use-config-record', () => ({
  hermesConfigCacheWriter: () => (config: Record<string, unknown>) => mocks.cache(config),
  useHermesConfigRecord: () => ({ data: mocks.loadedConfig })
}))

describe('BrowserRealProfilePanel', () => {
  beforeEach(() => {
    mocks.loadedConfig = { browser: { allow_private_urls: false }, model: { provider: 'nous' } }
    mocks.schemaOptions = undefined
    mocks.save.mockResolvedValue({ ok: true })
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders off for a config without the key and turns it on', async () => {
    render(<BrowserRealProfilePanel />)
    const toggle = screen.getByRole('switch', { name: 'Use My Real Browser Profile' })

    expect(toggle).toHaveProperty('ariaChecked', 'false')

    await act(async () => {
      fireEvent.click(toggle)
    })

    // Saves ONLY the toggled key (PUT deep-merges); the cache keeps the whole
    // merged record so sibling browser keys survive without a refetch.
    expect(mocks.save).toHaveBeenCalledWith({ browser: { use_real_profile: true } }, undefined)
    expect(mocks.cache).toHaveBeenCalledWith({
      browser: { allow_private_urls: false, use_real_profile: true },
      model: { provider: 'nous' }
    })
    expect(mocks.notify).toHaveBeenCalled()
  })

  it('turns an enabled toggle off', async () => {
    mocks.loadedConfig = { browser: { use_real_profile: true } }
    render(<BrowserRealProfilePanel />)
    const toggle = screen.getByRole('switch', { name: 'Use My Real Browser Profile' })

    expect(toggle).toHaveProperty('ariaChecked', 'true')

    await act(async () => {
      fireEvent.click(toggle)
    })

    expect(mocks.save).toHaveBeenCalledWith({ browser: { use_real_profile: false } }, undefined)
  })

  it('rolls the optimistic cache write back when the save fails', async () => {
    mocks.save.mockRejectedValue(new Error('boom'))
    render(<BrowserRealProfilePanel />)

    await act(async () => {
      fireEvent.click(screen.getByRole('switch', { name: 'Use My Real Browser Profile' }))
    })

    // Last cache write restores the original record.
    expect(mocks.cache).toHaveBeenLastCalledWith(mocks.loadedConfig)
    expect(mocks.notifyError).toHaveBeenCalled()
  })

  it('hides the browser selector while the toggle is off', () => {
    render(<BrowserRealProfilePanel />)

    expect(screen.queryByRole('combobox', { name: 'Browser to copy from' })).toBeNull()
  })

  it('saves the chosen real-profile browser sparsely', async () => {
    mocks.loadedConfig = { browser: { use_real_profile: true } }
    render(<BrowserRealProfilePanel />)
    const select = screen.getByRole('combobox', { name: 'Browser to copy from' })

    expect(select).toHaveProperty('value', '__hermes_empty__')
    expect(screen.getByRole('option', { name: 'Google Chrome' })).toBeTruthy()

    await act(async () => {
      fireEvent.change(select, { target: { value: 'chrome' } })
    })

    expect(mocks.save).toHaveBeenCalledWith({ browser: { real_profile_browser: 'chrome' } }, undefined)
    expect(mocks.cache).toHaveBeenCalledWith({ browser: { use_real_profile: true, real_profile_browser: 'chrome' } })
    expect(mocks.notify).toHaveBeenCalled()
  })

  it('maps System default back to an empty value', async () => {
    mocks.loadedConfig = { browser: { use_real_profile: true, real_profile_browser: 'brave' } }
    render(<BrowserRealProfilePanel />)

    await act(async () => {
      fireEvent.change(screen.getByRole('combobox', { name: 'Browser to copy from' }), {
        target: { value: '__hermes_empty__' }
      })
    })

    expect(mocks.save).toHaveBeenCalledWith({ browser: { real_profile_browser: '' } }, undefined)
  })

  it('lists only schema-reported (installed) browsers plus the saved one', () => {
    mocks.schemaOptions = ['', 'chrome', 'edge']
    mocks.loadedConfig = { browser: { use_real_profile: true, real_profile_browser: 'chromium' } }
    render(<BrowserRealProfilePanel />)

    const names = screen.getAllByRole('option').map(option => option.textContent)

    expect(names).toEqual(['System default', 'Google Chrome', 'Microsoft Edge', 'Chromium'])
  })
})
