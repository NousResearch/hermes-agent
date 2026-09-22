// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $chatFontFamily, resolveChatFontFamily } from '@/themes/chat-font'

import { ChatFontSetting } from './chat-font-setting'

const mocks = vi.hoisted(() => ({
  cache: vi.fn(),
  configUpdatedAt: 1,
  loadedConfig: {} as Record<string, unknown>,
  notifyError: vi.fn(),
  profileSwitch: null as null | (() => void),
  save: vi.fn(),
  writeScope: { connectionId: 'connection-a', profile: 'default' }
}))

vi.mock('@/hermes', () => ({
  saveHermesConfig: (config: Record<string, unknown>, scope?: unknown) => mocks.save(config, scope)
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      settings: {
        appearance: {
          chatFontDesc: 'Choose a font.',
          chatFontPlaceholder: 'OpenDyslexic or a CSS font stack',
          chatFontPreview: 'Preview',
          chatFontReset: 'Use theme font',
          chatFontSample: 'The quick brown fox',
          chatFontTitle: 'Chat Font'
        },
        config: { autosaveFailed: 'Autosave failed' }
      }
    }
  })
}))

vi.mock('@/store/notifications', () => ({
  notifyError: (...args: unknown[]) => mocks.notifyError(...args)
}))

vi.mock('../hooks/use-config-record', () => ({
  setHermesConfigCache: (config: Record<string, unknown>) => mocks.cache(config),
  useHermesConfigRecord: () => ({ data: mocks.loadedConfig, dataUpdatedAt: mocks.configUpdatedAt, writeScope: mocks.writeScope })
}))

vi.mock('../hooks/use-on-profile-switch', () => ({
  useOnProfileSwitch: (callback: () => void) => {
    mocks.profileSwitch = callback
  }
}))

async function flushAutosave() {
  await act(async () => {
    vi.advanceTimersByTime(550)
    await Promise.resolve()
    await Promise.resolve()
  })
}

describe('ChatFontSetting', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    mocks.configUpdatedAt = 1
    mocks.loadedConfig = { desktop: { font_family: '', repo_scan_enabled: true } }
    mocks.save.mockResolvedValue({ ok: true })
    mocks.profileSwitch = null
    $chatFontFamily.set('')
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
    vi.useRealTimers()
  })

  it('publishes the live family and persists only desktop.font_family, keeping sibling keys', async () => {
    render(<ChatFontSetting />)

    fireEvent.change(screen.getByRole('combobox', { name: 'Chat Font' }), { target: { value: 'OpenDyslexic' } })
    expect($chatFontFamily.get()).toBe('OpenDyslexic')

    await flushAutosave()

    expect(mocks.save).toHaveBeenCalledWith(
      { desktop: { font_family: 'OpenDyslexic' } },
      { connectionId: 'connection-a', profile: 'default' }
    )
    expect(mocks.cache).toHaveBeenCalledWith({ desktop: { font_family: 'OpenDyslexic', repo_scan_enabled: true } })
  })

  it('rolls back the optimistic family when autosave fails', async () => {
    mocks.loadedConfig = { desktop: { font_family: 'Lexend' } }
    mocks.save.mockRejectedValue(new Error('disk full'))
    render(<ChatFontSetting />)
    const input = screen.getByRole('combobox', { name: 'Chat Font' }) as HTMLInputElement

    fireEvent.change(input, { target: { value: 'OpenDyslexic' } })
    await flushAutosave()

    expect(input.value).toBe('Lexend')
    expect($chatFontFamily.get()).toBe('Lexend')
    expect(mocks.notifyError).toHaveBeenCalledWith(expect.any(Error), 'Autosave failed')
  })

  it('reseeds after a profile refetch reuses the cached config record', () => {
    const sharedConfig = { desktop: { font_family: 'Avenir' } }
    mocks.loadedConfig = sharedConfig
    const view = render(<ChatFontSetting />)

    expect($chatFontFamily.get()).toBe('Avenir')
    act(() => mocks.profileSwitch?.())
    expect($chatFontFamily.get()).toBe('')
    expect((screen.getByRole('combobox', { name: 'Chat Font' }) as HTMLInputElement).disabled).toBe(true)

    mocks.configUpdatedAt = 2
    view.rerender(<ChatFontSetting />)

    expect((screen.getByRole('combobox', { name: 'Chat Font' }) as HTMLInputElement).value).toBe('Avenir')
    expect($chatFontFamily.get()).toBe('Avenir')
  })
})

describe('resolveChatFontFamily', () => {
  const theme = '"Segoe UI", system-ui, sans-serif'

  it('layers a bare family in front of the theme stack and leaves the theme alone when empty', () => {
    expect(resolveChatFontFamily('', theme)).toBe(theme)
    expect(resolveChatFontFamily('  ', theme)).toBe(theme)
    expect(resolveChatFontFamily('OpenDyslexic', theme)).toBe(`'OpenDyslexic', ${theme}`)
    expect(resolveChatFontFamily("'Atkinson Hyperlegible', serif", theme)).toBe(
      `'Atkinson Hyperlegible', serif, ${theme}`
    )
  })
})
