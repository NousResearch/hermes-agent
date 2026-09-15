// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $chatFontFamily, resolveChatFontFamily } from '@/themes/chat-font'

import { ChatFontSetting } from './chat-font-setting'

const mocks = vi.hoisted(() => ({
  cache: vi.fn(),
  loadedConfig: {} as Record<string, unknown>,
  notifyError: vi.fn(),
  profileSwitch: null as null | (() => void),
  refetch: vi.fn(),
  save: vi.fn()
}))

vi.mock('@/hermes', () => ({
  saveHermesConfig: (config: Record<string, unknown>) => mocks.save(config)
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
  useHermesConfigRecord: () => ({ data: mocks.loadedConfig, refetch: mocks.refetch })
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
    mocks.loadedConfig = { desktop: { font_family: '', repo_scan_enabled: true } }
    mocks.refetch.mockResolvedValue({ data: mocks.loadedConfig, isSuccess: true })
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

    expect(mocks.save).toHaveBeenCalledWith({ desktop: { font_family: 'OpenDyslexic' } })
    expect(mocks.cache).toHaveBeenCalledWith({ desktop: { font_family: 'OpenDyslexic', repo_scan_enabled: true } })
  })

  it('reseeds when a profile refetch returns the same config object', async () => {
    const config = { desktop: { font_family: 'Avenir' } }
    mocks.loadedConfig = config
    mocks.refetch.mockResolvedValue({ data: config, isSuccess: true })
    render(<ChatFontSetting />)

    expect((screen.getByRole('combobox', { name: 'Chat Font' }) as HTMLInputElement).value).toBe('Avenir')

    await act(async () => {
      mocks.profileSwitch?.()
      await Promise.resolve()
    })

    const input = screen.getByRole('combobox', { name: 'Chat Font' }) as HTMLInputElement
    expect(mocks.refetch).toHaveBeenCalledWith({ cancelRefetch: false })
    expect(input.disabled).toBe(false)
    expect(input.value).toBe('Avenir')
    expect($chatFontFamily.get()).toBe('Avenir')
  })

  it('stays reset when a failed refetch retains stale data', async () => {
    const config = { desktop: { font_family: 'Avenir' } }
    mocks.loadedConfig = config
    mocks.refetch.mockResolvedValue({ data: config, isSuccess: false })
    render(<ChatFontSetting />)

    await act(async () => {
      mocks.profileSwitch?.()
      await Promise.resolve()
    })

    const input = screen.getByRole('combobox', { name: 'Chat Font' }) as HTMLInputElement
    expect(input.disabled).toBe(true)
    expect(input.value).toBe('')
    expect($chatFontFamily.get()).toBe('')
  })

  it('ignores a profile refetch that completes after unmount', async () => {
    const config = { desktop: { font_family: 'Avenir' } }
    let resolveRefetch!: (value: { data: typeof config; isSuccess: true }) => void
    mocks.loadedConfig = config
    mocks.refetch.mockReturnValue(
      new Promise(resolve => {
        resolveRefetch = resolve
      })
    )
    const view = render(<ChatFontSetting />)

    act(() => mocks.profileSwitch?.())
    view.unmount()
    $chatFontFamily.set('Lexend')
    await act(async () => {
      resolveRefetch({ data: config, isSuccess: true })
      await Promise.resolve()
    })

    expect($chatFontFamily.get()).toBe('Lexend')
  })

  it('ignores a late refetch from an older profile switch', async () => {
    const firstProfile = { desktop: { font_family: 'Avenir' } }
    const secondProfile = { desktop: { font_family: 'Lexend' } }
    let resolveFirst!: (value: { data: typeof firstProfile; isSuccess: true }) => void
    let resolveSecond!: (value: { data: typeof secondProfile; isSuccess: true }) => void
    mocks.refetch
      .mockReturnValueOnce(
        new Promise(resolve => {
          resolveFirst = resolve
        })
      )
      .mockReturnValueOnce(
        new Promise(resolve => {
          resolveSecond = resolve
        })
      )
    render(<ChatFontSetting />)

    act(() => mocks.profileSwitch?.())
    act(() => mocks.profileSwitch?.())

    await act(async () => {
      resolveSecond({ data: secondProfile, isSuccess: true })
      await Promise.resolve()
    })
    expect($chatFontFamily.get()).toBe('Lexend')

    await act(async () => {
      resolveFirst({ data: firstProfile, isSuccess: true })
      await Promise.resolve()
    })
    expect($chatFontFamily.get()).toBe('Lexend')
    expect((screen.getByRole('combobox', { name: 'Chat Font' }) as HTMLInputElement).value).toBe('Lexend')
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
