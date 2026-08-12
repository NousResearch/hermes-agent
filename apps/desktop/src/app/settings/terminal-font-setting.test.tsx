// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $terminalFontFamily } from '../right-sidebar/terminal/terminal-font'

import { TerminalFontSetting } from './terminal-font-setting'

const mocks = vi.hoisted(() => ({
  cache: vi.fn(),
  loadedConfig: {} as Record<string, unknown>,
  notifyError: vi.fn(),
  profileSwitch: null as null | (() => void),
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
          terminalFontDesc: 'Choose an installed font.',
          terminalFontPlaceholder: 'MesloLGS NF or a CSS font stack',
          terminalFontPreview: 'Glyph preview',
          terminalFontReset: 'Use default',
          terminalFontTitle: 'Terminal Font'
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
  useHermesConfigRecord: () => ({ data: mocks.loadedConfig })
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

describe('TerminalFontSetting', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    mocks.loadedConfig = {
      display: { skin: 'hermes' },
      terminal: { backend: 'local', cwd: '/workspace', font_family: '' }
    }
    mocks.save.mockResolvedValue({ ok: true })
    mocks.profileSwitch = null
    $terminalFontFamily.set('')
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
    vi.useRealTimers()
  })

  it('selects MesloLGS NF and persists only the terminal font field', async () => {
    render(<TerminalFontSetting />)
    const input = screen.getByRole('combobox', { name: 'Terminal Font' })

    fireEvent.change(input, { target: { value: 'MesloLGS NF' } })

    expect($terminalFontFamily.get()).toBe('MesloLGS NF')
    expect((screen.getByLabelText('Glyph preview') as HTMLElement).style.fontFamily).toContain('MesloLGS NF')

    await flushAutosave()

    expect(mocks.save).toHaveBeenCalledWith({
      display: { skin: 'hermes' },
      terminal: { backend: 'local', cwd: '/workspace', font_family: 'MesloLGS NF' }
    })
    expect(mocks.cache).toHaveBeenCalledWith(mocks.save.mock.calls[0][0])
  })

  it('accepts an arbitrary CSS stack and resets to the bundled default', async () => {
    mocks.loadedConfig = {
      terminal: { backend: 'local', font_family: "'Hack Nerd Font', monospace" }
    }
    render(<TerminalFontSetting />)
    const input = screen.getByRole('combobox', { name: 'Terminal Font' })

    expect((input as HTMLInputElement).value).toBe("'Hack Nerd Font', monospace")
    fireEvent.change(input, { target: { value: "'Custom Powerline', monospace" } })
    await flushAutosave()

    expect(mocks.save.mock.calls[0][0]).toMatchObject({
      terminal: { backend: 'local', font_family: "'Custom Powerline', monospace" }
    })

    fireEvent.click(screen.getByRole('button', { name: 'Use default' }))
    expect($terminalFontFamily.get()).toBe('')
    expect((screen.getByLabelText('Glyph preview') as HTMLElement).style.fontFamily).toContain('JetBrains Mono')
    await flushAutosave()

    expect(mocks.save.mock.calls[1][0]).toMatchObject({
      terminal: { backend: 'local', font_family: '' }
    })
  })

  it('rolls back the optimistic font when autosave fails', async () => {
    mocks.loadedConfig = { terminal: { font_family: 'MesloLGS NF' } }
    mocks.save.mockRejectedValue(new Error('disk full'))
    render(<TerminalFontSetting />)
    const input = screen.getByRole('combobox', { name: 'Terminal Font' })

    fireEvent.change(input, { target: { value: 'Hack Nerd Font' } })
    expect($terminalFontFamily.get()).toBe('Hack Nerd Font')
    await flushAutosave()

    expect((input as HTMLInputElement).value).toBe('MesloLGS NF')
    expect($terminalFontFamily.get()).toBe('MesloLGS NF')
    expect(mocks.notifyError).toHaveBeenCalledWith(expect.any(Error), 'Autosave failed')
  })

  it('drops the prior profile font and reseeds from the next profile', () => {
    mocks.loadedConfig = { terminal: { font_family: 'MesloLGS NF' } }
    const view = render(<TerminalFontSetting />)

    expect($terminalFontFamily.get()).toBe('MesloLGS NF')
    act(() => mocks.profileSwitch?.())
    expect($terminalFontFamily.get()).toBe('')
    expect((screen.getByRole('combobox', { name: 'Terminal Font' }) as HTMLInputElement).disabled).toBe(true)

    mocks.loadedConfig = { terminal: { font_family: 'Hack Nerd Font' } }
    view.rerender(<TerminalFontSetting />)

    expect((screen.getByRole('combobox', { name: 'Terminal Font' }) as HTMLInputElement).value).toBe('Hack Nerd Font')
    expect($terminalFontFamily.get()).toBe('Hack Nerd Font')
  })
})
