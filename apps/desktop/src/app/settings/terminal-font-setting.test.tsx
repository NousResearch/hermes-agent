// @vitest-environment jsdom
import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'
import { $settingsOwner } from '@/store/settings-scope'

import { $terminalFontFamily } from '../right-sidebar/terminal/terminal-font'

import { TerminalFontSetting } from './terminal-font-setting'

const $configRevision = atom(0)

const mocks = vi.hoisted(() => ({
  cache: vi.fn(),
  configUpdatedAt: 1,
  loadedConfig: {} as Record<string, unknown>,
  notifyError: vi.fn(),
  save: vi.fn()
}))

vi.mock('@/hermes', () => ({
  saveHermesConfig: (...args: unknown[]) => mocks.save(...args),
  setApiRequestProfile: vi.fn()
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
  hermesConfigCacheWriter: (scope: unknown) => (config: Record<string, unknown>) => mocks.cache(config, scope),
  useHermesConfigRecord: () => {
    useStore($configRevision)

    return { data: mocks.loadedConfig, dataUpdatedAt: mocks.configUpdatedAt }
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
    $activeGatewayProfile.set('default')
    mocks.configUpdatedAt = 1
    mocks.loadedConfig = {
      display: { skin: 'hermes' },
      terminal: { backend: 'local', cwd: '/workspace', font_family: '' }
    }
    mocks.save.mockResolvedValue({ ok: true })
    $connection.set({ baseUrl: 'http://127.0.0.1:3000', connectionId: 'local', mode: 'local' } as never)
    $terminalFontFamily.set('')
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
    vi.useRealTimers()
    $connection.set(null)
  })

  it('selects MesloLGS NF and persists only the terminal font field', async () => {
    render(<TerminalFontSetting />)
    const input = screen.getByRole('combobox', { name: 'Terminal Font' })

    fireEvent.change(input, { target: { value: 'MesloLGS NF' } })

    expect($terminalFontFamily.get()).toBe('MesloLGS NF')
    expect((screen.getByLabelText('Glyph preview') as HTMLElement).style.fontFamily).toContain('MesloLGS NF')

    await flushAutosave()

    // Only the font key goes over the wire (PUT deep-merges); the shared cache
    // gets the merged record so sibling terminal keys survive.
    const owner = $settingsOwner.get()
    expect(mocks.save).toHaveBeenCalledWith({ terminal: { font_family: 'MesloLGS NF' } }, owner)
    expect(mocks.cache).toHaveBeenCalledWith(
      {
        display: { skin: 'hermes' },
        terminal: { backend: 'local', cwd: '/workspace', font_family: 'MesloLGS NF' }
      },
      owner
    )
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

    expect(mocks.save.mock.calls[0][0]).toEqual({
      terminal: { font_family: "'Custom Powerline', monospace" }
    })

    fireEvent.click(screen.getByRole('button', { name: 'Use default' }))
    expect($terminalFontFamily.get()).toBe('')
    expect((screen.getByLabelText('Glyph preview') as HTMLElement).style.fontFamily).toContain('JetBrains Mono')
    await flushAutosave()

    expect(mocks.save.mock.calls[1][0]).toEqual({ terminal: { font_family: '' } })
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

  it('cancels a pending autosave when its settings owner is replaced', async () => {
    render(<TerminalFontSetting />)

    fireEvent.change(screen.getByRole('combobox', { name: 'Terminal Font' }), {
      target: { value: 'MesloLGS NF' }
    })
    act(() => {
      $connection.set({
        baseUrl: 'https://replacement.example',
        connectionId: 'replacement',
        mode: 'remote'
      } as never)
    })
    await flushAutosave()

    expect(mocks.save).not.toHaveBeenCalled()
  })

  it('drops the prior profile font and reseeds after a refetch reuses its cached record', () => {
    const sharedConfig = { terminal: { font_family: 'MesloLGS NF' } }
    mocks.loadedConfig = sharedConfig
    const view = render(<TerminalFontSetting />)

    expect($terminalFontFamily.get()).toBe('MesloLGS NF')
    act(() => {
      mocks.loadedConfig = undefined as never
      $activeGatewayProfile.set('research')
      $connection.set({
        baseUrl: 'http://127.0.0.1:3001',
        connectionId: 'local',
        mode: 'local',
        profile: 'research'
      } as never)
    })
    expect($terminalFontFamily.get()).toBe('')
    expect((screen.getByRole('combobox', { name: 'Terminal Font' }) as HTMLInputElement).disabled).toBe(true)

    mocks.configUpdatedAt = 2
    mocks.loadedConfig = sharedConfig
    act(() => $configRevision.set($configRevision.get() + 1))
    view.rerender(<TerminalFontSetting />)

    expect((screen.getByRole('combobox', { name: 'Terminal Font' }) as HTMLInputElement).value).toBe('MesloLGS NF')
    expect($terminalFontFamily.get()).toBe('MesloLGS NF')
  })
})
