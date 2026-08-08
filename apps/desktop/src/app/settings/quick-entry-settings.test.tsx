import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $quickEntry, QUICK_ENTRY_DEFAULT_SHORTCUT } from '@/store/quick-entry'

import { QuickEntrySettings } from './quick-entry-settings'

const getSettings = vi.fn()
const setSettings = vi.fn()

beforeEach(() => {
  const status = {
    enabled: true,
    error: null,
    registered: true,
    shortcut: QUICK_ENTRY_DEFAULT_SHORTCUT
  } as const

  $quickEntry.set(status)
  getSettings.mockResolvedValue(status)
  setSettings.mockResolvedValue(status)

  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      quickEntry: {
        getSettings,
        setSettings
      }
    }
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('QuickEntrySettings', () => {
  it('renders the default shortcut as readable Chinese text instead of Electron syntax', async () => {
    render(
      <I18nProvider configClient={null} initialLocale="zh">
        <QuickEntrySettings />
      </I18nProvider>
    )

    const input = screen.getByRole('textbox', {
      name: '快速输入快捷键'
    }) as HTMLInputElement

    await waitFor(() => {
      expect(input.value).toBe('Command 或 Control + Shift + Space')
    })

    expect(input.value).not.toContain('CommandOrControl')
    expect(input.placeholder).toBe('Command 或 Control + Shift + Space')
  })

  it('converts readable Chinese shortcut text back to Electron syntax before saving', async () => {
    render(
      <I18nProvider configClient={null} initialLocale="zh">
        <QuickEntrySettings />
      </I18nProvider>
    )

    const input = screen.getByRole('textbox', {
      name: '快速输入快捷键'
    })

    fireEvent.change(input, { target: { value: 'Command 或 Control + Alt + K' } })
    fireEvent.blur(input)

    await waitFor(() => {
      expect(setSettings).toHaveBeenCalledWith({ shortcut: 'CommandOrControl+Alt+K' })
    })
  })
})
