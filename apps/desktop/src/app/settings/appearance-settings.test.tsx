import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type I18nConfigClient, I18nProvider } from '@/i18n'

const notifyError = vi.fn()
const triggerHaptic = vi.fn()
const setMode = vi.fn()
const setTheme = vi.fn()
const setToolViewMode = vi.fn()

vi.mock('@/lib/haptics', () => ({
  triggerHaptic: (style: string) => triggerHaptic(style)
}))

vi.mock('@/store/notifications', () => ({
  notifyError: (error: unknown, fallback: string) => notifyError(error, fallback)
}))

vi.mock('@/store/tool-view', () => ({
  $toolViewMode: {},
  setToolViewMode: (mode: string) => setToolViewMode(mode)
}))

vi.mock('@nanostores/react', () => ({
  useStore: () => 'product'
}))

vi.mock('@/themes/context', () => ({
  useTheme: () => ({
    availableThemes: [
      {
        name: 'nous',
        label: 'Nous',
        description: 'Default theme',
        colors: {
          background: '#ffffff',
          foreground: '#111111',
          muted: '#eeeeee',
          mutedForeground: '#666666',
          border: '#dddddd',
          userBubble: '#f5f5f5',
          userBubbleBorder: '#dddddd'
        }
      }
    ],
    mode: 'light',
    setMode,
    setTheme,
    themeName: 'nous'
  })
}))

beforeEach(() => {
  notifyError.mockReset()
  triggerHaptic.mockReset()
  setMode.mockReset()
  setTheme.mockReset()
  setToolViewMode.mockReset()
})

afterEach(() => {
  cleanup()
})

async function renderAppearanceSettings(configClient: I18nConfigClient) {
  const { AppearanceSettings } = await import('./appearance-settings')

  render(
    <I18nProvider configClient={configClient}>
      <AppearanceSettings />
    </I18nProvider>
  )
}

describe('AppearanceSettings language selector', () => {
  it('saves Simplified Chinese through display.language while preserving latest config values', async () => {
    const saveConfig = vi.fn().mockResolvedValue({ ok: true })

    const configClient: I18nConfigClient = {
      getConfig: vi
        .fn()
        .mockResolvedValueOnce({ display: { language: 'en', skin: 'mono' }, terminal: { cwd: '/old' } })
        .mockResolvedValueOnce({ display: { language: 'en', skin: 'slate' }, terminal: { cwd: '/new' } }),
      saveConfig
    }

    await renderAppearanceSettings(configClient)

    await waitFor(() => expect(configClient.getConfig).toHaveBeenCalledTimes(1))
    fireEvent.click(screen.getByRole('button', { name: /Simplified Chinese/ }))

    await waitFor(() =>
      expect(saveConfig).toHaveBeenCalledWith({
        display: { language: 'zh', skin: 'slate' },
        terminal: { cwd: '/new' }
      })
    )
    expect(await screen.findByText('语言')).toBeTruthy()
    expect(screen.getByText('颜色模式')).toBeTruthy()
    expect(screen.getByText('工具调用显示')).toBeTruthy()
    expect(screen.getByText('玻璃感中性色，搭配 Nous 蓝色强调色')).toBeTruthy()
    expect(notifyError).not.toHaveBeenCalled()
  })
})
