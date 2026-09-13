import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { LoginStartupSettings } from './login-startup-settings'

const getSettings = vi.fn()
const setSettings = vi.fn()

beforeEach(() => {
  vi.stubGlobal('hermesDesktop', { loginStartup: { getSettings, setSettings } })
  getSettings.mockResolvedValue({ supported: true, openAtLogin: false })
  setSettings.mockImplementation(async enabled => ({ supported: true, openAtLogin: enabled }))
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.resetAllMocks()
})

const mount = (locale: string = 'en') =>
  render(
    <I18nProvider
      configClient={{
        getConfig: async () => ({ display: { language: locale } }),
        saveConfig: async () => ({ ok: true })
      }}
      initialLocale={locale}
    >
      <LoginStartupSettings />
    </I18nProvider>
  )

describe('Windows login startup settings', () => {
  it('reads OS state on each mount and saves only an explicit toggle', async () => {
    const view = mount()
    const toggle = await screen.findByRole('switch')
    await waitFor(() => expect(toggle.hasAttribute('disabled')).toBe(false))
    expect(toggle.getAttribute('aria-checked')).toBe('false')
    expect(setSettings).not.toHaveBeenCalled()
    fireEvent.click(toggle)
    await waitFor(() => expect(toggle.getAttribute('aria-checked')).toBe('true'))
    expect(setSettings).toHaveBeenCalledWith(true)

    view.unmount()
    getSettings.mockResolvedValue({ supported: true, openAtLogin: true })
    mount()
    await waitFor(() => expect(screen.getByRole('switch').getAttribute('aria-checked')).toBe('true'))
    fireEvent.click(screen.getByRole('switch'))
    await waitFor(() => expect(screen.getByRole('switch').getAttribute('aria-checked')).toBe('false'))
    expect(setSettings).toHaveBeenLastCalledWith(false)
  })

  it('does not claim enabled when Windows rejects registration', async () => {
    setSettings.mockResolvedValue({ supported: true, openAtLogin: false })
    mount()
    await waitFor(() => expect(screen.getByRole('switch').hasAttribute('disabled')).toBe(false))
    fireEvent.click(screen.getByRole('switch'))
    expect(await screen.findByRole('alert')).toBeTruthy()
    expect(screen.getByRole('switch').getAttribute('aria-checked')).toBe('false')
  })

  it('keeps confirmed state and reports IPC failures', async () => {
    setSettings.mockRejectedValue(new Error('OS write failed'))
    mount()
    await waitFor(() => expect(screen.getByRole('switch').hasAttribute('disabled')).toBe(false))
    fireEvent.click(screen.getByRole('switch'))
    expect(await screen.findByRole('alert')).toBeTruthy()
    expect(screen.getByRole('switch').getAttribute('aria-checked')).toBe('false')
  })

  it('does not offer startup for an unsupported build or platform', async () => {
    getSettings.mockResolvedValue({ supported: false, openAtLogin: false })
    mount()
    await waitFor(() => expect(screen.queryByRole('switch')).toBeNull())
    expect(setSettings).not.toHaveBeenCalled()
  })

  it('renders the startup choice in Arabic', async () => {
    mount('ar')
    const toggle = await screen.findByRole('switch')
    await waitFor(() => expect(toggle.hasAttribute('disabled')).toBe(false))
    expect(toggle.getAttribute('aria-label')).toMatch(/[\u0600-\u06ff]/)
  })
})
