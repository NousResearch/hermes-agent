import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

import { notifyError } from '@/store/notifications'

const loginItemGet = vi.fn()
const loginItemSet = vi.fn()

beforeEach(() => {
  loginItemGet.mockReset()
  loginItemSet.mockReset()
  loginItemGet.mockResolvedValue({ openAtLogin: false, supported: true })
  loginItemSet.mockResolvedValue({ openAtLogin: true, supported: true })

  vi.mocked(notifyError).mockClear()

  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { loginItem: { get: loginItemGet, set: loginItemSet } }
  })
})

describe('LoginItemSetting', () => {
  it('loads the current login-item state on mount', async () => {
    const { LoginItemSetting } = await import('./config-settings')

    render(<LoginItemSetting />)

    const toggle = await screen.findByRole('switch', { name: 'Launch Hermes Desktop at login' })
    expect(loginItemGet).toHaveBeenCalledOnce()
    expect(toggle.getAttribute('aria-checked')).toBe('false')
  })

  it('updates the login item when toggled on', async () => {
    const { LoginItemSetting } = await import('./config-settings')

    render(<LoginItemSetting />)

    const toggle = await screen.findByRole('switch', { name: 'Launch Hermes Desktop at login' })
    fireEvent.click(toggle)

    await waitFor(() => expect(loginItemSet).toHaveBeenCalledWith({ openAtLogin: true }))
    expect(toggle.getAttribute('aria-checked')).toBe('true')
  })

  it('reflects the authoritative state when set diverges from the request', async () => {
    // The write did not land (e.g. policy-blocked): Electron reports false.
    loginItemSet.mockResolvedValue({ openAtLogin: false, supported: true })
    const { LoginItemSetting } = await import('./config-settings')

    render(<LoginItemSetting />)

    const toggle = await screen.findByRole('switch', { name: 'Launch Hermes Desktop at login' })
    fireEvent.click(toggle)

    await waitFor(() => expect(loginItemSet).toHaveBeenCalled())
    expect(toggle.getAttribute('aria-checked')).toBe('false')
  })

  it('keeps the previous value when set rejects', async () => {
    loginItemSet.mockRejectedValue(new Error('bridge failure'))
    const { LoginItemSetting } = await import('./config-settings')

    render(<LoginItemSetting />)

    const toggle = await screen.findByRole('switch', { name: 'Launch Hermes Desktop at login' })
    fireEvent.click(toggle)

    await waitFor(() => expect(loginItemSet).toHaveBeenCalled())
    expect(toggle.getAttribute('aria-checked')).toBe('false')
  })

  it('surfaces a rejected write as a visible error notification', async () => {
    loginItemSet.mockRejectedValue(new Error('bridge failure'))
    const { LoginItemSetting } = await import('./config-settings')

    render(<LoginItemSetting />)

    const toggle = await screen.findByRole('switch', { name: 'Launch Hermes Desktop at login' })
    fireEvent.click(toggle)

    await waitFor(() => expect(loginItemSet).toHaveBeenCalled())
    expect(notifyError).toHaveBeenCalledWith(
      expect.any(Error),
      expect.stringContaining("Couldn't save")
    )
  })

  it('hides the toggle when the platform does not support login items', async () => {
    loginItemGet.mockResolvedValue({ openAtLogin: false, supported: false })
    const { LoginItemSetting } = await import('./config-settings')

    render(<LoginItemSetting />)

    await waitFor(() => expect(loginItemGet).toHaveBeenCalled())
    expect(screen.queryByRole('switch', { name: 'Launch Hermes Desktop at login' })).toBeNull()
  })

  it('treats a missing login-item API as unsupported (hidden)', async () => {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {}
    })
    const { LoginItemSetting } = await import('./config-settings')

    render(<LoginItemSetting />)

    expect(screen.queryByRole('switch', { name: 'Launch Hermes Desktop at login' })).toBeNull()
  })
})
