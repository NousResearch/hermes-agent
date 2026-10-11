import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import { en } from '@/i18n/en'

import { DownloadsSaveDirectSetting } from './downloads-save-direct-setting'

const errors = vi.hoisted(() => vi.fn())
vi.mock('@/store/notifications', () => ({ notifyError: errors }))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))

const original = window.hermesDesktop
const c = en.settings.config

function bridge(initial: boolean) {
  const listeners = new Set<(next: boolean) => void>()

  const api = {
    get: vi.fn(async () => initial),
    set: vi.fn(async (on: boolean) => {
      listeners.forEach(listener => listener(on))

      return on
    }),
    onChanged: (listener: (next: boolean) => void) => {
      listeners.add(listener)

      return () => {
        listeners.delete(listener)
      }
    }
  }

  window.hermesDesktop = { ...original, downloadSaveDirect: api }

  return { api, listeners }
}

afterEach(() => {
  cleanup()
  window.hermesDesktop = original
  vi.clearAllMocks()
})

test('renders nothing when the bridge is absent', () => {
  window.hermesDesktop = original

  const view = render(<DownloadsSaveDirectSetting id="setting-field-advanced-downloads-save-direct" />)

  expect(view.container.childElementCount).toBe(0)
})

test('mounts disabled, adopts main truth, and pushes toggles through the bridge', async () => {
  const { api } = bridge(false)

  const view = render(<DownloadsSaveDirectSetting id="setting-field-advanced-downloads-save-direct" />)

  const row = await screen.findByRole('switch', { name: c.downloadsSaveDirectTitle })
  await waitFor(() => expect(row.hasAttribute('disabled')).toBe(false))
  expect(row.getAttribute('aria-checked')).toBe('false')
  expect(api.get).toHaveBeenCalled()
  expect(api.set).not.toHaveBeenCalled()

  fireEvent.click(row)

  await waitFor(() => expect(api.set).toHaveBeenCalledWith(true))
  await waitFor(() =>
    expect(screen.getByRole('switch', { name: c.downloadsSaveDirectTitle }).getAttribute('aria-checked')).toBe('true')
  )
  view.unmount()
})

test('a broadcast from main updates the row without a renderer write', async () => {
  const { api, listeners } = bridge(false)

  const view = render(<DownloadsSaveDirectSetting id="setting-field-advanced-downloads-save-direct" />)

  await screen.findByRole('switch', { name: c.downloadsSaveDirectTitle })
  act(() => listeners.forEach(listener => listener(true)))

  expect(screen.getByRole('switch', { name: c.downloadsSaveDirectTitle }).getAttribute('aria-checked')).toBe('true')
  expect(api.set).not.toHaveBeenCalled()
  view.unmount()
  expect(listeners.size).toBe(0)
})

test('a failed toggle restores the previous state and notifies', async () => {
  const { api } = bridge(false)
  api.set.mockRejectedValue(new Error('IPC gone'))

  const view = render(<DownloadsSaveDirectSetting id="setting-field-advanced-downloads-save-direct" />)

  const row = await screen.findByRole('switch', { name: c.downloadsSaveDirectTitle })
  await waitFor(() => expect(row.hasAttribute('disabled')).toBe(false))

  fireEvent.click(row)

  await waitFor(() => expect(errors).toHaveBeenCalledWith(expect.any(Error), c.autosaveFailed))
  await waitFor(() =>
    expect(screen.getByRole('switch', { name: c.downloadsSaveDirectTitle }).getAttribute('aria-checked')).toBe('false')
  )
  view.unmount()
})
