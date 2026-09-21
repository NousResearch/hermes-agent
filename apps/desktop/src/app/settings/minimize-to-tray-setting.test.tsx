import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import { en } from '@/i18n/en'

import { MinimizeToTraySetting } from './minimize-to-tray-setting'

const errors = vi.hoisted(() => vi.fn())
vi.mock('@/store/notifications', () => ({ notifyError: errors }))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))

const original = window.hermesDesktop
const c = en.settings.config
interface Status {
  enabled: boolean
  available: boolean
}

function bridge(initial: Status) {
  const listeners = new Set<(next: Status) => void>()

  const api = {
    get: vi.fn(async () => initial),
    set: vi.fn(async (enabled: boolean) => {
      const next = { enabled, available: enabled }
      listeners.forEach(listener => listener(next))

      return next
    }),
    onChanged: (listener: (next: Status) => void) => {
      listeners.add(listener)

      return () => {
        listeners.delete(listener)
      }
    }
  }

  window.hermesDesktop = { ...original, minimizeToTray: api }

  return { api, listeners }
}

afterEach(() => {
  cleanup()
  window.hermesDesktop = original
  vi.clearAllMocks()
})

test('native truth drives both mounted settings rows and survives remount without renderer writes', async () => {
  const { api, listeners } = bridge({ enabled: false, available: false })
  const view = render(
    <>
      <MinimizeToTraySetting />
      <MinimizeToTraySetting />
    </>
  )
  const toggles = await screen.findAllByRole('switch', { name: c.minimizeToTrayTitle })
  await waitFor(() => expect(toggles.every(toggle => !toggle.hasAttribute('disabled'))).toBe(true))
  expect(api.set).not.toHaveBeenCalled()
  fireEvent.click(toggles[0])
  await waitFor(() => expect(toggles.every(toggle => toggle.getAttribute('aria-checked') === 'true')).toBe(true))
  expect(api.set).toHaveBeenCalledWith(true)
  act(() => listeners.forEach(listener => listener({ enabled: true, available: false })))
  expect(screen.getAllByText(c.minimizeToTrayUnavailable)).toHaveLength(2)
  view.unmount()
  expect(listeners.size).toBe(0)
  api.get.mockResolvedValue({ enabled: true, available: true })
  render(<MinimizeToTraySetting />)
  await waitFor(() => expect(screen.getByRole('switch').getAttribute('aria-checked')).toBe('true'))
  expect(api.set).toHaveBeenCalledTimes(1)
})

test('failed writes roll back visibly, and an older initial read cannot overwrite a peer update', async () => {
  const { api, listeners } = bridge({ enabled: false, available: false })
  let finishRead!: (status: Status) => void
  api.get.mockImplementation(
    () =>
      new Promise(resolve => {
        finishRead = resolve
      })
  )
  render(<MinimizeToTraySetting />)
  act(() => listeners.forEach(listener => listener({ enabled: true, available: true })))
  await act(async () => finishRead({ enabled: false, available: false }))
  const toggle = screen.getByRole('switch')
  expect(toggle.getAttribute('aria-checked')).toBe('true')
  const failure = new Error('Disk write failed')
  api.set.mockRejectedValue(failure)
  fireEvent.click(toggle)
  await waitFor(() => expect(errors).toHaveBeenCalledWith(failure, c.autosaveFailed))
  expect(toggle.getAttribute('aria-checked')).toBe('true')
})
