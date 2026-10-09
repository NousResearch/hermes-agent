import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'

import { en } from '@/i18n/en'
import { $hudAlwaysOnTop } from '@/store/hud'

import { TrayAccessSettings } from './tray-access-settings'

const errors = vi.hoisted(() => vi.fn())
vi.mock('@/store/notifications', () => ({ notifyError: errors }))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))

const original = window.hermesDesktop
const c = en.settings.config

interface Preferences {
  launchAtLogin: boolean
  openNewConversationOnClick: boolean
}

interface PinState {
  alwaysOnTop: boolean
}

function bridge(initial: Preferences, initialPin = true) {
  const listeners = new Set<(next: Preferences) => void>()
  const pinListeners = new Set<(next: PinState) => void>()
  let pin: PinState = { alwaysOnTop: initialPin }

  const trayPreferences = {
    get: vi.fn(async () => initial),
    set: vi.fn(async (patch: Partial<Preferences>) => {
      const next = { ...initial, ...patch }
      listeners.forEach(listener => listener(next))

      return next
    }),
    onChanged: (listener: (next: Preferences) => void) => {
      listeners.add(listener)

      return () => {
        listeners.delete(listener)
      }
    }
  }

  const alwaysOnTop = {
    get: vi.fn(async () => pin),
    set: vi.fn(async (on: boolean) => {
      pin = { alwaysOnTop: on }
      pinListeners.forEach(listener => listener(pin))

      return pin
    }),
    onChanged: (listener: (next: PinState) => void) => {
      pinListeners.add(listener)

      return () => {
        pinListeners.delete(listener)
      }
    }
  }

  window.hermesDesktop = {
    ...original,
    // Test double: only `alwaysOnTop` is real here, the rest of the hud
    // surface is carried over so nothing else in the tree sees a shape change.
    hud: { ...original?.hud, alwaysOnTop } as NonNullable<typeof window.hermesDesktop>['hud'],
    trayPreferences
  }

  return { alwaysOnTop, listeners, pinListeners, trayPreferences }
}

beforeEach(() => $hudAlwaysOnTop.set(true))

afterEach(() => {
  cleanup()
  window.hermesDesktop = original
  vi.clearAllMocks()
})

test('the quick-access rows write their own preference and nothing else', async () => {
  const { trayPreferences } = bridge({ launchAtLogin: false, openNewConversationOnClick: true })

  render(<TrayAccessSettings />)

  const newConversation = await screen.findByRole('switch', { name: c.trayNewConversationOnClickTitle })
  const launchAtLogin = screen.getByRole('switch', { name: c.launchAtLoginTitle })

  // Defaults come from main, not from an optimistic renderer guess.
  expect(newConversation.getAttribute('aria-checked')).toBe('true')
  expect(launchAtLogin.getAttribute('aria-checked')).toBe('false')
  expect(trayPreferences.set).not.toHaveBeenCalled()

  fireEvent.click(launchAtLogin)
  await waitFor(() => expect(trayPreferences.set).toHaveBeenCalledWith({ launchAtLogin: true }))

  fireEvent.click(newConversation)
  await waitFor(() => expect(trayPreferences.set).toHaveBeenCalledWith({ openNewConversationOnClick: false }))

  // Neither toggle may reach across into the other's patch.
  for (const [patch] of trayPreferences.set.mock.calls) {
    expect(Object.keys(patch)).toHaveLength(1)
  }
})

test('the pin row follows main’s persisted preference, live and in both directions', async () => {
  const { alwaysOnTop, pinListeners } = bridge({ launchAtLogin: false, openNewConversationOnClick: true }, true)

  render(<TrayAccessSettings />)

  const pin = await screen.findByRole('switch', { name: c.miniAssistantAlwaysOnTopTitle })
  expect(pin.getAttribute('aria-checked')).toBe('true')
  expect(alwaysOnTop.set).not.toHaveBeenCalled()

  fireEvent.click(pin)
  await waitFor(() => expect(alwaysOnTop.set).toHaveBeenCalledWith(false))
  // Optimistic: the row reflects the choice before main confirms.
  expect(pin.getAttribute('aria-checked')).toBe('false')

  // A change made elsewhere (the bar's own pin button) lands here too.
  act(() => pinListeners.forEach(listener => listener({ alwaysOnTop: true })))
  expect(pin.getAttribute('aria-checked')).toBe('true')
})

test('a failed preference write rolls the row back and surfaces the error', async () => {
  const { trayPreferences } = bridge({ launchAtLogin: false, openNewConversationOnClick: true })

  render(<TrayAccessSettings />)

  const launchAtLogin = await screen.findByRole('switch', { name: c.launchAtLoginTitle })
  const failure = new Error('Disk write failed')
  trayPreferences.set.mockRejectedValue(failure)

  fireEvent.click(launchAtLogin)
  await waitFor(() => expect(errors).toHaveBeenCalledWith(failure, c.autosaveFailed))
  expect(launchAtLogin.getAttribute('aria-checked')).toBe('false')
})

test('nothing renders when the shell exposes no tray preferences', () => {
  window.hermesDesktop = { ...original, trayPreferences: undefined }

  const view = render(<TrayAccessSettings />)

  expect(view.container.textContent).toBe('')
  expect(screen.queryByRole('switch')).toBeNull()
})
