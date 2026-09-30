// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'

import type { ScreenshotStatus } from '../../../electron/command-screenshot-types'

import { ScreenshotSettings } from './screenshot-settings'

vi.mock('@/i18n', () => ({ useI18n: () => ({ t: en }) }))

const withDefaults = (status: Partial<ScreenshotStatus>): ScreenshotStatus => ({
  destination: 'current-draft',
  bringToFront: false,
  ...status
} as ScreenshotStatus)

const copy = en.settings.screenshot

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(yes => {
    resolve = yes
  })

  return { promise, resolve }
}

function installBridge() {
  let onStatus: (status: ScreenshotStatus) => void = () => {}
  const unsubscribe = vi.fn()

  const api = {
    getSettings: vi.fn<() => Promise<ScreenshotStatus>>(),
    updateSettings: vi.fn<(patch: Record<string, unknown>) => Promise<ScreenshotStatus>>(),
    openPermissionSettings: vi.fn<(kind: 'input' | 'screen') => Promise<void>>().mockResolvedValue(undefined),
    onStatus: vi.fn((callback: (status: ScreenshotStatus) => void) => {
      onStatus = callback

      return unsubscribe
    })
  }

  vi.stubGlobal('hermesDesktop', { screenshot: api })

  return { api, emit: (status: ScreenshotStatus) => onStatus(status), unsubscribe }
}

async function click(element: HTMLElement) {
  await act(async () => fireEvent.click(element))
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('ScreenshotSettings', () => {
  it('requires opt-in and explicit permission retry before claiming the shortcut is ready', async () => {
    const { api, emit, unsubscribe } = installBridge()
    const initial = deferred<ScreenshotStatus>()
    const enabling = deferred<ScreenshotStatus>()
    api.getSettings.mockReturnValue(initial.promise)
    api.updateSettings.mockReturnValueOnce(enabling.promise)
    const view = render(<ScreenshotSettings />)
    const toggle = screen.getByRole('switch', { name: copy.enabledTitle })

    expect(toggle).toHaveProperty('disabled', true)
    expect(toggle).toHaveProperty('ariaChecked', 'false')
    expect(api.updateSettings).not.toHaveBeenCalled()
    await act(async () => initial.resolve(withDefaults({ enabled: false, state: 'disabled' })))
    await click(toggle)
    expect(api.updateSettings).toHaveBeenLastCalledWith({ enabled: true })
    expect(toggle).toHaveProperty('ariaChecked', 'false')
    expect(screen.queryByText(copy.ready)).toBeNull()
    await act(async () => enabling.resolve(withDefaults({ enabled: true, state: 'input-permission' })))
    expect(toggle).toHaveProperty('ariaChecked', 'true')
    expect(screen.getByText(copy.inputPermission)).toBeTruthy()
    expect(screen.queryByText(copy.ready)).toBeNull()

    await click(screen.getByRole('button', { name: copy.openSettings }))
    expect(api.openPermissionSettings).toHaveBeenLastCalledWith('input')
    expect(api.updateSettings).toHaveBeenCalledTimes(1)
    api.updateSettings.mockResolvedValueOnce(withDefaults({ enabled: true, state: 'screen-permission' }))
    await click(screen.getByRole('button', { name: copy.retry }))
    expect(api.updateSettings).toHaveBeenLastCalledWith({ enabled: true })
    expect(screen.getByText(copy.screenPermission)).toBeTruthy()
    await click(screen.getByRole('button', { name: copy.openSettings }))
    expect(api.openPermissionSettings).toHaveBeenLastCalledWith('screen')

    api.updateSettings.mockResolvedValueOnce(withDefaults({ enabled: true, state: 'starting' }))
    await click(screen.getByRole('button', { name: copy.retry }))
    expect(screen.getByText(copy.starting)).toBeTruthy()
    expect(screen.queryByText(copy.ready)).toBeNull()
    await act(async () => emit(withDefaults({ enabled: true, state: 'ready' })))
    expect(screen.getByText(copy.ready)).toBeTruthy()
    api.updateSettings.mockResolvedValueOnce(withDefaults({ enabled: false, state: 'disabled' }))
    await click(toggle)
    expect(api.updateSettings).toHaveBeenLastCalledWith({ enabled: false })
    expect(toggle).toHaveProperty('ariaChecked', 'false')
    expect(screen.queryByText(copy.ready)).toBeNull()
    view.unmount()
    expect(unsubscribe).toHaveBeenCalledOnce()
  })

  it('writes destination and bringToFront patches without touching the enabled toggle', async () => {
    const { api } = installBridge()
    api.getSettings.mockResolvedValue(
      withDefaults({ enabled: true, state: 'ready', destination: 'new-session', bringToFront: true })
    )
    let stored = withDefaults({ enabled: true, state: 'ready', destination: 'new-session', bringToFront: true })
    api.updateSettings.mockImplementation(async (patch: Record<string, unknown>) => {
      stored = withDefaults({ ...stored, ...patch } as ScreenshotStatus)

      return stored
    })
    render(<ScreenshotSettings />)
    const toggle = screen.getByRole('switch', { name: copy.enabledTitle })

    // The rows render the stored device prefs once the read lands.
    await screen.findByRole('button', { name: copy.destinationNewSession })
    expect(toggle).toHaveProperty('ariaChecked', 'true')

    // Switching destination writes only that field.
    await click(screen.getByRole('button', { name: copy.destinationCurrentDraft }))
    expect(api.updateSettings).toHaveBeenLastCalledWith({ destination: 'current-draft' })

    // The bring-to-front toggle writes only its own field.
    await click(screen.getByRole('switch', { name: copy.bringToFrontTitle }))
    expect(api.updateSettings).toHaveBeenLastCalledWith({ bringToFront: false })
  })

  it('keeps read, write, and permission failures recoverable without claiming success', async () => {
    const { api, emit } = installBridge()
    api.getSettings.mockRejectedValueOnce(new Error('IPC unavailable'))
    render(<ScreenshotSettings />)
    const toggle = screen.getByRole('switch', { name: copy.enabledTitle })
    expect(await screen.findByText(copy.loadFailed)).toBeTruthy()
    expect(toggle).toHaveProperty('disabled', true)
    expect(api.updateSettings).not.toHaveBeenCalled()

    api.getSettings.mockResolvedValueOnce(withDefaults({ enabled: false, state: 'disabled' }))
    await click(screen.getByRole('button', { name: copy.retry }))
    expect(toggle).toHaveProperty('disabled', false)
    api.updateSettings.mockImplementationOnce(async () => {
      emit(withDefaults({ enabled: false, state: 'disabled' }))
      throw new Error('Unconfirmed write')
    })
    await click(toggle)
    expect(screen.getByText(copy.saveFailed)).toBeTruthy()
    expect(toggle).toHaveProperty('ariaChecked', 'false')
    expect(screen.queryByText(copy.ready)).toBeNull()

    // A write may have landed even if IPC rejected: reread, don't guess.
    api.getSettings.mockResolvedValueOnce(withDefaults({ enabled: true, state: 'screen-permission' }))
    await click(screen.getByRole('button', { name: copy.retry }))
    expect(toggle).toHaveProperty('ariaChecked', 'true')
    api.openPermissionSettings.mockRejectedValueOnce(new Error('Cannot open settings'))
    await click(screen.getByRole('button', { name: copy.openSettings }))
    expect(screen.getByText(copy.permissionFailed)).toBeTruthy()
    expect(screen.queryByText(copy.ready)).toBeNull()

    // Manual permission recovery must restart the listener, not just reread.
    api.updateSettings.mockResolvedValueOnce(withDefaults({ enabled: true, state: 'unavailable' }))
    await click(screen.getByRole('button', { name: copy.retry }))
    expect(api.updateSettings).toHaveBeenCalledTimes(2)
    expect(screen.getByText(copy.unavailable)).toBeTruthy()
    api.updateSettings.mockResolvedValueOnce(withDefaults({ enabled: true, state: 'ready' }))
    await click(screen.getByRole('button', { name: copy.retry }))
    expect(screen.getByText(copy.ready)).toBeTruthy()
  })

  it('keeps newer native status over stale reads and hides without the native capability', async () => {
    const { api, emit, unsubscribe } = installBridge()
    const initial = deferred<ScreenshotStatus>()
    api.getSettings.mockReturnValue(initial.promise)
    const view = render(<ScreenshotSettings />)
    await act(async () => emit(withDefaults({ enabled: false, state: 'disabled' })))
    await act(async () => initial.resolve(withDefaults({ enabled: true, state: 'ready' })))
    expect(screen.getByRole('switch', { name: copy.enabledTitle })).toHaveProperty('ariaChecked', 'false')
    expect(screen.queryByText(copy.ready)).toBeNull()
    expect(api.updateSettings).not.toHaveBeenCalled()
    view.unmount()
    expect(unsubscribe).toHaveBeenCalledOnce()

    vi.stubGlobal('hermesDesktop', {})
    const unsupported = render(<ScreenshotSettings />)
    expect(unsupported.container.childElementCount).toBe(0)
  })
})
