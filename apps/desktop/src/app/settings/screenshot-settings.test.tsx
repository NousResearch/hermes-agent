// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ar } from '@/i18n/ar'
import { en } from '@/i18n/en'
import { ja } from '@/i18n/ja'
import { ru } from '@/i18n/ru'
import { zh } from '@/i18n/zh'
import { zhHant } from '@/i18n/zh-hant'

import type { ScreenshotStatus } from '../../../electron/command-screenshot-types'

import { ScreenshotSettings } from './screenshot-settings'

vi.mock('@/i18n', () => ({ useI18n: () => ({ t: en }) }))

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
    setEnabled: vi.fn<(enabled: boolean) => Promise<ScreenshotStatus>>(),
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
    api.setEnabled.mockReturnValueOnce(enabling.promise)
    const view = render(<ScreenshotSettings />)
    const toggle = screen.getByRole('switch', { name: copy.enabledTitle })

    expect(toggle).toHaveProperty('disabled', true)
    expect(toggle).toHaveProperty('ariaChecked', 'false')
    expect(api.setEnabled).not.toHaveBeenCalled()
    await act(async () => initial.resolve({ enabled: false, state: 'disabled' }))
    await click(toggle)
    expect(api.setEnabled).toHaveBeenLastCalledWith(true)
    expect(toggle).toHaveProperty('ariaChecked', 'false')
    expect(screen.queryByText(copy.ready)).toBeNull()
    await act(async () => enabling.resolve({ enabled: true, state: 'input-permission' }))
    expect(toggle).toHaveProperty('ariaChecked', 'true')
    expect(screen.getByText(copy.inputPermission)).toBeTruthy()
    expect(screen.queryByText(copy.ready)).toBeNull()

    await click(screen.getByRole('button', { name: copy.openSettings }))
    expect(api.openPermissionSettings).toHaveBeenLastCalledWith('input')
    expect(api.setEnabled).toHaveBeenCalledTimes(1)
    api.setEnabled.mockResolvedValueOnce({ enabled: true, state: 'screen-permission' })
    await click(screen.getByRole('button', { name: copy.retry }))
    expect(api.setEnabled).toHaveBeenLastCalledWith(true)
    expect(screen.getByText(copy.screenPermission)).toBeTruthy()
    await click(screen.getByRole('button', { name: copy.openSettings }))
    expect(api.openPermissionSettings).toHaveBeenLastCalledWith('screen')

    api.setEnabled.mockResolvedValueOnce({ enabled: true, state: 'starting' })
    await click(screen.getByRole('button', { name: copy.retry }))
    expect(screen.getByText(copy.starting)).toBeTruthy()
    expect(screen.queryByText(copy.ready)).toBeNull()
    await act(async () => emit({ enabled: true, state: 'ready' }))
    expect(screen.getByText(copy.ready)).toBeTruthy()
    api.setEnabled.mockResolvedValueOnce({ enabled: false, state: 'disabled' })
    await click(toggle)
    expect(api.setEnabled).toHaveBeenLastCalledWith(false)
    expect(toggle).toHaveProperty('ariaChecked', 'false')
    expect(screen.queryByText(copy.ready)).toBeNull()
    view.unmount()
    expect(unsubscribe).toHaveBeenCalledOnce()
  })

  it('keeps read, write, and permission failures recoverable without claiming success', async () => {
    const { api, emit } = installBridge()
    api.getSettings.mockRejectedValueOnce(new Error('IPC unavailable'))
    render(<ScreenshotSettings />)
    const toggle = screen.getByRole('switch', { name: copy.enabledTitle })
    expect(await screen.findByText(copy.loadFailed)).toBeTruthy()
    expect(toggle).toHaveProperty('disabled', true)
    expect(api.setEnabled).not.toHaveBeenCalled()

    api.getSettings.mockResolvedValueOnce({ enabled: false, state: 'disabled' })
    await click(screen.getByRole('button', { name: copy.retry }))
    expect(toggle).toHaveProperty('disabled', false)
    api.setEnabled.mockImplementationOnce(async () => {
      emit({ enabled: false, state: 'disabled' })
      throw new Error('Unconfirmed write')
    })
    await click(toggle)
    expect(screen.getByText(copy.saveFailed)).toBeTruthy()
    expect(toggle).toHaveProperty('ariaChecked', 'false')
    expect(screen.queryByText(copy.ready)).toBeNull()

    // A write may have landed even if IPC rejected: reread, don't guess.
    api.getSettings.mockResolvedValueOnce({ enabled: true, state: 'screen-permission' })
    await click(screen.getByRole('button', { name: copy.retry }))
    expect(toggle).toHaveProperty('ariaChecked', 'true')
    api.openPermissionSettings.mockRejectedValueOnce(new Error('Cannot open settings'))
    await click(screen.getByRole('button', { name: copy.openSettings }))
    expect(screen.getByText(copy.permissionFailed)).toBeTruthy()
    expect(screen.queryByText(copy.ready)).toBeNull()

    // Manual permission recovery must restart the listener, not just reread.
    api.setEnabled.mockResolvedValueOnce({ enabled: true, state: 'unavailable' })
    await click(screen.getByRole('button', { name: copy.retry }))
    expect(api.setEnabled).toHaveBeenCalledTimes(2)
    expect(screen.getByText(copy.unavailable)).toBeTruthy()
    api.setEnabled.mockResolvedValueOnce({ enabled: true, state: 'ready' })
    await click(screen.getByRole('button', { name: copy.retry }))
    expect(screen.getByText(copy.ready)).toBeTruthy()
  })

  it('keeps newer native status over stale reads and hides without the native capability', async () => {
    const { api, emit, unsubscribe } = installBridge()
    const initial = deferred<ScreenshotStatus>()
    api.getSettings.mockReturnValue(initial.promise)
    const view = render(<ScreenshotSettings />)
    await act(async () => emit({ enabled: false, state: 'disabled' }))
    await act(async () => initial.resolve({ enabled: true, state: 'ready' }))
    expect(screen.getByRole('switch')).toHaveProperty('ariaChecked', 'false')
    expect(screen.queryByText(copy.ready)).toBeNull()
    expect(api.setEnabled).not.toHaveBeenCalled()
    view.unmount()
    expect(unsubscribe).toHaveBeenCalledOnce()

    vi.stubGlobal('hermesDesktop', {})
    const unsupported = render(<ScreenshotSettings />)
    expect(unsupported.container.childElementCount).toBe(0)
  })

  it('provides every screenshot message in each locale', () => {
    for (const locale of [en, ja, zh, zhHant, ar, ru]) {
      expect(Object.keys(locale.settings.screenshot).sort()).toEqual(Object.keys(copy).sort())

      for (const text of Object.values(locale.settings.screenshot)) {
        expect(typeof text).toBe('string')
        expect(text.trim()).not.toBe('')
      }
    }
  })
})
