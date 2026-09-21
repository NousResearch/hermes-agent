// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'

import type { HudModifierStatus } from '../../../electron/hud-modifier-types'

import { HudModifierSettings } from './hud-modifier-settings'

vi.mock('@/i18n', () => ({ useI18n: () => ({ t: en }) }))
const copy = en.settings.hudModifier
const common = en.settings.screenshot

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

function bridge() {
  let emit = (_status: HudModifierStatus) => {}
  const unsubscribe = vi.fn()

  const api = {
    getSettings: vi.fn<() => Promise<HudModifierStatus>>().mockResolvedValue({ enabled: false, state: 'disabled' }),
    setEnabled: vi.fn<(enabled: boolean) => Promise<HudModifierStatus>>(),
    openPermissionSettings: vi.fn().mockResolvedValue(undefined),
    onStatus: vi.fn((listener: (status: HudModifierStatus) => void) => {
      emit = listener

      return unsubscribe
    })
  }

  vi.stubGlobal('hermesDesktop', { hudModifier: api })

  return { api, emit: (value: HudModifierStatus) => emit(value), unsubscribe }
}

it('requires explicit opt-in and native readiness, with permission recovery and stale-read protection', async () => {
  const { api, emit, unsubscribe } = bridge()
  let resolve!: (value: HudModifierStatus) => void
  api.getSettings.mockReturnValueOnce(
    new Promise(yes => {
      resolve = yes
    })
  )
  const view = render(<HudModifierSettings />)
  const toggle = screen.getByRole('switch', { name: copy.title })
  expect(toggle).toHaveProperty('disabled', true)
  expect(api.setEnabled).not.toHaveBeenCalled()
  await act(async () => {
    emit({ enabled: false, state: 'disabled' })
    resolve({ enabled: true, state: 'ready' })
  })
  expect(toggle).toHaveProperty('ariaChecked', 'false')
  api.setEnabled.mockResolvedValueOnce({ enabled: true, state: 'input-permission' })
  await act(async () => fireEvent.click(toggle))
  expect(api.setEnabled).toHaveBeenLastCalledWith(true)
  expect(screen.getByText(copy.permission)).toBeTruthy()
  await act(async () => fireEvent.click(screen.getByRole('button', { name: common.openSettings })))
  expect(api.openPermissionSettings).toHaveBeenCalledOnce()
  api.setEnabled.mockResolvedValueOnce({ enabled: true, state: 'starting' })
  await act(async () => fireEvent.click(screen.getByRole('button', { name: common.retry })))
  expect(api.setEnabled).toHaveBeenLastCalledWith(true)
  expect(screen.queryByRole('alert')).toBeNull()
  await act(async () => emit({ enabled: true, state: 'ready' }))
  expect(toggle).toHaveProperty('ariaChecked', 'true')
  expect(view.container.textContent).toBe(`${copy.title}${copy.description}`)
  expect(screen.queryByRole('alert')).toBeNull()
  api.setEnabled.mockResolvedValueOnce({ enabled: false, state: 'disabled' })
  await act(async () => fireEvent.click(toggle))
  expect(toggle).toHaveProperty('ariaChecked', 'false')
  view.unmount()
  expect(unsubscribe).toHaveBeenCalledOnce()
})

it('rereads after an unconfirmed write and keeps failed reads recoverable', async () => {
  const { api } = bridge()
  api.getSettings.mockRejectedValueOnce(new Error('IPC unavailable'))
  render(<HudModifierSettings />)
  expect(await screen.findByText(common.loadFailed)).toBeTruthy()
  await act(async () => fireEvent.click(screen.getByRole('button', { name: common.retry })))
  api.setEnabled.mockRejectedValueOnce(new Error('unconfirmed'))
  await act(async () => fireEvent.click(screen.getByRole('switch', { name: copy.title })))
  expect(screen.getByText(common.saveFailed)).toBeTruthy()
  api.getSettings.mockResolvedValueOnce({ enabled: true, state: 'unavailable' })
  await act(async () => fireEvent.click(screen.getByRole('button', { name: common.retry })))
  expect(api.setEnabled).toHaveBeenCalledOnce()
  expect(screen.getByText(copy.unavailable)).toBeTruthy()
})
