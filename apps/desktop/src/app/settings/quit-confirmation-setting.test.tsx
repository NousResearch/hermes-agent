import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { deferred } from '@/test/deferred'
import { stubMenuDomApis } from '@/test/jsdom'

import type { QuitConfirmationMode } from '../../../electron/quit-guard'

import { QuitConfirmationSetting } from './quit-confirmation-setting'

const copy = en.settings.quitConfirmation

const labels: Record<QuitConfirmationMode, string> = {
  never: copy.never,
  'while-working': copy.whileWorking,
  always: copy.always
}

const alternatives: Record<QuitConfirmationMode, QuitConfirmationMode> = {
  never: 'while-working',
  'while-working': 'always',
  always: 'never'
}

const modes: QuitConfirmationMode[] = ['never', 'while-working', 'always']

beforeEach(() => {
  stubMenuDomApis()
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it.each(modes)('keeps the native %s preference authoritative through read and write failures', async mode => {
  const initialRead = deferred<QuitConfirmationMode>()
  const failedWrite = deferred<QuitConfirmationMode>()
  const successfulWrite = deferred<QuitConfirmationMode>()

  const getQuitConfirmation = vi
    .fn<() => Promise<QuitConfirmationMode>>()
    .mockReturnValueOnce(initialRead.promise)
    .mockResolvedValue(mode)

  const setQuitConfirmation = vi
    .fn<(value: QuitConfirmationMode) => Promise<QuitConfirmationMode>>()
    .mockReturnValueOnce(failedWrite.promise)
    .mockReturnValueOnce(successfulWrite.promise)

  vi.stubGlobal('hermesDesktop', { settings: { getQuitConfirmation, setQuitConfirmation } })
  const view = render(<QuitConfirmationSetting />)
  const trigger = screen.getByRole('combobox', { name: copy.title })

  expect(trigger.hasAttribute('disabled')).toBe(true)
  expect(setQuitConfirmation).not.toHaveBeenCalled()

  await act(async () => initialRead.reject(new Error('read unavailable')))
  expect(screen.getByRole('alert').textContent).toBe(copy.loadFailed)
  expect(trigger.hasAttribute('disabled')).toBe(true)

  fireEvent.click(screen.getByRole('button', { name: en.common.retry }))
  await waitFor(() => expect(trigger.hasAttribute('disabled')).toBe(false))
  expect(trigger.textContent).toBe(labels[mode])
  expect(setQuitConfirmation).not.toHaveBeenCalled()

  const next = alternatives[mode]

  fireEvent.keyDown(trigger, { key: 'ArrowDown' })
  fireEvent.click(screen.getByRole('option', { name: labels[next] }))
  expect(setQuitConfirmation).toHaveBeenLastCalledWith(next)
  expect(trigger.hasAttribute('disabled')).toBe(true)
  expect(trigger.textContent).toBe(labels[mode])

  await act(async () => failedWrite.reject(new Error('disk full')))
  expect(screen.getByRole('alert').textContent).toBe(copy.saveFailed)
  expect(trigger.textContent).toBe(labels[mode])
  expect(trigger.hasAttribute('disabled')).toBe(false)

  fireEvent.keyDown(trigger, { key: 'ArrowDown' })
  fireEvent.click(screen.getByRole('option', { name: labels[next] }))
  await act(async () => successfulWrite.resolve(next))
  expect(trigger.textContent).toBe(labels[next])
  expect(trigger.hasAttribute('disabled')).toBe(false)
  expect(screen.queryByRole('alert')).toBeNull()

  const refreshRead = deferred<QuitConfirmationMode>()
  getQuitConfirmation.mockReturnValueOnce(refreshRead.promise)
  fireEvent.focus(window)
  expect(trigger.hasAttribute('disabled')).toBe(true)
  const callsWhileReading = getQuitConfirmation.mock.calls.length
  fireEvent.focus(window)
  expect(getQuitConfirmation).toHaveBeenCalledTimes(callsWhileReading)
  expect(setQuitConfirmation).toHaveBeenCalledTimes(2)

  await act(async () => refreshRead.resolve(mode))
  expect(trigger.textContent).toBe(labels[mode])
  expect(trigger.hasAttribute('disabled')).toBe(false)

  const staleRead = deferred<QuitConfirmationMode>()
  getQuitConfirmation.mockReturnValueOnce(staleRead.promise)
  fireEvent.focus(window)
  view.unmount()
  getQuitConfirmation.mockResolvedValue(next)
  const reopened = render(<QuitConfirmationSetting />)
  const reopenedTrigger = screen.getByRole('combobox', { name: copy.title })
  await waitFor(() => expect(reopenedTrigger.textContent).toBe(labels[next]))

  setQuitConfirmation.mockResolvedValueOnce(mode)
  fireEvent.keyDown(reopenedTrigger, { key: 'ArrowDown' })
  fireEvent.keyDown(screen.getByRole('option', { name: labels[mode] }), { key: 'Enter' })
  await waitFor(() => expect(reopenedTrigger.textContent).toBe(labels[mode]))
  await act(async () => staleRead.resolve(next))
  expect(reopenedTrigger.textContent).toBe(labels[mode])
  expect(setQuitConfirmation).toHaveBeenCalledTimes(3)

  const unmountedWrite = deferred<QuitConfirmationMode>()
  setQuitConfirmation.mockReturnValueOnce(unmountedWrite.promise)
  fireEvent.keyDown(reopenedTrigger, { key: 'ArrowDown' })
  fireEvent.keyDown(screen.getByRole('option', { name: labels[next] }), { key: 'Enter' })
  expect(reopenedTrigger.hasAttribute('disabled')).toBe(true)
  reopened.unmount()
  await act(async () => unmountedWrite.resolve(next))
  const afterSave = render(<QuitConfirmationSetting />)
  await waitFor(() => expect(screen.getByRole('combobox', { name: copy.title }).textContent).toBe(labels[next]))
  expect(setQuitConfirmation).toHaveBeenCalledTimes(4)

  afterSave.unmount()
  vi.stubGlobal('hermesDesktop', { settings: {} })
  render(<QuitConfirmationSetting />)
  expect(screen.queryByRole('combobox', { name: copy.title })).toBeNull()
})
