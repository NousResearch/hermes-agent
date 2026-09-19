import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import { en } from '@/i18n/en'
import { $updateApply, $updateOverlayOpen, $updateOverlayTarget, $updateStatus, applyUpdates, resetUpdateApplyState } from '@/store/updates'

import { UpdatesOverlay } from './updates-overlay'

afterEach((): void => {
  cleanup()
  $updateOverlayOpen.set(false)
  $updateStatus.set(null)
  resetUpdateApplyState()
  Reflect.deleteProperty(window, 'hermesDesktop')
  vi.restoreAllMocks()
})

it('shows manual recovery guidance without claiming the help command installs an update', async (): Promise<void> => {
  const message: string = 'Choose the intended branch or channel before updating this older checkout.'
  window.hermesDesktop = {
    updates: { apply: async (): Promise<unknown> => ({ ok: true, manual: true, command: 'hermes update --help', message }) }
  } as unknown as Window['hermesDesktop']
  $updateOverlayTarget.set('client')
  $updateOverlayOpen.set(true)
  $updateStatus.set({ supported: false, reason: 'source-probe-unavailable', message })
  await applyUpdates()
  expect($updateApply.get().message).toBe(message)
  await act(async (): Promise<void> => {
    render(<I18nProvider configClient={null} initialLocale="en"><UpdatesOverlay /></I18nProvider>)
  })
  expect(screen.getByText(message)).toBeTruthy()
  expect(screen.getByText('hermes update --help')).toBeTruthy()
  expect(screen.queryByText(en.updates.manualPickedUp)).toBeNull()
})
