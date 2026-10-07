import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import { en } from '@/i18n/en'
import {
  $backendUpdateApply,
  $updateApply,
  $updateOverlayOpen,
  $updateOverlayTarget,
  $updateStatus,
  applyUpdates,
  resetUpdateApplyState
} from '@/store/updates'

import { UpdatesOverlay } from './updates-overlay'

afterEach((): void => {
  cleanup()
  $updateOverlayOpen.set(false)
  $updateStatus.set(null)
  resetUpdateApplyState()
  Reflect.deleteProperty(window, 'rabbitDesktop')
  vi.restoreAllMocks()
})

it('shows manual recovery guidance without claiming the help command installs an update', async (): Promise<void> => {
  const message: string = 'Choose the intended branch or channel before updating this older checkout.'
  window.rabbitDesktop = {
    updates: {
      apply: async (): Promise<unknown> => ({ ok: true, manual: true, command: 'rabbit update --help', message })
    }
  } as unknown as Window['rabbitDesktop']
  $updateOverlayTarget.set('client')
  $updateOverlayOpen.set(true)
  $updateStatus.set({ supported: false, reason: 'source-probe-unavailable', message })
  await applyUpdates()
  expect($updateApply.get().message).toBe(message)
  await act(async (): Promise<void> => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <UpdatesOverlay />
      </I18nProvider>
    )
  })
  expect(screen.getByText(message)).toBeTruthy()
  expect(screen.getByText('rabbit update --help')).toBeTruthy()
  expect(screen.queryByText(en.updates.manualPickedUp)).toBeNull()
})

it('titles a command-less backend refusal honestly and offers nothing to copy', async (): Promise<void> => {
  const message: string = 'Rabbit updates are managed outside this dashboard in containerized environments.'
  $updateOverlayTarget.set('backend')
  $updateOverlayOpen.set(true)
  $backendUpdateApply.set({
    applying: false,
    stage: 'manual',
    message,
    percent: null,
    error: null,
    command: null,
    log: []
  })
  await act(async (): Promise<void> => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <UpdatesOverlay />
      </I18nProvider>
    )
  })
  expect(screen.getByText(en.updates.manualUnavailableTitle)).toBeTruthy()
  expect(screen.queryByText(en.updates.manualTitle)).toBeNull()
  expect(screen.getByText(message)).toBeTruthy()
  expect(screen.queryByText(en.updates.copy)).toBeNull()
})

it('names the backend, not a local install, when a remote refusal carries a bare command', async (): Promise<void> => {
  $updateOverlayTarget.set('backend')
  $updateOverlayOpen.set(true)
  $backendUpdateApply.set({
    applying: false,
    stage: 'manual',
    message: '',
    percent: null,
    error: null,
    command: 'docker pull seven0070/Rabbit-:latest',
    log: []
  })
  await act(async (): Promise<void> => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <UpdatesOverlay />
      </I18nProvider>
    )
  })
  expect(screen.getByText('docker pull seven0070/Rabbit-:latest')).toBeTruthy()
  expect(screen.getByText(en.updates.manualBodyBackend)).toBeTruthy()
  expect(screen.getByText(en.updates.manualPickedUpBackend)).toBeTruthy()
  expect(screen.queryByText(en.updates.manualBody)).toBeNull()
})

it('keeps the client title for a command-less client manual stage', async (): Promise<void> => {
  $updateOverlayTarget.set('client')
  $updateOverlayOpen.set(true)
  $updateApply.set({
    applying: false,
    stage: 'manual',
    message: 'Rabbit will pick up the new version next time you launch it.',
    percent: null,
    error: null,
    command: null,
    log: []
  })
  await act(async (): Promise<void> => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <UpdatesOverlay />
      </I18nProvider>
    )
  })
  expect(screen.getByText(en.updates.manualTitle)).toBeTruthy()
  expect(screen.queryByText(en.updates.manualUnavailableTitle)).toBeNull()
})
