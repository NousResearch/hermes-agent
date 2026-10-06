import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { runDebugShare } from '@/hermes'
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

vi.mock('@/hermes', async () => {
  const actual = await vi.importActual<typeof import('@/hermes')>('@/hermes')

  return { ...actual, runDebugShare: vi.fn() }
})

afterEach((): void => {
  cleanup()
  $updateOverlayOpen.set(false)
  $updateStatus.set(null)
  resetUpdateApplyState()
  Reflect.deleteProperty(window, 'hermesDesktop')
  vi.mocked(runDebugShare).mockReset()
  vi.restoreAllMocks()
})

it('runs Debug Share from an update error and makes each returned link copyable', async (): Promise<void> => {
  const reportUrl = 'https://paste.rs/report-id'
  const agentLogUrl = 'https://paste.rs/agent-log-id'
  const writeClipboard = vi.fn().mockResolvedValue(undefined)
  window.hermesDesktop = { writeClipboard } as unknown as Window['hermesDesktop']
  vi.mocked(runDebugShare).mockResolvedValue({
    auto_delete_seconds: 21600,
    failures: {},
    ok: true,
    redacted: true,
    urls: { Report: reportUrl, 'agent.log': agentLogUrl }
  })
  $updateOverlayTarget.set('client')
  $updateOverlayOpen.set(true)
  $updateStatus.set({ supported: false, reason: 'source-probe-unavailable', message: 'Update failed' })
  $updateApply.set({
    applying: false,
    stage: 'error',
    message: 'The update could not be installed.',
    percent: null,
    error: 'Update failed',
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

  fireEvent.click(screen.getByRole('button', { name: en.commandCenter.maintenance.debugShare }))

  expect(await screen.findByText(reportUrl)).toBeTruthy()
  expect(await screen.findByText(agentLogUrl)).toBeTruthy()
  expect(runDebugShare).toHaveBeenCalledOnce()

  const copyButtons = screen.getAllByRole('button', { name: en.commandCenter.maintenance.copyLink })
  expect(copyButtons).toHaveLength(2)
  fireEvent.click(copyButtons[0])
  await waitFor(() => expect(writeClipboard).toHaveBeenNthCalledWith(1, reportUrl))
  fireEvent.click(copyButtons[1])
  await waitFor(() => expect(writeClipboard).toHaveBeenNthCalledWith(2, agentLogUrl))
})

it('shows manual recovery guidance without claiming the help command installs an update', async (): Promise<void> => {
  const message: string = 'Choose the intended branch or channel before updating this older checkout.'
  window.hermesDesktop = {
    updates: {
      apply: async (): Promise<unknown> => ({ ok: true, manual: true, command: 'hermes update --help', message })
    }
  } as unknown as Window['hermesDesktop']
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
  expect(screen.getByText('hermes update --help')).toBeTruthy()
  expect(screen.queryByText(en.updates.manualPickedUp)).toBeNull()
})

it('titles a command-less backend refusal honestly and offers nothing to copy', async (): Promise<void> => {
  const message: string = 'Hermes updates are managed outside this dashboard in containerized environments.'
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
    command: 'docker pull nousresearch/hermes-agent:latest',
    log: []
  })
  await act(async (): Promise<void> => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <UpdatesOverlay />
      </I18nProvider>
    )
  })
  expect(screen.getByText('docker pull nousresearch/hermes-agent:latest')).toBeTruthy()
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
    message: 'Hermes will pick up the new version next time you launch it.',
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


it('shows concrete Debug Share failures without duplicating the generic error label', async (): Promise<void> => {
  vi.mocked(runDebugShare).mockResolvedValue({
    auto_delete_seconds: 21600,
    failures: { Report: 'upload failed' },
    ok: false,
    redacted: true,
    urls: {}
  })
  $updateOverlayTarget.set('client')
  $updateOverlayOpen.set(true)
  $updateStatus.set({ supported: false, reason: 'source-probe-unavailable', message: 'Update failed' })
  $updateApply.set({
    applying: false,
    stage: 'error',
    message: 'The update could not be installed.',
    percent: null,
    error: 'Update failed',
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

  fireEvent.click(screen.getByRole('button', { name: en.commandCenter.maintenance.debugShare }))

  const alert = await screen.findByRole('alert')
  expect(alert.textContent).toContain(en.commandCenter.maintenance.debugShareFailed)
  expect(alert.textContent).toContain('Report: upload failed')
  expect(alert.textContent).not.toContain(
    `${en.commandCenter.maintenance.debugShareFailed}: ${en.commandCenter.maintenance.debugShareFailed}`
  )
})
