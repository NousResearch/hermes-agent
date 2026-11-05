import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { SendDiagnosticsHost } from '@/components/send-diagnostics-dialog'
import { I18nProvider } from '@/i18n/context'
import { en } from '@/i18n/en'
import { $gateway } from '@/store/gateway'
import { $sendDiagnostics } from '@/store/send-diagnostics'
import {
  $backendUpdateApply,
  $updateApply,
  $updateOverlayOpen,
  $updateOverlayTarget,
  resetUpdateApplyState
} from '@/store/updates'

import { UpdatesOverlay } from './updates-overlay'

const originalGateway = $gateway.get()

afterEach((): void => {
  cleanup()
  $updateOverlayOpen.set(false)
  resetUpdateApplyState()
  $sendDiagnostics.set(null)
  $gateway.set(originalGateway)
  vi.restoreAllMocks()
})

it('a failed backend update shares its output from the backend and shows the link in the error box', async () => {
  const request = vi.fn().mockResolvedValue({ ok: true, upload_id: 'u1', view_url: 'https://nas.example/view/u1' })
  $gateway.set({ request } as never)
  $updateOverlayTarget.set('backend')
  $updateOverlayOpen.set(true)
  $backendUpdateApply.set({
    applying: false,
    command: null,
    error: 'apply-failed',
    log: [
      { at: 1, message: 'Pulling origin/main', stage: 'pull' },
      { at: 2, message: 'error: venv rebuild failed (exit 3)', stage: 'pull' }
    ],
    message: 'Backend update failed.',
    percent: null,
    stage: 'error'
  })

  await act(async () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <UpdatesOverlay />
        <SendDiagnosticsHost />
      </I18nProvider>
    )
  })

  // The updater output is readable in the box itself.
  expect(screen.getByText(/venv rebuild failed \(exit 3\)/)).toBeTruthy()

  fireEvent.click(screen.getByRole('button', { name: en.errors.sendDiagnostics }))
  // Consent is still explicit and per-upload, inline — not a second modal.
  expect(request).not.toHaveBeenCalled()
  expect(screen.queryByText(en.sendDiagnostics.title)).toBeNull()

  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: en.sendDiagnostics.upload }))
  })

  const [method, params] = request.mock.calls[0]
  expect(method).toBe('diagnostics.share_nous')
  expect(params.error_context).toContain('Backend update failed.')
  expect(params.extra_files['update-output.log']).toContain('venv rebuild failed (exit 3)')
  expect(screen.getByText('https://nas.example/view/u1')).toBeTruthy()
  expect(screen.getByText(en.updates.errorTitle)).toBeTruthy()
})

it('a report never outlives its failure: switching the update target drops it', async () => {
  $gateway.set({ request: vi.fn() } as never)
  $updateOverlayTarget.set('backend')
  $updateOverlayOpen.set(true)

  const failed = {
    applying: false,
    command: null,
    error: 'apply-failed',
    log: [{ at: 1, message: 'error: venv rebuild failed (exit 3)', stage: 'pull' as const }],
    message: 'Backend update failed.',
    percent: null,
    stage: 'error' as const
  }

  $backendUpdateApply.set(failed)
  $updateApply.set({ ...failed, message: 'Desktop update failed.' })

  await act(async () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <UpdatesOverlay />
      </I18nProvider>
    )
  })

  fireEvent.click(screen.getByRole('button', { name: en.errors.sendDiagnostics }))
  expect($sendDiagnostics.get()?.errorContext).toContain('Backend update failed.')

  await act(async () => $updateOverlayTarget.set('client'))

  expect($sendDiagnostics.get()).toBeNull()
  expect(screen.queryByRole('button', { name: en.sendDiagnostics.upload })).toBeNull()
  expect(screen.getByRole('button', { name: en.errors.sendDiagnostics })).toBeTruthy()
})
