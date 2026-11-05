import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import { en } from '@/i18n/en'
import {
  $backendUpdateApply,
  $updateApply,
  $updateOverlayOpen,
  $updateOverlayTarget,
  resetUpdateApplyState
} from '@/store/updates'

import { UpdatesOverlay } from './updates-overlay'

afterEach((): void => {
  cleanup()
  $updateOverlayOpen.set(false)
  resetUpdateApplyState()
})

const applying = {
  applying: true,
  stage: 'pull' as const,
  message: '',
  percent: null,
  error: null,
  command: null,
  log: []
}

async function renderOverlay(): Promise<void> {
  await act(async (): Promise<void> => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <UpdatesOverlay />
      </I18nProvider>
    )
  })
}

// A backend update restarts the backend, not this window: a browser tab or remote Desktop window
// stays open and reconnects, so it must never be told it will close.
it('tells a window updating its backend to stay open, never that it will close', async (): Promise<void> => {
  $updateOverlayTarget.set('backend')
  $updateOverlayOpen.set(true)
  $backendUpdateApply.set(applying)
  await renderOverlay()

  expect(screen.getByText(en.updates.applyingStayBackend)).toBeTruthy()
  expect(screen.queryByText(en.updates.applyingClose)).toBeNull()
})

it('still tells a self-updating app that its window will close', async (): Promise<void> => {
  $updateOverlayTarget.set('client')
  $updateOverlayOpen.set(true)
  $updateApply.set(applying)
  await renderOverlay()

  expect(screen.getByText(en.updates.applyingClose)).toBeTruthy()
  expect(screen.queryByText(en.updates.applyingStayBackend)).toBeNull()
})
