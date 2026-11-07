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

// The update's drain waits on approvals and questions in the chats under this overlay: a window
// updating its backend may hide the progress, a self-updating app (which restarts) may not.
it('lets a window updating its backend hide the progress, but not a self-updating app', async (): Promise<void> => {
  $updateOverlayTarget.set('backend')
  $updateOverlayOpen.set(true)
  $backendUpdateApply.set(applying)
  await renderOverlay()

  const close = screen.getByRole('button', { name: /close/i })
  await act(async (): Promise<void> => close.click())
  expect($updateOverlayOpen.get()).toBe(false)
  expect($backendUpdateApply.get().applying).toBe(true)

  cleanup()
  $updateOverlayTarget.set('client')
  $updateOverlayOpen.set(true)
  $updateApply.set(applying)
  await renderOverlay()
  expect(screen.queryByRole('button', { name: /close/i })).toBeNull()
})

it('still tells a self-updating app that its window will close', async (): Promise<void> => {
  $updateOverlayTarget.set('client')
  $updateOverlayOpen.set(true)
  $updateApply.set(applying)
  await renderOverlay()

  expect(screen.getByText(en.updates.applyingClose)).toBeTruthy()
  expect(screen.queryByText(en.updates.applyingStayBackend)).toBeNull()
})

// runBackendUpdate's poll still owns a backend apply in its restart stage: hiding it there keeps that state.
it('hides a backend update during its restart without resetting it', async (): Promise<void> => {
  $updateOverlayTarget.set('backend')
  $updateOverlayOpen.set(true)
  $backendUpdateApply.set({ ...applying, stage: 'restart' })
  await renderOverlay()

  await act(async (): Promise<void> => screen.getByRole('button', { name: /close/i }).click())

  expect($updateOverlayOpen.get()).toBe(false)
  expect($backendUpdateApply.get()).toMatchObject({ applying: true, stage: 'restart' })
})

it('dismissing a failed backend update clears only the backend state', async (): Promise<void> => {
  const clientError = { ...applying, applying: false, stage: 'error' as const, message: 'client failed' }
  $updateApply.set(clientError)
  $updateOverlayTarget.set('backend')
  $updateOverlayOpen.set(true)
  $backendUpdateApply.set({ ...clientError, message: 'backend failed' })
  await renderOverlay()

  await act(async (): Promise<void> => screen.getAllByRole('button', { name: /close/i })[0].click())

  expect($backendUpdateApply.get().stage).toBe('idle')
  expect($updateApply.get()).toEqual(clientError)
})

// A client's own outcome holds the overlay; a backend failure that arrived meanwhile waits behind it
// and shows when the person closes the client's view.
it.each([
  ['error', { stage: 'error' as const, message: 'client failed' }],
  ['manual', { stage: 'manual' as const, message: 'run this', command: 'run this' }]
])('closing a client %s shows a backend failure that waited behind it', async (_l, client) => {
  $updateApply.set({ ...applying, applying: false, ...client })
  $backendUpdateApply.set({ ...applying, applying: false, stage: 'error', message: 'backend failed' })
  $updateOverlayTarget.set('client')
  $updateOverlayOpen.set(true)
  await renderOverlay()

  await act(async (): Promise<void> => screen.getAllByRole('button', { name: /close/i })[0].click())

  expect($updateApply.get().stage).toBe('idle')
  expect($updateOverlayTarget.get()).toBe('backend')
  expect($updateOverlayOpen.get()).toBe(true)
  expect($backendUpdateApply.get().stage).toBe('error')
})
