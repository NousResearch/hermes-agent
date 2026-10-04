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
  Reflect.deleteProperty(window, 'hermesDesktop')
  vi.restoreAllMocks()
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

it('keeps the install actions in a non-scrolling footer below the changelog scroll area', async (): Promise<void> => {
  const commits = Array.from({ length: 3 }, (_, index) => ({
    sha: `000000000000000000000000000000000000000${index}`.slice(-40),
    summary: `fix(desktop): changelog row ${index} for a pending list`,
    author: 'hermes',
    at: 1_759_200_000 + index * 60
  }))

  $updateOverlayTarget.set('client')
  $updateOverlayOpen.set(true)
  $updateStatus.set({
    supported: true,
    updateAvailable: true,
    behind: commits.length + 6,
    commits,
    branch: 'main'
  })
  await act(async (): Promise<void> => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <UpdatesOverlay />
      </I18nProvider>
    )
  })

  // The dialog renders through a portal, so query from the dialog element
  // itself rather than the render container.
  const overlay = screen.getByRole('dialog')
  const scrollArea = overlay.querySelector('[data-slot="update-scroll-area"]')
  expect(scrollArea).toBeTruthy()
  expect(scrollArea?.className).toContain('overflow-y-auto')

  // The footer's pinning rests on utility classes across three elements, and the
  // footer only holds if the merged body class actually resolves to a flex column —
  // tailwind-merge lets the shell's `grid` win when the override drops `flex`. Assert
  // the merged class strings (jsdom has no layout engine, so these are the honest
  // ceiling): the body must be a flex column, and both the wrapper and the scroll
  // area must carry `flex-1`, or the footer has no flex context to divide (#128170).
  const body = overlay.firstElementChild as HTMLElement
  expect(body.className).toContain('flex')
  expect(body.className).toContain('flex-col')
  expect(body.className).not.toContain('grid ')
  const wrapper = scrollArea?.parentElement as HTMLElement
  expect(wrapper.className).toContain('flex-1')
  expect(scrollArea?.className).toContain('flex-1')

  // The actions sit in their own pinned footer, outside the scrolling box, so
  // a long changelog can never push them below the fold (#128170).
  const footer = overlay.querySelector('[data-slot="update-actions"]')
  expect(footer).toBeTruthy()
  expect(footer?.className).toContain('shrink-0')
  expect(scrollArea?.contains(footer as Node)).toBe(false)
  expect(footer?.contains(screen.getByRole('button', { name: en.updates.updateNow }))).toBe(true)
  expect(footer?.contains(screen.getByRole('button', { name: en.updates.maybeLater }))).toBe(true)
  expect(footer?.contains(screen.getByText(en.updates.moreChanges(6)))).toBe(true)
})
