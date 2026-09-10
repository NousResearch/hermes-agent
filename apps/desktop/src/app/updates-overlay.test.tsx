import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { BootFailureOverlay } from '@/components/boot-failure-overlay'
import { $desktopBoot } from '@/store/boot'
import { $desktopOnboarding } from '@/store/onboarding'
import { $backendUpdateApply, $backendUpdateStatus, $updateOverlayOpen, $updateOverlayTarget } from '@/store/updates'

import { UpdatesOverlay } from './updates-overlay'

beforeEach(() => {
  $desktopBoot.set({
    error: null,
    fakeMode: false,
    message: 'ready',
    phase: 'renderer.ready',
    progress: 100,
    running: false,
    timestamp: Date.now(),
    visible: false
  })
  $desktopOnboarding.set({
    configured: true,
    flow: { status: 'idle' },
    mode: 'oauth',
    providers: null,
    reason: null,
    requested: false,
    firstRunSkipped: false,
    manual: false,
    localEndpoint: false
  })
  $backendUpdateStatus.set({
    behind: 1,
    commits: [],
    fetchedAt: Date.now(),
    supported: true,
    updateAvailable: true
  })
  $backendUpdateApply.set({
    applying: true,
    stage: 'restart',
    message: 'Restarting backend…',
    percent: null,
    error: null,
    command: null,
    log: []
  })
  $updateOverlayTarget.set('backend')
  $updateOverlayOpen.set(true)
})

afterEach(() => {
  cleanup()
  $updateOverlayOpen.set(false)
})

describe('UpdatesOverlay recovery ownership', () => {
  it('releases its modal lock when a hard boot failure takes over', async () => {
    render(
      <>
        <UpdatesOverlay />
        <BootFailureOverlay />
      </>
    )

    await waitFor(() => expect(window.document.body.style.pointerEvents).toBe('none'))

    await act(async () => {
      $desktopBoot.set({
        ...$desktopBoot.get(),
        error: 'Your remote gateway session has expired.',
        message: 'Desktop boot failed',
        phase: 'renderer.error',
        running: false,
        visible: true
      })
    })

    await waitFor(() => expect($updateOverlayOpen.get()).toBe(false))
    await waitFor(() => expect(window.document.body.style.pointerEvents).not.toBe('none'))

    fireEvent.click(screen.getByRole('button', { name: /gateway settings/i }))
    expect(await screen.findByRole('button', { name: /back/i })).toBeTruthy()
  })
})
