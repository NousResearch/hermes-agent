import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import type { DesktopUpdateStatus } from '@/global'
import { I18nProvider } from '@/i18n/context'
import { $updateOverlayOpen, $updateOverlayTarget, $updateStatus, resetUpdateApplyState } from '@/store/updates'

import { UpdatesOverlay } from './updates-overlay'

async function renderUpdatesOverlay() {
  await act(async () => {
    render(
      <I18nProvider configClient={{ getConfig: async () => ({}), saveConfig: async () => ({ ok: true }) }}>
        <UpdatesOverlay />
      </I18nProvider>
    )
  })
}

// 2026-09-05: a checkout at the branch tip whose app.asar was still the
// previous build showed "You're all set" and never offered the rebuild. The
// version IPC already knew (bundleOutOfSync); the update check now carries it
// and the overlay treats it as an update whose only work is the rebuild.
describe('UpdatesOverlay: git-current checkout under a stale bundle', () => {
  afterEach(() => {
    cleanup()
    $updateOverlayOpen.set(false)
    $updateOverlayTarget.set('client')
    $updateStatus.set(null)
    resetUpdateApplyState()
  })

  it('offers the rebuild instead of "all set" when nothing is behind but the bundle is stale', async () => {
    $updateOverlayTarget.set('client')
    $updateOverlayOpen.set(true)
    $updateStatus.set({
      supported: true,
      updateAvailable: true,
      bundleOutOfSync: true,
      behind: 0,
      commits: []
    } satisfies DesktopUpdateStatus)

    await renderUpdatesOverlay()

    expect(screen.getByText('Desktop app needs a rebuild')).toBeTruthy()
    expect(screen.queryByText('You’re all set')).toBeNull()
    expect(screen.queryByText('New update available')).toBeNull()
    expect(screen.getByText('Update now')).toBeTruthy()
  })

  it('a bundle flag alone (older main process omitting updateAvailable) still counts as an update', async () => {
    $updateOverlayTarget.set('client')
    $updateOverlayOpen.set(true)
    $updateStatus.set({
      supported: true,
      bundleOutOfSync: true,
      behind: 0,
      commits: []
    } satisfies DesktopUpdateStatus)

    await renderUpdatesOverlay()

    expect(screen.getByText('Desktop app needs a rebuild')).toBeTruthy()
    expect(screen.queryByText('You’re all set')).toBeNull()
  })

  it('keeps "all set" when git is current and the bundle is current', async () => {
    $updateOverlayTarget.set('client')
    $updateOverlayOpen.set(true)
    $updateStatus.set({
      supported: true,
      updateAvailable: false,
      bundleOutOfSync: false,
      behind: 0,
      commits: []
    } satisfies DesktopUpdateStatus)

    await renderUpdatesOverlay()

    expect(screen.getByText('You’re all set')).toBeTruthy()
  })

  it('commits behind plus a stale bundle reads as a normal update, not a bare rebuild', async () => {
    $updateOverlayTarget.set('client')
    $updateOverlayOpen.set(true)
    $updateStatus.set({
      supported: true,
      updateAvailable: true,
      bundleOutOfSync: true,
      behind: 2,
      commits: []
    } satisfies DesktopUpdateStatus)

    await renderUpdatesOverlay()

    expect(screen.getByText('New update available')).toBeTruthy()
    expect(screen.queryByText('Desktop app needs a rebuild')).toBeNull()
  })

  it('an unknown commit count does not claim that only a rebuild is needed', async () => {
    $updateOverlayTarget.set('client')
    $updateOverlayOpen.set(true)
    $updateStatus.set({
      supported: true,
      updateAvailable: true,
      bundleOutOfSync: true,
      behind: null,
      commits: []
    } satisfies DesktopUpdateStatus)

    await renderUpdatesOverlay()

    expect(screen.getByText('New update available')).toBeTruthy()
    expect(screen.queryByText('Desktop app needs a rebuild')).toBeNull()
  })

})
