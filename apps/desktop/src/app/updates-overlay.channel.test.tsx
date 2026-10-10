import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import type { DesktopUpdateStatus } from '@/global'
import { I18nProvider } from '@/i18n/context'
import { $desktopVersion, $updateOverlayOpen, $updateOverlayTarget, $updateStatus } from '@/store/updates'

import { UpdatesOverlay } from './updates-overlay'

afterEach((): void => {
  cleanup()
  $updateOverlayOpen.set(false)
  $updateStatus.set(null)
  $desktopVersion.set(null)
})

async function openWith(status: DesktopUpdateStatus): Promise<void> {
  $desktopVersion.set({ appVersion: '0.21.6', electronVersion: '37', hermesRoot: '/h', nodeVersion: '22', platform: 'linux' })
  $updateStatus.set({ supported: true, behind: 0, mechanism: 'posix-handoff', ...status })
  $updateOverlayTarget.set('client')
  $updateOverlayOpen.set(true)
  await act(async (): Promise<void> => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <UpdatesOverlay />
      </I18nProvider>
    )
  })
}

it('offers Change only where Settings can switch; a custom branch or older runtime gets the label alone', async () => {
  await openWith({ channel: 'stable', channelSelectable: true })
  expect(screen.getByText('Channel')).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Change' })).toBeTruthy()
  cleanup()

  await openWith({ branch: 'feature/gui', channelSelectable: true })
  expect(screen.getByText(/Branch: feature\/gui/)).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Change' })).toBeNull()
  cleanup()

  // An older runtime cannot save a channel, so Settings shows no selector to land on.
  await openWith({ channel: 'stable' })
  expect(screen.getByText('Stable releases')).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Change' })).toBeNull()
})
