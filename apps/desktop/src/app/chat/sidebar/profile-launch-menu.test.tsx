import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $defaultProfileRoute } from '@/store/default-profile'
import { notify, notifyError } from '@/store/notifications'

import { ProfileLaunchContextMenu } from './profile-launch-menu'

vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

const openWindow = vi.fn()
const setDefault = vi.fn()

beforeEach(() => {
  openWindow.mockResolvedValue({ ok: true })
  setDefault.mockImplementation(async route => route)
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { openWindow, profile: { setDefault } }
  })
  $defaultProfileRoute.set(null)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  $defaultProfileRoute.set(null)
})

it('opens the clicked canonical route without selecting it and saves only that route as default', async () => {
  const select = vi.fn()
  const route = { connectionId: 'homelab', profile: 'designer' }
  render(
    <I18nProvider configClient={null}>
      <ProfileLaunchContextMenu {...route} label="Design studio">
        <button onClick={select} type="button">
          Design studio
        </button>
      </ProfileLaunchContextMenu>
    </I18nProvider>
  )

  fireEvent.contextMenu(screen.getByRole('button', { name: 'Design studio' }))
  await act(async () => fireEvent.click(screen.getByRole('menuitem', { name: 'Open in new window' })))
  expect(openWindow).toHaveBeenCalledWith(route)
  expect(select).not.toHaveBeenCalled()
  expect(setDefault).not.toHaveBeenCalled()

  fireEvent.contextMenu(screen.getByRole('button', { name: 'Design studio' }))
  await act(async () => fireEvent.click(screen.getByRole('menuitem', { name: 'Set as default' })))
  expect(setDefault).toHaveBeenCalledWith(route)
  expect($defaultProfileRoute.get()).toEqual(route)
  expect(select).not.toHaveBeenCalled()
  expect(notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'success' }))

  fireEvent.contextMenu(screen.getByRole('button', { name: 'Design studio' }))
  expect(screen.getByRole('menuitem', { name: 'Default profile' }).getAttribute('aria-disabled')).toBe('true')
})

it('distinguishes the same profile on another source and reports a rejected default write without success', async () => {
  const original = { connectionId: 'homelab', profile: 'designer' }
  $defaultProfileRoute.set(original)
  setDefault.mockRejectedValue(new Error('Disk is read-only'))
  render(
    <I18nProvider configClient={null}>
      <ProfileLaunchContextMenu connectionId="local" label="Local designer" profile="designer">
        <button type="button">Local designer</button>
      </ProfileLaunchContextMenu>
    </I18nProvider>
  )

  fireEvent.contextMenu(screen.getByRole('button', { name: 'Local designer' }))
  await act(async () => fireEvent.click(screen.getByRole('menuitem', { name: 'Set as default' })))
  expect(setDefault).toHaveBeenCalledWith({ connectionId: 'local', profile: 'designer' })
  expect($defaultProfileRoute.get()).toEqual(original)
  expect(notifyError).toHaveBeenCalled()
  expect(notify).not.toHaveBeenCalled()
})
