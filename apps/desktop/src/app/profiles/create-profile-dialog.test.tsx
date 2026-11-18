// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { browserDefaultProjectDirKey, browserFolderBasePath } from '@/lib/browser-settings'
import { browserStartupProfileKey } from '@/lib/browser-startup-profile'
import { $shellChoice } from '@/store/shell-variant'

import { CreateProfileDialog } from './create-profile-dialog'

const api = vi.hoisted(() => ({ createProfile: vi.fn(async () => ({})), updateProfileSoul: vi.fn(async () => ({})) }))
vi.mock('@/hermes', () => api)
afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  $shellChoice.set(null)
})

it.each([
  [false, 'my-profile'],
  [true, 'my-agent']
])('suggests a name in each shell’s vocabulary (redesign=%s)', (redesign, placeholder) => {
  $shellChoice.set(redesign)
  render(<CreateProfileDialog onClose={vi.fn()} open />)
  expect(screen.getByRole('textbox', { name: en.profiles.nameLabel }).getAttribute('placeholder')).toBe(placeholder)
})

it.each([true, false])(
  'creates the initial persona, retaining the older-backend fallback (soul_set=%s)',
  async applied => {
    api.createProfile.mockResolvedValueOnce({ soul_set: applied })
    const created = vi.fn()
    render(<CreateProfileDialog onClose={vi.fn()} onCreated={created} open />)
    fireEvent.change(screen.getByRole('textbox', { name: en.profiles.nameLabel }), { target: { value: 'new-persona' } })
    fireEvent.change(screen.getByRole('textbox', { name: /SOUL.md/ }), { target: { value: 'A new persona' } })
    fireEvent.click(screen.getByRole('button', { name: en.profiles.createAction }))
    await waitFor(() => expect(created).toHaveBeenCalledWith('new-persona'))
    expect(api.createProfile).toHaveBeenCalledWith({
      name: 'new-persona',
      clone_from: 'default',
      soul: 'A new persona'
    })

    if (applied) {
      expect(api.updateProfileSoul).not.toHaveBeenCalled()
    } else {
      expect(api.updateProfileSoul).toHaveBeenCalledWith('new-persona', 'A new persona', 'new-persona')
    }
  }
)

it('does not let a new browser profile inherit the prefs a deleted one left under its name', async () => {
  const win = window as Window & { __HERMES_BASE_PATH__?: string }
  const folderKey = browserDefaultProjectDirKey('/hermes', 'work')
  const startupKey = browserStartupProfileKey('/hermes')
  window.document.documentElement.dataset.hermesDesktopHost = 'browser'
  win.__HERMES_BASE_PATH__ = '/hermes'
  localStorage.setItem(folderKey, '/srv/deleted-work')
  localStorage.setItem(startupKey, JSON.stringify({ connectionId: null, profile: 'work' }))
  localStorage.setItem(browserDefaultProjectDirKey('/hermes', 'alpha'), '/srv/alpha')

  try {
    const created = vi.fn()
    render(<CreateProfileDialog onClose={vi.fn()} onCreated={created} open />)
    fireEvent.change(screen.getByRole('textbox', { name: en.profiles.nameLabel }), { target: { value: 'work' } })
    fireEvent.click(screen.getByRole('button', { name: en.profiles.createAction }))
    await waitFor(() => expect(created).toHaveBeenCalledWith('work'))

    expect(localStorage.getItem(folderKey)).toBeNull()
    expect(localStorage.getItem(startupKey)).toBeNull()
    expect(localStorage.getItem(browserDefaultProjectDirKey('/hermes', 'alpha'))).toBe('/srv/alpha')
  } finally {
    localStorage.clear()
    delete window.document.documentElement.dataset.hermesDesktopHost
    delete win.__HERMES_BASE_PATH__
  }
})

it('keeps duplicate creation and its older-backend persona write on the captured gateway', async () => {
  const win = window as Window & { __HERMES_BASE_PATH__?: string }
  const owner = { connectionId: 'gateway-a', profile: 'research' }
  const nextOwner = { connectionId: 'gateway-b', profile: 'research' }
  window.document.documentElement.dataset.hermesDesktopHost = 'browser'
  win.__HERMES_BASE_PATH__ = '/hermes'
  const oldKey = browserDefaultProjectDirKey(browserFolderBasePath(owner.connectionId), 'duplicate')
  const otherKey = browserDefaultProjectDirKey(browserFolderBasePath(nextOwner.connectionId), 'duplicate')
  localStorage.setItem(oldKey, '/deleted')
  localStorage.setItem(otherKey, '/keep')
  let finishCreate!: (value: { soul_set: boolean }) => void
  api.createProfile.mockImplementationOnce(
    () =>
      new Promise(resolve => {
        finishCreate = resolve
      })
  )
  const created = vi.fn()

  try {
    const { rerender } = render(
      <CreateProfileDialog initialCloneFrom="research" onClose={vi.fn()} onCreated={created} open scope={owner} />
    )

    fireEvent.change(screen.getByRole('textbox', { name: en.profiles.nameLabel }), { target: { value: 'duplicate' } })
    fireEvent.change(screen.getByRole('textbox', { name: /SOUL.md/ }), { target: { value: 'Research persona' } })
    fireEvent.click(screen.getByRole('button', { name: en.profiles.createAction }))
    rerender(
      <CreateProfileDialog initialCloneFrom="research" onClose={vi.fn()} onCreated={created} open scope={nextOwner} />
    )
    await act(async () => finishCreate({ soul_set: false }))
    await waitFor(() => expect(created).toHaveBeenCalledWith('duplicate'))
    expect(api.createProfile).toHaveBeenCalledWith(
      { name: 'duplicate', clone_from: 'research', soul: 'Research persona' },
      owner
    )
    expect(api.updateProfileSoul).toHaveBeenCalledWith('duplicate', 'Research persona', {
      connectionId: owner.connectionId,
      profile: 'duplicate'
    })
    expect(localStorage.getItem(oldKey)).toBeNull()
    expect(localStorage.getItem(otherKey)).toBe('/keep')
  } finally {
    localStorage.clear()
    delete window.document.documentElement.dataset.hermesDesktopHost
    delete win.__HERMES_BASE_PATH__
  }
})

it('closes itself shortly after a create, but never calls back once it has unmounted', async () => {
  const stays = vi.fn()
  const first = render(<CreateProfileDialog onClose={stays} onCreated={vi.fn()} open />)
  fireEvent.change(screen.getByRole('textbox', { name: en.profiles.nameLabel }), { target: { value: 'kept' } })
  fireEvent.click(screen.getByRole('button', { name: en.profiles.createAction }))
  await waitFor(() => expect(stays).toHaveBeenCalledOnce(), { timeout: 2000 })
  first.unmount()

  const gone = vi.fn()
  const created = vi.fn()
  const second = render(<CreateProfileDialog onClose={gone} onCreated={created} open />)
  fireEvent.change(screen.getByRole('textbox', { name: en.profiles.nameLabel }), { target: { value: 'left' } })
  fireEvent.click(screen.getByRole('button', { name: en.profiles.createAction }))
  await waitFor(() => expect(created).toHaveBeenCalledWith('left'))
  second.unmount()
  await new Promise(resolve => setTimeout(resolve, 1000))

  expect(gone).not.toHaveBeenCalled()
})
