// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { browserDefaultProjectDirKey } from '@/lib/browser-settings'
import { browserStartupProfileKey } from '@/lib/browser-startup-profile'

import { CreateProfileDialog } from './create-profile-dialog'

const api = vi.hoisted(() => ({ createProfile: vi.fn(async () => ({})), updateProfileSoul: vi.fn(async () => ({})) }))
vi.mock('@/hermes', () => api)
afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

it.each([true, false])(
  'creates the initial persona, retaining the older-backend fallback (soul_set=%s)',
  async applied => {
    api.createProfile.mockResolvedValueOnce({ soul_set: applied })
    const created = vi.fn()
    render(<CreateProfileDialog onClose={vi.fn()} onCreated={created} open />)
    fireEvent.change(screen.getByPlaceholderText('my-profile'), { target: { value: 'new-persona' } })
    fireEvent.change(screen.getByRole('textbox', { name: /SOUL.md/ }), { target: { value: 'A new persona' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create profile' }))
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
    fireEvent.change(screen.getByPlaceholderText('my-profile'), { target: { value: 'work' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create profile' }))
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
