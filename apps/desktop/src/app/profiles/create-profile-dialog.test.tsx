// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

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
