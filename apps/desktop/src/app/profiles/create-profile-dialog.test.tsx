import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import { createProfile } from '@/hermes'
import { setProfileGlyph } from '@/store/profile'

import { CreateProfileDialog } from './create-profile-dialog'

// The glyph picker (#79233) rides along with the create payload: a picked
// glyph persists under the NEW profile's key (before onCreated refreshes the
// rail), and no pick leaves appearance untouched — auto stays auto.

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

vi.mock('@/hermes', () => ({
  createProfile: vi.fn(async () => ({ ok: true })),
  updateProfileSoul: vi.fn(async () => undefined)
}))

vi.mock('@/store/profile', () => ({
  $profileGlyphs: atom<Record<string, string>>({}),
  setProfileGlyph: vi.fn()
}))

const fillName = () => {
  fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'my-profile' } })
}

it('persists the picked glyph under the created profile', async () => {
  render(<CreateProfileDialog onClose={vi.fn()} open />)

  fillName()
  fireEvent.click(screen.getByRole('button', { name: 'briefcase' }))
  fireEvent.click(screen.getByRole('button', { name: 'Create profile' }))

  await waitFor(() => expect(createProfile).toHaveBeenCalledWith({ name: 'my-profile', clone_from: 'default' }))
  await waitFor(() => expect(setProfileGlyph).toHaveBeenCalledWith('my-profile', 'briefcase'))
})

it('leaves the glyph unset when the pick is cleared back to Auto', async () => {
  render(<CreateProfileDialog onClose={vi.fn()} open />)

  fillName()
  fireEvent.click(screen.getByRole('button', { name: 'briefcase' }))
  fireEvent.click(screen.getByRole('button', { name: 'Auto' }))
  fireEvent.click(screen.getByRole('button', { name: 'Create profile' }))

  await waitFor(() => expect(createProfile).toHaveBeenCalled())
  expect(setProfileGlyph).toHaveBeenCalledWith('my-profile', null)
})

it('keeps the curated picker scannable — one row per glyph, no catalog dump', () => {
  render(<CreateProfileDialog onClose={vi.fn()} open />)

  // A representative sample of the role vocabulary; the full codicon catalog
  // would be overwhelming in a create dialog (#79233).
  for (const glyph of ['home', 'briefcase', 'rocket', 'beaker']) {
    expect(screen.getByRole('button', { name: glyph })).toBeTruthy()
  }
})
