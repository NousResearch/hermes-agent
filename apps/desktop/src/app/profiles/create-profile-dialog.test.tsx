import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createProfile } from '@/hermes'
import type { ProfileInfo } from '@/types/hermes'

import { CreateProfileDialog } from './create-profile-dialog'

// These pin the guardrail this dialog owes the fleet: the one free-text door that
// mints a profile must not silently scaffold a typo. `veste-frontend-dev` beside
// `vestr-frontend-dev` is the reported incident, verbatim.
//
// Real i18n (no provider → English), so the assertions read the copy a user sees.

afterEach(cleanup)

vi.mock('@/hermes', () => ({
  createProfile: vi.fn(async () => ({ name: 'x', ok: true, path: '/x' })),
  updateProfileSoul: vi.fn(async () => ({ ok: true }))
}))

function makeProfile(name: string, isDefault = false): ProfileInfo {
  return {
    has_env: false,
    is_default: isDefault,
    model: null,
    name,
    path: `/home/user/.hermes/profiles/${name}`,
    provider: null,
    skill_count: 0
  }
}

const PROFILES = [makeProfile('default', true), makeProfile('vestr-frontend-dev')]

function renderDialog() {
  const onUseExisting = vi.fn()

  render(
    <CreateProfileDialog onClose={vi.fn()} onCreated={vi.fn()} onUseExisting={onUseExisting} open profiles={PROFILES} />
  )

  return { onUseExisting }
}

const typeName = (value: string) => fireEvent.change(screen.getByLabelText('Name'), { target: { value } })

beforeEach(() => {
  vi.mocked(createProfile).mockClear()
})

describe('CreateProfileDialog typo guard', () => {
  it('flags a one-keystroke name and offers the profile that was meant', async () => {
    const { onUseExisting } = renderDialog()

    typeName('veste-frontend-dev')

    expect(await screen.findByText(/one character away from the existing profile “vestr-frontend-dev”/)).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Use vestr-frontend-dev' }))

    expect(onUseExisting).toHaveBeenCalledWith('vestr-frontend-dev')
    expect(createProfile).not.toHaveBeenCalled()
  })

  it('names the scaffold cost and still creates when the user insists', async () => {
    renderDialog()

    typeName('veste-frontend-dev')

    await screen.findByText(/scaffolds a stock SOUL\.md and boots its own backend/)
    fireEvent.click(screen.getByRole('button', { name: 'Create anyway' }))

    await waitFor(() =>
      expect(createProfile).toHaveBeenCalledWith({ name: 'veste-frontend-dev', clone_from: 'default' })
    )
  })

  it('refuses a name that is already taken instead of round-tripping the backend', async () => {
    renderDialog()

    typeName('vestr-frontend-dev')
    fireEvent.click(screen.getByRole('button', { name: 'Create profile' }))

    expect(await screen.findByText(/A profile named “vestr-frontend-dev” already exists/)).toBeTruthy()
    expect(createProfile).not.toHaveBeenCalled()
  })

  it('stays out of the way for a deliberate new name', async () => {
    renderDialog()

    typeName('scratch-space')

    expect(screen.queryByText(/one character away/)).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Create profile' }))

    await waitFor(() => expect(createProfile).toHaveBeenCalledWith({ name: 'scratch-space', clone_from: 'default' }))
  })
})
