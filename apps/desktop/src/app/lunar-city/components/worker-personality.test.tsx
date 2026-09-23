// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { getProfileSoul } from '@/api/profiles'
import type { ProfileSoul } from '@/types/hermes'

import { entityKey } from '../identity'
import type { LunarEntity } from '../model'
import { deriveWorkerPersonality } from '../worker-personality'

import { WorkerPersonality } from './worker-personality'

vi.mock('@/api/profiles', () => ({ getProfileSoul: vi.fn() }))
afterEach(() => { cleanup(); vi.resetAllMocks() })

function worker(connectionId: string): LunarEntity {
  const identity = { kind: 'profile', connectionId, profile: 'same-name' } as const

  return { key: entityKey(identity), identity, authority: 'authoritative', observedAt: 1, animation: 'idle', destination: 'library' }
}

function deferred() {
  let resolve!: (value: ProfileSoul) => void
  const promise = new Promise<ProfileSoul>(done => { resolve = done })

  return { promise, resolve }
}

it('loads only on request with exact scope and discards previous owner responses and text', async () => {
  const oldRead = deferred()
  const newRead = deferred()
  vi.mocked(getProfileSoul).mockReturnValueOnce(oldRead.promise).mockReturnValueOnce(newRead.promise)
  const preview = vi.fn()
  const oldEntity = worker('old-gateway')
  const { rerender } = render(<WorkerPersonality entity={oldEntity} onPreviewGesture={preview} />)
  expect(getProfileSoul).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Preview city gesture' }))
  expect(preview).toHaveBeenCalledWith(oldEntity.key, deriveWorkerPersonality(oldEntity).gesture)
  fireEvent.click(screen.getByRole('button', { name: 'Load profile SOUL' }))
  expect(getProfileSoul).toHaveBeenLastCalledWith('same-name', { connectionId: 'old-gateway', profile: 'same-name' })
  rerender(<WorkerPersonality entity={worker('new-gateway')} />)
  fireEvent.click(screen.getByRole('button', { name: 'Load profile SOUL' }))
  expect(getProfileSoul).toHaveBeenLastCalledWith('same-name', { connectionId: 'new-gateway', profile: 'same-name' })
  await act(async () => newRead.resolve({ exists: true, content: 'Current source instructions' }))
  await act(async () => oldRead.resolve({ exists: true, content: 'Wrong owner instructions' }))
  expect(screen.getByText('Current source instructions')).toBeTruthy()
  expect(screen.queryByText('Wrong owner instructions')).toBeNull()
  rerender(<WorkerPersonality entity={oldEntity} />)
  expect(screen.queryByText('Current source instructions')).toBeNull()
})
it('offers a scoped retry after failure and preserves absent source truth', async () => {
  vi.mocked(getProfileSoul).mockRejectedValueOnce(new Error('offline')).mockResolvedValueOnce({ exists: false, content: '' })
  render(<WorkerPersonality entity={worker('local')} />)
  fireEvent.click(screen.getByRole('button', { name: 'Load profile SOUL' }))
  fireEvent.click(await screen.findByRole('button', { name: 'Retry profile SOUL' }))
  expect(await screen.findByText('No SOUL file exists for this profile.')).toBeTruthy()
  expect(getProfileSoul).toHaveBeenLastCalledWith('same-name', { connectionId: 'local', profile: 'same-name' })
})
